# scheduler/constraints/hard_rules.py

import pandas as pd
import numpy as np
import re
from scheduler.types import PlanContext, ScheduleMatrix

"""
[Changes]
1. Shift Normalization: M->E, C->D, P->O applied before checking.
"""


def _get_normalized_df(sch: ScheduleMatrix):
    """
    제약 조건 계산을 위해 특수 근무를 표준 근무로 변환
    M -> E, C -> D, P -> O
    """
    if sch.df.empty: return sch.df
    # replace는 새로운 복사본을 반환함
    return sch.df.replace({'M': 'E', 'C': 'D', 'P': 'O'})


# 1. 근무 코드 유효성 (원본 코드로 검사 - 입력 자체가 유효한지)
def validate_shift_codes_vectorized(ctx: PlanContext, sch: ScheduleMatrix):
    # 허용된 모든 근무 코드
    valid_shifts = {"D", "E", "N", "O", "M", "C", "P"}
    df = sch.df
    if df.empty: return []

    invalid_mask = ~df.isin(valid_shifts)
    if invalid_mask.any().any():
        try:
            invalid_entries = df.where(invalid_mask).stack(future_stack=True)
        except TypeError:
            invalid_entries = df.where(invalid_mask).stack(dropna=False)

        return [
            ("INVALID_SHIFT_CODE", nid, d, f"Value: {val}")
            for (nid, d), val in invalid_entries.items()
        ]
    return []


# 2. 고정 근무(Lock) 유지 (원본 비교)
def validate_immutability_vectorized(ctx: PlanContext, sch: ScheduleMatrix):
    violations = []
    if not ctx.um_locks and not ctx.nurse_requests: return []

    try:
        sch_series = sch.df.stack(future_stack=True)
    except TypeError:
        sch_series = sch.df.stack(dropna=False)

    targets = [(ctx.um_locks, "UM_LOCK_VIOLATION"), (ctx.nurse_requests, "NURSE_REQUEST_VIOLATION")]
    for source_dict, code in targets:
        if not source_dict: continue
        constraint_series = pd.Series(source_dict)
        common_index = constraint_series.index.intersection(sch_series.index)
        if common_index.empty: continue

        constraint_subset = constraint_series.loc[common_index]
        sch_subset = sch_series.loc[common_index]

        diff_mask = sch_subset != constraint_subset
        if diff_mask.any():
            violated_items = constraint_subset[diff_mask]
            violations.extend(
                [(code, nid, d, f"Expected '{expected}' != Got '{sch_subset.get((nid, d))}'") for (nid, d), expected in
                 violated_items.items()])
    return violations


# --- 이하 함수들은 Normalized DF 사용 ---

# 3. 팀별 최소 인원
def validate_team_coverage_vectorized(ctx: PlanContext, sch: ScheduleMatrix):
    violations = []
    df = _get_normalized_df(sch)  # Normalized
    if df.empty: return []

    nurse_team_map = {n.id: n.team_id for n in ctx.nurses if n.team_id}
    if not nurse_team_map: return []

    df_with_team = df.copy()
    df_with_team["team_id"] = df.index.map(nurse_team_map)
    df_with_team = df_with_team.dropna(subset=["team_id"])

    for shift in ["D", "E", "N"]:
        shift_mask = (df_with_team.drop(columns=["team_id"]) == shift).astype(int)
        shift_mask["team_id"] = df_with_team["team_id"]
        team_counts = shift_mask.groupby("team_id").sum()

        zero_coverage = team_counts[team_counts == 0].stack()
        for (team_id, date), count in zero_coverage.items():
            violations.append(("TEAM_COVERAGE_FAIL", None, date, f"Team {team_id} has 0 '{shift}' staff"))
    return violations


# 4. 리더 배정
def validate_leader_requirement_vectorized(ctx: PlanContext, sch: ScheduleMatrix):
    violations = []
    df = _get_normalized_df(sch)  # Normalized
    valid_dates_set = set(df.columns)

    eligible_ids_map = {s: [n.id for n in ctx.nurses if (n.leader_eligible or {}).get(s, False) and n.id in df.index]
                        for s in ("D", "E", "N")}
    required_map = ctx.leader_required

    for s in ("D", "E", "N"):
        ids = eligible_ids_map[s]
        if ids:
            leader_counts = (df.loc[ids] == s).sum(axis=0)
        else:
            leader_counts = pd.Series(0, index=df.columns)

        dates_requiring_leader = [d for (d, sh), req in required_map.items() if
                                  sh == s and req and d in valid_dates_set]
        if not dates_requiring_leader: continue

        target_counts = leader_counts.reindex(dates_requiring_leader, fill_value=0)
        violated_dates = target_counts[target_counts == 0].index

        for d in violated_dates:
            violations.append(("LEADER_REQUIRED", None, d, f"Missing Leader for Shift '{s}'"))
    return violations


# 5. 특수 태그 검증
def validate_tags_vectorized(ctx: PlanContext, sch: ScheduleMatrix):
    violations = []
    df = _get_normalized_df(sch)  # Normalized (중요: 임산부가 M(->E)을 하는건 괜찮지만 N은 안됨)
    if df.empty: return []

    tag_rules = [
        ("pregnant", ["N"], "TAG_PREGNANT_N_BAN"),
        ("night_keep", ["D", "E"], "TAG_N_KEEP_BAN"),
        ("fixed_day", ["E", "N"], "TAG_D_KEEP_BAN"),
        ("fixed_evening", ["D", "N"], "TAG_E_KEEP_BAN"),
    ]

    for n in ctx.nurses:
        if n.id not in df.index: continue
        if not n.tags: continue
        row = df.loc[n.id]

        for tag, forbidden_shifts, code in tag_rules:
            if tag in n.tags:
                bad_mask = row.isin(forbidden_shifts)
                if bad_mask.any():
                    violated_dates = row.index[bad_mask]
                    for d in violated_dates:
                        violations.append((code, n.id, d, f"Tag '{tag}' forbids '{row[d]}'"))
    return violations


# 6. 금지 패턴 및 연속 근무
def validate_forbidden_sequences_vectorized(ctx: PlanContext, sch: ScheduleMatrix, strict: bool = True):
    violations = []
    df = _get_normalized_df(sch)  # Normalized
    if df.empty: return []

    df_prev = df.shift(1, axis=1)
    df_prev2 = df.shift(2, axis=1)

    mask_ed = (df_prev == 'E') & (df == 'D')
    mask_nd = (df_prev == 'N') & (df == 'D')
    mask_ne = (df_prev == 'N') & (df == 'E')
    mask_nod = (df_prev2 == 'N') & (df_prev == 'O') & (df == 'D')
    mask_non = (df_prev2 == 'N') & (df_prev == 'O') & (df == 'N')

    def extract_violations(mask, code, desc):
        if mask.any().any():
            try:
                stacked = mask.stack(future_stack=True)
            except TypeError:
                stacked = mask.stack()
            return [(code, nid, d, desc) for (nid, d), is_v in stacked.items() if is_v]
        return []

    violations.extend(extract_violations(mask_ed, "FORBID_ED", "Pattern E->D"))
    violations.extend(extract_violations(mask_nd, "FORBID_ND", "Pattern N->D"))
    violations.extend(extract_violations(mask_ne, "FORBID_NE", "Pattern N->E"))
    violations.extend(extract_violations(mask_nod, "FORBID_NOD", "Pattern N->O->D"))
    violations.extend(extract_violations(mask_non, "FORBID_NON", "Pattern N->O->N"))

    if strict:
        dates = df.columns
        regex_n_recovery = re.compile(r'N{2,}(?!OO)')
        for nid in df.index:
            row = df.loc[nid]
            safe_row = [str(x) if pd.notna(x) else "" for x in row]
            row_str = "".join(safe_row)
            for match in regex_n_recovery.finditer(row_str):
                end_idx = match.end() - 1
                if end_idx >= len(dates) - 2: continue
                if end_idx < len(dates):
                    violations.append(("N_RECOVERY_FAIL", nid, dates[end_idx], f"Consecutive N needs OO"))
            consec_work = 0
            consec_n = 0
            for c, s in enumerate(safe_row):
                current_date = dates[c]
                consec_n = (consec_n + 1) if s == 'N' else 0
                if consec_n >= 6: violations.append(("MAX_CONSEC_N", nid, current_date, f"Count: {consec_n}"))
                consec_work = (consec_work + 1) if s != 'O' else 0
                if consec_work >= 7: violations.append(("MAX_CONSEC_WORK", nid, current_date, f"Count: {consec_work}"))
    return violations


# 7. 프리셉터 (Normalized 불필요 - 매칭 여부만 확인하지만, 안전하게 Normalized 사용)
def validate_preceptor_pairing_pandas(ctx: PlanContext, sch: ScheduleMatrix):
    all_nurses = set(sch.df.index)
    pairs = {n.id: n.preceptor_id for n in ctx.nurses if
             n.preceptor_id and n.preceptor_id in all_nurses and n.id in all_nurses}
    if not pairs: return []

    df = _get_normalized_df(sch)  # M->E, C->D 변환 후 비교
    mentee_ids = list(pairs.keys())
    mentor_ids = [pairs[m] for m in mentee_ids]
    mentee_df = df.loc[mentee_ids]
    mentor_df = df.loc[mentor_ids]
    mentor_df.index = mentee_df.index

    mismatch_mask = mentee_df != mentor_df
    if mismatch_mask.any().any():
        try:
            stacked = mismatch_mask.stack(future_stack=True)
        except TypeError:
            stacked = mismatch_mask.stack()
        violations = []
        for (mentee_id, date) in stacked[stacked].index:
            mentor_id = pairs[mentee_id]
            s1 = mentee_df.at[mentee_id, date]
            s2 = mentor_df.at[mentee_id, date]
            violations.append(("PRECEPTOR_MISMATCH", mentee_id, date, f"{s1}!={s2}"))
        return violations
    return []


# 8. 월간 최대 횟수
def validate_max_shift_counts_vectorized(ctx: PlanContext, sch: ScheduleMatrix, strict: bool = True):
    if not strict: return []
    violations = []
    df = _get_normalized_df(sch)
    if df.empty: return []
    MAX_N_LIMIT = 15
    n_counts = (df == 'N').sum(axis=1)
    violators = n_counts[n_counts > MAX_N_LIMIT]
    for nurse_id, count in violators.items():
        violations.append(("MAX_MONTHLY_N_EXCEEDED", nurse_id, None, f"Count: {count}"))
    return violations


# 9. 전체 검증 통합
def validate_all(ctx: PlanContext, sch: ScheduleMatrix, strict: bool = True):
    if not sch.grid: return []

    df = sch.df
    if df.empty: return []

    # 채워진 비율이 10% 미만이면 검증 패스 (수동 입력 초기 단계로 간주)
    total_slots = df.size
    filled_slots = df.count().sum()  # non-NA count
    if total_slots > 0 and (filled_slots / total_slots) < 0.1:
        return []
    violations = []
    violations += validate_shift_codes_vectorized(ctx, sch)
    violations += validate_immutability_vectorized(ctx, sch)
    violations += validate_preceptor_pairing_pandas(ctx, sch)
    violations += validate_leader_requirement_vectorized(ctx, sch)
    violations += validate_tags_vectorized(ctx, sch)
    violations += validate_team_coverage_vectorized(ctx, sch)
    violations += validate_forbidden_sequences_vectorized(ctx, sch, strict=strict)
    violations += validate_max_shift_counts_vectorized(ctx, sch, strict=strict)
    return violations