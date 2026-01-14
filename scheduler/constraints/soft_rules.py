# scheduler/constraints/soft_rules.py

from __future__ import annotations
from statistics import mean, pstdev
import pandas as pd
import numpy as np

from scheduler.types import PlanContext, ScheduleMatrix, RuleScore
from scheduler.score_config import ScoreRule, ScoreConfig

def _get_normalized_df(sch: ScheduleMatrix):
    if sch.df.empty: return sch.df
    return sch.df.replace({'M': 'E', 'C': 'D', 'P': 'O'})

def count_shifts_all_nurses(sch: ScheduleMatrix) -> pd.DataFrame:
    df = _get_normalized_df(sch) # Normalized
    if df.empty: return pd.DataFrame()
    return df.apply(pd.Series.value_counts, axis=1).fillna(0).astype(int)

def _zscore(values):
    if not values: return []
    values = [float(v) for v in values]
    m = mean(values)
    sd = pstdev(values) if len(values) > 1 else 0.0
    return [(v - m) / sd for v in values] if sd != 0 else [0.0] * len(values)


# ---------------------------------------------------------
# Tier 2 & 3 Rule Calculation Logic
# ---------------------------------------------------------

def calculate_count_score(rule: ScoreRule, ctx: PlanContext, sch: ScheduleMatrix) -> tuple[float, dict]:
    score = 0.0
    shift = (rule.meta or {}).get("shift")
    df = _get_normalized_df(sch)
    if shift and not df.empty:
        counts = (df == shift).sum(axis=1)
        for n in ctx.nurses:
            c = counts.get(n.id, 0)
            if rule.max_value is not None and c > rule.max_value:
                score -= (c - rule.max_value) * rule.weight
            if rule.min_value is not None and c < rule.min_value:
                score -= (rule.min_value - c) * rule.weight
    return score, {"kind": "count", "shift": shift}


def calculate_sequence_score(rule: ScoreRule, ctx: PlanContext, sch: ScheduleMatrix) -> tuple[float, dict]:
    score = 0.0
    details = {}
    mode = (rule.meta or {}).get("mode", "penalty")
    df = _get_normalized_df(sch)

    if df.empty: return 0.0, {}

    # DataFrame 행을 순회 (느릴 수 있으나 정확함)
    for n in ctx.nurses:
        if n.id not in df.index: continue
        seq_list = df.loc[n.id].tolist()

        if hasattr(rule, "sequence_occurrences"):
            occurrences = rule.sequence_occurrences(seq_list)
            details.setdefault("pattern", rule.pattern or "")
        else:
            pat = rule.pattern or ""
            occurrences = "".join(seq_list).count(pat) if pat else 0
            details.setdefault("pattern", pat)

        if occurrences:
            delta = occurrences * rule.weight
            score += delta if mode == "bonus" else -delta

    details["mode"] = mode
    return score, details


def calculate_preference_score(rule: ScoreRule, ctx: PlanContext, sch: ScheduleMatrix) -> tuple[float, dict]:
    score = 0.0
    bonus_weight = rule.weight
    penalty_weight = rule.weight * 2
    df = _get_normalized_df(sch)

    for n in ctx.nurses:
        if n.id not in df.index: continue
        row = df.loc[n.id]

        # 가점 (Preferred)
        for p_shift in n.preferred:
            score += (row == p_shift).sum() * bonus_weight

        # 감점 (Avoid)
        for a_shift in n.avoid:
            score -= (row == a_shift).sum() * penalty_weight

    return score, {"kind": "preference_tier2"}


def calculate_fairness_score(ctx: PlanContext, sch: ScheduleMatrix, cfg: ScoreConfig) -> tuple[float, dict]:
    # count_shifts_all_nurses가 이미 Normalized DF를 사용함
    counts_df = count_shifts_all_nurses(sch)
    workloads = []

    for n in ctx.nurses:
        if n.id in counts_df.index:
            row = counts_df.loc[n.id]
            d_c, e_c, n_c = row.get("D", 0), row.get("E", 0), row.get("N", 0)
        else:
            d_c = e_c = n_c = 0

        raw_load = d_c + e_c + (n_c * 1.5)
        factor = n.short_work_factor if n.short_work_factor > 0 else 1.0
        normalized_load = raw_load / factor
        workloads.append(normalized_load)

    load_std = pstdev(workloads) if workloads else 0.0
    w = cfg.fairness_weights or {}
    weight = w.get("workload_balance", 10.0)
    score = -(load_std * weight)

    fairness_details = {
        "workload_std": load_std,
        "avg_workload": mean(workloads) if workloads else 0.0,
        "min_workload": min(workloads) if workloads else 0.0,
        "max_workload": max(workloads) if workloads else 0.0
    }
    return score, fairness_details