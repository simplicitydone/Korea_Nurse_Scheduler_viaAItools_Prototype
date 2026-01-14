# scheduler/solver.py

from __future__ import annotations
from typing import List, Tuple
import random
import time

from scheduler.data_loader import load_plan_context
from scheduler.generator import generate_initial_schedule
from scheduler.evaluator import evaluate_schedule
from scheduler.local_search import improve
from scheduler.score_config import load_score_config
from scheduler.constraints.hard_rules import validate_all
from scheduler.db import persist_schedule, ScheduleCandidate
from scheduler.db import ExecutionLog
from scheduler.ml_tuner import MLTuner
from scheduler.ai_learner import PreferenceLearner
from scheduler.db import Schedule

# DeepTuner는 선택 사항일 수 있으므로 import 에러 방지 처리 (옵션)
try:
    from scheduler.deep_tuner import DeepTuner
except ImportError:
    DeepTuner = None


def generate_schedule(
        session,
        plan_id: int,
        score_config_dict: dict,
        *,
        n_candidates: int | None = None,
        top_k: int | None = None,
        ls_iterations: int | None = None,
        generator_mode: str = "heuristic",
        optimization_mode: str = "manual_default",
        local_search_mode: str = "random",
        surrogate_mode: str = "none",
        seed: int | None = None,
):
    ctx = load_plan_context(session, plan_id)
    cfg = load_score_config(score_config_dict)

    # Feature 추출 (공통)
    feat_nurses = len(ctx.nurses)
    feat_days = len(ctx.dates)
    feat_constraints = len(ctx.um_locks) + len(ctx.leader_required)

    if seed is None:
        seed = random.SystemRandom().randint(0, 2 ** 31 - 1)

    # ---------------------------------------------------------
    # 1. Parameter & Weight Tuning Strategy
    # ---------------------------------------------------------
    method_used = "manual"

    # 기본값 (High Quality 기준)
    final_cand = 50      # 기존 30 -> 50
    final_iter = 2000    # 기존 500 -> 2000
    final_top_k = 5      # 기존 3 -> 5

    # A. Manual Mode (Default)
    if optimization_mode == "manual_default":
        method_used = "manual"
        # 위에서 설정한 High Quality 기본값 사용


    # B. ML Mode (Scikit-Learn)
    elif optimization_mode == "ml":
        method_used = "ml"
        tuner = MLTuner(session)
        # 1. 파라미터 예측
        s_cand, s_k, s_iter = tuner.suggest_parameters(feat_nurses, feat_days, feat_constraints)
        final_cand, final_iter, final_top_k = s_cand, s_iter, s_k

        # 2. 가중치 추천 (통계 기반) -> Config Override
        w_suggestions = tuner.suggest_weights(ctx.ward_id)
        if w_suggestions:
            for rid, weight in w_suggestions.items():
                if rid in cfg.rules:
                    # [Weight Safety] 가중치가 너무 작거나 크면 Clamp
                    safe_weight = max(0.1, min(100.0, weight))
                    cfg.rules[rid].weight = safe_weight

    # C. PyTorch Mode (AutoML)
    elif optimization_mode == "pytorch":
        if DeepTuner is None:
            raise RuntimeError("DeepTuner not available (missing pytorch or file)")

        method_used = "pytorch"
        dtuner = DeepTuner(session)
        pred = dtuner.predict(feat_nurses, feat_days, feat_constraints)

        if pred:
            # 1. 파라미터 적용
            final_cand = pred['param_n_candidates']
            final_iter = pred['param_ls_iterations']
            final_top_k = pred['param_top_k']

            # 2. 가중치 적용 (Fairness 등)
            if 'den_balance' in pred:
                cfg.fairness_weights['den_balance'] = max(0.1, min(100.0, float(pred['den_balance'])))
            if 'off_balance' in pred:
                cfg.fairness_weights['off_balance'] = max(0.1, min(100.0, float(pred['off_balance'])))
        else:
            # 학습 데이터 부족 시 Fallback (Balanced와 동일)
            method_used = "pytorch_fallback"
            final_cand, final_iter, final_top_k = 30, 500, 3

    # ---------------------------------------------------------
    # 2. Apply User Overrides (UI Input Priority)
    # ---------------------------------------------------------
    # 사용자가 UI에서 직접 입력한 값이 있으면(Not None), 무조건 그 값을 최우선으로 사용합니다.
    if n_candidates is not None: final_cand = n_candidates
    if top_k is not None: final_top_k = top_k
    if ls_iterations is not None: final_iter = ls_iterations

    # [Performance Safety Guard] 파라미터 상한선 적용
    final_cand = min(200, max(10, final_cand))  # 최대 200개 제한
    final_iter = min(10000, max(100, final_iter))  # 최대 10,000회 제한
    final_top_k = min(20, max(1, final_top_k))  # 최대 20개 제한

    # ------------------------------------------------
    # Phase 1: Generation & Selection
    # ------------------------------------------------
    t0 = time.perf_counter()
    candidates: List[Tuple[float, object, int]] = []

    fallback_candidates = []

    for i in range(final_cand):
        cand_seed = seed + i
        # 생성 모드는 최적화 모드와 무관하게 'balanced' 수준의 안정성을 위해 heuristic 사용 권장
        sch = generate_initial_schedule(
            ctx,
            seed=cand_seed,
            mode=generator_mode,
            session=session
        )

        # 1. Strict Mode Check
        violations = validate_all(ctx, sch, strict=True)
        is_valid = len(violations) == 0

        # 2. Relax Mode Check (Strict 실패 시)
        if not is_valid:
            relax_violations = validate_all(ctx, sch, strict=False)
            if len(relax_violations) == 0:
                is_valid = True
            else:
                # Relax 모드조차 실패하면 Fallback에 저장
                temp_score_res = evaluate_schedule(ctx, sch, cfg)
                fallback_candidates.append((len(relax_violations), temp_score_res.total, sch, cand_seed))
                continue

        # 유효한 스케줄(Strict or Relax Pass)만 점수 계산 및 등록
        score_result = evaluate_schedule(ctx, sch, cfg)
        score = score_result.total

        cand_row = ScheduleCandidate(
            plan_id=ctx.plan_id,
            seed=cand_seed,
            score_total=float(score),
            score_breakdown={
                "total": score_result.total,
                "fairness": score_result.fairness,
                "by_rule": [
                    {"rule_id": rs.rule_id, "score": rs.score, "details": rs.details}
                    for rs in score_result.by_rule
                ],
            },
        )
        session.add(cand_row)
        session.flush()

        candidates.append((float(score), sch, cand_row.id))

    # [Fallback Logic] 후보가 하나도 없을 때
    if not candidates:
        if fallback_candidates:
            # 위반 개수가 적은 순, 그 다음 점수가 높은 순으로 정렬
            fallback_candidates.sort(key=lambda x: (x[0], -x[1]))

            # 상위 1개 선택
            best_fallback = fallback_candidates[0]
            v_count, f_score, f_sch, f_seed = best_fallback

            # DB에 저장
            cand_row = ScheduleCandidate(
                plan_id=ctx.plan_id,
                seed=f_seed,
                score_total=float(f_score),
                score_breakdown={"note": f"Fallback selected with {v_count} violations"}
            )
            session.add(cand_row)
            session.flush()

            candidates.append((float(f_score), f_sch, cand_row.id))
            print(f"Warning: No valid schedules. Using fallback with {v_count} violations.")
        else:
            raise RuntimeError("Generator failed completely. Check constraints.")

    # 상위 top_k개 선택
    candidates.sort(key=lambda x: x[0], reverse=True)
    shortlisted = candidates[:max(1, final_top_k)]

    # 랭킹 업데이트
    for rank, (_, _, cand_id) in enumerate(shortlisted, start=1):
        session.query(ScheduleCandidate).filter(
            ScheduleCandidate.id == cand_id
        ).update({"rank": rank})
    session.flush()

    t1 = time.perf_counter()
    gen_time = t1 - t0
    initial_best_score = shortlisted[0][0]

    # ------------------------------------------------
    # Phase 2: Improvement (Local Search)
    # ------------------------------------------------
    t2 = time.perf_counter()
    best_score, best_sch, best_cid = shortlisted[0]

    for base_score, base_sch, cand_id in shortlisted:
        improved_sch, improved_score = improve(
            ctx,
            base_sch,
            cfg,
            session=session,
            candidate_id=cand_id,
            seed=seed,
            iterations=final_iter,
            local_search_mode=local_search_mode,
            surrogate_mode=surrogate_mode
        )
        session.query(ScheduleCandidate).filter(
            ScheduleCandidate.id == cand_id
        ).update({"score_total": float(improved_score)})

        if improved_score > best_score:
            best_score, best_sch, best_cid = float(improved_score), improved_sch, cand_id

    session.commit()

    t3 = time.perf_counter()
    imp_time = t3 - t2
    final_score = best_score

    # ------------------------------------------------
    # Phase 3: DB Logging
    # ------------------------------------------------
    total_slots = sum(cn.min_required for cn in ctx.coverage_needs)

    log_entry = ExecutionLog(
        ward_id=ctx.ward_id,
        method=method_used,  # [New] 실행 방식 기록
        n_nurses=feat_nurses,
        n_days=feat_days,
        n_constraints=feat_constraints,
        total_slots_required=total_slots,

        # 실제 사용된 최종 파라미터 기록
        param_n_candidates=final_cand,
        param_top_k=final_top_k,
        param_ls_iterations=final_iter,

        score_initial_best=initial_best_score,
        score_final=final_score,
        time_generation=gen_time,
        time_improvement=imp_time,
        time_total=(gen_time + imp_time)
    )
    session.add(log_entry)
    session.commit()

    schedule_id = persist_schedule(session, ctx, best_sch, best_score, cfg)
    return schedule_id


# [New Function] UI에서 "최종 저장(확정)" 버튼을 누를 때 호출해야 함
def learn_and_finalize(session, plan_id: int, schedule_id: int):
    """
    사용자가 수정을 마친 최종 스케줄을 저장할 때 호출.
    1. Preference Learning (가중치 조정)
    2. Demand Data Logging (수요 데이터 수집)
    """
    from scheduler.data_loader import load_plan_context
    from scheduler.db import Schedule, ScoreConfigModel
    from scheduler.score_config import load_score_config

    # 1. 데이터 로드
    ctx = load_plan_context(session, plan_id)

    # 최종 스케줄 (User Modified)
    user_schedule_row = session.query(Schedule).get(schedule_id)
    if not user_schedule_row:
        return False

    user_sch = _row_to_matrix(user_schedule_row, ctx)

    # 2. AI Learner 동작
    learner = PreferenceLearner(session, ctx.ward_id)

    # Demand Log (수요 데이터 수집)
    learner.log_demand_deviation(ctx, user_sch)

    return True


def _row_to_matrix(schedule_row, ctx):
    from scheduler.types import ScheduleMatrix
    sch = ScheduleMatrix()
    for assign in schedule_row.assignments:
        sch.set_shift(assign.nurse_id, assign.date, assign.shift_code)
    return sch