from __future__ import annotations
from scheduler.types import PlanContext, ScheduleMatrix, ScoreResult, RuleScore
from scheduler.score_config import ScoreConfig

# [변경] 로직은 constraints/soft_rules에서 가져옴
from scheduler.constraints.soft_rules import (
    calculate_count_score,
    calculate_sequence_score,
    calculate_preference_score,
    calculate_fairness_score
)

def evaluate_schedule(ctx: PlanContext, sch: ScheduleMatrix, cfg: ScoreConfig) -> ScoreResult:
    by_rule = []
    total = 0.0

    # 1. 개별 규칙 평가
    for rule in cfg.rules.values():
        if not rule.enabled:
            continue

        if rule.type == "count":
            score, details = calculate_count_score(rule, ctx, sch)
        elif rule.type == "sequence":
            score, details = calculate_sequence_score(rule, ctx, sch)
        elif rule.type == "preference":
            score, details = calculate_preference_score(rule, ctx, sch)
        else:
            score, details = 0.0, {}

        by_rule.append(RuleScore(rule_id=rule.id, score=score, details=details))
        total += score

    # 2. 공정성(Fairness) 평가
    fairness_score, fairness_details = calculate_fairness_score(ctx, sch, cfg)
    total += fairness_score
    by_rule.append(RuleScore(rule_id="FAIRNESS_MIN_SET", score=fairness_score, details=fairness_details))

    return ScoreResult(total=total, by_rule=by_rule, fairness=fairness_details)