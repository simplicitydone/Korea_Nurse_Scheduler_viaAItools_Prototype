from __future__ import annotations
from sqlalchemy.orm import Session
from datetime import date, timedelta # [Fix] timedelta 추가
import pandas as pd

from scheduler.db import ScoreConfigModel, ScoreWeightHistory, DemandLog
from scheduler.score_config import ScoreConfig, load_score_config
from scheduler.types import PlanContext, ScheduleMatrix
from scheduler.evaluator import evaluate_schedule


class PreferenceLearner:
    def __init__(self, session: Session, ward_id: str):
        self.session = session
        self.ward_id = ward_id

    def learn_from_feedback(self,
                            ctx: PlanContext,
                            ai_sch: ScheduleMatrix,
                            user_sch: ScheduleMatrix,
                            current_config: ScoreConfig):
        """
        AI 초안(ai_sch)과 사용자 최종본(user_sch)을 비교하여
        ScoreConfig의 가중치를 자동 조정하고 DB에 저장.
        """
        # 1. 두 스케줄의 점수 상세 내역 계산
        ai_eval = evaluate_schedule(ctx, ai_sch, current_config)
        user_eval = evaluate_schedule(ctx, user_sch, current_config)

        # 룰별 위반 횟수(또는 점수 감점량) 비교를 위한 딕셔너리
        ai_scores = {r.rule_id: r.score for r in ai_eval.by_rule}
        user_scores = {r.rule_id: r.score for r in user_eval.by_rule}

        updated_rules = []
        learning_rate = 0.1

        # 2. 규칙별 가중치 조정
        for rule_id, rule in current_config.rules.items():
            if not rule.enabled:
                continue

            s_ai = ai_scores.get(rule_id, 0.0)
            s_user = user_scores.get(rule_id, 0.0)

            # Case A: AI가 User보다 점수가 낮음 (AI가 더 많이 위반함) -> 가중치 증가
            if s_ai < s_user:
                delta = s_user - s_ai
                new_weight = rule.weight + (delta * learning_rate)
                new_weight = min(new_weight, 100.0)

                self._log_history(rule_id, rule.weight, new_weight, "User corrected AI violation")
                rule.weight = new_weight
                updated_rules.append(rule)

            # Case B: AI가 User보다 점수가 높음 (User가 규칙을 깸) -> 가중치 감소
            elif s_ai > s_user:
                delta = s_ai - s_user
                new_weight = rule.weight - (delta * learning_rate * 0.5)
                new_weight = max(new_weight, 0.1)

                self._log_history(rule_id, rule.weight, new_weight, "User relaxed constraint")
                rule.weight = new_weight
                updated_rules.append(rule)

        # 3. 변경된 Config 저장
        if updated_rules:
            self._save_new_config(current_config)

    def log_demand_deviation(self, ctx: PlanContext, final_sch: ScheduleMatrix):
        """
        시스템 제안 vs 실제 배정 차이를 상세 Context와 함께 로깅
        """
        df = final_sch.df
        if df.empty: return

        # 1. 날짜별 루프
        for d in ctx.dates:
            daily_counts = df[d].value_counts()

            # --- Feature Extraction ---
            wd = d.weekday()
            is_wknd = (wd >= 5)
            is_hol = ctx.is_holiday.get(d, False)
            hol_name = getattr(d, "holiday_name", None) if is_hol else None

            # 전날/다음날 휴일 여부 계산
            prev_d = d - timedelta(days=1)
            next_d = d + timedelta(days=1)
            is_before_hol = ctx.is_holiday.get(next_d, False)
            is_after_hol = ctx.is_holiday.get(prev_d, False)
            # --------------------------

            for shift in ["D", "E", "N"]:
                needed = next((cn.min_required for cn in ctx.coverage_needs
                               if cn.date == d and cn.shift == shift), 0)
                actual = daily_counts.get(shift, 0)

                log = DemandLog(
                    ward_id=self.ward_id,
                    plan_id=ctx.plan_id,
                    date=d,
                    shift_code=shift,
                    system_suggested=needed,
                    user_finalized=actual,
                    diff=actual - needed,
                    weekday=wd,
                    is_weekend=is_wknd,
                    is_holiday=is_hol,
                    holiday_name=hol_name,
                    is_day_before_holiday=is_before_hol,
                    is_day_after_holiday=is_after_hol
                )
                self.session.add(log)

        self.session.commit()

    def _log_history(self, rule_id, old, new, reason):
        h = ScoreWeightHistory(
            ward_id=self.ward_id,
            rule_id=rule_id,
            old_weight=old,
            new_weight=new,
            reason=reason
        )
        self.session.add(h)

    def _save_new_config(self, cfg: ScoreConfig):
        from scheduler.db import _score_config_to_dict
        cfg_dict = _score_config_to_dict(cfg)
        row = self.session.query(ScoreConfigModel).filter_by(ward_id=self.ward_id, active=True).first()
        if row:
            row.config_json = cfg_dict
            self.session.commit()