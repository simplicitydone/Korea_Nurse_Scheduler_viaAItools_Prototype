# scheduler/db.py

from __future__ import annotations

import os
from datetime import datetime, date
from typing import Optional, Dict, Any, List

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Date,
    DateTime,
    Boolean,
    Float,
    ForeignKey,
    UniqueConstraint,
    Text,
    Index,
    func,
    event,
)
from sqlalchemy.orm import (
    declarative_base,
    relationship,
    sessionmaker,
)

# SQLAlchemy 버전에 따른 JSON 타입 처리
try:
    from sqlalchemy import JSON
except ImportError:
    JSON = Text

# ---------------------------------------------------------
# Engine / Session
# ---------------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./nsp.db")

connect_args = {"check_same_thread": False}
if DATABASE_URL.startswith("sqlite"):
    connect_args["timeout"] = 60

# [Fix] 엔진을 먼저 생성합니다.
engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    future=True,
)

# [Fix] 엔진 생성 후 이벤트 리스너 등록 (순서 중요)
if DATABASE_URL.startswith("sqlite"):
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.close()

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, future=True)

Base = declarative_base()


def init_db():
    Base.metadata.create_all(bind=engine)


# ---------------------------------------------------------
# 1. Master Data (조직 구조)
# ---------------------------------------------------------
class Hospital(Base):
    __tablename__ = "hospitals"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False, unique=True)
    description = Column(String, nullable=True)
    wards = relationship("Ward", back_populates="hospital", cascade="all, delete-orphan")


class Ward(Base):
    __tablename__ = "wards"
    id = Column(String, primary_key=True)
    hospital_id = Column(Integer, ForeignKey("hospitals.id"), nullable=True)
    name = Column(String, nullable=False)
    timezone = Column(String, default="Asia/Seoul")
    active = Column(Boolean, default=True)

    hospital = relationship("Hospital", back_populates="wards")
    teams = relationship("Team", back_populates="ward")
    nurses = relationship("Nurse", back_populates="ward")
    holiday_calendar = relationship("HolidayCalendar", back_populates="ward")
    score_configs = relationship("ScoreConfigModel", back_populates="ward")

    __table_args__ = (Index("ix_ward_hospital", "hospital_id"),)


class Team(Base):
    __tablename__ = "teams"
    id = Column(Integer, primary_key=True, index=True)
    ward_id = Column(String, ForeignKey("wards.id"), nullable=False)
    code = Column(String, nullable=False)
    name = Column(String, nullable=False)
    active = Column(Boolean, default=True)

    ward = relationship("Ward", back_populates="teams")

    __table_args__ = (
        UniqueConstraint("ward_id", "code", name="uq_team_ward_code"),
        Index("ix_team_ward_active", "ward_id", "active"),
    )


class HolidayCalendar(Base):
    __tablename__ = "holiday_calendar"
    id = Column(Integer, primary_key=True, index=True)
    ward_id = Column(String, ForeignKey("wards.id"), nullable=False)
    date = Column(Date, nullable=False)
    name = Column(String, nullable=True)
    ward = relationship("Ward", back_populates="holiday_calendar")

    __table_args__ = (
        UniqueConstraint("ward_id", "date", name="uq_holiday_ward_date"),
        Index("ix_holiday_ward_date", "ward_id", "date"),
    )


# ---------------------------------------------------------
# 2. Personnel (인력 정보)
# ---------------------------------------------------------
class Nurse(Base):
    __tablename__ = "nurses"

    id = Column(Integer, primary_key=True, index=True)
    ward_id = Column(String, ForeignKey("wards.id"), nullable=False)
    employee_no = Column(String, nullable=False)
    name = Column(String, nullable=False)

    join_date = Column(Date, nullable=True)
    hire_year = Column(Integer, nullable=True)
    career_year = Column(Integer, nullable=True)

    team_id = Column(Integer, ForeignKey("teams.id"), nullable=True)
    active = Column(Boolean, default=True)

    level_weight = Column(Float, default=1.0)
    short_work_factor = Column(Float, default=1.0)

    # [New] Tier 1 & 2 Attributes
    tags = Column(JSON, default=list)  # ex: ["pregnant", "night_keep"]
    preferred_shifts = Column(JSON, default=list)  # ex: ["D", "E"]
    avoid_shifts = Column(JSON, default=list)  # ex: ["N"]

    # [Deprecated] Legacy fields (유지하되 사용 지양)
    pref_ratio = Column(JSON, default=dict)

    leader_eligible = Column(JSON, default=dict)

    preceptor_id = Column(Integer, ForeignKey("nurses.id"), nullable=True)
    preceptee_id = Column(Integer, ForeignKey("nurses.id"), nullable=True)
    is_preceptee = Column(Boolean, default=False)

    ward = relationship("Ward", back_populates="nurses")
    team = relationship("Team")
    preceptor = relationship("Nurse", remote_side=[id], foreign_keys=[preceptor_id], post_update=True, uselist=False)
    preceptee = relationship("Nurse", remote_side=[id], foreign_keys=[preceptee_id], post_update=True, uselist=False)
    avoids = relationship("NurseAvoid", back_populates="nurse", foreign_keys="NurseAvoid.nurse_id",
                          cascade="all, delete-orphan")

    __table_args__ = (
        UniqueConstraint("ward_id", "employee_no", name="uq_nurse_ward_employee_no"),
        Index("ix_nurse_lookup", "ward_id", "active"),
    )

    def _join_year_value(self) -> Optional[int]:
        if self.hire_year: return self.hire_year
        if self.join_date: return self.join_date.year
        return None

    def _career_year_value(self) -> int:
        if isinstance(self.career_year, int) and self.career_year > 0:
            return self.career_year
        jy = self._join_year_value()
        if jy:
            now = datetime.now().year
            return max(1, now - jy + 1)
        return 0

    def as_ui_dict(self, include_avoid_names: bool = False) -> Dict[str, Any]:
        team_str = self.team.code if self.team else None
        lvl = self.level_weight if self.level_weight is not None else 1.0
        swf = self.short_work_factor if self.short_work_factor is not None else 1.0

        d = {
            "id": self.id,
            "nurse_id": self.id,
            "ward_id": self.ward_id,
            "employee_no": self.employee_no,
            "staff_id": self.employee_no,
            "name": self.name,
            "join_year": self._join_year_value(),
            "years_exp": self._career_year_value(),
            "level": float(lvl),
            "level_weight": float(lvl),
            "short_work_factor": float(swf),
            "relative_work_days": float(swf),
            "leader_eligible": self.leader_eligible or {"D": False, "E": False, "N": False},
            "team_code": team_str,
            "team": team_str,
            "preceptor_id": self.preceptor_id,
            "preceptee_id": self.preceptee_id,
            "precept_partner": self.preceptor.name if self.preceptor else (
                self.preceptee.name if self.preceptee else ""),

            # [New] UI Data
            "tags": self.tags or [],
            "preferred_shifts": self.preferred_shifts or [],
            "avoid_shifts": self.avoid_shifts or []
        }

        le = d["leader_eligible"]
        d["leader_d"] = bool(le.get("D"))
        d["leader_e"] = bool(le.get("E"))
        d["leader_n"] = bool(le.get("N"))

        if include_avoid_names:
            names = []
            for a in self.avoids or []:
                if a.avoid_nurse and a.avoid_nurse.name:
                    names.append(a.avoid_nurse.name)
            d["avoid_names"] = names
            d["avoid_list"] = ", ".join(names)

        return d


class NurseAvoid(Base):
    __tablename__ = "nurse_avoids"
    id = Column(Integer, primary_key=True, index=True)
    ward_id = Column(String, ForeignKey("wards.id"), nullable=False)
    nurse_id = Column(Integer, ForeignKey("nurses.id"), nullable=False)
    avoid_nurse_id = Column(Integer, ForeignKey("nurses.id"), nullable=False)
    avoid_same_shift = Column(Boolean, default=True)
    avoid_same_team = Column(Boolean, default=True)
    nurse = relationship("Nurse", back_populates="avoids", foreign_keys=[nurse_id])
    avoid_nurse = relationship("Nurse", foreign_keys=[avoid_nurse_id])
    __table_args__ = (
        UniqueConstraint("ward_id", "nurse_id", "avoid_nurse_id", name="uq_avoid_pair"),
        Index("ix_avoid_ward_nurse", "ward_id", "nurse_id"),
    )


# ---------------------------------------------------------
# 3. Planning & AI Models
# ---------------------------------------------------------
class SchedulePlan(Base):
    __tablename__ = "schedule_plans"
    id = Column(Integer, primary_key=True, index=True)
    ward_id = Column(String, ForeignKey("wards.id"), nullable=False)
    start_date = Column(Date, nullable=False)
    end_date = Column(Date, nullable=False)
    status = Column(String, default="draft")
    created_by = Column(String, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    score_config_version = Column(String, nullable=True)
    coverage_requirements = relationship("CoverageRequirement", back_populates="plan", cascade="all, delete-orphan")
    leader_requirements = relationship("LeaderRequirement", back_populates="plan", cascade="all, delete-orphan")
    locks = relationship("ShiftLock", back_populates="plan", cascade="all, delete-orphan")
    __table_args__ = (Index("ix_plan_ward_date", "ward_id", "start_date"),)


class CoverageRequirement(Base):
    __tablename__ = "coverage_requirements"
    id = Column(Integer, primary_key=True, index=True)
    plan_id = Column(Integer, ForeignKey("schedule_plans.id"), nullable=False)
    date = Column(Date, nullable=False)
    shift_code = Column(String, nullable=False)
    min_required = Column(Integer, nullable=False)
    team_id = Column(Integer, ForeignKey("teams.id"), nullable=True)
    plan = relationship("SchedulePlan", back_populates="coverage_requirements")
    __table_args__ = (Index("ix_cov_plan_date_shift", "plan_id", "date", "shift_code"),)


class LeaderRequirement(Base):
    __tablename__ = "leader_requirements"
    id = Column(Integer, primary_key=True, index=True)
    plan_id = Column(Integer, ForeignKey("schedule_plans.id"), nullable=False)
    date = Column(Date, nullable=False)
    shift_code = Column(String, nullable=False)
    required = Column(Boolean, default=False)
    plan = relationship("SchedulePlan", back_populates="leader_requirements")
    __table_args__ = (Index("ix_leader_plan_date_shift", "plan_id", "date", "shift_code"),)


class ShiftLock(Base):
    __tablename__ = "shift_locks"
    id = Column(Integer, primary_key=True, index=True)
    plan_id = Column(Integer, ForeignKey("schedule_plans.id"), nullable=False)
    nurse_id = Column(Integer, ForeignKey("nurses.id"), nullable=False)
    date = Column(Date, nullable=False)
    shift_code = Column(String, nullable=False)
    lock_type = Column(String, nullable=False)
    created_by_role = Column(String, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    relaxable_in_local_search = Column(Boolean, default=False)
    plan = relationship("SchedulePlan", back_populates="locks")
    nurse = relationship("Nurse")
    __table_args__ = (
        Index("ix_lock_plan_nurse_date", "plan_id", "nurse_id", "date"),
        UniqueConstraint("plan_id", "nurse_id", "date", "lock_type", name="uq_lock_unique_per_type"),
    )


class ScoreConfigModel(Base):
    __tablename__ = "score_configs"
    id = Column(Integer, primary_key=True, index=True)
    ward_id = Column(String, ForeignKey("wards.id"), nullable=False)
    version_label = Column(String, nullable=False)
    config_json = Column(JSON, nullable=False)
    active = Column(Boolean, default=False)
    created_at = Column(DateTime, server_default=func.now())
    ward = relationship("Ward", back_populates="score_configs")
    __table_args__ = (
        UniqueConstraint("ward_id", "version_label", name="uq_scorecfg_ward_version"),
        Index("ix_scorecfg_ward_active", "ward_id", "active"),
    )


class ScheduleCandidate(Base):
    __tablename__ = "schedule_candidates"
    id = Column(Integer, primary_key=True, index=True)
    plan_id = Column(Integer, ForeignKey("schedule_plans.id"), nullable=False)
    seed = Column(Integer, nullable=False)
    score_total = Column(Float, default=0.0)
    score_breakdown = Column(JSON, nullable=True)
    rank = Column(Integer, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    __table_args__ = (Index("ix_cand_rank", "plan_id", "score_total"),)


class Schedule(Base):
    __tablename__ = "schedules"
    id = Column(Integer, primary_key=True, index=True)
    plan_id = Column(Integer, ForeignKey("schedule_plans.id"), nullable=False)
    engine_version = Column(String, default="v0.5.0")
    score_total = Column(Float, default=0.0)
    score_config_snapshot = Column(JSON, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    assignments = relationship("ScheduleAssignment", back_populates="schedule", cascade="all, delete-orphan")


class ScheduleAssignment(Base):
    __tablename__ = "schedule_assignments"
    id = Column(Integer, primary_key=True, index=True)
    schedule_id = Column(Integer, ForeignKey("schedules.id"), nullable=False)
    nurse_id = Column(Integer, ForeignKey("nurses.id"), nullable=False)
    date = Column(Date, nullable=False)
    shift_code = Column(String, nullable=False)
    is_um_lock_snapshot = Column(Boolean, default=False)
    is_nurse_request_snapshot = Column(Boolean, default=False)
    schedule = relationship("Schedule", back_populates="assignments")
    nurse = relationship("Nurse")
    __table_args__ = (
        UniqueConstraint("schedule_id", "nurse_id", "date", name="uq_assign_one_per_day"),
        Index("ix_assign_grid", "schedule_id", "date"),
    )


class LocalSearchLog(Base):
    __tablename__ = "local_search_logs"
    id = Column(Integer, primary_key=True, index=True)
    candidate_id = Column(Integer, ForeignKey("schedule_candidates.id"), nullable=True)
    operator = Column(String, nullable=False)
    delta_score = Column(Float, default=0.0)
    violations_count = Column(Integer, default=0)
    timestamp = Column(DateTime, server_default=func.now())
    __table_args__ = (Index("ix_lslog_candidate_time", "candidate_id", "timestamp"),)


class ExecutionLog(Base):
    __tablename__ = "execution_logs"
    id = Column(Integer, primary_key=True, index=True)
    ward_id = Column(String, nullable=False)
    method = Column(String, default="manual")
    n_nurses = Column(Integer)
    n_days = Column(Integer)
    n_constraints = Column(Integer)
    total_slots_required = Column(Integer, default=0)
    param_n_candidates = Column(Integer)
    param_top_k = Column(Integer)
    param_ls_iterations = Column(Integer)
    score_initial_best = Column(Float)
    score_final = Column(Float)
    time_generation = Column(Float)
    time_improvement = Column(Float)
    time_total = Column(Float)
    created_at = Column(DateTime, server_default=func.now())


class ScoreWeightHistory(Base):
    __tablename__ = "score_weight_history"
    id = Column(Integer, primary_key=True, index=True)
    ward_id = Column(String, nullable=False)
    rule_id = Column(String, nullable=False)
    old_weight = Column(Float, nullable=False)
    new_weight = Column(Float, nullable=False)
    reason = Column(String, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    __table_args__ = (Index("ix_weight_hist_ward", "ward_id"),)


class DemandLog(Base):
    __tablename__ = "demand_logs"
    id = Column(Integer, primary_key=True, index=True)
    ward_id = Column(String, nullable=False)
    plan_id = Column(Integer, nullable=False)
    date = Column(Date, nullable=False)
    shift_code = Column(String, nullable=False)
    system_suggested = Column(Integer, nullable=False)
    user_finalized = Column(Integer, nullable=False)
    diff = Column(Integer, nullable=False)
    weekday = Column(Integer, nullable=False)
    is_weekend = Column(Boolean, default=False)
    is_holiday = Column(Boolean, default=False)
    holiday_name = Column(String, nullable=True)
    is_day_before_holiday = Column(Boolean, default=False)
    is_day_after_holiday = Column(Boolean, default=False)
    created_at = Column(DateTime, server_default=func.now())
    __table_args__ = (Index("ix_demand_log_ward_date", "ward_id", "date"),)


class OperatorWeight(Base):
    __tablename__ = "operator_weights"
    id = Column(Integer, primary_key=True, index=True)
    ward_id = Column(String, nullable=False)
    operator_name = Column(String, nullable=False)
    weight = Column(Float, default=1.0)
    select_count = Column(Integer, default=0)
    success_count = Column(Integer, default=0)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    __table_args__ = (UniqueConstraint("ward_id", "operator_name", name="uq_op_weight_ward_name"),)


class ScheduleDailyNote(Base):
    """
    일별 특이사항/메모 (Daily Sheet용)
    """
    __tablename__ = "schedule_daily_notes"

    id = Column(Integer, primary_key=True, index=True)
    plan_id = Column(Integer, ForeignKey("schedule_plans.id"), nullable=False)
    date = Column(Date, nullable=False)
    note = Column(Text, nullable=True)

    # Audit용
    updated_by = Column(String, nullable=True)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    plan = relationship("SchedulePlan")

    __table_args__ = (
        UniqueConstraint("plan_id", "date", name="uq_daily_note_plan_date"),
        Index("ix_daily_note_plan_date", "plan_id", "date"),
    )


class LocalSearchExperience(Base):
    """
    [New] DQN 학습을 위한 (State, Action, Reward, Next_State) 로그 저장소
    """
    __tablename__ = "local_search_experiences"

    id = Column(Integer, primary_key=True, index=True)
    ward_id = Column(String, nullable=False)

    # State (Context)
    feature_progress = Column(Float)  # 진행률 (current_iter / max_iter)
    feature_violation_cnt = Column(Integer)  # 현재 하드 위반 수
    feature_score_norm = Column(Float)  # 정규화된 점수 (score / 1000 등)
    feature_fail_rate = Column(Float)  # 최근 10회 시도 중 실패율

    # Action
    action_index = Column(Integer)  # 선택한 연산자 인덱스

    # Reward
    reward = Column(Float)  # Delta Score

    created_at = Column(DateTime, server_default=func.now())


class EvaluationLog(Base):
    """
    [New] Surrogate Model 학습을 위한 (스케줄 특징, 실제 점수) 로그
    Local Search 중에 수집됨.
    """
    __tablename__ = "evaluation_logs"

    id = Column(Integer, primary_key=True, index=True)
    ward_id = Column(String, nullable=False)

    # MLP용 Hand-crafted Features (JSON으로 저장)
    # 예: [d_std, n_std, violation_count, ...]
    features = Column(JSON, nullable=False)

    # CNN용 Raw Data는 부피가 크므로, 별도 파일로 저장하거나
    # 여기서는 features JSON에 압축해서 넣는 방식을 선택 (초기 단계)

    # Target (Label)
    score = Column(Float, nullable=False)

    created_at = Column(DateTime, server_default=func.now())


# ---------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------
def _score_config_to_dict(cfg: Any) -> Dict[str, Any]:
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return cfg
    try:
        rules = []
        for r in cfg.rules.values():
            rules.append({
                "id": r.id,
                "enabled": r.enabled,
                "type": r.type,
                "weight": r.weight,
                "pattern": getattr(r, "pattern", None),
                "window": getattr(r, "window", None),
                "min_value": getattr(r, "min_value", None),
                "max_value": getattr(r, "max_value", None),
                "scope": getattr(r, "scope", "nurse"),
                "meta": getattr(r, "meta", {}) or {},
            })
        return {
            "version": getattr(cfg, "version", "v1"),
            "rules": rules,
            "fairness_weights": getattr(cfg, "fairness_weights", {}) or {},
        }
    except Exception:
        return {}


def persist_schedule(session, ctx, sch, total_score: float, cfg: Any) -> int:
    cfg_snapshot = _score_config_to_dict(cfg)
    schedule = Schedule(
        plan_id=ctx.plan_id,
        engine_version="v0.5.0",
        score_total=float(total_score),
        score_config_snapshot=cfg_snapshot,
    )
    session.add(schedule)
    session.flush()

    for n in ctx.nurses:
        for d in ctx.dates:
            shift = sch.get_shift(n.id, d)
            a = ScheduleAssignment(
                schedule_id=schedule.id,
                nurse_id=n.id,
                date=d,
                shift_code=shift,
                is_um_lock_snapshot=((n.id, d) in ctx.um_locks),
                is_nurse_request_snapshot=((n.id, d) in ctx.nurse_requests),
            )
            session.add(a)

    session.commit()
    return schedule.id


def cleanup_execution_logs(session, keep_limit=5000):
    total_count = session.query(ExecutionLog).count()
    if total_count < keep_limit:
        return
    recent_ids = [r.id for r in session.query(ExecutionLog.id).order_by(ExecutionLog.created_at.desc()).limit(1000)]
    best_ids = [r.id for r in session.query(ExecutionLog.id).order_by(ExecutionLog.score_final.desc()).limit(100)]
    keep_ids = set(recent_ids + best_ids)
    session.query(ExecutionLog).filter(ExecutionLog.id.notin_(keep_ids)).delete(synchronize_session=False)
    session.commit()


