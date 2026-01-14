# scheduler/main.py

from __future__ import annotations

from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, date, timedelta
from contextlib import asynccontextmanager
from pathlib import Path
from io import BytesIO
import json
import traceback
import calendar
import pandas as pd
import os

from fastapi import (
    FastAPI, Depends, HTTPException, Query, Path as FPath,
    Request, UploadFile, File, Body,
)
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, ConfigDict
from openpyxl import Workbook, load_workbook
from sqlalchemy.orm import joinedload, Session

# DB Components
from scheduler.db import (
    SessionLocal, init_db, ScoreConfigModel, Ward, SchedulePlan,
    ScheduleCandidate, Nurse, Team, NurseAvoid, Schedule, ScheduleDailyNote,
    ScheduleAssignment, ShiftLock, CoverageRequirement, HolidayCalendar, LeaderRequirement
)
from scheduler.solver import generate_schedule, learn_and_finalize
from scheduler.ml_tuner import MLTuner
from scheduler.ai_config_tuner import AIConfigTuner
from scheduler.data_loader import load_plan_context
from scheduler.constraints.hard_rules import validate_all
from scheduler.types import ScheduleMatrix
from scheduler.score_config import load_score_config
from scheduler.evaluator import evaluate_schedule
from scheduler.demand_models import DemandPredictor

BASE_DIR = Path(__file__).parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Excel Constants
NURSE_EXCEL_COLUMNS = [
    "사번", "이름", "입사년도", "연차", "팀", "업무레벨", "단축근무",
    "리더가능", "프리셉터(파트너)", "기피대상(이름)", "특수태그", "선호근무", "기피근무"
]

EXCEL_SAMPLE_DATA = [
    "S001", "김간호", 2020, 5, "A", 1.0, 1.0,
    "D,E", "박사수", "최신입", "pregnant", "D,E", "N"
]

EXCEL_DESC_DATA = [
    "필수", "필수", "선택(YYYY)", "자동계산", "선택", "기본1.0", "기본1.0",
    "콤마구분(D,E,N)", "이름", "이름(콤마)", "pregnant,night_keep 등", "D,E,N", "D,E,N"
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(lifespan=lifespan)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()



class ScheduleGenerateRequest(BaseModel):
    plan_id: int
    ward_id: Optional[str] = None
    optimization_mode: str = "manual_default"
    generator_mode: str = "heuristic"
    local_search_mode: str = "random"
    surrogate_mode: str = "none"
    n_candidates: Optional[int] = None
    iterations: Optional[int] = None
    top_k: Optional[int] = None
    seed: int = 0
    created_by: Optional[str] = None
    demand_mode: str = "rule"


class CreateEmptyPayload(BaseModel):
    month: str
    created_by: str


class ScheduleSavePayload(BaseModel):
    schedule: Dict[str, Any]
    record_history: bool = False
    changed_by: str
    notes: Optional[str] = None


class ScoreConfigPayload(BaseModel):
    version: str = "nsp"
    rules: List[Dict[str, Any]] = Field(default_factory=list)
    fairness_weights: Dict[str, float] = Field(default_factory=dict)


class ScoreConfigCreate(BaseModel):
    ward_id: str
    version_label: str
    config_json: ScoreConfigPayload
    active: bool = False


class ScoreConfigUpdate(BaseModel):
    version_label: Optional[str] = None
    config_json: Optional[ScoreConfigPayload] = None
    active: Optional[bool] = None


class ScoreConfigOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    ward_id: str
    version_label: str
    config_json: Dict[str, Any]
    active: bool
    created_at: Optional[datetime] = None


class ScheduleGenerateResponse(BaseModel):
    schedule_id: int
    used_score_config: Dict[str, Any]
    score: float = 0.0
    score_breakdown: Optional[Dict[str, Any]] = None


class NurseUpsertPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")
    employee_no: Optional[str] = None
    staff_id: Optional[str] = None
    name: str
    hire_year: Optional[int] = None
    join_year: Optional[int] = None
    career_year: Optional[int] = None
    years_exp: Optional[int] = None
    level_weight: Optional[float] = None
    level: Optional[float] = None
    short_work_factor: Optional[float] = None
    relative_work_days: Optional[float] = None
    team: Optional[str] = None
    team_code: Optional[str] = None
    leader_eligible: Optional[Dict[str, bool]] = None
    leader_d: Optional[bool] = None
    leader_e: Optional[bool] = None
    leader_n: Optional[bool] = None
    preceptor_id: Optional[int] = None
    precept_partner: Optional[str] = None
    preceptor_name: Optional[str] = None
    is_preceptee: Optional[bool] = None
    avoid_list: Optional[str] = None
    avoid_names: Optional[List[str]] = None
    tags: Optional[List[str]] = []
    preferred_shifts: Optional[List[str]] = []
    avoid_shifts: Optional[List[str]] = []


DEFAULT_SCORE_CONFIG_V1: Dict[str, Any] = {
    "version": "nsp_default",
    "rules": [
        {"id": "CONSEC_WORK_5_PEN", "enabled": True, "type": "sequence", "weight": 20.0, "pattern": "CONSEC_WORK",
         "scope": "nurse", "meta": {"kind": "consecutive_work", "threshold": 5}},
        {"id": "WORK_O_WORK_PEN", "enabled": True, "type": "sequence", "weight": 10.0, "pattern": "WOW",
         "scope": "nurse", "meta": {"kind": "sandwich_off"}},
        {"id": "N_COUNT_MAX_16", "enabled": True, "type": "count", "weight": 2.0, "max_value": 16, "scope": "nurse",
         "meta": {"shift": "N", "mode": "penalty"}},
    ],
    "fairness_weights": {"den_balance": 30.0, "off_balance": 20.0, "weekend_holiday_balance": 40.0},
}

DEFAULT_WARD_ID = "WARD1"


def _ensure_ward_exists(db, ward_id: str):
    ward = db.query(Ward).filter(Ward.id == ward_id).first()
    if not ward:
        raise HTTPException(status_code=404, detail=f"Ward not found: {ward_id}")
    return ward


def _deactivate_all_for_ward(db, ward_id: str):
    db.query(ScoreConfigModel).filter(
        ScoreConfigModel.ward_id == ward_id, ScoreConfigModel.active == True
    ).update({"active": False})


def _get_active_score_config_dict(db, ward_id: str) -> Dict[str, Any]:
    cfg = db.query(ScoreConfigModel).filter(
        ScoreConfigModel.ward_id == ward_id, ScoreConfigModel.active == True
    ).order_by(ScoreConfigModel.created_at.desc()).first()
    if cfg and isinstance(cfg.config_json, dict):
        return cfg.config_json
    return DEFAULT_SCORE_CONFIG_V1


def _get_or_create_ward(db, ward_id: str = DEFAULT_WARD_ID) -> Ward:
    ward = db.query(Ward).filter(Ward.id == ward_id).first()
    if not ward:
        ward = Ward(id=ward_id, name=ward_id)
        db.add(ward);
        db.commit();
        db.refresh(ward)
    return ward


def _get_or_create_team(db, ward: Ward, team_code: Optional[str]) -> Optional[Team]:
    if not team_code: return None
    team_code = team_code.strip()
    if not team_code: return None
    team = db.query(Team).filter(Team.ward_id == ward.id, Team.code == team_code).first()
    if team: return team
    team = Team(ward_id=ward.id, code=team_code, name=team_code, active=True)
    db.add(team);
    db.commit();
    db.refresh(team)
    return team


def _normalize_leader_eligible(payload: Dict[str, Any]) -> Dict[str, bool]:
    base = {"D": False, "E": False, "N": False, "O": False}
    le = payload.get("leader_eligible") or {}
    for k in ("D", "E", "N", "O"):
        if k in le: base[k] = bool(le[k])
    if "leader_d" in payload: base["D"] = bool(payload["leader_d"])
    if "leader_e" in payload: base["E"] = bool(payload["leader_e"])
    if "leader_n" in payload: base["N"] = bool(payload["leader_n"])
    return base


def _parse_label_to_employee_or_name(label: str) -> Tuple[Optional[str], str]:
    raw = (label or "").strip()
    if not raw: return None, ""
    if "(" in raw and ")" in raw:
        inside = raw[raw.find("(") + 1: raw.rfind(")")]
        emp = inside.strip() or None
        name_part = raw[: raw.find("(")].strip()
        return emp, name_part
    if "/" in raw:
        parts = [p.strip() for p in raw.split("/") if p.strip()]
        if len(parts) == 2:
            p1, p2 = parts
            if any(ch.isdigit() for ch in p2): return p2, p1
            if any(ch.isdigit() for ch in p1): return p1, p2
    if any(ch.isdigit() for ch in raw): return raw, ""
    return None, raw


def _find_nurse_by_label(db, ward: Ward, label: str) -> Optional[Nurse]:
    emp, name_candidate = _parse_label_to_employee_or_name(label)
    q = db.query(Nurse).filter(Nurse.ward_id == ward.id)
    if emp:
        n = q.filter(Nurse.employee_no == emp).first()
        if n: return n
    if name_candidate:
        n = q.filter(Nurse.name == name_candidate).first()
        if n: return n
    raw = (label or "").strip()
    if raw:
        n = q.filter(Nurse.name == raw).first()
        if n: return n
    return None


def _validate_relationship_conflict(db, ward: Ward, payload: Dict[str, Any], current_nurse_id: Optional[int] = None):
    partner_id = payload.get("preceptor_id")
    partner_name = payload.get("precept_partner") or payload.get("preceptor_name")
    target_partner_id = None
    if partner_id:
        target_partner_id = int(partner_id)
    elif partner_name:
        p_obj = _find_nurse_by_label(db, ward, partner_name)
        if p_obj: target_partner_id = p_obj.id

    if not target_partner_id: return

    avoid_names = _parse_avoid_names_from_payload(payload)
    if not avoid_names: return

    avoid_ids = set()
    for name in avoid_names:
        a_obj = _find_nurse_by_label(db, ward, name)
        if a_obj: avoid_ids.add(a_obj.id)

    if target_partner_id in avoid_ids:
        if current_nurse_id and target_partner_id == current_nurse_id: return
        raise HTTPException(400,
                            f"Conflict: Nurse ID {target_partner_id} cannot be both a partner and an avoid target.")


def _sync_preceptor_relation(db, ward: Ward, nurse: Nurse, payload: Dict[str, Any]):
    if nurse.preceptor_id:
        old = db.query(Nurse).filter(Nurse.id == nurse.preceptor_id).first()
        if old: old.preceptee_id = None
        nurse.preceptor_id = None;
        nurse.is_preceptee = False;
        nurse.level_weight = 0.7
    if nurse.preceptee_id:
        old = db.query(Nurse).filter(Nurse.id == nurse.preceptee_id).first()
        if old: old.preceptor_id = None; old.is_preceptee = False; old.level_weight = 0.7
        nurse.preceptee_id = None

    preceptor_id = payload.get("preceptor_id")
    partner_name = payload.get("precept_partner") or payload.get("preceptor_name")
    target_nurse: Optional[Nurse] = None
    if preceptor_id:
        target_nurse = db.query(Nurse).filter(Nurse.ward_id == ward.id, Nurse.id == int(preceptor_id)).first()
    elif partner_name:
        target_nurse = _find_nurse_by_label(db, ward, partner_name)

    if not target_nurse or target_nurse.id == nurse.id: return

    my_exp = nurse.career_year or 0
    target_exp = target_nurse.career_year or 0

    if my_exp <= target_exp:
        nurse.preceptor_id = target_nurse.id;
        nurse.is_preceptee = True;
        nurse.level_weight = 0.0
        target_nurse.preceptee_id = nurse.id;
        target_nurse.is_preceptee = False
        if target_nurse.level_weight < 1.0: target_nurse.level_weight = 1.0
    else:
        nurse.preceptee_id = target_nurse.id;
        nurse.is_preceptee = False
        if nurse.level_weight < 1.0: nurse.level_weight = 1.0
        target_nurse.preceptor_id = nurse.id;
        target_nurse.is_preceptee = True;
        target_nurse.level_weight = 0.0


def _parse_avoid_names_from_payload(payload: Dict[str, Any]) -> List[str]:
    names: List[str] = []
    avoid_names = payload.get("avoid_names") or []
    if isinstance(avoid_names, list):
        for v in avoid_names:
            if isinstance(v, str) and v.strip(): names.append(v.strip())
    avoid_list = payload.get("avoid_list")
    if isinstance(avoid_list, str) and avoid_list.strip():
        raw = avoid_list.replace("/", ",")
        for part in raw.split(","):
            p = part.strip()
            if p: names.append(p)
    return list(set(names))


def _sync_avoid_relations(db, ward: Ward, nurse: Nurse, payload: Dict[str, Any]):
    db.query(NurseAvoid).filter(NurseAvoid.ward_id == ward.id, NurseAvoid.nurse_id == nurse.id).delete(
        synchronize_session=False)
    names = _parse_avoid_names_from_payload(payload)
    if not names: return
    for label in names:
        other = _find_nurse_by_label(db, ward, label)
        if not other or other.id == nurse.id: continue
        na = NurseAvoid(ward_id=ward.id, nurse_id=nurse.id, avoid_nurse_id=other.id, avoid_same_shift=True,
                        avoid_same_team=True)
        db.add(na)


# ---------------------------
# API Routes: Basic Info
# ---------------------------
@app.get("/api/ward")
def get_ward_info(ward_id: str = Query(DEFAULT_WARD_ID), db=Depends(get_db)):
    ward = _get_or_create_ward(db, ward_id)
    return {"id": ward.id, "name": ward.name, "ward_name": ward.name}


# ---------------------------
# API Routes: ML/Deep Tuner
# ---------------------------
@app.get("/api/ml/status")
def get_ml_status_api(db=Depends(get_db)):
    tuner = MLTuner(db)
    return tuner.get_training_status()


@app.post("/api/deep/train")
def train_deep_model_api(db=Depends(get_db)):
    try:
        from scheduler.deep_tuner import DeepTuner
        tuner = DeepTuner(db)
        tuner.ensure_data()
        success = tuner.train()
        return {"success": success, "message": "Deep Model 학습 완료" if success else "실패"}
    except ImportError:
        return {"success": False, "message": "PyTorch 모듈이 없습니다."}


@app.post("/api/ml/train")
def train_ml_model_api(db=Depends(get_db)):
    tuner = MLTuner(db)
    tuner.ensure_data_for_training()
    success = tuner.train()
    return {"success": success, "message": "ML Model 학습 완료"}


@app.post("/api/demand/apply")
def apply_demand_forecast_api(
        plan_id: int = Body(..., embed=True),
        mode: str = Body(..., embed=True),
        team_count: int = Body(..., embed=True),
        db: Session = Depends(get_db)
):
    """
    선택된 AI 모델(mode)을 사용하여 해당 Plan의 전체 기간에 대한
    CoverageRequirement와 ScheduleDailyNote를 생성/갱신합니다.
    """
    plan = db.query(SchedulePlan).get(plan_id)
    if not plan: raise HTTPException(404, "Plan not found")

    # 1. Baseline Calculation (User Requirement: D,E=T+1, N=T)
    base_d = team_count + 1
    base_e = team_count + 1
    base_n = team_count
    base_needs = {'D': base_d, 'E': base_e, 'N': base_n}

    # 2. Holiday Info
    holidays = db.query(HolidayCalendar).filter(
        HolidayCalendar.ward_id == plan.ward_id,
        HolidayCalendar.date >= plan.start_date,
        HolidayCalendar.date <= plan.end_date
    ).all()
    holiday_dates = {h.date for h in holidays}

    # 3. Predictor Init
    predictor = DemandPredictor(mode)

    # 4. Loop Dates
    curr = plan.start_date

    # 기존 데이터 클리어 (Coverage & Notes)
    db.query(CoverageRequirement).filter(CoverageRequirement.plan_id == plan.id).delete()
    db.query(ScheduleDailyNote).filter(ScheduleDailyNote.plan_id == plan.id).delete()

    while curr <= plan.end_date:
        # Predict
        adjusted, note_text = predictor.predict(curr, holiday_dates, base_needs)

        # Save Coverage
        for s, count in adjusted.items():
            db.add(CoverageRequirement(
                plan_id=plan.id, date=curr, shift_code=s, min_required=count
            ))
            # Leader Requirement 자동 추가
            db.add(LeaderRequirement(
                plan_id=plan.id, date=curr, shift_code=s, required=True
            ))

        # Save Note (if any)
        if note_text:
            db.add(ScheduleDailyNote(
                plan_id=plan.id, date=curr, note=note_text, updated_by="AI_Forecaster"
            ))

        curr += timedelta(days=1)

    db.commit()
    return {"status": "ok", "message": f"{mode.upper()} 기반 인력 배치 완료"}

# ---------------------------
# API Routes: Schedule
# ---------------------------
@app.get("/api/schedules/by_month")
def get_schedule_by_month(
        month: str = Query(..., pattern=r"^\d{4}-\d{2}$"),
        db=Depends(get_db)
):
    y, m = map(int, month.split("-"))
    start_dt = date(y, m, 1)

    plan = db.query(SchedulePlan).filter(SchedulePlan.start_date == start_dt).first()
    if not plan:
        return {"exists": False}

    schedule = (
        db.query(Schedule)
        .filter(Schedule.plan_id == plan.id)
        .order_by(Schedule.created_at.desc())
        .first()
    )

    if not schedule:
        return {"exists": False}

    grid = {}
    assignments = db.query(ScheduleAssignment).filter(ScheduleAssignment.schedule_id == schedule.id).all()
    for a in assignments:
        d_str = a.date.isoformat()
        if d_str not in grid: grid[d_str] = {}
        if a.shift_code not in grid[d_str]: grid[d_str][a.shift_code] = []
        grid[d_str][a.shift_code].append(a.nurse_id)

    locks = {}
    db_locks = db.query(ShiftLock).filter(ShiftLock.plan_id == plan.id).all()
    for l in db_locks:
        d_str = l.date.isoformat()
        if d_str not in locks: locks[d_str] = {}
        locks[d_str][str(l.nurse_id)] = True

    config_snapshot = schedule.score_config_snapshot or {}

    # Calculate Breakdown (Score Details & Violations)
    score_breakdown = {}
    try:
        ctx = load_plan_context(db, plan.id)
        cfg = load_score_config(config_snapshot)

        # ScheduleMatrix 복원
        sch_matrix = ScheduleMatrix()
        for a in assignments:
            sch_matrix.set_shift(a.nurse_id, a.date, a.shift_code)

        # Soft Rule 평가
        eval_res = evaluate_schedule(ctx, sch_matrix, cfg)

        # Hard Rule 검증
        hard_violations = validate_all(ctx, sch_matrix, strict=True)

        # Breakdown 통합
        by_rule = []
        # Hard Rule 추가
        for code, nid, v_date, desc in hard_violations:
            n_name = next((n.name for n in ctx.nurses if n.id == nid), "Unknown")
            d_str = v_date.strftime("%Y-%m-%d") if v_date else "All"
            by_rule.append({
                "rule_id": f"HARD_{code}",
                "score": -999.0,
                "details": {"desc": desc, "nurse_id": nid, "date": d_str}
            })

        # Soft Rule 추가
        for r in eval_res.by_rule:
            by_rule.append({"rule_id": r.rule_id, "score": r.score, "details": r.details})

        score_breakdown = {
            "total": eval_res.total,
            "fairness": eval_res.fairness,
            "by_rule": by_rule
        }

    except Exception as e:
        print(f"Error calculating breakdown on load: {e}")
        traceback.print_exc()

    return {
        "exists": True,
        "schedule_id": schedule.id,
        "plan_id": plan.id,
        "schedule": {
            "id": schedule.id,
            "version": str(schedule.id),
            "generated_at": schedule.created_at.isoformat(),
            "score_total": schedule.score_total,
            "score_breakdown": score_breakdown,
            "grid": grid,
            "config": config_snapshot,
            "locks": locks
        }
    }


@app.post("/api/schedules/create_empty")
def create_empty_schedule(payload: CreateEmptyPayload, db=Depends(get_db)):
    y, m = map(int, payload.month.split("-"))
    import calendar
    start_dt = date(y, m, 1)
    last_day = calendar.monthrange(y, m)[1]
    end_dt = date(y, m, last_day)

    plan = db.query(SchedulePlan).filter(SchedulePlan.start_date == start_dt).first()
    if not plan:
        plan = SchedulePlan(
            ward_id=DEFAULT_WARD_ID,
            start_date=start_dt,
            end_date=end_dt,
            status="draft",
            created_by=payload.created_by
        )
        db.add(plan)
        db.commit()
        db.refresh(plan)

    new_schedule = Schedule(plan_id=plan.id, engine_version="nsp", score_total=0.0)
    db.add(new_schedule)
    db.commit()
    db.refresh(new_schedule)

    return {
        "schedule_id": new_schedule.id,
        "plan_id": plan.id,
        "schedule": {"id": new_schedule.id, "grid": {}, "config": {}, "locks": {}}
    }


@app.post("/api/schedules/generate", response_model=ScheduleGenerateResponse)
def generate_schedule_api(payload: ScheduleGenerateRequest, db=Depends(get_db)):
    plan = db.query(SchedulePlan).filter(SchedulePlan.id == payload.plan_id).first()
    if not plan: raise HTTPException(404, "Plan not found")
    ward_id = payload.ward_id or plan.ward_id
    _ensure_ward_exists(db, ward_id)

    # 1. 활성 Config
    score_cfg_dict = _get_active_score_config_dict(db, ward_id)

    # 2. AI Tuner (저장된 User Intention 기반 자동 튜닝)
    last_schedule = db.query(Schedule).filter(Schedule.plan_id == plan.id).order_by(Schedule.created_at.desc()).first()
    user_intention = {}
    if last_schedule and last_schedule.score_config_snapshot:
        user_intention = last_schedule.score_config_snapshot.get("user_intention", {})

    if user_intention:
        from scheduler.ai_config_tuner import AIConfigTuner
        tuner = AIConfigTuner()

        # 의도에 맞는 가중치 추천
        recommended_weights = tuner.tune_weights_by_intention(user_intention)

        # Config에 반영
        if "fairness_weights" not in score_cfg_dict: score_cfg_dict["fairness_weights"] = {}
        if "w_workload_balance" in recommended_weights:
            score_cfg_dict["fairness_weights"]["workload_balance"] = recommended_weights["w_workload_balance"]

        for rule in score_cfg_dict.get("rules", []):
            if rule.get("type") == "preference":
                if "w_preference" in recommended_weights:
                    rule["weight"] = recommended_weights["w_preference"]

    try:
        schedule_id = generate_schedule(
            session=db,
            plan_id=payload.plan_id,
            score_config_dict=score_cfg_dict,
            n_candidates=payload.n_candidates,
            top_k=payload.top_k,
            ls_iterations=payload.iterations,

            # 모드 전달
            optimization_mode=payload.optimization_mode,  # 튜닝
            generator_mode=payload.generator_mode,  # 생성
            local_search_mode=payload.local_search_mode,  # [New] 탐색
            surrogate_mode=payload.surrogate_mode,
            seed=payload.seed
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Solver Error: {str(e)}")

    created = db.query(Schedule).filter(Schedule.id == schedule_id).first()
    if not created:
        raise HTTPException(500, "Schedule created but not found in DB")

    # [Fix] Calculate Full Breakdown for immediate dashboard display
    score_breakdown = {}
    try:
        ctx = load_plan_context(db, plan.id)
        cfg = load_score_config(score_cfg_dict)

        sch_matrix = ScheduleMatrix()
        assigns = db.query(ScheduleAssignment).filter(ScheduleAssignment.schedule_id == schedule_id).all()
        for a in assigns:
            sch_matrix.set_shift(a.nurse_id, a.date, a.shift_code)

        eval_res = evaluate_schedule(ctx, sch_matrix, cfg)
        hard_violations = validate_all(ctx, sch_matrix, strict=True)

        by_rule = []
        for code, nid, v_date, desc in hard_violations:
            n_name = next((n.name for n in ctx.nurses if n.id == nid), "Unknown")
            d_str = v_date.strftime("%Y-%m-%d") if v_date else "All"
            by_rule.append({
                "rule_id": f"HARD_{code}",
                "score": -999.0,
                "details": {"desc": desc, "nurse_name": n_name, "date": d_str}
            })

        for r in eval_res.by_rule:
            by_rule.append({"rule_id": r.rule_id, "score": r.score, "details": r.details})

        score_breakdown = {
            "total": eval_res.total,
            "fairness": eval_res.fairness,
            "by_rule": by_rule
        }

        created.score_breakdown = score_breakdown
        db.commit()

    except Exception as e:
        print(f"Error calculating breakdown: {e}")

    return ScheduleGenerateResponse(
        schedule_id=schedule_id,
        used_score_config=score_cfg_dict,
        score=created.score_total,
        score_breakdown=score_breakdown  # [Fix] Return detail
    )


@app.post("/api/schedules/{schedule_id}/save")
def save_schedule_endpoint(schedule_id: int, payload: ScheduleSavePayload, db=Depends(get_db)):
    try:
        curr_sched = db.query(Schedule).filter(Schedule.id == schedule_id).first()
        if not curr_sched:
            raise HTTPException(404, "Schedule not found")

        plan_id = curr_sched.plan_id
        data = payload.schedule
        grid = data.get("grid") or {}
        locks = data.get("locks") or {}
        config = data.get("config") or {}

        # 1. Update DB CoverageRequirements (and LeaderRequirements)
        if config and payload.changed_by != "nurse":
            shift_needs = config.get("shift_needs") or {}
            curr_sched.score_config_snapshot = config

            # Clear Coverage
            db.query(CoverageRequirement).filter(CoverageRequirement.plan_id == plan_id).delete(
                synchronize_session=False)

            # [Fix] Clear Leader Requirement
            db.query(LeaderRequirement).filter(LeaderRequirement.plan_id == plan_id).delete(synchronize_session=False)

            plan = db.query(SchedulePlan).filter(SchedulePlan.id == plan_id).first()
            start_date = plan.start_date
            end_date = plan.end_date

            holidays = db.query(HolidayCalendar).filter(
                HolidayCalendar.ward_id == plan.ward_id,
                HolidayCalendar.date >= start_date,
                HolidayCalendar.date <= end_date
            ).all()
            holiday_dates = {h.date for h in holidays}

            curr = start_date

            while curr <= end_date:
                is_hol = (curr in holiday_dates)
                wd = curr.weekday()
                day_type = "weekday"
                if is_hol:
                    day_type = "holiday"
                elif wd == 5:
                    day_type = "saturday"
                elif wd == 6:
                    day_type = "sunday"

                needs_map = shift_needs.get(day_type, {})
                for shift, count in needs_map.items():
                    if count > 0:
                        # Coverage 추가
                        db.add(CoverageRequirement(
                            plan_id=plan_id, date=curr, shift_code=shift, min_required=int(count)
                        ))
                        # [Fix] Leader Requirement 추가 (D, E, N이면 필수)
                        if shift in ["D", "E", "N"]:
                            db.add(LeaderRequirement(
                                plan_id=plan_id, date=curr, shift_code=shift, required=True
                            ))
                curr += timedelta(days=1)

            # db.add_all 대신 루프 내 add 사용 (가독성 위해)

        # 2. Update Assignments
        db.query(ScheduleAssignment).filter(ScheduleAssignment.schedule_id == schedule_id).delete(
            synchronize_session=False)

        new_assigns = []
        assigned_map = {}

        for d_str, shifts in grid.items():
            if not shifts: continue
            try:
                curr_date = datetime.strptime(d_str, "%Y-%m-%d").date()
            except ValueError:
                continue

            for shift_code, nurse_ids in shifts.items():
                if not nurse_ids: continue
                for nid in nurse_ids:
                    if not nid: continue
                    try:
                        nid_int = int(nid)
                    except:
                        continue

                    # [Fix] 맵에 기록
                    assigned_map[(d_str, nid_int)] = shift_code

                    is_locked = False
                    if d_str in locks and locks[d_str]:
                        if str(nid) in locks[d_str] and locks[d_str][str(nid)]:
                            is_locked = True

                    is_request = (payload.changed_by == "nurse")
                    new_assigns.append(ScheduleAssignment(
                        schedule_id=schedule_id, nurse_id=nid_int, date=curr_date,
                        shift_code=shift_code, is_um_lock_snapshot=is_locked,
                        is_nurse_request_snapshot=is_request
                    ))
        if new_assigns: db.add_all(new_assigns)

        # 3. Update Locks
        if payload.changed_by != "nurse":
            db.query(ShiftLock).filter(ShiftLock.plan_id == plan_id).delete(synchronize_session=False)
            new_locks = []
            for d_str, nurse_map in locks.items():
                if not nurse_map: continue
                try:
                    curr_date = datetime.strptime(d_str, "%Y-%m-%d").date()
                except ValueError:
                    continue

                for nid_str, locked in nurse_map.items():
                    if locked:
                        try:
                            nid_int = int(nid_str)
                            # [Fix] "FIX" 대신 실제 배정된 근무 코드를 가져옴
                            target_shift = assigned_map.get((d_str, nid_int))

                            # 배정된 근무가 있을 때만 Lock 생성 (혹은 기본값 처리)
                            if target_shift:
                                new_locks.append(ShiftLock(
                                    plan_id=plan_id, nurse_id=nid_int, date=curr_date,
                                    shift_code=target_shift,  # <-- 수정됨 (기존 "FIX")
                                    lock_type="UM", created_by_role="admin"
                                ))
                        except:
                            continue
            if new_locks: db.add_all(new_locks)

        db.commit()

        # [Fix] Re-evaluate
        ctx = load_plan_context(db, plan_id)
        cfg = load_score_config(config)

        sch = ScheduleMatrix()
        for (d_str, nid), s_code in assigned_map.items():
            try:
                d_date = datetime.strptime(d_str, "%Y-%m-%d").date()
            except:
                continue
            sch.set_shift(nid, d_date, s_code)

        eval_res = evaluate_schedule(ctx, sch, cfg)
        hard_violations = validate_all(ctx, sch, strict=True)

        violation_msgs = []
        for code, nid, v_date, desc in hard_violations:
            n_name = next((n.name for n in ctx.nurses if n.id == nid), "Unknown")
            d_str = v_date.strftime("%Y-%m-%d") if v_date else "All"
            violation_msgs.append(f"[{code}] {n_name}({d_str}): {desc}")

        # Trigger Learning
        if payload.changed_by != "nurse":
            try:
                learn_and_finalize(db, plan_id, schedule_id)
            except:
                pass

        # Build Breakdown for Dashboard
        by_rule = []
        for code, nid, v_date, desc in hard_violations:
            n_name = next((n.name for n in ctx.nurses if n.id == nid), "Unknown")
            d_str = v_date.strftime("%Y-%m-%d") if v_date else "All"
            by_rule.append({
                "rule_id": f"HARD_{code}",
                "score": -999.0,
                "details": {"desc": desc, "nurse_name": n_name, "date": d_str}
            })

        for r in eval_res.by_rule:
            by_rule.append({"rule_id": r.rule_id, "score": r.score, "details": r.details})

        score_breakdown = {
            "total": eval_res.total,
            "fairness": eval_res.fairness,
            "by_rule": by_rule
        }

        return {
            "ok": True,
            "version": str(schedule_id),
            "violations": violation_msgs,
            "score_total": eval_res.total,
            "score_breakdown": score_breakdown  # [New] Return Full Detail
        }

    except Exception as e:
        db.rollback()
        print("========== SAVE SCHEDULE ERROR ==========")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


# ---------------------------
# API: Daily View & Note
# ---------------------------
@app.get("/api/schedules/{schedule_id}/daily")
def get_daily_view(schedule_id: int, db=Depends(get_db)):
    """
    일별 근무표 데이터 (날짜별 D/E/N 멤버, 리더 표시, 메모)
    """
    schedule = db.query(Schedule).filter(Schedule.id == schedule_id).first()
    if not schedule:
        raise HTTPException(404, "Schedule not found")

    plan_id = schedule.plan_id

    # 1. 배정 데이터 조회 (Nurse 정보 Join)
    assigns = (
        db.query(ScheduleAssignment, Nurse)
        .join(Nurse, ScheduleAssignment.nurse_id == Nurse.id)
        .filter(ScheduleAssignment.schedule_id == schedule_id)
        .all()
    )

    # 2. 메모 데이터 조회
    notes_db = db.query(ScheduleDailyNote).filter(ScheduleDailyNote.plan_id == plan_id).all()
    notes_map = {n.date.isoformat(): n.note for n in notes_db}

    # 3. 데이터 구조화
    # daily_data = { "2024-05-01": { "D": [nurse_obj...], "E": [...], "note": "..." } }
    daily_map = {}

    for assign, nurse in assigns:
        d_str = assign.date.isoformat()
        s_code = assign.shift_code

        if d_str not in daily_map:
            daily_map[d_str] = {"date": d_str, "shifts": {"D": [], "E": [], "N": []}, "note": notes_map.get(d_str, "")}

        if s_code in ["D", "E", "N"]:
            # 리더 적격 여부 판단
            is_leader = False
            if nurse.leader_eligible and nurse.leader_eligible.get(s_code):
                is_leader = True

            daily_map[d_str]["shifts"][s_code].append({
                "name": nurse.name,
                "is_leader": is_leader,
                "level": nurse.level_weight
            })

    # 정렬: 날짜순 -> 근무자 이름순(또는 레벨순)
    result = []
    for d_str in sorted(daily_map.keys()):
        row = daily_map[d_str]
        # 근무자 정렬 (리더 우선, 그 다음 이름)
        for s in ["D", "E", "N"]:
            row["shifts"][s].sort(key=lambda x: (not x["is_leader"], x["name"]))
        result.append(row)

    return result


@app.post("/api/schedules/{schedule_id}/daily/note")
def update_daily_note(
        schedule_id: int,
        payload: dict = Body(...),  # { "date": "2024-05-01", "note": "..." }
        db=Depends(get_db)
):
    date_str = payload.get("date")
    note_text = payload.get("note")

    schedule = db.query(Schedule).filter(Schedule.id == schedule_id).first()
    if not schedule: raise HTTPException(404, "Schedule not found")

    target_date = datetime.strptime(date_str, "%Y-%m-%d").date()

    # Upsert Note
    daily_note = db.query(ScheduleDailyNote).filter(
        ScheduleDailyNote.plan_id == schedule.plan_id,
        ScheduleDailyNote.date == target_date
    ).first()

    if not daily_note:
        daily_note = ScheduleDailyNote(
            plan_id=schedule.plan_id,
            date=target_date,
            note=note_text,
            updated_by="admin"  # 추후 로그인 유저로 변경
        )
        db.add(daily_note)
    else:
        daily_note.note = note_text
        daily_note.updated_by = "admin"

    db.commit()
    return {"ok": True}

# ---------------------------
# API Routes: ScoreConfig
# ---------------------------
@app.get("/api/score-config/active", response_model=ScoreConfigOut)
def get_active_score_config(ward_id: str = Query(...), db=Depends(get_db)):
    _ensure_ward_exists(db, ward_id)
    cfg = db.query(ScoreConfigModel).filter(ScoreConfigModel.ward_id == ward_id,
                                            ScoreConfigModel.active == True).order_by(
        ScoreConfigModel.created_at.desc()).first()
    if not cfg:
        return ScoreConfigOut(
            id=0, ward_id=ward_id, version_label="default",
            config_json=DEFAULT_SCORE_CONFIG_V1, active=True, created_at=datetime.now()
        )
    return cfg


@app.get("/api/score-config", response_model=List[ScoreConfigOut])
def list_score_configs(ward_id: str = Query(...), db=Depends(get_db)):
    _ensure_ward_exists(db, ward_id)
    return db.query(ScoreConfigModel).filter(ScoreConfigModel.ward_id == ward_id).order_by(
        ScoreConfigModel.created_at.desc()).all()


@app.get("/api/score-config/{config_id}", response_model=ScoreConfigOut)
def read_score_config(config_id: int = FPath(...), db=Depends(get_db)):
    cfg = db.query(ScoreConfigModel).filter(ScoreConfigModel.id == config_id).first()
    if not cfg: raise HTTPException(404, "Score config not found")
    return cfg


@app.post("/api/score-config", response_model=ScoreConfigOut)
def create_score_config(payload: ScoreConfigCreate, db=Depends(get_db)):
    _ensure_ward_exists(db, payload.ward_id)
    exists = db.query(ScoreConfigModel).filter(ScoreConfigModel.ward_id == payload.ward_id,
                                               ScoreConfigModel.version_label == payload.version_label).first()
    if exists: raise HTTPException(409, "version_label already exists")
    if payload.active: _deactivate_all_for_ward(db, payload.ward_id)
    cfg = ScoreConfigModel(ward_id=payload.ward_id, version_label=payload.version_label,
                           config_json=payload.config_json.model_dump(), active=payload.active)
    db.add(cfg);
    db.commit();
    db.refresh(cfg)
    return cfg


@app.put("/api/score-config/{config_id}", response_model=ScoreConfigOut)
def update_score_config(payload: ScoreConfigUpdate, config_id: int = FPath(...), db=Depends(get_db)):
    cfg = db.query(ScoreConfigModel).filter(ScoreConfigModel.id == config_id).first()
    if not cfg: raise HTTPException(404, "Score config not found")
    if payload.version_label:
        dup = db.query(ScoreConfigModel).filter(ScoreConfigModel.ward_id == cfg.ward_id,
                                                ScoreConfigModel.version_label == payload.version_label,
                                                ScoreConfigModel.id != cfg.id).first()
        if dup: raise HTTPException(409, "version_label already exists")
        cfg.version_label = payload.version_label
    if payload.config_json is not None: cfg.config_json = payload.config_json.model_dump()
    if payload.active is not None:
        if payload.active: _deactivate_all_for_ward(db, cfg.ward_id)
        cfg.active = payload.active
    db.commit();
    db.refresh(cfg)
    return cfg


@app.put("/api/score-config/{config_id}/activate", response_model=ScoreConfigOut)
def activate_score_config(config_id: int = FPath(...), db=Depends(get_db)):
    cfg = db.query(ScoreConfigModel).filter(ScoreConfigModel.id == config_id).first()
    if not cfg: raise HTTPException(404, "Score config not found")
    _deactivate_all_for_ward(db, cfg.ward_id)
    cfg.active = True
    db.commit();
    db.refresh(cfg)
    return cfg


@app.delete("/api/score-config/{config_id}")
def delete_score_config(config_id: int = FPath(...), db=Depends(get_db)):
    cfg = db.query(ScoreConfigModel).filter(ScoreConfigModel.id == config_id).first()
    if not cfg: raise HTTPException(404, "Score config not found")
    db.delete(cfg);
    db.commit()
    return {"ok": True}


# ---------------------------
# API Routes: Nurse
# ---------------------------
@app.get("/api/nurses")
def list_nurses(ward_id: str = Query(DEFAULT_WARD_ID), db=Depends(get_db)):
    ward = _get_or_create_ward(db, ward_id)
    nurses = db.query(Nurse).filter(Nurse.ward_id == ward.id).order_by(Nurse.employee_no, Nurse.id).all()
    return [n.as_ui_dict(include_avoid_names=True) for n in nurses]


@app.get("/api/nurses/duplicate_check")
def check_nurse_duplicate(staff_id: str = Query(..., alias="staff_id"), ward_id: str = Query(DEFAULT_WARD_ID),
                          db=Depends(get_db)):
    ward = _get_or_create_ward(db, ward_id)
    nurse = db.query(Nurse).filter(Nurse.ward_id == ward.id, Nurse.employee_no == staff_id).first()
    if not nurse: return {"exists": False}
    return {"exists": True, "nurse": nurse.as_ui_dict(include_avoid_names=True)}


@app.post("/api/nurses")
def create_nurse(payload: NurseUpsertPayload = Body(...), ward_id: str = Query(DEFAULT_WARD_ID), db=Depends(get_db)):
    ward = _get_or_create_ward(db, ward_id)
    data = payload.model_dump()
    employee_no = data.get("employee_no") or data.get("staff_id")

    if not employee_no:
        raise HTTPException(400, "Employee number is required")

    dup = db.query(Nurse).filter(Nurse.ward_id == ward.id, Nurse.employee_no == employee_no).first()
    if dup:
        raise HTTPException(409, "Nurse with this employee number already exists")

    hire_year = data.get("hire_year") or data.get("join_year")
    career_year = data.get("career_year") or data.get("years_exp")
    level_weight = data.get("level_weight") or data.get("level") or 1.0
    short_work_factor = data.get("short_work_factor") or data.get("relative_work_days") or 1.0
    leader_eligible = _normalize_leader_eligible(data)
    team_code = data.get("team_code") or data.get("team")
    team = _get_or_create_team(db, ward, team_code)

    tags = data.get("tags") or []
    preferred_shifts = data.get("preferred_shifts") or []
    avoid_shifts = data.get("avoid_shifts") or []

    nurse = Nurse(
        ward_id=ward.id,
        employee_no=employee_no,
        name=data.get("name"),
        hire_year=hire_year,
        career_year=career_year,
        level_weight=level_weight,
        short_work_factor=short_work_factor,
        leader_eligible=leader_eligible,
        team_id=team.id if team else None,
        tags=tags,
        preferred_shifts=preferred_shifts,
        avoid_shifts=avoid_shifts,
        pref_ratio={}
    )

    db.add(nurse)
    db.flush()

    _validate_relationship_conflict(db, ward, data, current_nurse_id=nurse.id)
    _sync_preceptor_relation(db, ward, nurse, data)
    _sync_avoid_relations(db, ward, nurse, data)

    db.commit()
    db.refresh(nurse)
    return nurse.as_ui_dict(include_avoid_names=True)


@app.put("/api/nurses/{nurse_id}")
def update_nurse(nurse_id: int = FPath(...), payload: NurseUpsertPayload = Body(...),
                 ward_id: str = Query(DEFAULT_WARD_ID), db=Depends(get_db)):
    ward = _get_or_create_ward(db, ward_id)
    nurse = db.query(Nurse).filter(Nurse.id == nurse_id, Nurse.ward_id == ward.id).first()
    if not nurse:
        raise HTTPException(404, "Nurse not found")

    data = payload.model_dump()
    _validate_relationship_conflict(db, ward, data, current_nurse_id=nurse.id)

    nurse.employee_no = data.get("employee_no") or data.get("staff_id") or nurse.employee_no
    nurse.name = data.get("name") or nurse.name
    nurse.hire_year = data.get("hire_year") or data.get("join_year") or nurse.hire_year
    nurse.career_year = data.get("career_year") or data.get("years_exp") or nurse.career_year
    nurse.level_weight = data.get("level_weight") or data.get("level") or nurse.level_weight
    nurse.short_work_factor = data.get("short_work_factor") or data.get("relative_work_days") or nurse.short_work_factor
    nurse.leader_eligible = _normalize_leader_eligible(data)

    team_code = data.get("team_code") or data.get("team")
    if team_code is not None:
        team = _get_or_create_team(db, ward, team_code)
        nurse.team_id = team.id if team else None

    if "tags" in data and data["tags"] is not None:
        nurse.tags = data["tags"]
    if "preferred_shifts" in data and data["preferred_shifts"] is not None:
        nurse.preferred_shifts = data["preferred_shifts"]
    if "avoid_shifts" in data and data["avoid_shifts"] is not None:
        nurse.avoid_shifts = data["avoid_shifts"]

    _sync_preceptor_relation(db, ward, nurse, data)
    _sync_avoid_relations(db, ward, nurse, data)

    db.commit()
    db.refresh(nurse)
    return nurse.as_ui_dict(include_avoid_names=True)


@app.delete("/api/nurses/{nurse_id}")
def delete_nurse(nurse_id: int = FPath(...), ward_id: str = Query(DEFAULT_WARD_ID), db=Depends(get_db)):
    ward = _get_or_create_ward(db, ward_id)
    nurse = db.query(Nurse).filter(Nurse.id == nurse_id, Nurse.ward_id == ward.id).first()
    if not nurse: raise HTTPException(404, "not found")

    db.query(NurseAvoid).filter(
        NurseAvoid.ward_id == ward.id,
        (NurseAvoid.nurse_id == nurse.id) | (NurseAvoid.avoid_nurse_id == nurse.id)
    ).delete(synchronize_session=False)

    others = db.query(Nurse).filter(Nurse.ward_id == ward.id,
                                    (Nurse.preceptor_id == nurse.id) | (Nurse.preceptee_id == nurse.id)).all()
    for o in others:
        if o.preceptor_id == nurse.id: o.preceptor_id = None
        if o.preceptee_id == nurse.id: o.preceptee_id = None

    db.delete(nurse);
    db.commit()
    return {"ok": True}


@app.get("/api/nurses/download")
def download_nurses_excel(db: Session = Depends(get_db)):
    nurses = db.query(Nurse).filter_by(active=True).all()
    rows = []
    rows.append(EXCEL_SAMPLE_DATA)
    rows.append(EXCEL_DESC_DATA)

    for n in nurses:
        tags_str = ", ".join(n.tags) if n.tags else ""
        pref_str = ", ".join(n.preferred_shifts) if n.preferred_shifts else ""
        avoid_str = ", ".join(n.avoid_shifts) if n.avoid_shifts else ""

        leader_s = []
        le = n.leader_eligible or {}
        if le.get("D"): leader_s.append("D")
        if le.get("E"): leader_s.append("E")
        if le.get("N"): leader_s.append("N")

        row = [
            n.employee_no,
            n.name,
            n.join_date.year if n.join_date else (n.hire_year or ""),
            n.career_year,
            n.team.code if n.team else "",
            n.level_weight,
            n.short_work_factor,
            ",".join(leader_s),
            n.preceptor.name if n.preceptor else "",
            ", ".join([a.avoid_nurse.name for a in n.avoids]) if n.avoids else "",
            tags_str,
            pref_str,
            avoid_str
        ]
        rows.append(row)

    df = pd.DataFrame(rows, columns=NURSE_EXCEL_COLUMNS)
    filename = f"nurses_{datetime.now().strftime('%Y%m%d')}.xlsx"
    path = f"./temp/{filename}"
    os.makedirs("./temp", exist_ok=True)
    df.to_excel(path, index=False)
    return FileResponse(path, filename=filename)


@app.get("/api/schedules/{schedule_id}/download_excel")
def download_schedule_excel(schedule_id: int, db=Depends(get_db)):
    schedule = db.query(Schedule).filter(Schedule.id == schedule_id).first()
    if not schedule: raise HTTPException(status_code=404, detail="Schedule not found")

    plan = db.query(SchedulePlan).filter(SchedulePlan.id == schedule.plan_id).first()
    assignments = db.query(ScheduleAssignment).filter(ScheduleAssignment.schedule_id == schedule_id).all()
    nurses = db.query(Nurse).filter(Nurse.ward_id == plan.ward_id).order_by(Nurse.employee_no).all()

    grid = {}
    for a in assignments:
        nid = a.nurse_id
        d_str = a.date.isoformat()
        if nid not in grid: grid[nid] = {}
        grid[nid][d_str] = a.shift_code

    wb = Workbook();
    ws = wb.active;
    ws.title = "Schedule"
    year = plan.start_date.year;
    month = plan.start_date.month
    last_day = calendar.monthrange(year, month)[1]

    headers = ["사번", "이름", "팀"]
    date_keys = []
    for day in range(1, last_day + 1):
        dt = date(year, month, day)
        headers.append(f"{day}일")
        date_keys.append(dt.isoformat())
    ws.append(headers)

    for nurse in nurses:
        row = [nurse.employee_no, nurse.name, nurse.team.code if nurse.team else ""]
        nurse_grid = grid.get(nurse.id, {})
        for d_str in date_keys:
            row.append(nurse_grid.get(d_str, ""))
        ws.append(row)

    buf = BytesIO();
    wb.save(buf);
    buf.seek(0)
    filename = f"schedule_{year}_{month:02d}.xlsx"
    return StreamingResponse(
        buf, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/api/upload_excel")
async def upload_nurses_excel(file: UploadFile = File(...), db: Session = Depends(get_db)):
    contents = await file.read()
    # [수정] BytesIO로 감싸서 FutureWarning 방지
    df = pd.read_excel(BytesIO(contents))

    # 헤더(1행), 예시(2행), 설명(3행) 구조 처리
    # 데이터가 충분할 때만 앞의 2줄(예시, 설명) 제거
    if len(df) >= 2:
        df = df.iloc[2:].reset_index(drop=True)

    created = 0
    updated = 0
    current_year = datetime.now().year

    for _, row in df.iterrows():
        emp_no = str(row.get("사번", "")).strip()
        name = str(row.get("이름", "")).strip()

        # 유효성 검사 (헤더나 설명 행이 남아있을 경우, 혹은 빈 값 대비)
        if not emp_no or not name or emp_no in ["S001", "필수"] or name == "nan":
            continue

        # ---------------------------------------------------------
        # 1. [수정] 입사년도 및 연차 상호 자동 계산 로직
        # ---------------------------------------------------------
        raw_join = row.get("입사년도")
        raw_exp = row.get("연차")

        hire_year = None
        career_year = 0

        # 입사년도 파싱
        try:
            if not pd.isna(raw_join):
                hire_year = int(float(raw_join))
        except:
            hire_year = None

        # 연차 파싱
        try:
            if not pd.isna(raw_exp):
                career_year = int(float(raw_exp))
        except:
            career_year = 0

        # 입사년도가 있고 연차가 0이면 -> 연차 자동 계산
        if career_year == 0 and hire_year:
            career_year = max(1, current_year - hire_year + 1)

        # 연차가 있고 입사년도가 없으면 -> 입사년도 역산
        if hire_year is None and career_year > 0:
            hire_year = current_year - career_year + 1

        # ---------------------------------------------------------
        # 2. 업무 레벨 파싱
        # ---------------------------------------------------------
        level = 1.0
        raw_level = row.get("업무레벨")
        if pd.isna(raw_level) or str(raw_level).strip() == "":
            if career_year <= 2:
                level = 0.7
            elif career_year <= 5:
                level = 1.0
            elif career_year <= 10:
                level = 1.1
            else:
                level = 1.2
        else:
            try:
                level = float(raw_level)
            except:
                level = 1.0

        # ---------------------------------------------------------
        # 3. [수정] 리더 파싱 (NAN 문자열 버그 수정)
        # ---------------------------------------------------------
        raw_leader = row.get("리더가능")
        if pd.isna(raw_leader):
            l_str = ""
        else:
            l_str = str(raw_leader).upper().strip()
            # pandas가 빈 셀을 NaN으로 읽은 것을 문자열로 변환하면 "NAN"이 됨
            # "NAN" 안에 "N"이 포함되어 N 리더로 인식되는 문제 방지
            if l_str == "NAN": l_str = ""

        leader_eligible = {
            "D": "D" in l_str,
            "E": "E" in l_str,
            "N": "N" in l_str
        }

        # ---------------------------------------------------------
        # 4. 리스트 및 기타 필드 파싱
        # ---------------------------------------------------------
        def parse_list(s):
            if pd.isna(s): return []
            val = str(s).strip()
            if val.lower() == "nan": return []
            return [x.strip() for x in val.split(",") if x.strip()]

        tags = parse_list(row.get("특수태그"))
        prefs = parse_list(row.get("선호근무"))
        avoids = parse_list(row.get("기피근무"))

        # 팀 정보 처리
        team_name = str(row.get("팀", "")).strip()
        if team_name.lower() == "nan": team_name = ""

        team = None
        if team_name:
            ward = _get_or_create_ward(db, DEFAULT_WARD_ID)
            team = _get_or_create_team(db, ward, team_name)

        # ---------------------------------------------------------
        # 5. DB 업데이트 (Upsert)
        # ---------------------------------------------------------
        nurse = db.query(Nurse).filter_by(employee_no=emp_no, ward_id=DEFAULT_WARD_ID).first()
        if not nurse:
            nurse = Nurse(
                ward_id=DEFAULT_WARD_ID,
                employee_no=emp_no,
                name=name,
                active=True
            )
            db.add(nurse)
            created += 1
        else:
            updated += 1
            nurse.name = name

        # 계산된 값 적용
        nurse.hire_year = hire_year
        nurse.career_year = career_year
        nurse.level_weight = level

        # 단축근무
        swf_val = row.get("단축근무")
        try:
            nurse.short_work_factor = float(swf_val) if not pd.isna(swf_val) else 1.0
        except:
            nurse.short_work_factor = 1.0

        nurse.leader_eligible = leader_eligible
        nurse.tags = tags
        nurse.preferred_shifts = prefs
        nurse.avoid_shifts = avoids

        if team:
            nurse.team_id = team.id

    db.commit()
    return {"created": created, "updated": updated}


@app.get("/", response_class=HTMLResponse)
def root(request: Request):
    return templates.TemplateResponse("schedule_um.html", {"request": request, "ward_id": "WARD1"})


@app.get("/schedule", include_in_schema=False)
def redirect_schedule():
    return RedirectResponse(url="/schedule/um")


@app.get("/schedule/um", response_class=HTMLResponse)
def schedule_um_page(request: Request):
    return templates.TemplateResponse("schedule_um.html", {"request": request, "ward_id": "WARD1"})


@app.get("/personnel", response_class=HTMLResponse)
def personnel_page(request: Request):
    return templates.TemplateResponse("personnel.html", {"request": request})


@app.get("/schedule/nurse", response_class=HTMLResponse)
def schedule_nurse_page(request: Request, nurse_id: Optional[int] = Query(None), db=Depends(get_db)):
    if not nurse_id:
        n = db.query(Nurse).filter(Nurse.ward_id == "WARD1").first()
        current_id = n.id if n else 0
    else:
        current_id = nurse_id
    now = datetime.now()
    return templates.TemplateResponse("schedule_nurse.html", {
        "request": request, "page_mode": "nurse_public", "user": {"role": "nurse"},
        "current_nurse_id": current_id, "current_ward": "WARD1",
        "initial_year": now.year, "initial_month": now.month
    })

@app.get("/schedule/daily", response_class=HTMLResponse)
def schedule_daily_page(request: Request):
    return templates.TemplateResponse("schedule_daily.html", {"request": request})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)