# scheduler/data_loader.py

from __future__ import annotations
from datetime import timedelta

from scheduler.types import PlanContext, NurseInfo, CoverageNeed


def load_plan_context(session, plan_id: int) -> PlanContext:
    from scheduler.db import (
        SchedulePlan, Nurse, CoverageRequirement, LeaderRequirement,
        ShiftLock, HolidayCalendar, Team, Schedule, ScheduleAssignment
    )

    plan = session.query(SchedulePlan).get(plan_id)
    if plan is None:
        raise ValueError(f"SchedulePlan not found: plan_id={plan_id}")

    dates = []
    d = plan.start_date
    while d <= plan.end_date:
        dates.append(d)
        d += timedelta(days=1)

    # [New] Previous Schedule Loading
    previous_schedule = {}
    lookback_start = plan.start_date - timedelta(days=7)
    lookback_end = plan.start_date - timedelta(days=1)

    prev_assignments = (
        session.query(ScheduleAssignment)
        .join(Schedule)
        .join(Nurse)
        .filter(
            Nurse.ward_id == plan.ward_id,
            ScheduleAssignment.date >= lookback_start,
            ScheduleAssignment.date <= lookback_end
        )
        .order_by(Schedule.created_at.asc())
        .all()
    )

    for a in prev_assignments:
        previous_schedule[(a.nurse_id, a.date)] = a.shift_code

    nurses = []
    q = session.query(Nurse).filter_by(ward_id=plan.ward_id, active=True).all()
    for n in q:
        tags_set = set(n.tags) if n.tags else set()
        pref_set = set(n.preferred_shifts) if n.preferred_shifts else set()
        avoid_set = set(n.avoid_shifts) if n.avoid_shifts else set()

        nurses.append(NurseInfo(
            id=n.id,
            employee_no=n.employee_no,
            name=n.name,
            ward_id=n.ward_id,
            team_id=str(n.team_id) if n.team_id is not None else None,
            level_weight=n.level_weight if n.level_weight is not None else 1.0,
            short_work_factor=n.short_work_factor if n.short_work_factor is not None else 1.0,
            tags=tags_set,
            preferred=pref_set,
            avoid=avoid_set,
            leader_eligible=n.leader_eligible or {},
            preceptor_id=n.preceptor_id,
            preceptee_id=n.preceptee_id,
            is_preceptee=bool(n.is_preceptee),
            avoid_ids=[a.avoid_nurse_id for a in getattr(n, "avoids", [])],
        ))

    coverage_needs = []
    for r in session.query(CoverageRequirement).filter_by(plan_id=plan_id).all():
        coverage_needs.append(
            CoverageNeed(date=r.date, shift=r.shift_code, min_required=r.min_required)
        )

    leader_required = {
        (lr.date, lr.shift_code): bool(lr.required)
        for lr in session.query(LeaderRequirement).filter_by(plan_id=plan_id).all()
    }

    um_locks = {
        (l.nurse_id, l.date): l.shift_code
        for l in session.query(ShiftLock).filter_by(plan_id=plan_id, lock_type="UM").all()
    }

    nurse_requests = {
        (l.nurse_id, l.date): l.shift_code
        for l in session.query(ShiftLock).filter_by(plan_id=plan_id, lock_type="NURSE").all()
    }

    is_holiday = {
        h.date: True
        for h in session.query(HolidayCalendar).filter_by(ward_id=plan.ward_id).all()
    }
    is_weekend = {dd: (dd.weekday() >= 5) for dd in dates}

    teams = [
        str(t.id)
        for t in session.query(Team).filter_by(ward_id=plan.ward_id, active=True).all()
    ]

    return PlanContext(
        plan_id=plan.id,
        ward_id=plan.ward_id,
        start=plan.start_date,
        end=plan.end_date,
        dates=dates,
        nurses=nurses,
        teams=teams,
        coverage_needs=coverage_needs,
        leader_required=leader_required,
        um_locks=um_locks,
        nurse_requests=nurse_requests,
        is_holiday=is_holiday,
        is_weekend=is_weekend,
        previous_schedule=previous_schedule
    )