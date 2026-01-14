# scheduler/generator.py

from __future__ import annotations
from datetime import timedelta
import random
import numpy as np
import torch
import os
from collections import defaultdict

from scheduler.types import PlanContext, ScheduleMatrix
from scheduler.constraints.hard_rules import validate_all
from scheduler.db import Schedule, ScheduleAssignment, Nurse
from scheduler.ai_models import ShiftLSTM

# 근무 코드 매핑 (AI 모델용)
SHIFT_TO_IDX = {'D': 0, 'E': 1, 'N': 2, 'O': 3}
IDX_TO_SHIFT = {0: 'D', 1: 'E', 2: 'N', 3: 'O'}


def _is_safe_assignment(ctx: PlanContext, sch: ScheduleMatrix, nurse_id: int, date, shift: str,
                        strict: bool = True) -> bool:
    """
    Hard Rule 위반 여부 검사
    strict=True: 모든 패턴 검사 (Generator 초기)
    strict=False: Tier 1(절대 불가) 제약만 검사 (Coverage 강제 배정 시)
    """
    try:
        d_idx = ctx.dates.index(date)
    except ValueError:
        return True

    nurse = next((n for n in ctx.nurses if n.id == nurse_id), None)
    if not nurse: return True

    # [Tier 1] Tag-based Hard Constraints (절대 타협 불가)
    if "pregnant" in nurse.tags and shift == "N": return False
    if "night_keep" in nurse.tags and shift in ["D", "E"]: return False
    if "fixed_day" in nurse.tags and shift in ["E", "N"]: return False
    if "fixed_evening" in nurse.tags and shift in ["D", "N"]: return False

    if not strict:
        return True  # 강제 배정 모드에서는 여기까지만 통과하면 OK

    # [Month Boundary Check]
    prev_1 = sch.get_shift(nurse_id, date - timedelta(days=1))

    # 1. 근무 패턴 체크
    if shift == 'D':
        if prev_1 == 'E': return False
        if prev_1 == 'N': return False
        if prev_1 == 'O':
            prev_2 = sch.get_shift(nurse_id, date - timedelta(days=2))
            if prev_2 == 'N': return False  # N-O-D

    if shift == 'E':
        if prev_1 == 'N': return False

    # N-O-N (Pong-dang) check
    if shift == 'N':
        if prev_1 == 'O':
            prev_2 = sch.get_shift(nurse_id, date - timedelta(days=2))
            if prev_2 == 'N': return False  # N-O-N

    # 2. 연속 근무 제한
    if shift != 'O':
        consec_work = 0
        for i in range(1, 8):
            if d_idx - i < 0: break
            past_s = sch.get_shift(nurse_id, ctx.dates[d_idx - i])
            if past_s != 'O':
                consec_work += 1
            else:
                break
        if consec_work >= 6: return False  # Max 6일 연속

    if shift == 'N':
        consec_n = 0
        for i in range(1, 6):
            if d_idx - i < 0: break
            past_s = sch.get_shift(nurse_id, ctx.dates[d_idx - i])
            if past_s == 'N':
                consec_n += 1
            else:
                break
        if consec_n >= 4: return False  # Max 4일 연속 N

    return True


def _try_repair_violations(ctx: PlanContext, sch: ScheduleMatrix, violations: list, rng: random.Random) -> bool:
    modified = False
    rng.shuffle(violations)

    for v in violations:
        _, nurse_id, date, _ = v
        if nurse_id is None or date is None: continue
        if (nurse_id, date) in ctx.um_locks or (nurse_id, date) in ctx.nurse_requests: continue

        curr = sch.get_shift(nurse_id, date)
        candidates = ["O", "D", "E", "N"]
        if "O" in candidates:
            candidates.remove("O")
            candidates.insert(0, "O")  # O를 우선 시도하여 패턴 끊기

        for new_s in candidates:
            if new_s == curr: continue
            if _is_safe_assignment(ctx, sch, nurse_id, date, new_s, strict=True):
                sch.set_shift(nurse_id, date, new_s)
                modified = True
                break
        if modified: break
    return modified


# ---------------------------------------------------------
# Generators Classes
# ---------------------------------------------------------

class BaseGenerator:
    def __init__(self, ctx: PlanContext):
        self.ctx = ctx
        self.shifts = ['D', 'E', 'N', 'O']

    def get_probs(self, nurse_id, date, history_seq):
        return {s: 0.25 for s in self.shifts}


class MarkovGenerator(BaseGenerator):
    def __init__(self, ctx, session):
        super().__init__(ctx)
        self.valid_data = False
        self.transition_matrix = self._build_matrix(session, ctx.ward_id)

    def _get_default_matrix(self):
        return {
            'D': {'D': 0.5, 'E': 0.3, 'N': 0.0, 'O': 0.2},
            'E': {'D': 0.0, 'E': 0.5, 'N': 0.3, 'O': 0.2},
            'N': {'D': 0.0, 'E': 0.0, 'N': 0.1, 'O': 0.9},
            'O': {'D': 0.6, 'E': 0.3, 'N': 0.1, 'O': 0.0}
        }

    def _build_matrix(self, session, ward_id):
        counts = {s: {ns: 0 for ns in self.shifts} for s in self.shifts}
        records = session.query(ScheduleAssignment).join(Schedule).join(Nurse) \
            .filter(Nurse.ward_id == ward_id) \
            .order_by(ScheduleAssignment.nurse_id, ScheduleAssignment.date) \
            .limit(10000).all()

        if not records:
            return self._get_default_matrix()

        self.valid_data = True
        prev_map = {}
        for r in records:
            curr = r.shift_code
            if curr == 'M':
                curr = 'E'
            elif curr == 'C':
                curr = 'D'
            elif curr == 'P':
                curr = 'O'
            if curr not in self.shifts: continue
            if r.nurse_id in prev_map:
                prev = prev_map[r.nurse_id]
                if prev in self.shifts:
                    counts[prev][curr] += 1
            prev_map[r.nurse_id] = curr

        matrix = {}
        for s in self.shifts:
            total = sum(counts[s].values())
            if total == 0:
                matrix[s] = self._get_default_matrix()[s]
            else:
                matrix[s] = {ns: (counts[s][ns] + 1) / (total + 4) for ns in self.shifts}
        return matrix

    def get_probs(self, nurse_id, date, history_seq):
        if not history_seq: return {s: 0.25 for s in self.shifts}
        prev = history_seq[-1]
        if prev not in self.transition_matrix: return {s: 0.25 for s in self.shifts}
        return self.transition_matrix[prev]


class DeepGenerator(BaseGenerator):
    def __init__(self, ctx, session):
        super().__init__(ctx)
        self.model = self._load_model()
        self.window_size = 7

    def _load_model(self):
        model = ShiftLSTM(num_shifts=4)
        model_path = "lstm_model.pth"
        if os.path.exists(model_path):
            try:
                model.load_state_dict(torch.load(model_path))
                model.eval()
            except:
                pass
        return model

    def get_probs(self, nurse_id, date, history_seq):
        if len(history_seq) < self.window_size:
            return {s: 0.25 for s in self.shifts}
        seq = history_seq[-self.window_size:]
        clean_seq = []
        for s in seq:
            if s == 'M':
                s = 'E'
            elif s == 'C':
                s = 'D'
            elif s == 'P':
                s = 'O'
            clean_seq.append(s)
        idxs = [SHIFT_TO_IDX.get(s, 3) for s in clean_seq]
        with torch.no_grad():
            inp = torch.tensor([idxs], dtype=torch.long)
            logits = self.model(inp)
            probs = torch.softmax(logits, dim=1).numpy()[0]
        return {IDX_TO_SHIFT[i]: p for i, p in enumerate(probs)}


# ---------------------------------------------------------
# Generation Core Logic
# ---------------------------------------------------------

def _generate_with_ai(ctx: PlanContext, sch: ScheduleMatrix, rng: random.Random, generator: BaseGenerator):
    # 1. Lock 고정
    for n in ctx.nurses:
        for d in ctx.dates:
            if (n.id, d) in ctx.um_locks:
                sch.set_shift(n.id, d, ctx.um_locks[(n.id, d)])
            elif (n.id, d) in ctx.nurse_requests:
                sch.set_shift(n.id, d, ctx.nurse_requests[(n.id, d)])

    sorted_dates = sorted(ctx.dates)

    # History Buffer 초기화
    history_buffers = defaultdict(list)
    start_lookback = ctx.start - timedelta(days=7)
    for n in ctx.nurses:
        relevant_prev = []
        for i in range(7):
            target_d = start_lookback + timedelta(days=i)
            if (n.id, target_d) in ctx.previous_schedule:
                relevant_prev.append(ctx.previous_schedule[(n.id, target_d)])
            else:
                relevant_prev.append('O')
        history_buffers[n.id] = relevant_prev

    # [Fix] AI 생성 시에도 'O' 남발 방지 로직 강화
    for d in sorted_dates:
        for n in ctx.nurses:
            if sch.get_shift(n.id, d) != 'O':
                if (n.id, d) in sch.grid:
                    history_buffers[n.id].append(sch.grid[(n.id, d)])
                    continue

            # 1. AI 확률
            probs_map = generator.get_probs(n.id, d, history_buffers[n.id])

            # 2. Hard Rule 필터링 (Work Shift 우선)
            valid_shifts = []
            valid_probs = []

            # D, E, N 먼저 시도
            for s in ['D', 'E', 'N']:
                p = probs_map.get(s, 0.0)
                if _is_safe_assignment(ctx, sch, n.id, d, s, strict=True):
                    valid_shifts.append(s)
                    valid_probs.append(p)

            # Work Shift가 불가능하면 O 시도
            if not valid_shifts:
                if _is_safe_assignment(ctx, sch, n.id, d, 'O', strict=True):
                    chosen = 'O'
                else:
                    # O마저 안되면... 일단 O (Tier 1 위반 등 상황)
                    chosen = 'O'
            else:
                # Work Shift 가능하면 그 중에서 확률 기반 선택
                total_p = sum(valid_probs)
                if total_p > 0:
                    norm_probs = np.array([p / total_p for p in valid_probs], dtype=np.float64)
                    norm_probs /= norm_probs.sum()
                    chosen = np.random.choice(valid_shifts, p=norm_probs)
                else:
                    chosen = rng.choice(valid_shifts)

            sch.set_shift(n.id, d, chosen)
            history_buffers[n.id].append(chosen)

    return sch


def _patch_mandatory_requirements(ctx: PlanContext, sch: ScheduleMatrix, rng: random.Random):
    # AI 생성 후 필수 인원 부족분 채우기 (Coverage Repair)
    shifts = ['D', 'E', 'N']
    for d in ctx.dates:
        needs = {need.shift: need.min_required for need in ctx.coverage_needs if need.date == d}
        current_counts = {s: 0 for s in shifts}

        for n in ctx.nurses:
            s = sch.get_shift(n.id, d)
            if s in current_counts: current_counts[s] += 1

        for s in shifts:
            req = needs.get(s, 0)
            while current_counts[s] < req:
                shortage = req - current_counts[s]
                # 'O'인 사람 중 태그 제약(Tier 1)만 통과하면 강제 배정
                candidates = [n for n in ctx.nurses if sch.get_shift(n.id, d) == 'O']
                rng.shuffle(candidates)

                filled = False
                for n in candidates:
                    if (n.id, d) in ctx.um_locks: continue

                    # [Fix] strict=False (패턴 무시, 태그만 준수)
                    if _is_safe_assignment(ctx, sch, n.id, d, s, strict=False):
                        sch.set_shift(n.id, d, s)
                        current_counts[s] += 1
                        filled = True
                        break

                if not filled: break  # 더 이상 채울 사람이 없음


def _heuristic_generate_core(ctx: PlanContext, rng: random.Random) -> ScheduleMatrix:
    sch = ScheduleMatrix()

    # 1. Lock 고정
    for n in ctx.nurses:
        for d in ctx.dates:
            if (n.id, d) in ctx.um_locks:
                sch.set_shift(n.id, d, ctx.um_locks[(n.id, d)])
            elif (n.id, d) in ctx.nurse_requests:
                sch.set_shift(n.id, d, ctx.nurse_requests[(n.id, d)])

    # 2. Coverage Fill (Forceful)
    shifts = ['D', 'E', 'N']

    # Workload Balancing (간단한 카운터)
    workload_counter = {n.id: 0 for n in ctx.nurses}

    for d in ctx.dates:
        needs = {need.shift: need.min_required for need in ctx.coverage_needs if need.date == d}
        assigned_count = {'D': 0, 'E': 0, 'N': 0}

        # 이미 배정된 인원 파악
        assigned_nurses = set()
        for n in ctx.nurses:
            s = sch.grid.get((n.id, d))
            if s in shifts:
                assigned_count[s] += 1
                assigned_nurses.add(n.id)
                workload_counter[n.id] += 1
            elif s == "O":
                assigned_nurses.add(n.id)

        # 배정 안 된 인원
        free_nurses = [n for n in ctx.nurses if n.id not in assigned_nurses]
        # 근무 적은 순으로 정렬 (Fairness)
        free_nurses.sort(key=lambda x: (workload_counter[x.id], rng.random()))

        # Shift Assignment
        for s in ['D', 'E', 'N']:
            req = needs.get(s, 0)

            # 1차 시도: Safe Assignment
            while assigned_count[s] < req and free_nurses:
                found = False
                for i, n in enumerate(free_nurses):
                    if _is_safe_assignment(ctx, sch, n.id, d, s, strict=True):
                        sch.set_shift(n.id, d, s)
                        assigned_count[s] += 1
                        workload_counter[n.id] += 1
                        free_nurses.pop(i)
                        found = True
                        break

                # [Fix] 2차 시도: Coverage 부족 시 Force Assignment (Strict=False)
                if not found:
                    for i, n in enumerate(free_nurses):
                        if _is_safe_assignment(ctx, sch, n.id, d, s, strict=False):
                            sch.set_shift(n.id, d, s)
                            assigned_count[s] += 1
                            workload_counter[n.id] += 1
                            free_nurses.pop(i)
                            found = True
                            break

                if not found: break  # 도저히 넣을 사람이 없음

        # 나머지는 OFF
        for n in free_nurses:
            sch.set_shift(n.id, d, "O")

    return sch


def generate_initial_schedule(
        ctx: PlanContext,
        seed: int = 0,
        max_retries: int = 100,
        mode: str = "heuristic",
        session=None
) -> ScheduleMatrix:
    rng = random.Random(seed)
    sch = ScheduleMatrix()

    if ctx.previous_schedule:
        sch.grid.update(ctx.previous_schedule)

    # 1. Generate based on Mode
    if mode == "markov" and session:
        gen = MarkovGenerator(ctx, session)
        if gen.valid_data:
            sch = _generate_with_ai(ctx, sch, rng, gen)
            _patch_mandatory_requirements(ctx, sch, rng)
        else:
            sch = _heuristic_generate_core(ctx, rng)

    elif mode == "deep" and session:
        gen = DeepGenerator(ctx, session)
        sch = _generate_with_ai(ctx, sch, rng, gen)
        _patch_mandatory_requirements(ctx, sch, rng)

    else:  # heuristic
        sch = _heuristic_generate_core(ctx, rng)

    # 2. Repair Phase
    violations = validate_all(ctx, sch, strict=True)
    if not violations: return sch

    for i in range(max_retries):
        repaired = _try_repair_violations(ctx, sch, violations, rng)

        if not repaired:
            n = rng.choice(ctx.nurses)
            d = rng.choice(ctx.dates)
            if (n.id, d) not in ctx.um_locks:
                sch.set_shift(n.id, d, rng.choice(["D", "E", "N", "O"]))

        violations = validate_all(ctx, sch, strict=True)
        if not violations: return sch

    return sch