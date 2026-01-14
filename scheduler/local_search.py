# scheduler/local_search.py

from __future__ import annotations

import random
from copy import deepcopy
from typing import Optional, Tuple, List, Dict
import os
import torch
import numpy as np

from scheduler.types import PlanContext, ScheduleMatrix
from scheduler.score_config import ScoreConfig
from scheduler.evaluator import evaluate_schedule
from scheduler.constraints.hard_rules import validate_all
from scheduler.db import LocalSearchLog, OperatorWeight, LocalSearchExperience, EvaluationLog
from scheduler.ai_models import OperatorDQN
from scheduler.features import FeatureExtractor
from scheduler.surrogate_models import SurrogateManager


# ---------------------------------------------------------
# Helper Functions & Operators
# ---------------------------------------------------------
def _check_safety_locally(sch, nurse_id, date, new_shift, ctx_dates):
    if new_shift == 'O': return True
    try:
        d_idx = ctx_dates.index(date)
    except ValueError:
        return True

    # 전날 체크
    if d_idx > 0:
        prev = sch.get_shift(nurse_id, ctx_dates[d_idx - 1])
        if new_shift == 'D' and prev in ['E', 'N']: return False
        if new_shift == 'E' and prev == 'N': return False

    # 다음날 체크
    if d_idx < len(ctx_dates) - 1:
        next_s = sch.get_shift(nurse_id, ctx_dates[d_idx + 1])
        if next_s == 'D' and new_shift in ['E', 'N']: return False
        if next_s == 'E' and new_shift == 'N': return False

    return True


def block_mutation_1_2_days(ctx, sch, rng, **kwargs):
    new = deepcopy(sch)
    for _ in range(5):
        n = rng.choice(ctx.nurses)
        start_idx = rng.randrange(0, len(ctx.dates))
        length = rng.choice([1, 2])
        valid_found = False

        for i in range(start_idx, min(start_idx + length, len(ctx.dates))):
            d = ctx.dates[i]
            if (n.id, d) in ctx.um_locks or (n.id, d) in ctx.nurse_requests: continue

            curr = new.get_shift(n.id, d)
            cands = [s for s in ["D", "E", "N", "O"] if s != curr]
            rng.shuffle(cands)

            for c in cands:
                if _check_safety_locally(new, n.id, d, c, ctx.dates):
                    new.set_shift(n.id, d, c)
                    valid_found = True
                    break
        if valid_found: return new
    return new


def team_move(ctx, sch, rng, **kwargs):
    new = deepcopy(sch)
    team_map = kwargs.get('team_map')
    if not team_map: return new

    t_keys = list(team_map.keys())
    if not t_keys: return new

    for _ in range(10):
        tid = rng.choice(t_keys)
        mems = team_map[tid]
        if len(mems) < 2: continue

        a, b = rng.sample(mems, 2)
        d = rng.choice(ctx.dates)

        if (a.id, d) in ctx.um_locks or (b.id, d) in ctx.um_locks: continue

        sa, sb = sch.get_shift(a.id, d), sch.get_shift(b.id, d)
        if sa == sb: continue

        if _check_safety_locally(sch, a.id, d, sb, ctx.dates) and \
                _check_safety_locally(sch, b.id, d, sa, ctx.dates):
            new.set_shift(a.id, d, sb)
            new.set_shift(b.id, d, sa)
            return new
    return new


def night_pattern_mutation(ctx, sch, rng, **kwargs):
    new = deepcopy(sch)
    patterns = [["N", "N", "O", "O"], ["N", "N", "N", "O"]]

    for _ in range(5):
        n = rng.choice(ctx.nurses)
        pat = rng.choice(patterns)
        length = len(pat)
        if len(ctx.dates) < length: continue

        start = rng.randrange(0, len(ctx.dates) - length + 1)

        # Lock check
        safe = True
        for k in range(length):
            if (n.id, ctx.dates[start + k]) in ctx.um_locks:
                safe = False;
                break
        if not safe: continue

        for k in range(length):
            new.set_shift(n.id, ctx.dates[start + k], pat[k])
        return new
    return new


# 연산자 목록
OPS_LIST = [
    ("block_1_2", block_mutation_1_2_days),
    ("team_move", team_move),
    ("night_pattern", night_pattern_mutation)
]
OPS_MAP = {name: func for name, func in OPS_LIST}
NUM_OPS = len(OPS_LIST)
DQN_MODEL_PATH = "operator_dqn.pth"


# ---------------------------------------------------------
# RL Agent Classes
# ---------------------------------------------------------

class BaseAgent:
    def select_action(self, state_features: list) -> int:
        raise NotImplementedError

    def update(self, action_idx, reward, next_state_features):
        pass

    def save(self):
        pass


class RandomAgent(BaseAgent):
    def __init__(self, rng):
        self.rng = rng

    def select_action(self, state_features):
        return self.rng.randint(0, NUM_OPS - 1)


class MABAgent(BaseAgent):
    """
    Multi-Armed Bandit (Epsilon-Greedy)
    - 상태(Context) 무시, 오직 평균 보상만 추적
    """

    def __init__(self, session, ward_id, rng, epsilon=0.2, alpha=0.1):
        self.session = session
        self.ward_id = ward_id
        self.rng = rng
        self.epsilon = epsilon
        self.alpha = alpha

        # Initialize Weights
        self.q_values = [1.0] * NUM_OPS
        self.counts = [0] * NUM_OPS

        if session:
            self._load_from_db()

    def _load_from_db(self):
        rows = self.session.query(OperatorWeight).filter_by(ward_id=self.ward_id).all()
        w_map = {r.operator_name: r.weight for r in rows}
        for i, (name, _) in enumerate(OPS_LIST):
            if name in w_map:
                self.q_values[i] = w_map[name]

    def select_action(self, state_features):
        if self.rng.random() < self.epsilon:
            return self.rng.randint(0, NUM_OPS - 1)

        # Argmax
        max_q = max(self.q_values)
        candidates = [i for i, q in enumerate(self.q_values) if q == max_q]
        return self.rng.choice(candidates)

    def update(self, action_idx, reward, next_state):
        # Q(a) = Q(a) + alpha * (r - Q(a))
        old_q = self.q_values[action_idx]
        # Reward Scailing: 0~100 사이로 보정 (너무 크면 발산함)
        scaled_reward = max(-10, min(10, reward))

        new_q = old_q + self.alpha * (scaled_reward - old_q)
        self.q_values[action_idx] = max(0.1, new_q)
        self.counts[action_idx] += 1

    def save(self):
        if not self.session: return
        try:
            for i, (name, _) in enumerate(OPS_LIST):
                obj = self.session.query(OperatorWeight).filter_by(ward_id=self.ward_id, operator_name=name).first()
                if not obj:
                    obj = OperatorWeight(ward_id=self.ward_id, operator_name=name)
                    self.session.add(obj)
                obj.weight = self.q_values[i]
                obj.select_count += self.counts[i]  # 누적
            self.session.commit()
        except Exception:
            self.session.rollback()


class DQNAgent(BaseAgent):
    """
    Contextual Bandit / RL (DQN)
    - 상태(Context)를 보고 최적의 연산자 선택
    """

    def __init__(self, session, ward_id, rng, epsilon=0.2):
        self.session = session
        self.ward_id = ward_id
        self.rng = rng
        self.epsilon = epsilon

        self.model = OperatorDQN(input_dim=4, output_dim=NUM_OPS)
        self._load_model()

        self.experiences = []  # Buffer for batch save

    def _load_model(self):
        if os.path.exists(DQN_MODEL_PATH):
            try:
                self.model.load_state_dict(torch.load(DQN_MODEL_PATH))
                self.model.eval()
            except:
                pass

    def select_action(self, state_features):
        if self.rng.random() < self.epsilon:
            return self.rng.randint(0, NUM_OPS - 1)

        # Inference
        with torch.no_grad():
            state_tensor = torch.tensor([state_features], dtype=torch.float32)
            q_values = self.model(state_tensor)
            action = torch.argmax(q_values).item()
        return action

    def update(self, action_idx, reward, next_state_features):
        # 학습은 Offline(배치)로 진행하므로, 여기서는 경험(Experience)을 수집하여 DB에 저장

        # Scaling
        scaled_reward = max(-50, min(50, reward))  # Clip reward

        if self.session:
            exp = LocalSearchExperience(
                ward_id=self.ward_id,
                feature_progress=next_state_features[0],
                feature_violation_cnt=next_state_features[1],
                feature_score_norm=next_state_features[2],
                feature_fail_rate=next_state_features[3],
                action_index=action_idx,
                reward=scaled_reward
            )
            self.experiences.append(exp)

            # Batch Flush to DB (100개 단위로 저장)
            if len(self.experiences) >= 100:
                self._flush_experiences()

    def _flush_experiences(self):
        try:
            self.session.bulk_save_objects(self.experiences)
            self.session.commit()
        except:
            self.session.rollback()
        finally:
            self.experiences = []

    def save(self):
        self._flush_experiences()


# ---------------------------------------------------------
# 3. Main Logic: Improve
# ---------------------------------------------------------

def _get_state_features(step, max_steps, current_score, violations, recent_fails_count):
    """
    Context Feature Extraction (4-dim)
    """
    progress = step / max_steps if max_steps > 0 else 0
    viol_cnt = len(violations)
    # Score Normalization: 0 근처로 맞춤 (상대적 비교용)
    score_norm = np.tanh(current_score / 1000.0)
    # Recent Fail Rate (last 10 steps)
    fail_rate = recent_fails_count / 10.0 if recent_fails_count else 0.0

    return [progress, viol_cnt, score_norm, fail_rate]


def improve(
        ctx: PlanContext,
        sch: ScheduleMatrix,
        cfg: ScoreConfig,
        *,
        session=None,
        candidate_id: Optional[int] = None,
        seed: Optional[int] = None,
        iterations: int = 500,
        local_search_mode: str = "random",
        surrogate_mode: str = "none"  # [New] Surrogate Mode
) -> Tuple[ScheduleMatrix, float]:
    if seed is None:
        seed = random.SystemRandom().randint(0, 2 ** 31 - 1)
    rng = random.Random(seed)

    best = sch
    best_score = evaluate_schedule(ctx, best, cfg).total

    # Team Map Caching
    team_map = {}
    for n in ctx.nurses:
        if n.team_id:
            team_map.setdefault(n.team_id, []).append(n)

    # 1. Select Agent Strategy
    if local_search_mode == "dqn":
        agent = DQNAgent(session, ctx.ward_id, rng)
    elif local_search_mode == "mab":
        agent = MABAgent(session, ctx.ward_id, rng)
    else:
        agent = RandomAgent(rng)

    # [New] Surrogate Manager Init
    surrogate = SurrogateManager(session, mode=surrogate_mode)

    recent_fails = 0
    logs_buffer = []
    eval_buffer = []  # [New] Surrogate Training Data Buffer

    for i in range(iterations):
        # 2. Get Context (State)
        curr_violations = validate_all(ctx, best)
        curr_score = best_score

        state_feat = _get_state_features(i, iterations, curr_score, curr_violations, recent_fails)

        # 3. Select Action
        op_idx = agent.select_action(state_feat)
        op_name, op_func = OPS_LIST[op_idx]

        # 4. Apply
        cand = op_func(ctx, best, rng, team_map=team_map)

        # [New] Surrogate Check (Early Rejection)
        # 5% 확률로 무작위 통과(Exploration)를 허용하여 모델 편향 방지
        if surrogate.is_ready and rng.random() > 0.05:
            # 점수가 현재보다 현저히 나쁠 것으로 예측되면 스킵
            if surrogate.should_skip(ctx, cand, best_score, threshold=50.0):
                # 스킵 시 에이전트에게 약간의 패널티(시간 낭비 방지 보상) 부여
                agent.update(op_idx, -5.0, state_feat)
                continue

        # 5. Evaluate (Real Expensive Calculation)
        cand_violations = validate_all(ctx, cand)

        if cand_violations:
            reward = -10.0  # Penalty for invalid state
            agent.update(op_idx, reward, state_feat)
            recent_fails = min(10, recent_fails + 1)
            continue

        cand_score = evaluate_schedule(ctx, cand, cfg).total
        delta = cand_score - best_score

        # [New] Collect Data for Surrogate Training (MLP Features)
        if session and surrogate_mode != "none":
            # P0: MLP용 Handcrafted Features 수집
            feats = FeatureExtractor.extract_handcrafted(ctx, cand)
            # CNN용 Raw Data 수집은 용량 문제로 여기선 제외 (필요 시 별도 로직)

            eval_buffer.append(EvaluationLog(
                ward_id=ctx.ward_id,
                features=feats,
                score=cand_score
            ))

            # Batch Save Evaluation Logs
            if len(eval_buffer) >= 200:
                try:
                    session.bulk_save_objects(eval_buffer)
                    session.commit()
                except Exception:
                    session.rollback()
                finally:
                    eval_buffer = []

        # 6. Update Agent (Feedback)
        if delta > 0:
            best, best_score = cand, cand_score
            reward = delta
            recent_fails = 0

            # Log successful move
            if session and candidate_id:
                logs_buffer.append(LocalSearchLog(
                    candidate_id=candidate_id, operator=op_name,
                    delta_score=float(delta), violations_count=0
                ))
        else:
            reward = 0.0
            recent_fails = min(10, recent_fails + 1)

        # Update Agent (Next state approximation)
        next_state_feat = _get_state_features(i + 1, iterations, best_score, [], recent_fails)
        agent.update(op_idx, reward, next_state_feat)

        # Batch Log Flush
        if session and len(logs_buffer) >= 500:
            try:
                session.bulk_save_objects(logs_buffer)
                session.commit()
            except Exception:
                session.rollback()
            finally:
                logs_buffer = []

    # Finalize
    if hasattr(agent, 'save'):
        agent.save()

    # Flush remaining logs
    if session:
        try:
            if logs_buffer:
                session.bulk_save_objects(logs_buffer)
            if eval_buffer:
                session.bulk_save_objects(eval_buffer)
            session.flush()
        except Exception:
            session.rollback()

    return best, float(best_score)