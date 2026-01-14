# scheduler/train_dqn_학습 로직 구현 필요.py

import torch
import torch.nn as nn
import torch.optim as optim
from scheduler.db import SessionLocal, LocalSearchExperience
from scheduler.ai_models import OperatorDQN


def train_dqn():
    session = SessionLocal()
    # 1. 데이터 로드 (최근 10000개)
    exps = session.query(LocalSearchExperience).order_by(LocalSearchExperience.id.desc()).limit(10000).all()
    if len(exps) < 100:
        print("Not enough data to train DQN.")
        return

    # 2. 데이터셋 변환
    X = []
    y = []  # DQN은 Q-Target 계산이 필요하지만, 여기서는 Simplified Monte-Carlo 방식(Reward 직접 학습)으로 예시
    # 정석적인 DQN은 (s, a, r, s')가 필요하나, DB엔 s, a, r만 있음.
    # Contextual Bandit 관점에서는 CrossEntropyLoss나 MSE로 Q(s,a) -> r 학습 가능.

    for e in exps:
        state = [e.feature_progress, e.feature_violation_cnt, e.feature_score_norm, e.feature_fail_rate]
        target = [0.0] * 3  # Operator 개수
        # 해당 Action의 Q-Value만 Reward로 업데이트하도록 Masking하거나 Loss 조정 필요
        # 간단히: Action Index에 해당하는 값을 Reward로 설정하고 학습
        X.append(state)
        y.append([e.action_index, e.reward])

    # ... 학습 로직 구현 ...
    # 모델 저장
    torch.save(model.state_dict(), "operator_dqn.pth")