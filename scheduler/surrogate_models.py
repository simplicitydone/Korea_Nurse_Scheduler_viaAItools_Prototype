# scheduler/surrogate_models.py

import torch
import torch.nn as nn
import torch.optim as optim
import joblib
import os
import numpy as np
from sqlalchemy.orm import Session
from scheduler.db import EvaluationLog
from scheduler.features import FeatureExtractor

MODEL_PATH_MLP = "surrogate_mlp.pth"
MODEL_PATH_CNN = "surrogate_cnn.pth"


# --- Models ---
class ScoreMLP(nn.Module):
    def __init__(self, input_dim):
        super(ScoreMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


class ScheduleCNN(nn.Module):
    def __init__(self):
        super(ScheduleCNN, self).__init__()
        # Input: (4, Nurse, Day)
        self.conv1 = nn.Conv2d(4, 16, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # Global Pooling
        self.fc = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)  # (Batch, 32, 1, 1)
        x = x.view(x.size(0), -1)  # Flatten
        return self.fc(x)


# --- Manager ---
class SurrogateManager:
    def __init__(self, session: Session, mode="none"):
        self.session = session
        self.mode = mode  # 'none', 'mlp', 'cnn'
        self.model = None
        self.is_ready = False
        self._load_model()

    def _load_model(self):
        if self.mode == "mlp" and os.path.exists(MODEL_PATH_MLP):
            try:
                # Input dim은 feature 수에 따라 다름 (예: 7)
                self.model = ScoreMLP(input_dim=7)
                self.model.load_state_dict(torch.load(MODEL_PATH_MLP))
                self.model.eval()
                self.is_ready = True
            except:
                pass
        elif self.mode == "cnn" and os.path.exists(MODEL_PATH_CNN):
            try:
                self.model = ScheduleCNN()
                self.model.load_state_dict(torch.load(MODEL_PATH_CNN))
                self.model.eval()
                self.is_ready = True
            except:
                pass

    def should_skip(self, ctx, sch, current_best_score, threshold=50.0):
        """
        True를 반환하면 Evaluator를 건너뛰고 이 후보를 버림 (Pruning).
        False를 반환하면 정밀 평가 수행.
        """
        if not self.is_ready: return False  # 모델 없으면 무조건 평가

        pred_score = 0.0
        with torch.no_grad():
            if self.mode == "mlp":
                feats = FeatureExtractor.extract_handcrafted(ctx, sch)
                tensor = torch.tensor([feats], dtype=torch.float32)
                pred_score = self.model(tensor).item()
            elif self.mode == "cnn":
                tensor = FeatureExtractor.extract_tensor(ctx, sch).unsqueeze(0)
                pred_score = self.model(tensor).item()

        # 예측 점수가 현재 최고 점수보다 현저히 낮으면 스킵 (Conservative Pruning)
        # 점수가 높을수록 좋다고 가정 (음수 점수 체계라면 logic 주의)
        # 현재 시스템: 0에 가까울수록 좋음 (음수).
        # 예: Best -100, Pred -500 -> 차이 400 > Threshold -> Skip
        if pred_score < (current_best_score - threshold):
            return True
        return False

    def train(self):
        """DB의 EvaluationLog를 사용하여 학습"""
        logs = self.session.query(EvaluationLog).order_by(EvaluationLog.id.desc()).limit(2000).all()
        if len(logs) < 100: return False

        if self.mode == "mlp":
            X = [log.features for log in logs]  # JSON list
            y = [[log.score] for log in logs]

            X_t = torch.tensor(X, dtype=torch.float32)
            y_t = torch.tensor(y, dtype=torch.float32)

            model = ScoreMLP(input_dim=len(X[0]))
            opt = optim.Adam(model.parameters(), lr=0.001)
            loss_fn = nn.MSELoss()

            model.train()
            for _ in range(100):
                opt.zero_grad()
                out = model(X_t)
                loss = loss_fn(out, y_t)
                loss.backward()
                opt.step()

            torch.save(model.state_dict(), MODEL_PATH_MLP)
            self.model = model
            self.is_ready = True
            return True

        elif self.mode == "cnn":
            # CNN 학습은 Raw Data가 필요하므로 현재 EvaluationLog 구조로는 한계가 있음.
            # 추후 EvaluationLog에 Raw Grid를 저장하거나,
            # 온라인 학습(메모리 버퍼) 방식을 써야 함.
            # 여기서는 Placeholder.
            print("CNN training requires raw grid data storage.")
            return False

        return False