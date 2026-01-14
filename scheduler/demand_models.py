# scheduler/demand_models.py

import joblib
import os
import numpy as np
import torch
import torch.nn as nn
from datetime import date
from sklearn.ensemble import RandomForestRegressor
from scheduler.features import get_date_features, features_to_vector, explain_features

MODEL_PATH_ML = "demand_rf.pkl"
MODEL_PATH_DL = "demand_lstm.pth"


class DemandPredictor:
    def __init__(self, mode="rule"):
        self.mode = mode
        self.ml_model = None
        self.dl_model = None
        self._load_models()

    def _load_models(self):
        if self.mode == "ml" and os.path.exists(MODEL_PATH_ML):
            try:
                self.ml_model = joblib.load(MODEL_PATH_ML)
            except:
                pass
        elif self.mode == "deep" and os.path.exists(MODEL_PATH_DL):
            try:
                # Simple LSTM definition needs to match training
                self.dl_model = DemandLSTM(input_size=6, hidden_size=32)
                self.dl_model.load_state_dict(torch.load(MODEL_PATH_DL))
                self.dl_model.eval()
            except:
                pass

    def predict(self, target_date: date, holiday_set: set, base_needs: dict) -> tuple[dict, str]:
        """
        Return: (adjusted_needs_dict, explanation_string)
        base_needs: {'D': 4, 'E': 4, 'N': 3} (Team Count 기반)
        """
        features = get_date_features(target_date, holiday_set)

        # 1. Rule / None (기본값 사용)
        if self.mode == "rule" or (self.mode == "ml" and not self.ml_model) or (
                self.mode == "deep" and not self.dl_model):
            # 기본 로직만 적용 (주말/공휴일 등) - 여기선 베이스라인 그대로 리턴하되,
            # 만약 "None" 모드여도 기본적인 주말 감축 로직을 원한다면 여기에 추가
            return base_needs, ""

        # 2. ML (RandomForest)
        if self.mode == "ml":
            vec = features_to_vector(features)
            # 모델은 '변동분(Delta)'을 예측한다고 가정 (-1, 0, +1)
            # D, E, N 각각 예측
            deltas = self.ml_model.predict([vec])[0]  # shape (3,) -> [d_delta, e_delta, n_delta]

            # 정수로 반올림
            d_delta = int(round(deltas[0]))
            e_delta = int(round(deltas[1]))
            n_delta = int(round(deltas[2]))

            adjusted = {
                'D': max(1, base_needs['D'] + d_delta),
                'E': max(1, base_needs['E'] + e_delta),
                'N': max(1, base_needs['N'] + n_delta)
            }

            # 설명 생성 (가장 큰 변동 기준)
            max_delta = max([d_delta, e_delta, n_delta], key=abs)
            note = explain_features(features, max_delta)
            return adjusted, note

        # 3. Deep (LSTM) - 시계열 문맥 필요 (구현 복잡도상 단일 시점 추론으로 단순화하거나 ML과 유사하게 처리)
        # 여기서는 ML과 유사하게 Feature 기반 추론으로 처리 (Sequence 처리는 training 단계에서 복잡함)
        if self.mode == "deep":
            vec = torch.tensor([features_to_vector(features)], dtype=torch.float32)
            with torch.no_grad():
                out = self.dl_model(vec.unsqueeze(0))  # (1, 1, input) -> (1, 3)
                deltas = out.numpy()[0]

            d_delta = int(round(deltas[0]))
            e_delta = int(round(deltas[1]))
            n_delta = int(round(deltas[2]))

            adjusted = {
                'D': max(1, base_needs['D'] + d_delta),
                'E': max(1, base_needs['E'] + e_delta),
                'N': max(1, base_needs['N'] + n_delta)
            }
            max_delta = max([d_delta, e_delta, n_delta], key=abs)
            note = explain_features(features, max_delta)
            return adjusted, note

        return base_needs, ""


# For Deep Learning
class DemandLSTM(nn.Module):
    def __init__(self, input_size=6, hidden_size=32):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 3)  # Output: D, E, N deltas

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])