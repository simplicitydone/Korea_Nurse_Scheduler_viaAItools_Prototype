# scheduler/deep_tuner.py

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os  # [Fix] Added import
from sqlalchemy.orm import Session
from scheduler.db import ExecutionLog, ScoreWeightHistory
import random

MODEL_FILE = "deep_model.pth"

class MetaSolverNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MetaSolverNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

class DeepTuner:
    def __init__(self, session: Session):
        self.session = session
        self.model = None
        self.is_trained = False
        self.target_keys = [
            "param_n_candidates", "param_top_k", "param_ls_iterations",
            "den_balance", "off_balance"
        ]
        self._load_model()

    def _load_model(self):
        if os.path.exists(MODEL_FILE):
            try:
                self.model = MetaSolverNet(3, 5)
                self.model.load_state_dict(torch.load(MODEL_FILE))
                self.model.eval()
                self.is_trained = True
                print("DeepTuner model loaded.")
            except Exception as e:
                print(f"Deep load failed: {e}")

    def _save_model(self):
        if self.model:
            torch.save(self.model.state_dict(), MODEL_FILE)

    def ensure_data(self):
        try:
            count = self.session.query(ExecutionLog).count()
            if count >= 10: return

            logs = []
            for _ in range(15 - count):
                n_cand = random.choice([10, 20, 30, 50])
                n_iter = random.choice([200, 500, 1000])
                top_k = random.choice([1, 3, 5])

                init_score = 100.0 - (1000.0 / n_cand)
                delta = (np.log(n_iter) * 5) * (1 + top_k * 0.1)
                final_score = init_score + delta + random.uniform(-2, 2)

                logs.append(ExecutionLog(
                    ward_id="SYNTHETIC",
                    method="synthetic",
                    n_nurses=random.choice([10, 15, 20]),
                    n_days=30,
                    n_constraints=random.randint(2, 10),
                    param_n_candidates=n_cand,
                    param_top_k=top_k,
                    param_ls_iterations=n_iter,
                    score_initial_best=init_score,
                    score_final=final_score,
                    time_generation=0.5,
                    time_improvement=0.5,
                    time_total=1.0
                ))

            if logs:
                self.session.add_all(logs)
                self.session.commit()
        except Exception as e:
            self.session.rollback()
            print(f"DeepTuner data ensure error: {e}")

    def train(self):
        self.ensure_data()
        try:
            logs = pd.read_sql(self.session.query(ExecutionLog).statement, self.session.bind)
        except Exception:
            return False

        if len(logs) < 5: return False

        threshold = logs['score_final'].quantile(0.5)
        good_logs = logs[logs['score_final'] >= threshold].copy()

        avg_weights = self._get_avg_weights(good_logs['ward_id'].unique())

        X_data = []
        y_data = []

        for _, row in good_logs.iterrows():
            x_norm = [
                min(row['n_nurses'] / 50.0, 1.0),
                min(row['n_days'] / 31.0, 1.0),
                min(row['n_constraints'] / 20.0, 1.0)
            ]
            X_data.append(x_norm)

            ward_weights = avg_weights.get(row['ward_id'], {})
            w_den = ward_weights.get('den_balance', 1.0)
            w_off = ward_weights.get('off_balance', 1.0)

            y_norm = [
                row['param_n_candidates'] / 100.0,
                row['param_top_k'] / 10.0,
                np.log1p(row['param_ls_iterations']),
                w_den / 50.0,
                w_off / 50.0
            ]
            y_data.append(y_norm)

        if not X_data: return False

        X_tensor = torch.tensor(X_data, dtype=torch.float32)
        y_tensor = torch.tensor(y_data, dtype=torch.float32)

        input_dim = 3
        output_dim = len(self.target_keys)

        self.model = MetaSolverNet(input_dim, output_dim)
        optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(200):
            optimizer.zero_grad()
            out = self.model(X_tensor)
            loss = criterion(out, y_tensor)
            loss.backward()
            optimizer.step()

        self.is_trained = True
        self._save_model()
        return True

    def predict(self, n_nurses, n_days, n_constraints):
        if not self.is_trained:
            if not self.train():
                return None

        self.model.eval()
        with torch.no_grad():
            x_norm = [
                min(n_nurses / 50.0, 1.0),
                min(n_days / 31.0, 1.0),
                min(n_constraints / 20.0, 1.0)
            ]
            x = torch.tensor([x_norm], dtype=torch.float32)
            out = self.model(x).numpy()[0]

        pred_cand = out[0] * 100.0
        pred_topk = out[1] * 10.0
        pred_iter = np.expm1(out[2])
        pred_w_den = out[3] * 50.0
        pred_w_off = out[4] * 50.0

        res = {
            "param_n_candidates": int(max(10, pred_cand)),
            "param_top_k": int(max(1, pred_topk)),
            "param_ls_iterations": int(max(200, pred_iter)),
            "den_balance": max(0.1, pred_w_den),
            "off_balance": max(0.1, pred_w_off)
        }

        return res

    def _get_avg_weights(self, ward_ids):
        res = {}
        for wid in ward_ids:
            if wid == "SYNTHETIC": continue
            rows = self.session.query(ScoreWeightHistory).filter_by(ward_id=wid).all()
            temp = {}
            for r in rows:
                if r.rule_id in ['den_balance', 'off_balance']:
                    temp.setdefault(r.rule_id, []).append(r.new_weight)
            if temp:
                res[wid] = {k: np.mean(v) for k, v in temp.items()}
        return res