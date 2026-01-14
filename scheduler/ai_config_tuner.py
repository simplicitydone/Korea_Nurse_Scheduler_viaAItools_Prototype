# scheduler/ai_config_tuner.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random


class MetricPredictor(nn.Module):
    """
    [Surrogate Model]
    Input: Weights Vector
    Output: Metrics Vector (Violations, StdDev)
    """

    def __init__(self, num_params, num_metrics):
        super(MetricPredictor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(num_params, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_metrics)
        )

    def forward(self, weights):
        return self.net(weights)


class AIConfigTuner:
    def __init__(self, param_keys: list = None, metric_keys: list = None):
        self.param_keys = param_keys or ["w_preference", "w_workload_balance"]
        self.metric_keys = metric_keys or ["pref_violations", "workload_std"]

        self.model = MetricPredictor(len(self.param_keys), len(self.metric_keys))
        self.is_trained = False

    def ensure_data(self):
        """
        [New] 초기 Cold Start 문제를 해결하기 위해 가상 데이터 주입
        """
        if self.is_trained: return

        # Synthetic Data Generation
        # w_pref(0) -> pref_viol(0), w_work(1) -> work_std(1)
        X_syn = []
        y_syn = []

        for _ in range(200):  # 200건의 가상 데이터
            # Random Weights (1.0 ~ 100.0)
            wp = random.uniform(1.0, 100.0)
            ww = random.uniform(1.0, 100.0)

            # Simulated Metrics (Inverse relationship + Noise)
            # 가중치가 높으면 metric(나쁜값)은 작아짐
            viol = max(0, (1000 / wp) + random.gauss(0, 2))
            std = max(0, (100 / ww) + random.gauss(0, 0.5))

            X_syn.append([wp, ww])
            y_syn.append([viol, std])

        self.train(X_syn, y_syn, is_synthetic=True)

    def train(self, X_data, y_data, is_synthetic=False):
        """
        X_data: List of weight vectors
        y_data: List of metric vectors
        """
        if not X_data: return False

        X = torch.tensor(X_data, dtype=torch.float32)
        y = torch.tensor(y_data, dtype=torch.float32)

        optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        self.model.train()
        epochs = 100 if is_synthetic else 500

        for _ in range(epochs):
            optimizer.zero_grad()
            preds = self.model(X)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()

        self.is_trained = True
        return True

    def tune_weights_by_intention(self, intention: dict):
        if not self.is_trained:
            self.ensure_data()

        self.model.eval()

        optimized_params = torch.tensor([10.0] * len(self.param_keys), dtype=torch.float32, requires_grad=True)
        optimizer = optim.Adam([optimized_params], lr=1.0)

        target_metrics = torch.zeros(len(self.metric_keys))

        idx_pref = self.metric_keys.index('pref_violations')
        idx_std = self.metric_keys.index('workload_std')

        if intention.get("preference") == "high":
            target_metrics[idx_pref] = 0.0
        else:
            target_metrics[idx_pref] = 10.0

        if intention.get("fairness") == "high":
            target_metrics[idx_std] = 0.1
        else:
            target_metrics[idx_std] = 5.0

        for _ in range(100):
            optimizer.zero_grad()
            predicted = self.model(optimized_params)
            loss = torch.mean((predicted - target_metrics) ** 2)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                optimized_params.clamp_(min=1.0, max=100.0)

        result = optimized_params.detach().numpy()
        return {k: float(v) for k, v in zip(self.param_keys, result)}