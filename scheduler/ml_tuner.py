# scheduler/ml_tuner.py

from __future__ import annotations
import pandas as pd
import numpy as np
import os  # [Fix] Added import
import joblib
from sklearn.ensemble import RandomForestRegressor
from sqlalchemy.orm import Session
from sqlalchemy import func
from scheduler.db import ExecutionLog, ScoreWeightHistory
import random

MODEL_PATH = "ml_model.pkl"


class MLTuner:
    def __init__(self, session: Session):
        self.session = session
        self.is_trained = False

        # Models
        self.model_gen_score = RandomForestRegressor(n_estimators=50, random_state=42)
        self.model_imp_delta = RandomForestRegressor(n_estimators=50, random_state=42)

        # Try Loading
        self._load_model()

    def _load_model(self):
        if os.path.exists(MODEL_PATH):
            try:
                data = joblib.load(MODEL_PATH)
                self.model_gen_score = data['gen']
                self.model_imp_delta = data['imp']
                self.is_trained = True
                print("MLTuner model loaded from disk.")
            except Exception as e:
                print(f"Failed to load ML model: {e}")

    def _save_model(self):
        try:
            joblib.dump({
                'gen': self.model_gen_score,
                'imp': self.model_imp_delta
            }, MODEL_PATH)
        except Exception as e:
            print(f"Failed to save ML model: {e}")

    def ensure_data_for_training(self):
        try:
            count = self.session.query(ExecutionLog).count()
            if count >= 10: return

            logs = []
            for _ in range(15 - count):
                n_cand = random.choice([10, 20, 30, 50, 80])
                n_iter = random.choice([200, 500, 1000, 2000])
                top_k = random.choice([1, 3, 5])

                gen_time = n_cand * 0.1 + random.uniform(0, 0.5)
                init_score = 100 - (1000 / n_cand) + random.uniform(-5, 5)

                imp_time = n_iter * top_k * 0.005 + random.uniform(0, 0.5)
                delta = (np.log(n_iter) * 5) * (1 + top_k * 0.1) + random.uniform(-2, 2)

                logs.append(ExecutionLog(
                    ward_id="SYNTHETIC",
                    method="synthetic",
                    n_nurses=15, n_days=30, n_constraints=5,
                    param_n_candidates=n_cand,
                    param_top_k=top_k,
                    param_ls_iterations=n_iter,
                    score_initial_best=init_score,
                    score_final=init_score + delta,
                    time_generation=gen_time,
                    time_improvement=imp_time,
                    time_total=gen_time + imp_time
                ))

            if logs:
                self.session.add_all(logs)
                self.session.commit()
        except Exception as e:
            self.session.rollback()
            print(f"Error ensuring data: {e}")

    def train(self) -> bool:
        self.ensure_data_for_training()
        try:
            query = self.session.query(ExecutionLog).statement
            df = pd.read_sql(query, self.session.bind)
        except Exception:
            return False

        if len(df) < 5: return False
        df = df.fillna(0)

        X_base = df[['n_nurses', 'n_days', 'n_constraints']]

        X_gen = pd.concat([X_base, df[['param_n_candidates']]], axis=1)
        self.model_gen_score.fit(X_gen, df['score_initial_best'])

        df['delta_score'] = df['score_final'] - df['score_initial_best']
        X_imp = pd.concat([X_base, df[['param_ls_iterations', 'param_top_k', 'score_initial_best']]], axis=1)
        self.model_imp_delta.fit(X_imp, df['delta_score'])

        self.is_trained = True
        self._save_model()
        return True

    def suggest_parameters(self, n_nurses, n_days, n_constraints, mode="balanced") -> tuple[int, int, int]:
        if not self.is_trained:
            if not self.train():
                return self._rule_based_fallback(n_nurses * n_days, mode)

        candidates_range = [30, 50, 80, 100]
        iterations_range = [1000, 2000, 3000, 5000]
        top_k_range = [3, 5]

        best_params = None
        best_score = -float('inf')

        for n_cand in candidates_range:
            X_gen = pd.DataFrame([[n_nurses, n_days, n_constraints, n_cand]],
                                 columns=['n_nurses', 'n_days', 'n_constraints', 'param_n_candidates'])
            pred_init = self.model_gen_score.predict(X_gen)[0]

            for top_k in top_k_range:
                for n_iter in iterations_range:
                    X_imp = pd.DataFrame([[n_nurses, n_days, n_constraints, n_iter, top_k, pred_init]],
                                         columns=['n_nurses', 'n_days', 'n_constraints', 'param_ls_iterations',
                                                  'param_top_k', 'score_initial_best'])
                    pred_delta = self.model_imp_delta.predict(X_imp)[0]
                    final = pred_init + pred_delta

                    if final > best_score:
                        best_score = final
                        best_params = (n_cand, top_k, n_iter)

        return best_params if best_params else (50, 5, 2000)

    def suggest_weights(self, ward_id: str) -> dict:
        try:
            rows = (
                self.session.query(
                    ScoreWeightHistory.rule_id,
                    func.avg(ScoreWeightHistory.new_weight).label("avg_weight")
                )
                .filter(ScoreWeightHistory.ward_id == ward_id)
                .group_by(ScoreWeightHistory.rule_id)
                .all()
            )
            suggestions = {}
            for r in rows:
                if r.rule_id and r.avg_weight is not None:
                    suggestions[r.rule_id] = float(r.avg_weight)
            return suggestions
        except Exception:
            return {}

    def _rule_based_fallback(self, complexity, mode):
        if mode == "fast":
            return (10, 2, 200)
        elif mode == "quality":
            return (50, 5, 2000)
        else:
            return (30, 3, 500)