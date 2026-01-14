# scheduler/train_deep.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from sqlalchemy.orm import Session
from collections import defaultdict

from scheduler.db import SessionLocal, ScheduleAssignment, Schedule, Nurse, SchedulePlan
from scheduler.ai_models import ShiftLSTM
from scheduler.generator import generate_initial_schedule
from scheduler.data_loader import load_plan_context
from scheduler.types import ShiftCode

# Config
MODEL_PATH = "lstm_model.pth"
MIN_SAMPLES_FOR_REAL_TRAIN = 500  # 실제 데이터가 이보다 적으면 가상 데이터 생성
SYNTHETIC_EPISODES = 50  # 가상으로 생성할 스케줄 개수
WINDOW_SIZE = 7  # 입력 시퀀스 길이
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.01

SHIFT_TO_IDX = {'D': 0, 'E': 1, 'N': 2, 'O': 3}


def get_latest_plan_id(session):
    plan = session.query(SchedulePlan).order_by(SchedulePlan.id.desc()).first()
    return plan.id if plan else None


def prepare_sequences(assignments):
    """
    (nurse_id, date) 순으로 정렬된 데이터를 시퀀스로 변환
    """
    # 1. 간호사별 시계열 정리
    nurse_shifts = defaultdict(list)
    for r in assignments:
        s = r.shift_code
        if s == 'M':
            s = 'E'
        elif s == 'C':
            s = 'D'
        elif s == 'P':
            s = 'O'

        if s in SHIFT_TO_IDX:
            nurse_shifts[r.nurse_id].append(SHIFT_TO_IDX[s])

    # 2. Sliding Window로 데이터셋 생성
    X, y = [], []
    for nid, shifts in nurse_shifts.items():
        if len(shifts) <= WINDOW_SIZE: continue

        for i in range(len(shifts) - WINDOW_SIZE):
            seq_in = shifts[i: i + WINDOW_SIZE]
            target = shifts[i + WINDOW_SIZE]
            X.append(seq_in)
            y.append(target)

    return X, y


def generate_synthetic_data(session):
    """
    데이터 부족 시 Heuristic Generator를 이용해 가상 데이터 생성
    """
    print("⚠️ Not enough real data. Generating synthetic data for warm-up...")

    plan_id = get_latest_plan_id(session)
    if not plan_id:
        print("❌ No Schedule Plan found. Cannot generate synthetic data.")
        return []

    try:
        ctx = load_plan_context(session, plan_id)
    except Exception as e:
        print(f"❌ Failed to load context: {e}")
        return []

    synthetic_assignments = []

    for _ in range(SYNTHETIC_EPISODES):
        # Heuristic 모드로 생성 (규칙 준수 스케줄)
        sch = generate_initial_schedule(ctx, seed=None, mode="heuristic")

        # DataFrame을 ScheduleAssignment 객체 리스트처럼 변환 (구조만 맞춤)
        # 실제 DB 객체가 아니라 namedtuple이나 dict 유사 객체로 만듦
        class MockAssignment:
            def __init__(self, nid, sc):
                self.nurse_id = nid
                self.shift_code = sc

        # 날짜 순서대로 정렬하기 위해 dates 순회
        for d in ctx.dates:
            for n in ctx.nurses:
                shift = sch.get_shift(n.id, d)
                synthetic_assignments.append(MockAssignment(n.id, shift))

    print(f"✅ Generated {len(synthetic_assignments)} synthetic shift records.")
    return synthetic_assignments


def train_model():
    session = SessionLocal()
    try:
        # 1. 실제 데이터 로드
        real_data = (
            session.query(ScheduleAssignment)
            .join(Schedule)
            .join(Nurse)
            .order_by(ScheduleAssignment.nurse_id, ScheduleAssignment.date)
            .all()
        )

        # 2. Cold Start 체크
        assignments = real_data
        if len(real_data) < MIN_SAMPLES_FOR_REAL_TRAIN:
            # 가상 데이터 생성 및 합치기
            syn_data = generate_synthetic_data(session)
            assignments = real_data + syn_data  # 리스트 합치기

        if not assignments:
            print("❌ No data available for training.")
            return

        # 3. 데이터셋 구성
        X_raw, y_raw = prepare_sequences(assignments)

        if not X_raw:
            print("❌ Not enough sequences formed.")
            return

        X_tensor = torch.tensor(X_raw, dtype=torch.long)
        y_tensor = torch.tensor(y_raw, dtype=torch.long)

        # 4. 모델 초기화
        model = ShiftLSTM(num_shifts=4)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # 기존 모델이 있다면 로드해서 이어서 학습 (Fine-tuning)
        if os.path.exists(MODEL_PATH):
            try:
                model.load_state_dict(torch.load(MODEL_PATH))
                print("🔄 Loaded existing model for fine-tuning.")
            except:
                print("⚠️ Failed to load existing model. Starting fresh.")

        # 5. 학습 루프
        model.train()
        print(f"🚀 Starting training on {len(X_raw)} sequences...")

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        for epoch in range(EPOCHS):
            total_loss = 0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = model(batch_X)  # (Batch, 4)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {total_loss / len(loader):.4f}")

        # 6. 저장
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"💾 Model saved to {MODEL_PATH}")

    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        session.close()


if __name__ == "__main__":
    train_model()