# scheduler/features.py

from datetime import timedelta, date
import math
import numpy as np
import torch
from scheduler.types import ScheduleMatrix, PlanContext

def get_date_features(target_date: date, holiday_dates: set) -> dict:
    """
    날짜에 대한 문맥 Feature를 추출합니다.
    """
    features = {}

    # 1. 기본 달력 정보
    features['month'] = target_date.month
    features['weekday'] = target_date.weekday()  # 0=Mon, 6=Sun
    features['day'] = target_date.day

    # 2. 휴일 관련
    is_hol = target_date in holiday_dates
    features['is_holiday'] = 1 if is_hol else 0
    features['is_weekend'] = 1 if target_date.weekday() >= 5 else 0

    # 3. 전후 관계 (Lag/Lead)
    prev_1 = target_date - timedelta(days=1)
    prev_2 = target_date - timedelta(days=2)
    next_1 = target_date + timedelta(days=1)

    is_prev_hol = prev_1 in holiday_dates or prev_1.weekday() >= 5
    is_next_hol = next_1 in holiday_dates or next_1.weekday() >= 5

    # 연휴 후유증 (연휴/주말 다음날 평일)
    features['is_post_holiday'] = 1 if (is_prev_hol and not is_hol and not features['is_weekend']) else 0

    # 징검다리 휴일 (샌드위치)
    features['is_sandwich'] = 1 if (is_prev_hol and is_next_hol and not is_hol) else 0

    # 4. 월요일 특성 (주말 직후 월요일)
    features['is_monday'] = 1 if target_date.weekday() == 0 else 0

    return features


def explain_features(features: dict, prediction_delta: int) -> str:
    """
    예측 결과(증감)에 대한 이유를 텍스트로 생성
    """
    if prediction_delta == 0:
        return ""

    reasons = []
    action = "추가" if prediction_delta > 0 else "감원"

    if features['is_post_holiday']:
        reasons.append("연휴/주말 다음날")
    elif features['is_monday']:
        reasons.append("월요일 입원 증가 예상")
    elif features['is_sandwich']:
        reasons.append("징검다리 휴일(환자 감소 예상)")
    elif features['is_holiday']:
        reasons.append("공휴일")
    elif features['is_weekend']:
        reasons.append("주말")

    if not reasons:
        return ""  # 특별한 이유 없으면 메모 남기지 않음

    return f"[AI] {', '.join(reasons)}으로 인원 {action} ({prediction_delta:+d}명)"


def features_to_vector(features: dict) -> list:
    """ML 모델 입력용 벡터 변환"""
    # 순서 중요
    return [
        features['month'],
        features['weekday'],
        features['is_holiday'],
        features['is_weekend'],
        features['is_post_holiday'],
        features['is_sandwich']
    ]


class FeatureExtractor:
    @staticmethod
    def extract_handcrafted(ctx: PlanContext, sch: ScheduleMatrix) -> list:
        """
        [P0: MLP용] 통계적 특징 추출 (가벼운 연산)
        Return: List of floats
        """
        # 1. 근무 통계
        df = sch.df.replace({'M': 'E', 'C': 'D', 'P': 'O'})  # Normalize
        if df.empty: return [0.0] * 10

        # 각 간호사별 근무 횟수
        counts = df.apply(pd.Series.value_counts, axis=1).fillna(0)

        # 표준편차 (공정성 지표)
        std_d = counts['D'].std() if 'D' in counts else 0
        std_n = counts['N'].std() if 'N' in counts else 0

        # 합계 (전체 커버리지 지표)
        total_d = counts['D'].sum() if 'D' in counts else 0
        total_n = counts['N'].sum() if 'N' in counts else 0

        # 주말 근무 편차
        weekend_cols = [d for d in ctx.dates if ctx.is_weekend.get(d, False)]
        if weekend_cols:
            wh_counts = (df[weekend_cols] != 'O').sum(axis=1)
            std_wh = wh_counts.std()
        else:
            std_wh = 0

        # 연속 근무 위반 근사치 (정밀 계산은 비싸므로 간단히 체크)
        # 여기서는 생략하거나 간단한 패턴 매칭만 수행

        return [
            float(std_d), float(std_n), float(std_wh),
            float(total_d), float(total_n),
            len(ctx.nurses), len(ctx.dates)
        ]

    @staticmethod
    def extract_tensor(ctx: PlanContext, sch: ScheduleMatrix) -> torch.Tensor:
        """
        [P1: CNN용] 스케줄 그리드를 이미지 텐서로 변환
        Shape: (Channels=4, Height=Nurses, Width=Days)
        Channels: D, E, N, O (One-hot encoding)
        """
        # Mapping: D=0, E=1, N=2, O=3
        mapping = {'D': 0, 'E': 1, 'N': 2, 'O': 3, 'M': 1, 'C': 0, 'P': 3}

        n_nurses = len(ctx.nurses)
        n_days = len(ctx.dates)

        # 초기화 (Nurse x Day)
        grid = np.full((n_nurses, n_days), 3, dtype=int)  # Default O

        # Fill data
        # sch.grid는 dict {(nid, date): shift}
        # 순서 보장을 위해 ctx.nurses와 ctx.dates 인덱스 사용
        nurse_idx_map = {n.id: i for i, n in enumerate(ctx.nurses)}
        date_idx_map = {d: i for i, d in enumerate(ctx.dates)}

        for (nid, d), shift in sch.grid.items():
            if nid in nurse_idx_map and d in date_idx_map:
                r = nurse_idx_map[nid]
                c = date_idx_map[d]
                grid[r, c] = mapping.get(shift, 3)

        # One-hot encoding to (N, D, 4)
        # Then transpose to (4, N, D) for PyTorch Conv2d
        tensor = torch.tensor(grid, dtype=torch.long)
        one_hot = torch.nn.functional.one_hot(tensor, num_classes=4)  # (N, D, 4)
        one_hot = one_hot.permute(2, 0, 1).float()  # (4, N, D)

        return one_hot


import pandas as pd