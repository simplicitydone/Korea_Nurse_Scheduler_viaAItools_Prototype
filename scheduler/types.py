# scheduler/types.py
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Tuple, Literal, Set

import pandas as pd

ShiftCode = Literal["D", "E", "N", "O"]


@dataclass
class NurseInfo:
    id: int
    employee_no: str
    name: str
    ward_id: str
    team_id: Optional[str] = None
    level_weight: float = 1.0

    # [New] 단축근무 비율 (기본 1.0)
    short_work_factor: float = 1.0

    # [New] Tier 1 & 2 Attributes (Set for O(1) lookup)
    tags: Set[str] = field(default_factory=set)  # ex: {"pregnant", "night_keep"}
    preferred: Set[str] = field(default_factory=set)  # ex: {"D", "E"}
    avoid: Set[str] = field(default_factory=set)  # ex: {"N"}

    leader_eligible: Dict[ShiftCode, bool] = field(
        default_factory=lambda: {"D": False, "E": False, "N": False, "O": False}
    )
    preceptor_id: Optional[int] = None
    preceptee_id: Optional[int] = None
    is_preceptee: bool = False
    avoid_ids: List[int] = field(default_factory=list)


@dataclass
class CoverageNeed:
    date: date
    shift: ShiftCode
    min_required: int


@dataclass
class PlanContext:
    plan_id: int
    ward_id: str
    start: date
    end: date
    dates: List[date]
    nurses: List[NurseInfo]
    teams: List[str]
    coverage_needs: List[CoverageNeed]
    leader_required: Dict[Tuple[date, ShiftCode], bool]
    um_locks: Dict[Tuple[int, date], ShiftCode]
    nurse_requests: Dict[Tuple[int, date], ShiftCode]
    is_holiday: Dict[date, bool]
    is_weekend: Dict[date, bool]

    # [Fix] Generator 오류 해결을 위한 필수 필드
    previous_schedule: Dict[Tuple[int, date], ShiftCode] = field(default_factory=dict)


@dataclass
class ScheduleMatrix:
    # Primary storage: (nurse_id, date) -> ShiftCode
    grid: Dict[Tuple[int, date], ShiftCode] = field(default_factory=dict)

    # [Cache] DataFrame 캐싱을 위한 내부 변수
    _df_cache: Optional[pd.DataFrame] = field(default=None, init=False, repr=False)

    @property
    def df(self) -> pd.DataFrame:
        if self._df_cache is not None:
            return self._df_cache

        if not self.grid:
            return pd.DataFrame()

        try:
            ser = pd.Series(self.grid)
            df_ = ser.unstack(level=1)
            self._df_cache = df_
            return df_
        except Exception:
            return pd.DataFrame()

    def get_shift(self, nurse_id: int, d: date) -> ShiftCode:
        return self.grid.get((nurse_id, d), "O")

    def set_shift(self, nurse_id: int, d: date, s: ShiftCode):
        if self.grid.get((nurse_id, d)) != s:
            self.grid[(nurse_id, d)] = s
            self._df_cache = None

    def normalize(self):
        self._df_cache = None

    def to_sequence(self, nurse_id: int, dates: List[date]) -> List[ShiftCode]:
        return [self.grid.get((nurse_id, d), "O") for d in dates]

    def to_string(self, nurse_id: int, dates: List[date]) -> str:
        return "".join(self.grid.get((nurse_id, d), "O") for d in dates)


@dataclass
class RuleScore:
    rule_id: str
    score: float
    details: dict


@dataclass
class ScoreResult:
    total: float
    by_rule: List[RuleScore]
    fairness: dict