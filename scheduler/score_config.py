# scheduler/score_config.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Literal, List

from scheduler.types import ShiftCode

RuleType = Literal["count", "sequence", "preference", "fairness"]


@dataclass
class ScoreRule:
    id: str
    enabled: bool = True
    type: RuleType = "count"
    weight: float = 1.0
    scope: str = "nurse"

    min_value: Optional[int] = None
    max_value: Optional[int] = None
    pattern: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def _count_runs_ge(self, seq: List[ShiftCode], *, predicate, threshold: int) -> int:
        if threshold <= 0: return 0
        count = 0
        run = 0
        for s in seq:
            if predicate(s):
                run += 1
            else:
                if run >= threshold: count += 1
                run = 0
        if run >= threshold: count += 1
        return count

    def _count_sandwich_work_off_work(self, seq: List[ShiftCode]) -> int:
        if len(seq) < 3: return 0
        c = 0
        for i in range(len(seq) - 2):
            a, b, c3 = seq[i], seq[i + 1], seq[i + 2]
            if a != "O" and b == "O" and c3 != "O": c += 1
        return c

    def sequence_occurrences(self, seq: List[ShiftCode]) -> int:
        kind = (self.meta or {}).get("kind")
        threshold = int((self.meta or {}).get("threshold") or 0)

        if kind == "consecutive_work":
            t = threshold or 1
            return self._count_runs_ge(seq, predicate=lambda s: s != "O", threshold=t)
        if kind == "consecutive_night":
            t = threshold or 1
            return self._count_runs_ge(seq, predicate=lambda s: s == "N", threshold=t)
        if kind == "consecutive_off":
            t = threshold or 1
            return self._count_runs_ge(seq, predicate=lambda s: s == "O", threshold=t)
        if kind == "sandwich_off":
            return self._count_sandwich_work_off_work(seq)

        pat = (self.pattern or "").strip()
        if pat == "CONSEC_WORK":
            return self._count_runs_ge(seq, predicate=lambda s: s != "O", threshold=threshold or 1)
        if pat == "CONSEC_N":
            return self._count_runs_ge(seq, predicate=lambda s: s == "N", threshold=threshold or 1)
        if pat == "CONSEC_O":
            return self._count_runs_ge(seq, predicate=lambda s: s == "O", threshold=threshold or 1)
        if pat == "WOW":
            return self._count_sandwich_work_off_work(seq)

        if not pat: return 0
        seq_str = "".join(seq)
        return seq_str.count(pat)


@dataclass
class ScoreConfig:
    version: str = "v1"
    rules: Dict[str, ScoreRule] = field(default_factory=dict)
    fairness_weights: Dict[str, float] = field(default_factory=dict)


def load_score_config(config_dict: Dict[str, Any]) -> ScoreConfig:
    version = str(config_dict.get("version") or "v1")
    fairness_weights = dict(config_dict.get("fairness_weights") or {})

    rules_list = config_dict.get("rules") or []
    rules_map: Dict[str, ScoreRule] = {}

    for r in rules_list:
        if not isinstance(r, dict): continue
        rid = str(r.get("id") or "")
        if not rid: continue

        rule = ScoreRule(
            id=rid,
            enabled=bool(r.get("enabled", True)),
            type=r.get("type") or "count",
            weight=float(r.get("weight") or 1.0),
            scope=str(r.get("scope") or "nurse"),
            min_value=r.get("min_value"),
            max_value=r.get("max_value"),
            pattern=r.get("pattern"),
            meta=dict(r.get("meta") or {}),
        )
        rules_map[rid] = rule

    return ScoreConfig(version=version, rules=rules_map, fairness_weights=fairness_weights)