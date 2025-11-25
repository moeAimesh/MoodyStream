"""Simple must-have heuristics per emotion."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Sequence

import numpy as np

BLOCK_SPECS = [
    ("brow_raise", 7),
    ("brow_lower", 5),
    ("lid_aperture", 7),
    ("cheek_raise", 7),
    ("nose_flare", 5),
    ("mouth_corner", 8),
    ("mouth_depressor", 7),
]

SURPRISE_MOUTH_OPEN_MIN = 0.10
FEAR_LID_OPEN_MIN = 0.03
SAD_DROP_MIN = 0.20
SAD_AVG_MIN = 0.27
SAD_MOUTH_OPEN_MAX = 0.16


class EmotionRules:
    """Defines specific geometric rules for different emotions."""
    def __init__(self):
        cursor = 0
        self.slices: Dict[str, slice] = {}
        for name, length in BLOCK_SPECS:
            self.slices[name] = slice(cursor, cursor + length)
            cursor += length

    def _get_block(self, features: np.ndarray, name: str) -> np.ndarray:
        return features[self.slices[name]]

    def surprise(self, features: np.ndarray) -> bool:
        mouth = self._get_block(features, "mouth_corner")
        return mouth[0] >= SURPRISE_MOUTH_OPEN_MIN

    def fear(self, features: np.ndarray) -> bool:
        lid = self._get_block(features, "lid_aperture")
        return float(np.mean(lid[:2])) >= FEAR_LID_OPEN_MIN

    def happy(self, features: np.ndarray) -> bool:
        mouth = self._get_block(features, "mouth_corner")
        cheek = self._get_block(features, "cheek_raise")
        return mouth[2] < 0.45 and float(np.mean(cheek[:2])) > 0

    def sad(self, features: np.ndarray) -> bool:
        mouth_drop = self._get_block(features, "mouth_depressor")
        mouth_geom = self._get_block(features, "mouth_corner")
        left_ok = mouth_drop[0] >= SAD_DROP_MIN if mouth_drop.size >= 1 else False
        right_ok = mouth_drop[1] >= SAD_DROP_MIN if mouth_drop.size >= 2 else False
        avg_ok = mouth_drop[2] >= SAD_AVG_MIN if mouth_drop.size >= 3 else False
        open_ok = mouth_geom[0] <= SAD_MOUTH_OPEN_MAX if mouth_geom.size >= 1 else True
        return left_ok and right_ok and avg_ok and open_ok

    def neutral(self, features: np.ndarray) -> bool:
        return True


@dataclass
class EmotionHeuristicScorer:
    """Validates emotion predictions against a set of geometric rules."""
    min_confidence: float = 0.45
    debug: bool = False
    debug_interval: int = 200
    neutral_baseline: np.ndarray | None = None
    rules: EmotionRules = field(default_factory=EmotionRules, repr=False)
    total_dim: int = field(init=False)

    def __post_init__(self):
        self.total_dim = self.rules.slices["mouth_depressor"].stop
        self._debug_counts: Dict[str, int] = defaultdict(int)
        self._debug_total = 0

    def validate(self, label: str, features: np.ndarray) -> bool:
        """
        Hard validation step: returns True only if the AU conditions for the predicted
        label are met. Unknown labels are accepted by default.
        """
        if features is None:
            return True
        if label == "happy":
            return True
        vec = np.asarray(features, dtype=float).flatten()
        if vec.size < self.total_dim or not np.all(np.isfinite(vec)):
            return True

        rule_method = getattr(self.rules, label, None)
        if rule_method is None:
            return True  # Default to True if no specific rule exists

        is_valid = rule_method(vec)

        if not is_valid:
            self._maybe_log_debug([f"{label}_block"])

        return is_valid

    def _maybe_log_debug(self, actions: Sequence[str]) -> None:
        if not self.debug or not actions:
            return
        self._debug_total += 1
        for act in actions:
            self._debug_counts[act] += 1
        if self._debug_total % max(1, self.debug_interval) == 0:
            summary = ", ".join(f"{k}:{v}" for k, v in sorted(self._debug_counts.items()))
            print(f"[heuristics] adjustments={self._debug_total} -> {summary}")
