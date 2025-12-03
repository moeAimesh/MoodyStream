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
FEAR_LID_OPEN_MIN = 0.02
SAD_DROP_MIN = 0.17
SAD_AVG_MIN = 0.23
SAD_MOUTH_OPEN_MAX = 0.16
HAPPY_MOUTH_CURVE_MAX = 0.55  # vorher 0.35/0.45
HAPPY_CHEEK_MEAN_MIN = -0.05  # vorher > 0


@dataclass
class HeuristicThresholds:
    """Configurable thresholds for AU-based heuristics."""

    surprise_mouth_open_min: float = SURPRISE_MOUTH_OPEN_MIN
    fear_lid_open_min: float = FEAR_LID_OPEN_MIN
    sad_drop_min: float = SAD_DROP_MIN
    sad_avg_min: float = SAD_AVG_MIN
    sad_mouth_open_max: float = SAD_MOUTH_OPEN_MAX
    happy_mouth_curve_max: float = HAPPY_MOUTH_CURVE_MAX
    happy_cheek_mean_min: float = HAPPY_CHEEK_MEAN_MIN


class EmotionRules:
    """Defines specific geometric rules for different emotions."""

    def __init__(self, thresholds: HeuristicThresholds | None = None):
        cursor = 0
        self.slices: Dict[str, slice] = {}
        for name, length in BLOCK_SPECS:
            self.slices[name] = slice(cursor, cursor + length)
            cursor += length
        self.total_dim = cursor
        self.thresholds = thresholds or HeuristicThresholds()

    def _get_block(self, features: np.ndarray, name: str) -> np.ndarray:
        return features[self.slices[name]]

    def surprise(self, features: np.ndarray) -> bool:
        mouth = self._get_block(features, "mouth_corner")
        return mouth[0] >= self.thresholds.surprise_mouth_open_min

    def fear(self, features: np.ndarray) -> bool:
        lid = self._get_block(features, "lid_aperture")
        return float(np.mean(lid[:2])) >= self.thresholds.fear_lid_open_min

    def happy(self, features: np.ndarray) -> bool:
        mouth = self._get_block(features, "mouth_corner")
        cheek = self._get_block(features, "cheek_raise")
        return (
            mouth[2] <= self.thresholds.happy_mouth_curve_max
            and float(np.mean(cheek[:2])) >= self.thresholds.happy_cheek_mean_min
        )

    def sad(self, features: np.ndarray) -> bool:
        mouth_drop = self._get_block(features, "mouth_depressor")
        mouth_geom = self._get_block(features, "mouth_corner")
        left_ok = mouth_drop[0] >= self.thresholds.sad_drop_min if mouth_drop.size >= 1 else False
        right_ok = mouth_drop[1] >= self.thresholds.sad_drop_min if mouth_drop.size >= 2 else False
        avg_ok = mouth_drop[2] >= self.thresholds.sad_avg_min if mouth_drop.size >= 3 else False
        open_ok = mouth_geom[0] <= self.thresholds.sad_mouth_open_max if mouth_geom.size >= 1 else True
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
    thresholds: HeuristicThresholds = field(default_factory=HeuristicThresholds)
    rules: EmotionRules = field(init=False, repr=False)
    total_dim: int = field(init=False)

    def __post_init__(self):
        self.rules = EmotionRules(self.thresholds)
        self.total_dim = self.rules.total_dim
        self._debug_counts: Dict[str, int] = defaultdict(int)
        self._debug_total = 0

    def set_thresholds(self, thresholds: HeuristicThresholds) -> None:
        """Update thresholds (e.g., loaded from user setup or model file)."""
        self.thresholds = thresholds
        self.rules.thresholds = thresholds

    def validate(self, label: str, features: np.ndarray) -> bool:
        """
        Hard validation step: returns True only if the AU conditions for the predicted
        label are met. Unknown labels are accepted by default.
        """
        if features is None:
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


def compute_thresholds_from_samples(
    feature_vectors: Dict[str, Sequence[Sequence[float]]]
) -> HeuristicThresholds | None:
    """
    Derive user-specific heuristic thresholds from recorded AU feature samples.
    Uses robust percentiles to stay tolerant to outliers. Returns None if no valid
    vectors are available.
    """
    if not feature_vectors:
        return None

    defaults = HeuristicThresholds()
    rules = EmotionRules(defaults)

    def iter_valid(raw):
        for vec in raw or []:
            arr = np.asarray(vec, dtype=float).flatten()
            if arr.size >= rules.total_dim and np.all(np.isfinite(arr)):
                yield arr

    def percentile_or_default(values, pct, default):
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return default
        return float(np.percentile(arr, pct))

    surprise_mouth = [
        rules._get_block(vec, "mouth_corner")[0]
        for vec in iter_valid(feature_vectors.get("surprise"))
        if rules._get_block(vec, "mouth_corner").size >= 1
    ]
    fear_lid = [
        float(np.mean(rules._get_block(vec, "lid_aperture")[:2]))
        for vec in iter_valid(feature_vectors.get("fear"))
        if rules._get_block(vec, "lid_aperture").size >= 2
    ]
    happy_mouth_curve = [
        rules._get_block(vec, "mouth_corner")[2]
        for vec in iter_valid(feature_vectors.get("happy"))
        if rules._get_block(vec, "mouth_corner").size >= 3
    ]
    happy_cheek_mean = [
        float(np.mean(rules._get_block(vec, "cheek_raise")[:2]))
        for vec in iter_valid(feature_vectors.get("happy"))
        if rules._get_block(vec, "cheek_raise").size >= 2
    ]
    sad_drop = []
    sad_avg = []
    sad_mouth_open = []
    for vec in iter_valid(feature_vectors.get("sad")):
        mouth_drop = rules._get_block(vec, "mouth_depressor")
        mouth_geom = rules._get_block(vec, "mouth_corner")
        if mouth_drop.size >= 1:
            sad_drop.append(mouth_drop[0])
        if mouth_drop.size >= 2:
            sad_drop.append(mouth_drop[1])
        if mouth_drop.size >= 3:
            sad_avg.append(mouth_drop[2])
        if mouth_geom.size >= 1:
            sad_mouth_open.append(mouth_geom[0])

    # Softened percentiles to stay permissive while adapting to user geometry.
    thresholds = HeuristicThresholds(
        surprise_mouth_open_min=percentile_or_default(surprise_mouth, 25, defaults.surprise_mouth_open_min),
        fear_lid_open_min=percentile_or_default(fear_lid, 25, defaults.fear_lid_open_min),
        sad_drop_min=percentile_or_default(sad_drop, 25, defaults.sad_drop_min),
        sad_avg_min=percentile_or_default(sad_avg, 25, defaults.sad_avg_min),
        sad_mouth_open_max=percentile_or_default(sad_mouth_open, 80, defaults.sad_mouth_open_max),
        happy_mouth_curve_max=percentile_or_default(happy_mouth_curve, 80, defaults.happy_mouth_curve_max),
        happy_cheek_mean_min=percentile_or_default(happy_cheek_mean, 20, defaults.happy_cheek_mean_min),
    )

    return thresholds
