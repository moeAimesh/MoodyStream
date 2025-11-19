"""Rule-based heuristic booster for AU feature blocks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

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


@dataclass
class HeuristicConfig:
    min_confidence: float = 0.45


class EmotionHeuristicScorer:
    """Compute lightweight AU heuristics that nudge classifier probabilities."""

    def __init__(self, config: HeuristicConfig | None = None):
        self.config = config or HeuristicConfig()
        self.slices: Dict[str, slice] = {}
        cursor = 0
        for name, length in BLOCK_SPECS:
            self.slices[name] = slice(cursor, cursor + length)
            cursor += length
        self.total_dim = cursor

    def score(self, features: np.ndarray) -> Dict[str, float]:
        vec = np.asarray(features, dtype=float)
        if vec.ndim != 1 or vec.size < self.total_dim:
            return {}

        brow_raise = vec[self.slices["brow_raise"]]
        brow_lower = vec[self.slices["brow_lower"]]
        lid = vec[self.slices["lid_aperture"]]
        cheek = vec[self.slices["cheek_raise"]]
        nose = vec[self.slices["nose_flare"]]
        mouth_corner = vec[self.slices["mouth_corner"]]
        mouth_depressor = vec[self.slices["mouth_depressor"]]

        brow_raise_avg = float(np.mean(brow_raise[:2]))
        brow_raise_inner = float(np.max(brow_raise[:2]))
        brow_lower_inner = brow_lower[0]
        lid_open_avg = float(np.mean(lid[:2]))
        cheek_lift_avg = float(np.mean(cheek[:2]))
        nose_width = nose[0]
        mouth_open = mouth_corner[0]
        mouth_sym = abs(mouth_corner[4])
        mouth_curvature = mouth_corner[7]
        mouth_drop = mouth_depressor[2]

        scores = {
            "angry": 0.0,
            "disgust": 0.0,
            "fear": 0.0,
            "happy": 0.0,
            "sad": 0.0,
            "surprise": 0.0,
            "neutral": 0.0,
        }

        # helper activations
        def high(val, low, high):
            if high <= low:
                return 1.0 if val >= high else 0.0
            return float(np.clip((val - low) / (high - low), 0.0, 1.0))

        def low(val, low, high):
            if high <= low:
                return 1.0 if val <= high else 0.0
            return float(np.clip((high - val) / (high - low), 0.0, 1.0))

        brow_compression = low(brow_lower_inner, 0.12, 0.28)
        lid_narrow = low(lid_open_avg, 0.05, 0.12)
        mouth_closed = low(mouth_open, 0.04, 0.12)

        nose_flare = high(nose_width, 0.15, 0.32)
        lip_raise = low(mouth_drop, 0.25, 0.4)
        mouth_drop_high = high(mouth_drop, 0.35, 0.6)
        cheek_lift = high(cheek_lift_avg, 0.015, 0.06)
        brow_raise_high = high(brow_raise_avg, 0.05, 0.1)
        brow_raise_mid = high(brow_raise_avg, 0.035, 0.08)
        inner_brow_raise = high(brow_raise_inner, 0.04, 0.08)
        mouth_open_high = high(mouth_open, 0.06, 0.18)

        sad_hint = min(1.0, mouth_drop_high + inner_brow_raise)

        scores["angry"] = 0.6 * brow_compression + 0.25 * lid_narrow + 0.15 * mouth_closed
        scores["disgust"] = 0.65 * nose_flare + 0.25 * lip_raise + 0.1 * high(mouth_sym, 0.02, 0.12)
        scores["fear"] = 0.45 * high(lid_open_avg, 0.08, 0.18) + 0.35 * brow_raise_mid + 0.2 * high(mouth_open, 0.03, 0.12)
        scores["happy"] = 0.55 * low(mouth_drop, 0.18, 0.35) + 0.35 * cheek_lift + 0.1 * high(-mouth_curvature, 0.0, 0.05)
        scores["sad"] = 0.65 * mouth_drop_high + 0.35 * inner_brow_raise
        scores["surprise"] = 0.4 * brow_raise_high + 0.4 * high(lid_open_avg, 0.1, 0.2) + 0.2 * mouth_open_high

        # pair balancing
        if mouth_open_high > 0.6:
            scores["surprise"] += 0.15
        if mouth_drop > 0.4 or lid_narrow > 0.5:
            scores["fear"] += 0.1
        if nose_flare > 0.5 and lip_raise > 0.4:
            scores["disgust"] += 0.1
        if brow_compression > 0.6:
            scores["angry"] += 0.1
        if sad_hint > 0.8:
            scores["sad"] += 0.1
        if mouth_drop < 0.2 and inner_brow_raise < 0.3:
            scores["neutral"] += 0.2

        activity = sum(min(1.0, scores[label]) for label in ("angry", "disgust", "fear", "happy", "sad", "surprise"))
        scores["neutral"] += max(0.0, 1.0 - 0.5 * activity)

        return scores

    def min_score(self) -> float:
        return self.config.min_confidence
