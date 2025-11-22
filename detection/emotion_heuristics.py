"""Simple must-have heuristics per emotion."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
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
FEAR_LID_OPEN_MIN = 0.065
DISGUST_NOSTRIL_WIDTH_MIN = 0.45
DISGUST_BRIDGE_GAP_MAX = 0.03
SAD_DROP_MIN = 0.24
SAD_AVG_MIN = 0.29
SAD_MOUTH_OPEN_MAX = 0.16


class EmotionRules:
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

    def disgust(self, features: np.ndarray) -> bool:
        nose = self._get_block(features, "nose_flare")
        width_ok = nose[0] >= DISGUST_NOSTRIL_WIDTH_MIN
        bridge_ok = nose[4] <= DISGUST_BRIDGE_GAP_MAX if nose.size >= 5 else False
        return width_ok and bridge_ok

    def happy(self, features: np.ndarray) -> bool:
        mouth = self._get_block(features, "mouth_corner")
        cheek = self._get_block(features, "cheek_raise")
        return mouth[2] < 0.4 and float(np.mean(cheek[:2])) > 0.02

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
    min_confidence: float = 0.45
    debug: bool = False
    debug_interval: int = 200

    def __post_init__(self):
        self.rules = EmotionRules()
        cursor = 0
        self.slices: Dict[str, slice] = {}
        for name, length in BLOCK_SPECS:
            self.slices[name] = slice(cursor, cursor + length)
            cursor += length
        self.total_dim = cursor
        self._debug_counts: Dict[str, int] = defaultdict(int)
        self._debug_total = 0

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

        actions: list[str] = []
        if label == "surprise":
            mouth = self._get_block(vec, "mouth_corner")
            ok = mouth.size >= 1 and mouth[0] >= SURPRISE_MOUTH_OPEN_MIN
            if not ok:
                actions.append("surprise_block")
            self._maybe_log_debug(actions)
            return ok
        if label == "fear":
            lid = self._get_block(vec, "lid_aperture")
            ok = lid.size >= 2 and float(np.mean(lid[:2])) >= FEAR_LID_OPEN_MIN
            if not ok:
                actions.append("fear_block")
            self._maybe_log_debug(actions)
            return ok
        if label == "disgust":
            nose = self._get_block(vec, "nose_flare")
            width_ok = nose.size >= 1 and nose[0] >= DISGUST_NOSTRIL_WIDTH_MIN
            bridge_ok = nose.size >= 5 and nose[4] <= DISGUST_BRIDGE_GAP_MAX
            ok = width_ok and bridge_ok
            if not ok:
                actions.append("disgust_block")
            self._maybe_log_debug(actions)
            return ok
        if label == "sad":
            mouth_drop = self._get_block(vec, "mouth_depressor")
            mouth_geom = self._get_block(vec, "mouth_corner")
            left_ok = mouth_drop.size >= 1 and mouth_drop[0] >= SAD_DROP_MIN
            right_ok = mouth_drop.size >= 2 and mouth_drop[1] >= SAD_DROP_MIN
            avg_ok = mouth_drop.size >= 3 and mouth_drop[2] >= SAD_AVG_MIN
            open_ok = mouth_geom.size < 1 or mouth_geom[0] <= SAD_MOUTH_OPEN_MAX
            ok = left_ok and right_ok and avg_ok and open_ok
            if not ok:
                actions.append("sad_block")
            self._maybe_log_debug(actions)
            return ok
        return True

    def _get_block(self, vec: np.ndarray, name: str) -> np.ndarray:
        slc = self.slices.get(name)
        return vec[slc] if slc else np.array([])

    def _maybe_log_debug(self, actions: Sequence[str]) -> None:
        if not self.debug or not actions:
            return
        self._debug_total += 1
        for act in actions:
            self._debug_counts[act] += 1
        if self._debug_total % max(1, self.debug_interval) == 0:
            summary = ", ".join(f"{k}:{v}" for k, v in sorted(self._debug_counts.items()))
            print(f"[heuristics] adjustments={self._debug_total} -> {summary}")
