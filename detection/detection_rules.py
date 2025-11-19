"""Reject frames that violate basic detection requirements (motion, size, visibility)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from utils.settings import (
    MIN_EYE_VISIBILITY,
    MIN_FACE_AREA_RATIO,
    MOTION_HOLD_FRAMES,
    MOTION_MAX_RATIO,
)


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
class _TrackState:
    center: Tuple[float, float] | None = None
    diag: float | None = None
    cooldown: int = 0


@dataclass
class DetectionRules:
    """Enforce motion/visibility constraints before running emotion classification."""

    motion_ratio: float = MOTION_MAX_RATIO
    motion_cooldown: int = MOTION_HOLD_FRAMES
    min_face_ratio: float = MIN_FACE_AREA_RATIO
    eye_threshold: float = MIN_EYE_VISIBILITY
    _track_state: Dict[int, _TrackState] = field(default_factory=dict)

    def __post_init__(self):
        cursor = 0
        self.slices: Dict[str, slice] = {}
        for name, length in BLOCK_SPECS:
            self.slices[name] = slice(cursor, cursor + length)
            cursor += length
        self.total_dim = cursor

    def should_skip(
        self,
        track_id: Optional[int],
        bbox: Optional[Tuple[int, int, int, int]],
        frame_shape: Optional[Tuple[int, int]],
        features: Optional[np.ndarray],
    ) -> Optional[str]:
        if bbox is None or frame_shape is None:
            return "missing_bbox"
        if not self._check_face_size(bbox, frame_shape):
            return "face_too_small"
        if not self._check_motion(track_id, bbox):
            return "face_moving"
        if not self._has_full_face(features):
            return "partial_face"
        return None

    def _check_face_size(self, bbox, frame_shape) -> bool:
        frame_h, frame_w = frame_shape
        if frame_h <= 0 or frame_w <= 0:
            return False
        _, _, w, h = bbox
        area_ratio = (w * h) / float(frame_w * frame_h)
        return area_ratio >= self.min_face_ratio

    def _check_motion(self, track_id: Optional[int], bbox) -> bool:
        if track_id is None:
            return True
        state = self._track_state.setdefault(track_id, _TrackState())
        x, y, w, h = bbox
        center = (x + w / 2.0, y + h / 2.0)
        diag = float(np.hypot(w, h)) or 1.0

        if state.cooldown > 0:
            state.cooldown -= 1
            state.center = center
            state.diag = diag
            return False

        if state.center is not None and state.diag:
            dist = float(np.hypot(center[0] - state.center[0], center[1] - state.center[1]))
            if dist > self.motion_ratio * state.diag:
                state.cooldown = self.motion_cooldown
                state.center = center
                state.diag = diag
                return False

        state.center = center
        state.diag = diag
        return True

    def _has_full_face(self, features: Optional[np.ndarray]) -> bool:
        if features is None:
            return False
        vec = np.asarray(features, dtype=float).flatten()
        if vec.size < self.total_dim or not np.all(np.isfinite(vec)):
            return False

        lid = vec[self.slices["lid_aperture"]]
        if not self._visible_pair(lid[0], lid[1], self.eye_threshold):
            return False

        brow = vec[self.slices["brow_raise"]]
        if not self._visible_pair(brow[0], brow[1], self.eye_threshold * 0.5):
            return False

        mouth_lr = vec[self.slices["mouth_corner"]][2:4]
        if not self._visible_pair(mouth_lr[0], mouth_lr[1], self.eye_threshold * 0.6):
            return False

        return True

    def _visible_pair(self, left: float, right: float, threshold: float, max_ratio: float = 3.0) -> bool:
        if left < threshold or right < threshold:
            return False
        denom = max(min(left, right), 1e-6)
        ratio = max(left, right) / denom
        return ratio <= max_ratio
