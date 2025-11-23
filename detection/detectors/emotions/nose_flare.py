"""Nasal flare / wrinkling (AU9 proxy)."""

from __future__ import annotations

import numpy as np


def nose_flare_features(coords, face_width: float, face_height: float) -> np.ndarray:
    """Ratio of nostril width and lift relative to facial size."""
    face_width = face_width or 1.0
    face_height = face_height or 1.0
    left_nostril = np.array(coords["LEFT_NOSTRIL"])
    right_nostril = np.array(coords["RIGHT_NOSTRIL"])

    nostril_width = abs(right_nostril[0] - left_nostril[0]) / face_width
    nostril_lift = (coords["NOSE_TIP"][1] - (left_nostril[1] + right_nostril[1]) / 2.0) / face_height

    nostril_vec = right_nostril - left_nostril
    angle = np.arctan2(nostril_vec[1], nostril_vec[0]) if np.linalg.norm(nostril_vec) > 0 else 0.0

    bridge_gap = (coords["NOSE_TIP"][1] - coords.get("NOSE_BRIDGE", coords["NOSE_TIP"])[1]) / face_height

    return np.array(
        [nostril_width, nostril_lift, float(np.cos(angle)), float(np.sin(angle)), bridge_gap],
        dtype=np.float32,
    )
