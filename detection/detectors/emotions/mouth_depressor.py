"""Mouth corner depressor / chin raiser (AU15/AU17)."""

from __future__ import annotations

import numpy as np


def mouth_depressor_features(coords, face_height: float, nose_y: float, chin_y: float) -> np.ndarray:
    """Measure downward mouth corner motion and chin raise."""
    face_height = face_height or 1.0
    denom = (chin_y - nose_y) or 1.0
    left_drop = (coords["MOUTH_LEFT"][1] - nose_y) / denom
    right_drop = (coords["MOUTH_RIGHT"][1] - nose_y) / denom
    avg_drop = (left_drop + right_drop) / 2.0
    chin_raise = (chin_y - coords["CHIN"][1]) / face_height

    lip_thickness = abs(coords["MOUTH_BOTTOM"][1] - coords["MOUTH_TOP"][1]) / face_height
    chin_vector = np.array(coords["CHIN"]) - np.array(coords["MOUTH_BOTTOM"])
    chin_angle = np.arctan2(chin_vector[1], chin_vector[0]) if np.linalg.norm(chin_vector) > 0 else 0.0

    return np.array(
        [
            left_drop,
            right_drop,
            avg_drop,
            chin_raise,
            lip_thickness,
            float(np.cos(chin_angle)),
            float(np.sin(chin_angle)),
        ],
        dtype=np.float32,
    )
