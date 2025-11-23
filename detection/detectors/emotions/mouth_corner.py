"""Mouth corner lift / stretch (AU12/AU20 proxy)."""

from __future__ import annotations

import numpy as np


def mouth_corner_features(coords, face_dims, nose_y: float, chin_y: float) -> np.ndarray:
    """Return normalized mouth opening, width, and corner lift."""
    face_width, face_height = face_dims
    face_width = face_width or 1.0
    face_height = face_height or 1.0

    mouth_open = abs(coords["MOUTH_BOTTOM"][1] - coords["MOUTH_TOP"][1]) / face_height
    mouth_width = abs(coords["MOUTH_RIGHT"][0] - coords["MOUTH_LEFT"][0]) / face_width

    denom = (chin_y - nose_y) or 1.0
    left_corner_rel = (coords["MOUTH_LEFT"][1] - nose_y) / denom
    right_corner_rel = (coords["MOUTH_RIGHT"][1] - nose_y) / denom
    symmetry = left_corner_rel - right_corner_rel

    mouth_vec = np.array(coords["MOUTH_RIGHT"]) - np.array(coords["MOUTH_LEFT"])
    angle = np.arctan2(mouth_vec[1], mouth_vec[0]) if np.linalg.norm(mouth_vec) > 0 else 0.0

    center = (np.array(coords["MOUTH_LEFT"]) + np.array(coords["MOUTH_RIGHT"])) / 2.0
    curvature = (coords["MOUTH_TOP"][1] + coords["MOUTH_BOTTOM"][1]) / 2.0 - center[1]
    curvature /= face_height

    return np.array(
        [
            mouth_open,
            mouth_width,
            left_corner_rel,
            right_corner_rel,
            symmetry,
            float(np.cos(angle)),
            float(np.sin(angle)),
            curvature,
        ],
        dtype=np.float32,
    )
