"""Forehead/brow compression features (AU4 proxy)."""

from __future__ import annotations

import numpy as np


def brow_lower_features(coords, face_width: float, face_height: float) -> np.ndarray:
    """Return closeness of inner brows and brow-to-forehead drop."""
    face_width = face_width or 1.0
    face_height = face_height or 1.0
    inner_dist = abs(coords["RIGHT_INNER_BROW"][0] - coords["LEFT_INNER_BROW"][0]) / face_width
    brow_avg_y = (coords["LEFT_BROW"][1] + coords["RIGHT_BROW"][1]) / 2.0
    forehead_gap = (coords["FOREHEAD"][1] - brow_avg_y) / face_height

    nose_bridge = coords.get("NOSE_BRIDGE", coords["NOSE_TIP"])
    brow_drop = (brow_avg_y - nose_bridge[1]) / face_height

    inner_vec = np.array(coords["RIGHT_INNER_BROW"]) - np.array(coords["LEFT_INNER_BROW"])
    angle = np.arctan2(inner_vec[1], inner_vec[0]) if np.linalg.norm(inner_vec) > 0 else 0.0

    return np.array(
        [inner_dist, forehead_gap, brow_drop, float(np.cos(angle)), float(np.sin(angle))],
        dtype=np.float32,
    )
