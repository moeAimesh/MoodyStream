"""Eyebrow raise (AU1/AU2 proxy) features."""

from __future__ import annotations

import numpy as np


def _mean_point(coords, keys, fallback_key):
    pts = [coords[k] for k in keys if k in coords]
    if not pts:
        return np.array(coords[fallback_key])
    return np.mean(pts, axis=0)


def brow_raise_features(coords, face_width: float, face_height: float) -> np.ndarray:
    """Return enhanced eyebrow-raise metrics using multiple samples."""
    face_width = face_width or 1.0
    face_height = face_height or 1.0

    left_brow = _mean_point(
        coords,
        ["LEFT_BROW", "LEFT_INNER_BROW", "LEFT_BROW_OUTER"],
        "LEFT_BROW",
    )
    right_brow = _mean_point(
        coords,
        ["RIGHT_BROW", "RIGHT_INNER_BROW", "RIGHT_BROW_OUTER"],
        "RIGHT_BROW",
    )
    left_eye = _mean_point(
        coords,
        ["LEFT_EYE_TOP", "LEFT_EYE_TOP2", "LEFT_EYE_TOP3"],
        "LEFT_EYE_TOP",
    )
    right_eye = _mean_point(
        coords,
        ["RIGHT_EYE_TOP", "RIGHT_EYE_TOP2", "RIGHT_EYE_TOP3"],
        "RIGHT_EYE_TOP",
    )

    left_raise = (left_eye[1] - left_brow[1]) / face_height
    right_raise = (right_eye[1] - right_brow[1]) / face_height
    avg_raise = (left_raise + right_raise) / 2.0

    left_span = (
        coords.get("LEFT_BROW_OUTER", left_brow)[0]
        - coords.get("LEFT_INNER_BROW", left_brow)[0]
    ) / face_width
    right_span = (
        coords.get("RIGHT_INNER_BROW", right_brow)[0]
        - coords.get("RIGHT_BROW_OUTER", right_brow)[0]
    ) / face_width

    brow_vec = np.array(coords["RIGHT_BROW"]) - np.array(coords["LEFT_BROW"])
    angle = np.arctan2(brow_vec[1], brow_vec[0]) if np.linalg.norm(brow_vec) > 0 else 0.0
    brow_angle_cos = float(np.cos(angle))
    brow_angle_sin = float(np.sin(angle))

    return np.array(
        [left_raise, right_raise, avg_raise, left_span, right_span, brow_angle_cos, brow_angle_sin],
        dtype=np.float32,
    )
