"""Cheek raiser / AU6 helpers."""

from __future__ import annotations

import numpy as np


def _mean_point(coords, keys, fallback_key):
    pts = [coords[k] for k in keys if k in coords]
    if not pts:
        return np.array(coords[fallback_key])
    return np.mean(pts, axis=0)


def cheek_raise_features(coords, face_width: float, face_height: float) -> np.ndarray:
    """Measure cheek lift relative to eyes and mouth corners."""
    face_height = face_height or 1.0
    face_width = face_width or 1.0

    left_eye_center = (
        coords["LEFT_EYE_TOP"][1]
        + coords["LEFT_EYE_BOTTOM"][1]
        + coords.get("LEFT_EYE_TOP2", coords["LEFT_EYE_TOP"])[1]
    ) / 3.0
    right_eye_center = (
        coords["RIGHT_EYE_TOP"][1]
        + coords["RIGHT_EYE_BOTTOM"][1]
        + coords.get("RIGHT_EYE_TOP2", coords["RIGHT_EYE_TOP"])[1]
    ) / 3.0

    left_cheek = _mean_point(coords, ["LEFT_CHEEK", "LEFT_CHEEK_LOWER"], "LEFT_CHEEK")
    right_cheek = _mean_point(coords, ["RIGHT_CHEEK", "RIGHT_CHEEK_LOWER"], "RIGHT_CHEEK")

    left_cheek_lift = (left_eye_center - left_cheek[1]) / face_height
    right_cheek_lift = (right_eye_center - right_cheek[1]) / face_height
    symmetry = left_cheek_lift - right_cheek_lift

    left_corner_gap = (coords["MOUTH_LEFT"][1] - left_cheek[1]) / face_height
    right_corner_gap = (coords["MOUTH_RIGHT"][1] - right_cheek[1]) / face_height

    left_corner_dist = abs(coords["MOUTH_LEFT"][0] - left_cheek[0]) / face_width
    right_corner_dist = abs(coords["MOUTH_RIGHT"][0] - right_cheek[0]) / face_width

    return np.array(
        [
            left_cheek_lift,
            right_cheek_lift,
            symmetry,
            left_corner_gap,
            right_corner_gap,
            left_corner_dist,
            right_corner_dist,
        ],
        dtype=np.float32,
    )
