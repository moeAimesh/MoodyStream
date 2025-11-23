"""Eyelid aperture / lid tightening (AU5/AU7 proxy)."""

from __future__ import annotations

import numpy as np


def _mean_point(coords, keys, fallback_key):
    pts = [coords[k] for k in keys if k in coords]
    if not pts:
        return np.array(coords[fallback_key])
    return np.mean(pts, axis=0)


def lid_aperture_features(coords, face_width: float, face_height: float) -> np.ndarray:
    """Return normalized lid opening plus tilt metrics for both eyes."""
    face_height = face_height or 1.0
    face_width = face_width or 1.0

    left_top = _mean_point(
        coords,
        ["LEFT_EYE_TOP", "LEFT_EYE_TOP2", "LEFT_EYE_TOP3"],
        "LEFT_EYE_TOP",
    )
    left_bottom = _mean_point(
        coords,
        ["LEFT_EYE_BOTTOM", "LEFT_EYE_BOTTOM2", "LEFT_EYE_BOTTOM3"],
        "LEFT_EYE_BOTTOM",
    )
    right_top = _mean_point(
        coords,
        ["RIGHT_EYE_TOP", "RIGHT_EYE_TOP2", "RIGHT_EYE_TOP3"],
        "RIGHT_EYE_TOP",
    )
    right_bottom = _mean_point(
        coords,
        ["RIGHT_EYE_BOTTOM", "RIGHT_EYE_BOTTOM2", "RIGHT_EYE_BOTTOM3"],
        "RIGHT_EYE_BOTTOM",
    )

    left_open = abs(left_bottom[1] - left_top[1]) / face_height
    right_open = abs(right_bottom[1] - right_top[1]) / face_height
    asym = left_open - right_open

    def _lid_angle(top, bottom, outer_key, inner_key):
        horizontal = coords[outer_key][0] - coords[inner_key][0]
        return np.arctan2(top[1] - bottom[1], horizontal if abs(horizontal) > 1e-3 else 1e-3)

    left_angle = _lid_angle(left_top, left_bottom, "LEFT_EYE_OUTER", "LEFT_EYE_INNER")
    right_angle = _lid_angle(right_top, right_bottom, "RIGHT_EYE_OUTER", "RIGHT_EYE_INNER")

    left_width_ratio = abs(coords["LEFT_EYE_OUTER"][0] - coords["LEFT_EYE_INNER"][0]) / face_width
    right_width_ratio = abs(coords["RIGHT_EYE_OUTER"][0] - coords["RIGHT_EYE_INNER"][0]) / face_width

    return np.array(
        [
            left_open,
            right_open,
            asym,
            float(np.cos(left_angle)),
            float(np.cos(right_angle)),
            left_width_ratio,
            right_width_ratio,
        ],
        dtype=np.float32,
    )
