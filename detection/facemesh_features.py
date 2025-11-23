"""Feature extraction using MediaPipe FaceMesh."""

from __future__ import annotations

from functools import lru_cache
import math
from typing import Optional, Sequence, Tuple

import cv2
import mediapipe as mp
import numpy as np

from detection.detectors.emotions import (
    brow_raise_features,
    brow_lower_features,
    cheek_raise_features,
    lid_aperture_features,
    mouth_corner_features,
    mouth_depressor_features,
    nose_flare_features,
)
from utils.mediapipe_fix import apply_fix
from utils.settings import PLOT_AU

apply_fix()


@lru_cache(maxsize=1)
def _get_facemesh() -> mp.solutions.face_mesh.FaceMesh:
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


FACIAL_LANDMARKS = {
    "LEFT_EYE_TOP": 159,
    "LEFT_EYE_TOP2": 158,
    "LEFT_EYE_TOP3": 160,
    "LEFT_EYE_BOTTOM": 145,
    "LEFT_EYE_BOTTOM2": 144,
    "LEFT_EYE_BOTTOM3": 153,
    "LEFT_EYE_OUTER": 33,
    "LEFT_EYE_INNER": 133,
    "RIGHT_EYE_TOP": 386,
    "RIGHT_EYE_TOP2": 387,
    "RIGHT_EYE_TOP3": 385,
    "RIGHT_EYE_BOTTOM": 374,
    "RIGHT_EYE_BOTTOM2": 373,
    "RIGHT_EYE_BOTTOM3": 380,
    "RIGHT_EYE_OUTER": 263,
    "RIGHT_EYE_INNER": 362,
    "LEFT_BROW": 105,
    "LEFT_BROW_OUTER": 46,
    "LEFT_INNER_BROW": 107,
    "RIGHT_BROW": 334,
    "RIGHT_BROW_OUTER": 276,
    "RIGHT_INNER_BROW": 336,
    "FOREHEAD": 10,
    "LEFT_CHEEK": 50,
    "LEFT_CHEEK_LOWER": 205,
    "RIGHT_CHEEK": 280,
    "RIGHT_CHEEK_LOWER": 425,
    "MOUTH_LEFT": 61,
    "MOUTH_RIGHT": 291,
    "MOUTH_TOP": 13,
    "MOUTH_BOTTOM": 14,
    "NOSE_TIP": 1,
    "NOSE_BRIDGE": 6,
    "CHIN": 152,
    "LEFT_NOSTRIL": 98,
    "RIGHT_NOSTRIL": 327,
}

AU_PLOT_GROUPS = {
    "brow_raise": {
        "color": (0, 255, 0),
        "keys": [
            "LEFT_BROW",
            "LEFT_BROW_OUTER",
            "RIGHT_BROW",
            "RIGHT_BROW_OUTER",
            "LEFT_EYE_TOP",
            "RIGHT_EYE_TOP",
        ],
    },
    "brow_lower": {"color": (0, 180, 0), "keys": ["LEFT_INNER_BROW", "RIGHT_INNER_BROW", "FOREHEAD"]},
    "lid": {
        "color": (255, 255, 0),
        "keys": [
            "LEFT_EYE_TOP",
            "LEFT_EYE_TOP2",
            "LEFT_EYE_BOTTOM",
            "LEFT_EYE_BOTTOM2",
            "RIGHT_EYE_TOP",
            "RIGHT_EYE_TOP2",
            "RIGHT_EYE_BOTTOM",
            "RIGHT_EYE_BOTTOM2",
        ],
    },
    "cheek": {
        "color": (255, 200, 0),
        "keys": ["LEFT_CHEEK", "LEFT_CHEEK_LOWER", "RIGHT_CHEEK", "RIGHT_CHEEK_LOWER"],
    },
    "nose": {"color": (255, 0, 0), "keys": ["LEFT_NOSTRIL", "RIGHT_NOSTRIL", "NOSE_TIP"]},
    "mouth_corner": {"color": (255, 0, 255), "keys": ["MOUTH_LEFT", "MOUTH_RIGHT"]},
    "mouth_vertical": {"color": (200, 0, 200), "keys": ["MOUTH_TOP", "MOUTH_BOTTOM", "CHIN"]},
}


def _landmark_xy(landmark, width, height) -> Tuple[float, float]:
    return landmark.x * width, landmark.y * height


def _extract_roi(frame, bbox: Optional[Tuple[int, int, int, int]]):
    if bbox:
        x, y, w, h = bbox
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(frame.shape[1], x + w)
        y2 = min(frame.shape[0], y + h)
        roi = frame[y1:y2, x1:x2]
    else:
        roi = frame
        x1 = y1 = 0
    return roi, (x1, y1)


def _process_facemesh(roi):
    if roi.size == 0:
        return None
    rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    mesh = _get_facemesh()
    results = mesh.process(rgb)
    if not results.multi_face_landmarks:
        return None
    return results.multi_face_landmarks[0]


def extract_facemesh_features(frame, bbox: Optional[Tuple[int, int, int, int]] = None) -> Optional[np.ndarray]:
    """
    Compute a small set of normalized facial-action features.
    Returns None if no face mesh could be detected.
    """
    roi, _ = _extract_roi(frame, bbox)
    face_landmarks = _process_facemesh(roi)
    if face_landmarks is None:
        return None
    h, w, _ = roi.shape

    coords = {}
    for key, idx in FACIAL_LANDMARKS.items():
        landmark = face_landmarks.landmark[idx]
        coords[key] = _landmark_xy(landmark, w, h)

    # basic facial size
    xs = [p.x * w for p in face_landmarks.landmark]
    ys = [p.y * h for p in face_landmarks.landmark]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    width = width or 1.0
    height = height or 1.0

    nose_y = coords["NOSE_TIP"][1]
    chin_y = coords["CHIN"][1]

    blocks = [
        brow_raise_features(coords, width, height),
        brow_lower_features(coords, width, height),
        lid_aperture_features(coords, width, height),
        cheek_raise_features(coords, width, height),
        nose_flare_features(coords, width, height),
        mouth_corner_features(coords, (width, height), nose_y, chin_y),
        mouth_depressor_features(coords, height, nose_y, chin_y),
    ]
    if PLOT_AU:
        _plot_au_points(roi, coords)
    return np.concatenate(blocks).astype(np.float32, copy=False)


def _plot_au_points(roi, coords):
    overlay = roi.copy()
    for group in AU_PLOT_GROUPS.values():
        color = group["color"]
        for key in group["keys"]:
            if key not in coords:
                continue
            x, y = coords[key]
            cv2.circle(
                overlay,
                (int(x), int(y)),
                3,
                color,
                thickness=-1,
            )
    cv2.addWeighted(overlay, 0.4, roi, 0.6, 0, roi)


def estimate_head_tilt(frame, bbox: Optional[Tuple[int, int, int, int]] = None) -> Optional[float]:
    """Estimate head tilt angle in degrees (positive -> tilt to the right)."""
    roi, _ = _extract_roi(frame, bbox)
    face_landmarks = _process_facemesh(roi)
    if face_landmarks is None:
        return None
    w = roi.shape[1] or 1.0
    h = roi.shape[0] or 1.0
    left = face_landmarks.landmark[FACIAL_LANDMARKS["LEFT_EYE_OUTER"]]
    right = face_landmarks.landmark[FACIAL_LANDMARKS["RIGHT_EYE_OUTER"]]
    x_left, y_left = left.x * w, left.y * h
    x_right, y_right = right.x * w, right.y * h
    dx = x_right - x_left
    dy = y_right - y_left
    if dx == 0:
        return None
    angle = math.degrees(math.atan2(dy, dx))
    return angle
