"""Mediapipe-based face detection helpers."""

from __future__ import annotations

import cv2
import mediapipe as mp

from utils.mediapipe_fix import apply_fix
from utils.settings import (
    FACE_DETECTOR_PAD_X,
    FACE_DETECTOR_PAD_Y_BOTTOM,
    FACE_DETECTOR_PAD_Y_TOP,
)

apply_fix()

_mp_face = mp.solutions.face_detection
_detector = _mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)


def _detection_to_bbox(det, frame_shape):
    H, W, _ = frame_shape
    box = det.location_data.relative_bounding_box
    x, y, w, h = box.xmin, box.ymin, box.width, box.height
    if w <= 0 or h <= 0:
        return None

    pad_x = int(w * W * FACE_DETECTOR_PAD_X)
    pad_y_top = int(h * H * FACE_DETECTOR_PAD_Y_TOP)
    pad_y_bottom = int(h * H * FACE_DETECTOR_PAD_Y_BOTTOM)

    x1 = max(0, int(x * W) - pad_x)
    y1 = max(0, int(y * H) - pad_y_top)
    x2 = min(W, int((x + w) * W) + pad_x)
    y2 = min(H, int((y + h) * H) + pad_y_bottom)
    width = x2 - x1
    height = y2 - y1
    if width <= 0 or height <= 0:
        return None
    return (x1, y1, width, height)


def detect_faces(frame):
    """Return list of padded bounding boxes for detected faces."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = _detector.process(rgb)
    boxes = []
    if result.detections:
        for det in result.detections:
            bbox = _detection_to_bbox(det, frame.shape)
            if bbox:
                boxes.append(bbox)
    return boxes
