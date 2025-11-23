"""Generate helpful camera feedback messages during the setup."""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from detection.facemesh_features import estimate_head_tilt

FaceBox = Optional[Tuple[int, int, int, int]]

MIN_FACE_RATIO = 0.08
MIN_BRIGHTNESS = 70
TILT_THRESHOLD_DEG = 12.0


def analyze_capture_feedback(frame: np.ndarray, bbox: FaceBox) -> List[str]:
    """Return a list of feedback strings to overlay on the camera preview."""
    if frame is None or frame.size == 0:
        return []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    messages: List[str] = []

    if bbox is None:
        messages.append("No face detected. Move into view.")
        if brightness < MIN_BRIGHTNESS:
            messages.append("Improve lighting so your face is brighter.")
        return messages

    fh, fw = frame.shape[:2]
    x, y, w, h = bbox
    face_ratio = (w * h) / float(fw * fh if fw and fh else 1)
    if face_ratio < MIN_FACE_RATIO:
        messages.append("Move closer to the camera.")

    if brightness < MIN_BRIGHTNESS:
        messages.append("Increase lighting—your face is too dark.")

    tilt = estimate_head_tilt(frame, bbox)
    if tilt is not None and abs(tilt) > TILT_THRESHOLD_DEG:
        direction = "right" if tilt > 0 else "left"
        messages.append(f"Tilt your head slightly to the {direction}.")

    return messages
