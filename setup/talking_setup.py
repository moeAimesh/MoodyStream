"""
Calibrate a per-user talking threshold based on mouth-opening variance.

Workflow:
- Reuse the existing preview window (Emotion Profiling).
- Show a short countdown with on-screen text.
- Let the user read a longer paragraph for a fixed duration (auto-stop).
- Collect mouth_open (mouth_corner[0]) values from FaceMesh-based AU features.
- Compute variance over a sliding window to derive a speaking threshold.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

from detection.face_detection import detect_faces
from detection.facemesh_features import extract_facemesh_features
from detection.preprocessing import preprocess_face
from utils.settings import TALKING_MIN_SAMPLES, TALKING_THRESHOLD_DEFAULT, TALKING_THRESHOLD_MAX, TALKING_WINDOW


DEFAULT_READING_TEXT = (
    "On a quiet morning by the river, a wooden boat drifted slowly past the reeds while the town woke "
    "up. Voices from the market blended with the sound of water, and a light breeze carried the smell "
    "of fresh bread across the square."
)
COUNTDOWN_STEPS: List[Tuple[str, float]] = [
    ("Read the following text out loud", 1.1),
    ("Ready?", 0.8),
    ("3", 0.7),
    ("2", 0.7),
    ("1", 0.7),
    ("Go!", 0.7),
]


def _compute_mouth_open_features(frame, bbox) -> Optional[float]:
    """Extract mouth_open ratio (mouth_corner[0]) from AU features."""
    feats = extract_facemesh_features(frame, bbox)
    if feats is None or feats.size < 1:
        return None
    # mouth_corner block starts after brow/lid/cheek/nose => known slice in facemesh_features
    # but here we rely on facemesh_features ordering: mouth_corner block starts at index 31 (7+5+7+7+5=31)
    # safer: slice based on mouth_corner length (8)
    start = 7 + 5 + 7 + 7 + 5  # up to nose_flare
    mouth_block = feats[start : start + 8]
    return float(mouth_block[0]) if mouth_block.size >= 1 else None


def compute_talking_threshold(values: List[float], window: int = TALKING_WINDOW) -> float:
    """Compute a per-user talking variance threshold from mouth_open history."""
    if len(values) < window:
        return TALKING_THRESHOLD_DEFAULT
    vals = np.array(values, dtype=float)
    # sliding window variance
    buf: Deque[float] = deque(maxlen=window)
    variances: List[float] = []
    for v in vals:
        buf.append(v)
        if len(buf) == window:
            variances.append(float(np.var(buf)))
    if not variances:
        return TALKING_THRESHOLD_DEFAULT
    variances_np = np.array(variances, dtype=float)
    gate = float(np.percentile(variances_np, 90))  # high percentile of observed jitter
    gate = float(np.clip(gate, TALKING_THRESHOLD_DEFAULT, TALKING_THRESHOLD_MAX))
    return gate


def _wrap_text_lines(text: str, max_chars: int = 52) -> List[str]:
    """Wrap a paragraph into shorter lines for on-screen display."""
    words = text.split()
    lines: List[str] = []
    current: List[str] = []
    for word in words:
        candidate = " ".join(current + [word])
        if len(candidate) > max_chars and current:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return lines


def _draw_multiline_text(
    frame,
    text: str,
    origin: Tuple[int, int] = (20, 50),
    max_chars: int = 52,
    line_height: int = 28,
    color: Tuple[int, int, int] = (0, 255, 255),
    thickness: int = 2,
    scale: float = 0.7,
) -> None:
    """Render wrapped text lines onto the frame."""
    lines = _wrap_text_lines(text, max_chars=max_chars)
    x, y = origin
    for idx, line in enumerate(lines):
        y_offset = y + idx * line_height
        cv2.putText(frame, line, (x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def _show_countdown(cap: cv2.VideoCapture, window_name: str, steps: List[Tuple[str, float]]) -> bool:
    """Display a short countdown in the existing window. Returns False on abort."""
    for text, duration in steps:
        start = time.time()
        while time.time() - start < duration:
            ret, frame = cap.read()
            if not ret:
                continue
            overlay = frame.copy()
            _draw_multiline_text(
                overlay,
                text=text,
                origin=(40, 80),
                max_chars=40,
                line_height=36,
                color=(0, 255, 255),
                thickness=2,
                scale=1.0,
            )
            cv2.imshow(window_name, overlay)
            key = cv2.waitKey(1) & 0xFF
            if key in {27, ord("q"), ord("Q")}:
                return False
    return True


def _overlay_timer(frame, remaining: float) -> None:
    """Draw a simple timer in the top-right corner."""
    remaining = max(0.0, remaining)
    text = f"{remaining:04.1f}s"
    height, width = frame.shape[:2]
    cv2.putText(
        frame,
        text,
        (width - 130, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )


def check_talking_data(values: List[float], min_samples: int = TALKING_MIN_SAMPLES) -> Tuple[bool, str]:
    """Validate collected talking samples and return status."""
    if not values:
        return False, "Keine Talking-Samples erfasst."
    if len(values) < min_samples:
        return False, f"Zu wenige Talking-Samples ({len(values)}/{min_samples}). Bitte erneut aufnehmen."
    if not np.all(np.isfinite(values)):
        return False, "Ungueltige Werte in Talking-Samples (NaN/inf)."
    return True, "Talking-Samples OK."


def run_talking_setup(
    camera_index: int = 0,
    window: int = TALKING_WINDOW,
    cap: Optional[cv2.VideoCapture] = None,
    window_name: str = "Emotion Profiling",
    prompt_text: str = "",
    capture_duration: float = 11.0,
    countdown_steps: Optional[List[Tuple[str, float]]] = None,
) -> Dict[str, float]:
    """
    Open camera, play a short countdown, and collect a fixed-duration reading sample.
    Threshold computation stays identical (variance over sliding window).
    Reuses the provided capture + window to avoid creating new windows.
    """
    own_cap = False
    if cap is None:
        cap = cv2.VideoCapture(camera_index)
        own_cap = True
    if not cap.isOpened():
        raise RuntimeError(f"Kamera-Index {camera_index} konnte nicht geoeffnet werden.")

    mouth_values: List[float] = []
    reading_text = prompt_text or DEFAULT_READING_TEXT
    steps = countdown_steps or COUNTDOWN_STEPS

    print("Talking-Kalibrierung: Countdown startet, bitte gleich laut vorlesen.")
    print("ESC oder q zum Abbrechen.")

    try:
        if not _show_countdown(cap, window_name, steps):
            return {"threshold": TALKING_THRESHOLD_DEFAULT, "samples": 0}

        start_time = time.time()
        end_time = start_time + capture_duration
        while time.time() < end_time:
            ret, frame = cap.read()
            if not ret:
                continue
            boxes = detect_faces(frame)
            if boxes:
                bbox = boxes[0]
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)
                mouth_open = _compute_mouth_open_features(frame, bbox)
                if mouth_open is not None:
                    mouth_values.append(mouth_open)
            _draw_multiline_text(
                frame,
                text=reading_text,
                origin=(20, 60),
                max_chars=60,
                line_height=26,
                color=(0, 255, 255),
                thickness=2,
                scale=0.65,
            )
            _overlay_timer(frame, end_time - time.time())
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in {27, ord("q"), ord("Q")}:
                mouth_values.clear()
                break
    finally:
        if own_cap:
            cap.release()
            cv2.destroyAllWindows()

    ok, msg = check_talking_data(mouth_values)
    if not ok:
        print(f"Fehler: {msg}")
        return {"threshold": TALKING_THRESHOLD_DEFAULT, "samples": len(mouth_values)}

    threshold = compute_talking_threshold(mouth_values, window=window)
    print(f"Talking-Threshold berechnet: {threshold:.4f} (Samples: {len(mouth_values)})")
    return {"threshold": threshold, "samples": len(mouth_values)}


if __name__ == "__main__":
    result = run_talking_setup()
    print(result)
