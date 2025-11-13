"""Single-face detection + tracking helper."""

from __future__ import annotations

from typing import Dict, Optional

import cv2

from detection.face_detection import detect_faces
from detection.tracking_types import TrackInfo, random_color
from utils.settings import FACE_DETECT_INTERVAL_SINGLE, SINGLE_FACE_TRACKER


class SingleFaceTracker:
    """Wraps OpenCV single-object tracker with periodic face detection."""

    def __init__(self):
        self.tracker = None
        self.track: Optional[TrackInfo] = None
        self.last_detection_frame = -FACE_DETECT_INTERVAL_SINGLE

    def update(self, frame, frame_idx, now) -> Dict[int, TrackInfo]:
        tracks: Dict[int, TrackInfo] = {}
        need_detection = (
            self.tracker is None
            or frame_idx - self.last_detection_frame >= FACE_DETECT_INTERVAL_SINGLE
        )

        if need_detection:
            boxes = detect_faces(frame)
            if boxes:
                bbox = boxes[0]
                self.tracker = _create_tracker(SINGLE_FACE_TRACKER)
                self.tracker.init(frame, tuple(bbox))
                color = self.track.color if self.track else random_color()
                self.track = TrackInfo(track_id=0, bbox=bbox, color=color, last_seen=now)
                self.last_detection_frame = frame_idx

        if self.tracker and self.track:
            ok, bbox = self.tracker.update(frame)
            if ok:
                bbox = tuple(int(v) for v in bbox)
                self.track.bbox = bbox
                self.track.last_seen = now
                tracks[0] = self.track
            else:
                self.tracker = None
                self.track = None

        return tracks


def _create_tracker(name: str):
    name = name.upper()
    if name == "CSRT":
        return cv2.TrackerCSRT_create()
    if name == "KCF":
        return cv2.TrackerKCF_create()
    if name == "MOSSE":
        return cv2.TrackerMOSSE_create()
    raise ValueError(f"Unsupported tracker type: {name}")
