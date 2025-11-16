"""Helpers for publishing frames to a pyvirtualcam-based virtual webcam."""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

__all__ = [
    "VirtualCamError",
    "VirtualCamPublisher",
    "resolve_cam_dims",
    "resolve_cam_fps",
]


class VirtualCamError(RuntimeError):
    """Raised when a virtual camera output cannot be created."""


class VirtualCamPublisher:
    """Handles streaming frames to a pyvirtualcam backend."""

    def __init__(self, width: int, height: int, fps: int):
        try:
            import pyvirtualcam
            from pyvirtualcam import PixelFormat
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise VirtualCamError(
                "pyvirtualcam fehlt. Installiere es mit `pip install pyvirtualcam`."
            ) from exc

        self._width = int(width)
        self._height = int(height)
        self._camera = pyvirtualcam.Camera(
            width=self._width,
            height=self._height,
            fps=int(max(1, fps)),
            fmt=PixelFormat.BGR,
        )
        print(
            f"[virtual-cam] Sende Stream {self._width}x{self._height}@{fps} "
            f"an {self._camera.device}."
        )

    def send(self, frame: np.ndarray) -> None:
        if frame.shape[1] != self._width or frame.shape[0] != self._height:
            frame = cv2.resize(frame, (self._width, self._height))
        self._camera.send(frame)
        self._camera.sleep_until_next_frame()

    def close(self) -> None:
        self._camera.close()


def resolve_cam_dims(frame: np.ndarray, width: Optional[int], height: Optional[int]) -> Tuple[int, int]:
    """Return effective output dimensions, falling back to the current frame."""
    frame_h, frame_w = frame.shape[:2]
    return int(width or frame_w), int(height or frame_h)


def resolve_cam_fps(cap: cv2.VideoCapture, fps_override: Optional[int]) -> int:
    """Return output FPS, favoring overrides and sane defaults."""
    if fps_override:
        return int(fps_override)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    if fps <= 0:
        return 30
    return int(round(fps))

