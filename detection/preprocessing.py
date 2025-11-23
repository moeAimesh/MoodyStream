"""Shared face preprocessing helpers."""

from __future__ import annotations

import cv2
import numpy as np

from utils.settings import PREPROCESS_PAD_RATIO

_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def extract_face(frame, bbox, pad_ratio: float = PREPROCESS_PAD_RATIO):
    """Extract face region with optional padding to keep some context."""
    x, y, w, h = bbox
    h_frame, w_frame, _ = frame.shape
    pad_w = int(w * pad_ratio)
    pad_h = int(h * pad_ratio)
    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(w_frame, x + w + pad_w)
    y2 = min(h_frame, y + h + pad_h)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop


def _balance_white(rgb: np.ndarray) -> np.ndarray:
    """Apply a simple gray-world white balance."""
    img = rgb.astype(np.float32)
    mean = img.mean(axis=(0, 1), keepdims=True)
    scale = mean.mean() / (mean + 1e-6)
    balanced = np.clip(img * scale, 0, 255)
    return balanced.astype(np.uint8)


def _enhance_contrast(rgb: np.ndarray) -> np.ndarray:
    """Use CLAHE on the luminance channel to reduce lighting variance."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = _CLAHE.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def _denoise(rgb: np.ndarray) -> np.ndarray:
    """Preserve edges while knocking down sensor noise."""
    return cv2.bilateralFilter(rgb, d=5, sigmaColor=50, sigmaSpace=50)


def _normalize_range(rgb: np.ndarray) -> np.ndarray:
    """Stretch pixel range to the full [0, 255] interval."""
    return cv2.normalize(rgb, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)


def preprocess_face(frame, bbox):
    """Crop, enhance, and resize faces before feature extraction."""
    crop = extract_face(frame, bbox)
    if crop is None:
        return None
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    rgb = _balance_white(rgb)
    rgb = _enhance_contrast(rgb)
    rgb = _denoise(rgb)
    rgb = _normalize_range(rgb)
    return cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
