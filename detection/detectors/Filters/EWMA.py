"""Exponentially Weighted Moving Average filter for emotion probabilities."""

from __future__ import annotations

import numpy as np


class EWMAFilter:
    """Maintain an EMA over probability vectors and emit confident labels."""

    def __init__(self, classes, alpha=0.3, threshold=0.6):
        self.classes = classes
        self.alpha = alpha
        self.threshold = threshold
        self.state = np.zeros(len(classes), dtype=float)

    def update(self, probs: np.ndarray):
        """Update state with new probability vector."""
        probs = np.asarray(probs, dtype=float)
        if probs.sum() == 0:
            return None, 0.0
        probs = probs / probs.sum()
        if not self.state.any():
            self.state = probs
        else:
            self.state = self.alpha * probs + (1.0 - self.alpha) * self.state
        idx = int(np.argmax(self.state))
        confidence = float(self.state[idx])
        if confidence >= self.threshold:
            return self.classes[idx], confidence
        return None, confidence
