"""Lightweight Hidden Markov Model filter for emotion smoothing."""

from __future__ import annotations

import numpy as np


class HiddenMarkovModelFilter:
    """Applies a simple HMM with fixed transition probabilities."""

    def __init__(self, classes, stay_prob=0.85):
        self.classes = classes
        self.num_states = len(classes)
        stay = np.clip(stay_prob, 0.0, 0.99)
        change_prob = (1.0 - stay) / max(1, self.num_states - 1)
        self.transition = np.full((self.num_states, self.num_states), change_prob, dtype=float)
        np.fill_diagonal(self.transition, stay)
        self.state = np.full(self.num_states, 1.0 / self.num_states, dtype=float)

    def update(self, measurement: np.ndarray):
        measurement = np.asarray(measurement, dtype=float)
        if measurement.sum() == 0:
            measurement = np.full_like(self.state, 1.0 / self.num_states)
        else:
            measurement = measurement / measurement.sum()

        prior = self.transition @ self.state
        posterior = prior * measurement
        total = posterior.sum()
        if total == 0:
            posterior = prior
            total = posterior.sum()
        self.state = posterior / total

        idx = int(np.argmax(self.state))
        return self.classes[idx], float(self.state[idx])
