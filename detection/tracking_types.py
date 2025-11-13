"""Shared data structures for tracking logic."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class TrackInfo:
    track_id: int
    bbox: Tuple[int, int, int, int]
    color: Tuple[int, int, int]
    last_seen: float
    last_enqueued: float = 0.0
    emotion: Optional[str] = None


def random_color() -> Tuple[int, int, int]:
    """Generate a bright-ish random color for drawing boxes."""
    return tuple(random.randint(80, 255) for _ in range(3))
