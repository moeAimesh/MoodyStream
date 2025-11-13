"""Centralized project paths, constants, and shared configuration."""

from pathlib import Path
from typing import List

# Canonical base folders
BASE_DIR = Path(__file__).resolve().parents[1]
SETUP_DIR = BASE_DIR / "setup"
SOUNDS_DIR = BASE_DIR / "sounds"
PROFILES_DIR = SETUP_DIR / "profiles"

# Frequently accessed resources
SOUND_CACHE_DIR = SOUNDS_DIR / "sound_cache"
REST_FACE_MODEL_PATH = SETUP_DIR / "rest_face_model.json"
SOUND_MAP_PATH = SOUNDS_DIR / "sound_map.json"
SETUP_CONFIG_PATH = SETUP_DIR / "setup_config.json"

# Optional behaviour keys (emotions + gestures) used by sound setup
ALLOWED_BEHAVIOUR_KEYS: List[str] = [
    "happy",
    "sad",
    "angry",
    "surprise",
    "neutral",
    "thumbsup",
    "peace",
    "wave",
    "fist",
    "ok",
    "laugh",
]

# Default start URL for the myinstants integration
MYINSTANTS_URL = "https://www.myinstants.com/en/index/de/"


def rest_face_model_file() -> str:
    """Return the canonical rest-face model path as a string for legacy callers."""
    return str(REST_FACE_MODEL_PATH)


# --- Backwards compatibility aliases (older modules still import these names) ---
BASE = BASE_DIR
CACHE = SOUND_CACHE_DIR
PROFILES = PROFILES_DIR
