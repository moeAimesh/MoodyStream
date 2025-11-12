"""Global path helpers reused by setup and detection modules."""

from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parents[1]
SETUP_DIR = BASE_DIR / "setup"
SOUNDS_DIR = BASE_DIR / "sounds"
PROFILES_DIR = SETUP_DIR / "profiles"

# Frequently accessed resources
SOUND_CACHE_DIR = SOUNDS_DIR / "sound_cache"
REST_FACE_MODEL_PATH = SETUP_DIR / "rest_face_model.json"


def rest_face_model_file() -> str:
    """Return the canonical rest-face model path as a string for legacy callers."""
    return str(REST_FACE_MODEL_PATH)
