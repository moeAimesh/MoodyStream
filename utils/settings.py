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

# Setup toggles
FACE_SETUP_ENABLED = False  # Set False to reuse snapshots and skip new recordings
IMPORTANCE_TEST = "permutation"  # "ablation", "permutation", or "coef"
PLOT_AU = True  # Debug: overlay AU landmark points on crops during setup/detection
TRAINED_MODEL = "svm"  # "logreg", "svm", "random_forest", "lightgbm"
EMOTION_FILTER = "HMM"  # "EWMA", "HMM", or "none"
EMOTION_SWITCH_FRAMES: int | None = None  # require N consistent frames before switching label; None disables
EWMA_ALPHA = 0.3  # smoothing factor for EWMA filter
EWMA_THRESHOLD = 0.55  # minimum confidence before EWMA commits a label
HMM_STAY_PROB = 0.85  # probability to remain in same state per step for HMM smoothing

# Detection crop padding controls the context kept around faces
FACE_DETECTOR_PAD_X = 0.20  # horizontal padding (fraction of bbox width); increase for wider cheeks/ears
FACE_DETECTOR_PAD_Y_TOP = 0.25  # upward padding (fraction of bbox height); increase to keep more forehead
FACE_DETECTOR_PAD_Y_BOTTOM = 0.10  # downward padding (fraction of bbox height); raise for chin coverage
PREPROCESS_PAD_RATIO = 0.20  # final crop padding before DeepFace/AU; increase if preprocessing needs more context

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

# Tracking configuration
TRACKING_MODE = "none"  # "single" or "multi" or "none"
SINGLE_FACE_TRACKER = "CSRT"  # CSRT, KCF, MOSSE
FACE_DETECT_INTERVAL_SINGLE = 5  # frames between detections in single mode
FACE_DETECT_INTERVAL_MULTI = 1  # run detection every N frames in multi mode
MAX_MISSING_FRAMES = 10  # tolerance before dropping a track
