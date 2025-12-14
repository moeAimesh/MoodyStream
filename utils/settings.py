"""Centralized project paths, constants, and shared configuration."""

from pathlib import Path
from typing import List

# Canonical base folders
BASE_DIR = Path(__file__).resolve().parents[1]
SETUP_DIR = BASE_DIR / "setup"
SOUNDS_DIR = BASE_DIR / "sounds"

# Frequently accessed resources
SOUND_CACHE_DIR = SOUNDS_DIR / "sound_cache"
REST_FACE_MODEL_PATH = SETUP_DIR / "rest_face_model.json"
SOUND_MAP_PATH = SOUNDS_DIR / "sound_map.json"
SETUP_CONFIG_PATH = SETUP_DIR / "setup_config.json"

# Setup toggles
FACE_SETUP_ENABLED = True  # Set False to reuse snapshots and skip new recordings
IMPORTANCE_TEST = "ablation"  # "ablation", "permutation", or "coef"
PLOT_AU = True  # Debug: overlay AU landmark points on crops during setup/detection
TRAINED_MODEL = "svm"  # "logreg", "svm", "random_forest", "lightgbm"
EMOTION_FILTER = "HMM"  # "HMM" or "none"
EMOTION_SWITCH_FRAMES: int | None = None  # require N consistent frames before switching label; None disables
HMM_STAY_PROB = 0.50  # probability to remain in same state per step for HMM smoothing
CLASSIFIER_CONFIDENCE = 0.50
HEURISTIC_DEBUG = True  # If True, log how often AU heuristics override model probs
HEURISTIC_DEBUG_INTERVAL = 200  # print summary every N adjustments
# Per-threshold scaling for personalized heuristic thresholds (1.0 = as recorded)
HEURISTIC_THRESH_WEIGHTS = {
    "surprise_mouth_open_min": 0.3,
    "fear_lid_open_min": 0.4,
    "sad_drop_min": 0.4,
    "sad_avg_min": 0.4,
    "sad_mouth_open_max": 0.4,
    "happy_mouth_curve_max": 0.1,
    "happy_cheek_mean_min": 2.5,
}
HEURISTIC_PROB_DEBUG = True  # If True, print model vs post-heuristic probabilities per frame
CLUSTER_RADIUS_QUANTILE = 0.99  # quantile for cluster boundary (e.g., 0.99 keeps 99% of samples)
CLUSTER_NEIGHBOR_RATIO = 0.5  # max radius is this fraction of the nearest cluster distance
TRIGGER_TIME_GESTURE_SEC = 1.0  # seconds a gesture must persist before playing sound
TRIGGER_TIME_EMOTION_SEC = 0.0  # seconds an emotion must persist before playing sound

# Detection crop padding controls the context kept around faces
FACE_DETECTOR_PAD_X = 0.20  # horizontal padding (fraction of bbox width); increase for wider cheeks/ears
FACE_DETECTOR_PAD_Y_TOP = 0.25  # upward padding (fraction of bbox height); increase to keep more forehead
FACE_DETECTOR_PAD_Y_BOTTOM = 0.10  # downward padding (fraction of bbox height); raise for chin coverage
PREPROCESS_PAD_RATIO = 0.20  # final crop padding before DeepFace/AU; increase if preprocessing needs more context

# Optional behaviour keys (emotions + gestures) used by sound setup
ALLOWED_BEHAVIOUR_KEYS: List[str] = [
    "happy",
    "sad",
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
