import json
import time
from collections import defaultdict, deque
from pathlib import Path
from statistics import mode
from typing import Optional, Union

import numpy as np
from deepface import DeepFace

from utils.settings import REST_FACE_MODEL_PATH


class EmotionRecognition:
    """Handles DeepFace analysis with per-track smoothing and rest-face comparison."""

    def __init__(
        self,
        model_path: Union[str, Path] = REST_FACE_MODEL_PATH,
        threshold: float = 10.0,
        interval: float = 0.3,
        history_size: int = 7,
    ):
        self.model_path = Path(model_path)
        self.threshold = threshold
        self.interval = interval
        self.history_size = history_size

        self.last_analysis = defaultdict(float)  # context_id -> timestamp
        self.recent_emotions = defaultdict(lambda: deque(maxlen=self.history_size))
        self.stable_emotion = {}

        self.mean_vector = None
        if self.model_path.exists():
            with self.model_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                self.mean_vector = np.array(data["mean_vector"])
            print(f"✅ Rest-Face mean_vector geladen ({self.model_path})")
        else:
            print("⚠️ Kein Rest-Face-Modell gefunden. Bitte Kalibrierung ausführen!")

    def analyze_frame(self, frame, track_id: Optional[int] = None) -> Optional[str]:
        """Analyze a frame for a given track (default context if None)."""
        key = track_id if track_id is not None else "_default"
        now = time.time()

        if now - self.last_analysis[key] < self.interval:
            return self.stable_emotion.get(key)

        self.last_analysis[key] = now

        try:
            result = DeepFace.analyze(
                frame,
                actions=["emotion"],
                enforce_detection=False,
                detector_backend="mediapipe",
            )
            emotion_scores = result[0]["emotion"]
            dominant = result[0]["dominant_emotion"]
            v_current = np.array(list(emotion_scores.values()))

            if self.mean_vector is None:
                emotion = dominant
            else:
                distance = np.linalg.norm(v_current - self.mean_vector)
                emotion = "neutral" if distance < self.threshold else dominant

            self.recent_emotions[key].append(emotion)
            stable = mode(self.recent_emotions[key])
            self.stable_emotion[key] = stable
            return stable

        except Exception as exc:  # pragma: no cover
            print(f"⚠️ Fehler bei Emotionserkennung: {exc}")
            return self.stable_emotion.get(key)
