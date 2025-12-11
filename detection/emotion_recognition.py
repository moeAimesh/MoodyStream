import base64
import json
import time
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import Optional, Union

import joblib
import numpy as np
from deepface import DeepFace

from detection.detectors.Filters import EWMAFilter, HiddenMarkovModelFilter
from detection.emotion_heuristics import (
    EmotionHeuristicScorer,
    HeuristicThresholds,
    compute_thresholds_from_samples,
)
from utils.settings import (
    EMOTION_FILTER,
    EMOTION_SWITCH_FRAMES,
    EWMA_ALPHA,
    EWMA_THRESHOLD,
    HMM_STAY_PROB,
    REST_FACE_MODEL_PATH,
    HEURISTIC_DEBUG,
    HEURISTIC_DEBUG_INTERVAL,
)

CLASSIFIER_CONFIDENCE = 0.35
DEFAULT_GATE = 0.40
DEFAULT_MARGIN = 0.01
STATE_TTL_SECONDS = 600  # purge per-track state after N seconds of inactivity
STATE_PURGE_INTERVAL = 60  # throttle how often we scan for stale tracks


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
        self.stable_emotion = {}
        self.filter_mode = (EMOTION_FILTER or "none").lower()
        self.filters = {}
        self.switch_frames = EMOTION_SWITCH_FRAMES
        self.switch_state = {}
        self.emotion_classes = ["fear", "happy", "sad", "surprise", "neutral"]
        self.class_index = {label: idx for idx, label in enumerate(self.emotion_classes)}

        self.profile_means = {}
        self.mean_vector = None
        self.distance_stats = {"mean": None, "std": None}
        self.profile_stats = {}
        self.classifier = None
        self.classifier_dim = None
        self.classifier_feature_len = None
        self.scaler_mean = None
        self.scaler_scale = None
        self.classifier_model = None
        self.classifier_model_type = None
        self.classifier_model_classes = None
        self.neutral_feature_mean = None
        self.confidence_gate = DEFAULT_GATE
        self.confidence_margin = DEFAULT_MARGIN
        self.state_ttl = STATE_TTL_SECONDS
        self._last_purge = 0.0
        self.heuristics = EmotionHeuristicScorer(
            debug=HEURISTIC_DEBUG,
            debug_interval=HEURISTIC_DEBUG_INTERVAL,
            neutral_baseline=None,
        )
        self._load_profiles()

    def analyze_frame(self, frame, features=None, track_id: Optional[int] = None) -> Optional[str]:
        """Analyze a frame for a given track (default context if None)."""
        key = track_id if track_id is not None else "_default"
        now = time.time()
        self._purge_stale_states(now)

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
            feature_vec_raw = np.array(features, dtype=float) if features is not None else None
            feature_vec = feature_vec_raw
            if feature_vec is not None and self.neutral_feature_mean is not None:
                if feature_vec.shape[0] == self.neutral_feature_mean.shape[0]:
                    feature_vec = feature_vec - self.neutral_feature_mean
            combined_vector = None
            if self.classifier_feature_len is None:
                combined_vector = v_current
            elif (
                feature_vec is not None
                and self.classifier_feature_len is not None
                and feature_vec.shape[0] == self.classifier_feature_len - len(v_current)
            ):
                combined_vector = np.concatenate([v_current, feature_vec])
            elif self.classifier_feature_len == len(v_current):
                combined_vector = v_current
            if (
                combined_vector is not None
                and self.classifier_feature_len is not None
                and combined_vector.shape[0] != self.classifier_feature_len
            ):
                combined_vector = None

            emotion = None
            classifier_probs = None
            confidence = None
            if self.classifier_model is not None and combined_vector is not None:
                _, _, classifier_probs = self._predict_classifier_model(combined_vector)
            elif self.classifier is not None and combined_vector is not None:
                _, _, classifier_probs = self._predict_classifier_linear(combined_vector)

            if classifier_probs is not None:
                top2 = np.partition(classifier_probs, -2)[-2:]
                idx = int(np.argmax(classifier_probs))
                confidence = float(classifier_probs[idx])
                margin = confidence - float(top2[0] if classifier_probs.size > 1 else 0.0)
                required_gate = max(CLASSIFIER_CONFIDENCE, self.confidence_gate)
                required_margin = self.confidence_margin
                candidate = None
                # Einheitliche Confidence-Policy: Gate + Margin für alle Klassen
                if confidence >= required_gate and margin >= required_margin:
                    candidate = self.emotion_classes[idx]
                else:
                    classifier_probs = None
                if candidate and feature_vec_raw is not None:
                    if not self.heuristics.validate(candidate, feature_vec_raw):
                        candidate = None
                        classifier_probs = None
                emotion = candidate

            if emotion is None:
                emotion = "neutral"

            measurement = None
            if classifier_probs is not None:
                measurement = classifier_probs
            elif emotion:
                measurement = self._label_to_vector(emotion)
            filtered = self._apply_filter(key, emotion, measurement)
            return filtered

        except Exception as exc:  # pragma: no cover
            print(f"⚠️ Fehler bei Emotionserkennung: {exc}")
            return self.stable_emotion.get(key)

    def _load_profiles(self):
        if not self.model_path.exists():
            print("⚠️ Kein Rest-Face-/Profil-Modell gefunden. Bitte Kalibrierung ausführen!")
            return
        with self.model_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        heuristic_thresholds = None
        heuristic_blob = data.get("heuristic_thresholds")
        if heuristic_blob:
            defaults = HeuristicThresholds()
            heuristic_thresholds = HeuristicThresholds(
                surprise_mouth_open_min=heuristic_blob.get(
                    "surprise_mouth_open_min", defaults.surprise_mouth_open_min
                ),
                fear_lid_open_min=heuristic_blob.get("fear_lid_open_min", defaults.fear_lid_open_min),
                sad_drop_min=heuristic_blob.get("sad_drop_min", defaults.sad_drop_min),
                sad_avg_min=heuristic_blob.get("sad_avg_min", defaults.sad_avg_min),
                sad_mouth_open_max=heuristic_blob.get("sad_mouth_open_max", defaults.sad_mouth_open_max),
                happy_mouth_curve_max=heuristic_blob.get(
                    "happy_mouth_curve_max", defaults.happy_mouth_curve_max
                ),
                happy_cheek_mean_min=heuristic_blob.get(
                    "happy_cheek_mean_min", defaults.happy_cheek_mean_min
                ),
            )

        profiles = data.get("profiles")
        if profiles:
            for emotion, payload in profiles.items():
                mean_vec = np.array(payload["mean_vector"])
                std = payload.get("distance_std") or 1.0
                self.profile_means[emotion] = {"mean": mean_vec, "std": std}
                self.profile_stats[emotion] = {
                    "mean": payload.get("distance_mean"),
                    "std": std,
                }
                if emotion == "neutral":
                    self.distance_stats["mean"] = payload.get("distance_mean")
                    self.distance_stats["std"] = std
            print(f"✅ {len(self.profile_means)} Emotion-Profile geladen ({self.model_path})")
        elif "mean_vector" in data:
            self.mean_vector = np.array(data["mean_vector"])
            self.distance_stats["mean"] = data.get("distance_mean")
            self.distance_stats["std"] = data.get("distance_std")
            print(f"✅ Rest-Face mean_vector geladen ({self.model_path})")
        else:
            print("⚠️ Modell-Datei enthält keine nutzbaren Profile.")

        if heuristic_thresholds is None and profiles:
            feature_vectors = {
                emotion: payload.get("feature_vectors")
                for emotion, payload in profiles.items()
                if isinstance(payload, dict) and payload.get("feature_vectors")
            }
            heuristic_thresholds = compute_thresholds_from_samples(feature_vectors)

        classifier = data.get("classifier")
        if not classifier:
            if heuristic_thresholds:
                self.heuristics.set_thresholds(heuristic_thresholds)
                print("ƒo. AU-Heuristik aus Profil-Features kalibriert.")
            return

        self.classifier_feature_len = classifier.get("feature_length")
        neutral_mean = data.get("neutral_feature_mean")
        if neutral_mean is not None:
            self.neutral_feature_mean = np.array(neutral_mean, dtype=float)
        thresholds = data.get("thresholds", {})
        self.confidence_gate = thresholds.get("gate", DEFAULT_GATE)
        self.confidence_margin = thresholds.get("margin", DEFAULT_MARGIN)
        mean = classifier.get("scaler_mean")
        scale = classifier.get("scaler_scale")
        if mean is not None and scale is not None:
            self.scaler_mean = np.array(mean)
            self.scaler_scale = np.array(scale)

        model_blob = classifier.get("model_blob")
        if model_blob:
            try:
                buffer = BytesIO(base64.b64decode(model_blob))
                self.classifier_model = joblib.load(buffer)
                self.classifier_model_type = classifier.get("model_type", "logreg")
                self.classifier_model_classes = classifier.get("model_classes")
            except Exception as exc:
                print(f"⚠️ Klassifikator konnte nicht geladen werden: {exc}")
        elif "coef" in classifier:
            self.classifier = {
                "classes": classifier["classes"],
                "coef": np.array(classifier["coef"]),
                "intercept": np.array(classifier["intercept"]),
            }
            self.classifier_dim = self.classifier["coef"].shape[1]
        print("✅ Personalisierter Klassifikator geladen.")
        if heuristic_thresholds:
            self.heuristics.set_thresholds(heuristic_thresholds)
            print("✅ AU-Heuristik aus Profil-Features kalibriert.")

    def _predict_profile(self, vector):
        best_label = None
        best_score = None
        best_dist = None
        for emotion, stats in self.profile_means.items():
            mean_vec = stats["mean"]
            std = stats.get("std") or 1.0
            dist = np.linalg.norm(vector - mean_vec)
            norm_dist = dist / std
            if best_score is None or norm_dist < best_score:
                best_score = norm_dist
                best_label = emotion
                best_dist = dist
        return best_label, best_dist, best_score

    def _predict_classifier_model(self, vector):
        vector = self._scale_vector(vector)
        probs = self.classifier_model.predict_proba([vector])[0]
        classes_attr = getattr(self.classifier_model, "classes_", None)
        if classes_attr is not None:
            classes = list(classes_attr)
        elif self.classifier_model_classes:
            classes = list(self.classifier_model_classes)
        else:
            classes = []
        if not classes:
            raise RuntimeError("Classifier-Klassen sind nicht definiert.")
        idx = int(np.argmax(probs))
        label = str(classes[idx])
        mapped = np.zeros(len(self.emotion_classes), dtype=float)
        for c_idx, class_label in enumerate(classes):
            mapped[self._label_index(class_label)] = probs[c_idx]
        return label, float(probs[idx]), mapped

    def _predict_classifier_linear(self, vector):
        vector = self._scale_vector(vector)
        coef = self.classifier["coef"]
        intercept = self.classifier["intercept"]
        logits = coef @ vector + intercept
        logits = logits - np.max(logits)
        exp = np.exp(logits)
        probs = exp / np.sum(exp)
        idx = int(np.argmax(probs))
        label = self.classifier["classes"][idx]
        mapped = np.zeros(len(self.emotion_classes), dtype=float)
        for c_idx, class_label in enumerate(self.classifier["classes"]):
            mapped[self._label_index(class_label)] = probs[c_idx]
        return label, float(probs[idx]), mapped

    def _scale_vector(self, vector):
        if self.scaler_mean is not None and self.scaler_scale is not None:
            denom = np.where(self.scaler_scale == 0, 1.0, self.scaler_scale)
            return (vector - self.scaler_mean) / denom
        return vector

    def _maybe_report_distance(self, distance, key, emotion):
        stats = self.profile_stats.get(emotion, self.distance_stats)
        mean = stats.get("mean")
        std = stats.get("std") or 0
        if mean is None or std is None:
            return
        if std == 0:
            std = 1e-6
        if distance > mean + 3 * std:
            print(
                f"⚠️ Track {key}: Abstand {distance:.2f} weicht stark vom {emotion}-Profil ab "
                f"(μ={mean:.2f}, σ={std:.2f})"
            )

    def _label_index(self, label: str) -> int:
        return self.class_index.get(label, self.class_index["neutral"])

    def _label_to_vector(self, label: str) -> np.ndarray:
        vec = np.zeros(len(self.emotion_classes), dtype=float)
        vec[self._label_index(label)] = 1.0
        return vec

    def _get_filter(self, key):
        if self.filter_mode == "none":
            return None
        if key not in self.filters:
            if self.filter_mode == "ewma":
                self.filters[key] = EWMAFilter(
                    self.emotion_classes,
                    alpha=EWMA_ALPHA,
                    threshold=EWMA_THRESHOLD,
                )
            elif self.filter_mode == "hmm":
                self.filters[key] = HiddenMarkovModelFilter(
                    self.emotion_classes,
                    stay_prob=HMM_STAY_PROB,
                )
            else:
                self.filters[key] = None
        return self.filters[key]

    def _apply_filter(self, key, label, measurement: Optional[np.ndarray]):
        if measurement is None and label:
            measurement = self._label_to_vector(label)
        filt = self._get_filter(key)
        filtered = label if filt is None else None
        if filt is not None and measurement is not None:
            candidate, _ = filt.update(measurement)
            filtered = candidate
        filtered = self._apply_switch(key, filtered)
        if filtered is not None:
            self.stable_emotion[key] = filtered
        return filtered or self.stable_emotion.get(key)

    def _apply_switch(self, key, candidate: Optional[str]) -> Optional[str]:
        if candidate is None:
            return self.stable_emotion.get(key)
        if self.switch_frames is None or self.switch_frames <= 0:
            self.switch_state.pop(key, None)
            return candidate

        current = self.stable_emotion.get(key)
        if current is None:
            self.switch_state.pop(key, None)
            return candidate
        if candidate == current:
            state = self.switch_state.get(key)
            if state:
                state["pending"] = None
                state["count"] = 0
            return candidate

        state = self.switch_state.setdefault(key, {"pending": None, "count": 0})
        if state["pending"] != candidate:
            state["pending"] = candidate
            state["count"] = 1
            return current
        state["count"] += 1
        if state["count"] >= self.switch_frames:
            state["pending"] = None
            state["count"] = 0
            return candidate
        return current

    def _purge_stale_states(self, now: float):
        """Drop per-track state that has been inactive longer than TTL."""
        if self.state_ttl is None or self.state_ttl <= 0:
            return
        if now - self._last_purge < STATE_PURGE_INTERVAL:
            return
        cutoff = now - self.state_ttl
        for key, ts in list(self.last_analysis.items()):
            if key == "_default" or ts >= cutoff:
                continue
            self.drop_track_state(key)
        self._last_purge = now

    def drop_track_state(self, track_id: Optional[int] = None):
        """Explicitly remove cached state for a track (used when a track is dropped)."""
        key = track_id if track_id is not None else "_default"
        self.last_analysis.pop(key, None)
        self.stable_emotion.pop(key, None)
        self.switch_state.pop(key, None)
        self.filters.pop(key, None)
