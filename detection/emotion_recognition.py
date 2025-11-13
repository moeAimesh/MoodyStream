import json
import time
from pathlib import Path
import numpy as np
from deepface import DeepFace
from collections import deque
from statistics import mode

from utils.settings import REST_FACE_MODEL_PATH

class EmotionRecognition:
    """
    Erkennt stabile Emotionen mit Glättung + Rest-Face-Abgleich.
    """

    def __init__(self, model_path=REST_FACE_MODEL_PATH,
                 threshold=10.0, interval=0.3, history_size=7):
        """
        threshold: Distanz-Schwelle für Rest-Face.
        interval: Analyseintervall (Sekunden).
        history_size: Wie viele letzte Emotionen gespeichert werden.
        """
        self.model_path = Path(model_path)
        self.threshold = threshold
        self.interval = interval
        self.last_analysis = 0
        self.recent_emotions = deque(maxlen=history_size)
        self.mean_vector = None
        self.stable_emotion = None

        if self.model_path.exists():
            with self.model_path.open("r") as f:
                data = json.load(f)
                self.mean_vector = np.array(data["mean_vector"])
            print(f"✅ Rest-Face mean_vector geladen ({self.model_path})")
        else:
            print("⚠️ Kein Rest-Face-Modell gefunden. Bitte Kalibrierung ausführen!")

    def analyze_frame(self, frame):
        """
        Analysiert ein einzelnes Frame (alle interval Sekunden).
        Kombiniert DeepFace-Ergebnis mit Rest-Face-Abstand und Glättung.
        """
        now = time.time()
        if now - self.last_analysis < self.interval:
            return self.stable_emotion  # zu früh, altes Ergebnis zurückgeben

        self.last_analysis = now

        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, detector_backend= 'mediapipe')
            emotion_scores = result[0]["emotion"]
            dominant = result[0]["dominant_emotion"]
            v_current = np.array(list(emotion_scores.values()))

            # Wenn kein Rest-Face existiert → direkt Emotion
            if self.mean_vector is None:
                emotion = dominant
            else:
                # Distanz zur neutralen Emotion berechnen
                distance = np.linalg.norm(v_current - self.mean_vector)
                emotion = "neutral" if distance < self.threshold else dominant

            # Glättung: aktuelle Emotion in Verlauf einfügen
            self.recent_emotions.append(emotion)
            self.stable_emotion = mode(self.recent_emotions)

            return self.stable_emotion

        except Exception as e:
            print(f"⚠️ Fehler bei Emotionserkennung: {e}")
            return self.stable_emotion
