"""Aufgabe: Aus Emotions-Scores eine diskrete Emotion ableiten (mit Schwellenwerten, Hysterese, Mehrframe-Mehrheit).

Eingaben: Scores von face_analyzer.

Ausgaben: z. B. "laugh", "angry" oder None.

Ziel: Stabile Entscheidungen, nicht bei jedem Frame springen."""

import os
import json
import numpy as np
from deepface import DeepFace


class EmotionRecognition:
    """
    Ermittelt die aktuelle Emotion aus dem Kamerabild
    und erkennt automatisch, ob es sich nur um das Rest-Face handelt.
    """

    def __init__(self, model_path="setup/rest_face_model.json", threshold=10.0):
        """
        Lädt das gespeicherte Rest-Face-Modell und legt den Schwellenwert fest.
        threshold: Euklidische Distanz, ab der eine Emotion als „echt“ gilt.
        """
        self.threshold = threshold
        self.mean_vector = None

        if os.path.exists(model_path):
            with open(model_path, "r") as f:
                data = json.load(f)
                self.mean_vector = np.array(data["mean_vector"])
            print(f"✅ Rest-Face mean_vector geladen ({model_path})")
        else:
            print("⚠️ Kein Rest-Face-Modell gefunden. Bitte Kalibrierung ausführen!")

    def analyze_frame(self, frame):
        """
        Analysiert ein einzelnes Kamerabild mit DeepFace.
        Gibt 'neutral' zurück, wenn Rest-Face erkannt wird,
        sonst die dominante Emotion.
        """
        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            emotion_scores = result[0]["emotion"]
            dominant = result[0]["dominant_emotion"]

            v_current = np.array(list(emotion_scores.values()))

            # Wenn kein Rest-Face vorhanden, gib die Emotion direkt zurück
            if self.mean_vector is None:
                return dominant

            # Distanz zum gespeicherten Rest-Face berechnen
            distance = np.linalg.norm(v_current - self.mean_vector)

            if distance < self.threshold:
                # innerhalb der Neutralzone → kein Trigger
                print(f"🧊 Rest-Face erkannt (Distanz={distance:.2f})")
                return "neutral"
            else:
                # klare Emotion erkannt
                print(f"🔥 Emotion erkannt: {dominant} (Distanz={distance:.2f})")
                return dominant

        except Exception as e:
            print(f"⚠️ Fehler bei Emotionserkennung: {e}")
            return None
