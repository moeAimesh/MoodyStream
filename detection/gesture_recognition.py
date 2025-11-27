"""Aufgabe: Hand-/Körpergesten (z. B. Thumbs Up) mit MediaPipe erkennen.

Eingaben: Frame.
Ausgaben: symbolischer String (z. B. "thumbsup" oder None).
"""
from detection.detectors.gestures.gesture_rules import validate_gesture
from detection.detectors.gestures.model_inference import classify_gesture
import cv2
import mediapipe as mp
import numpy as np
from utils.mediapipe_fix import apply_fix

apply_fix()

# Configure MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize hand detection
_hands = None

def get_hands():
    """Singleton für die Hands-Instanz."""
    global _hands
    if _hands is None:
        _hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=1
        )
    return _hands

def detect_gestures(frame):
    """Analysiert das Bild und gibt eine Liste erkannter Gesten zurück."""
    h, w, _ = frame.shape
    gestures = []

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = get_hands().process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            # 1. Hand zeichnen
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

            # 2. SVM-Vorhersage
            gesture = classify_gesture(hand_landmarks)

            # 3. Regel-basierte Prüfung
            if validate_gesture(gesture, hand_landmarks, w, h):
                final_gesture = gesture
            else:
                final_gesture = None

            # 4. Geste auf das Video schreiben
            cv2.putText(
                frame,
                f"Geste: {final_gesture}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

            # 5. Ergebnis speichern
            gestures.append(final_gesture)

    return gestures
