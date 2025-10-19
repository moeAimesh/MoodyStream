"""Aufgabe: Hand-/Körpergesten (z. B. Thumbs Up) mit MediaPipe erkennen.

Eingaben: Frame.

Ausgaben: symbolischer String (z. B. "thumbsup" oder None).

Tipp: hier nur die Erkennung kapseln; Mapping zu Sounds macht der Mapper."""


import cv2
import mediapipe as mp
import numpy as np
from detection.detectors.gestures.thumbs_up import detect_thumbsup
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
            max_num_hands=2,  # Wieder auf 2 Hände gesetzt
            min_detection_confidence=0.7,  # Höhere Konfidenz für genauere Erkennung
            min_tracking_confidence=0.7,  # Höhere Konfidenz für stabileres Tracking
            model_complexity=1  # Mittlere Modellkomplexität für Balance
        )
    return _hands

def detect_gestures(frame):
    """Analysiert das Bild und gibt eine Liste erkannter Gesten zurück."""
    h, w, _ = frame.shape
    gestures = []

    # Verwende die Hands-Instanz
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = get_hands().process(rgb)


    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Zeichne die Hand (optional)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Prüfe, ob Daumen hoch erkannt wurde
            if detect_thumbsup(hand_landmarks, w, h):
                gestures.append("thumbsup")

    return gestures
