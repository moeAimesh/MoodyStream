"""Aufgabe: Zentrale Loop über die Webcam.

pro Frame: Gesichter/Emotionen analysieren, Gesten erkennen,

Entscheidung treffen (Mapper),

passenden Sound abspielen,

Live-Feedback in GUI melden.

Eingaben: Frames (OpenCV), aktuelles Profil, Mapper-Ergebnisse.

Ausgaben: Trigger-Events („play: ok“), Logs.

Wichtig: Nicht blockieren → Threads verwenden (Kamera, Audio, GUI getrennt"""

import cv2, time
from detection.gesture_recognition import detect_gestures
from sounds.play_sound import play
from utils.mediapipe_fix import apply_fix

apply_fix()

def start_detection():
    cap = cv2.VideoCapture(0)
    thumb_start_time = None
    sound_played = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gestures = detect_gestures(frame)
        current_time = time.time()

        if "thumbsup" in gestures:
            if thumb_start_time is None:
                thumb_start_time = current_time
                sound_played = False
            elif (current_time - thumb_start_time) >= 1 and not sound_played:
                play("sounds/sound_cache/Rammus_Select.mp3")
                sound_played = True
        else:
            thumb_start_time = None
            sound_played = False

        if "thumbsup" in gestures:
            cv2.putText(frame, "👍 Daumen hoch erkannt!",
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2)

        cv2.imshow("Gesture Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
