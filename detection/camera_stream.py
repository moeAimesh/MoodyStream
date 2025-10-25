"""
Zentrale Loop über die Webcam.

pro Frame:
- Gesichter/Emotionen analysieren
- Gesten erkennen
- Entscheidung treffen (Mapper)
- Passenden Sound abspielen
- Live-Feedback in GUI melden

Wichtig:
Nicht blockieren → Threads für Kamera, Audio, GUI.
"""

import cv2, time
from detection.gesture_recognition import detect_gestures
from detection.emotion_recognition import EmotionRecognition
from sounds.play_sound import play
from utils.mediapipe_fix import apply_fix
from detection.crop_face import crop_face
apply_fix()


def start_detection():
    cap = cv2.VideoCapture(0)
    thumb_start_time = None
    sound_played = False

    er = EmotionRecognition(threshold=10)
    print("🎥 Kamera gestartet – Gesten- und Emotionserkennung aktiv!")

    while True:
        ret, frame_full = cap.read()
        if not ret:
            break

        # 🧩 Cropping nur für Emotion
        face_crop = crop_face(frame_full, draw_box=True)

        # ✋ GESTENERKENNUNG – nutzt das ganze Frame!
        gestures = detect_gestures(frame_full)
        current_time = time.time()

        # 🖐️ Beispiel: Daumen-hoch-Geste → Sound
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

        # 🧠 EMOTION-ERKENNUNG – nutzt nur das Gesicht!
        if face_crop is not None:
            emotion = er.analyze_frame(face_crop)
        else:
            emotion = None

        # 💬 VISUELLES FEEDBACK
        if "thumbsup" in gestures:
            cv2.putText(frame_full, "👍 Daumen hoch erkannt!",
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2)
        if emotion and emotion != "neutral":
            cv2.putText(frame_full, f"Emotion: {emotion}",
                        (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 255), 2)

        cv2.imshow("MOODY Detection", frame_full)

        # ESC = Abbrechen
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("🛑 Erkennung gestoppt.")

