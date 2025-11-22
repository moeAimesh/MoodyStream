"""
Zentrale Loop über die Webcam.

pro Frame:

Gesichter/Emotionen analysieren
Gesten erkennen
Entscheidung treffen (Mapper)
Passenden Sound abspielen
Live-Feedback in GUI melden
Wichtig:
Nicht blockieren → Threads für Kamera, Audio, GUI.
"""

import cv2, time
from detection.gesture_recognition import detect_gestures
from detection.emotion_recognition import EmotionRecognition
from sounds.play_sound import play
from utils.mediapipe_fix import apply_fix
from detection.crop_face import crop_face
import matplotlib.pyplot as plt
from detection.emotion_mapper import map_emotion_to_sound
from detection.gesture_mapper import map_gesture_to_sound

apply_fix()

FPS_BEFORE_GESTURE = []
FPS_BEFORE = []
FPS_AFTER = []

def visualise_avg_fps():
    # Plot
    plt.plot(FPS_BEFORE, label="FPS_before")
    plt.plot(FPS_BEFORE_GESTURE, label="FPS_BEFORE_GESTURE")
    plt.plot(FPS_AFTER, label="FPS_after")
    plt.xlabel("Frame")
    plt.ylabel("FPS")
    plt.legend()
    plt.title("FPS Vergleich: vor und nach Emotionserkennung")
    plt.show()

# Averages
avg_before = sum(FPS_BEFORE) / len(FPS_BEFORE) if FPS_BEFORE else 0
avg_before_gesture = sum(FPS_BEFORE_GESTURE) / len(FPS_BEFORE_GESTURE) if FPS_BEFORE_GESTURE else 0
avg_after = sum(FPS_AFTER) / len(FPS_AFTER) if FPS_AFTER else 0

print("\n==== Durchschnittliche FPS ====")
print(f"FPS_BEFORE_GESTURE: {avg_before_gesture:.1f}")
print(f"FPS_BEFORE:         {avg_before:.1f}")
print(f"FPS_AFTER:          {avg_after:.1f}")
print("================================\n")
def start_detection():
    cap = cv2.VideoCapture(0)
    thumb_start_time = None
    sound_played = False

    er = EmotionRecognition(threshold=10)
    last_emotion = None  # merkt sich die letzte abgespielte Emotion
    print("🎥 Kamera gestartet – Gesten- und Emotionserkennung aktiv!")
    
    while True:
        loop_start = time.perf_counter()
        ret, frame_full = cap.read()
        if not ret:
            break

        # 🧩 Cropping nur für Emotion
        face_crop = None
        face = crop_face(frame_full, draw_box=True)
        if face is not None :
            rgbface= cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face_crop= cv2.resize(rgbface, (224, 224), interpolation=cv2.INTER_AREA)
        else:
            print("no face could have been cropped")
        elapsed_befor_gesture = (time.perf_counter() - loop_start) * 1000
        fps_befor_gesture = 1000 / elapsed_befor_gesture if elapsed_befor_gesture else 0

        FPS_BEFORE_GESTURE.append(fps_befor_gesture)
        cv2.putText(frame_full, f"FPS_befor_gesture: {fps_befor_gesture:.1f}",
            (150, 200), cv2.FONT_HERSHEY_SIMPLEX,
            1, (255, 0, 0), 2)
        
        # ✋ GESTENERKENNUNG – nutzt das ganze Frame!
        gestures = detect_gestures(frame_full)
        current_time = time.time()

        # 🖐️ Beispiel: Geste (z. B. "thumbsup") → Sound über Mapper
        if gestures:
            if thumb_start_time is None:
                thumb_start_time = current_time
                sound_played = False
            elif (current_time - thumb_start_time) >= 1 and not sound_played:
                # Mapper entscheidet, welche Geste welchen Sound bekommt
                g_key, g_path = map_gesture_to_sound(gestures)
                if g_path:
                    play(g_path)
                    sound_played = True
        else:
            thumb_start_time = None
            sound_played = False


        elapsed = (time.perf_counter() - loop_start) * 1000
        fps_before = 1000 / elapsed if elapsed else 0
        FPS_BEFORE.append(fps_before)
        cv2.putText(frame_full, f"FPS_before: {fps_before:.1f}",
                    (150, 150), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 0, 0), 2)
        
        
        # 🧠 EMOTION-ERKENNUNG – nutzt nur das Gesicht!
        emotion = None
        if face_crop is not None:
            emotion = er.analyze_frame(face_crop)

        # 🎵 EMOTION → SOUND (über Mapper, nur bei Wechsel & nicht neutral)
        if emotion != last_emotion:
            if emotion and emotion != "neutral":
                sound_key, sound_path = map_emotion_to_sound([emotion])
                if sound_path:
                    play(sound_path)
            last_emotion = emotion
            
        # 💬 VISUELLES FEEDBACK
        if "thumbsup" in gestures:
            cv2.putText(frame_full, "👍 Daumen hoch erkannt!",
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2)
        if emotion and emotion != "neutral":
            cv2.putText(frame_full, f"Emotion: {emotion}",
                        (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 255), 2)
            
        elapsed_after = (time.perf_counter() - loop_start) * 1000
        fps_after = 1000 / elapsed_after if elapsed_after else 0

        FPS_AFTER.append(fps_after)
        cv2.putText(frame_full, f"FPS_after: {fps_after:.1f}",
            (150, 200), cv2.FONT_HERSHEY_SIMPLEX,
            1, (255, 0, 0), 2)

        cv2.imshow("MOODY Detection", frame_full)

        # ESC = Abbrechen
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("🛑 Erkennung gestoppt.")
    print("show current avg fps before vs after")
    visualise_avg_fps()