from __future__ import annotations
 
import time
from typing import Optional
 
import cv2
 
from detection.emotion_recognition import EmotionRecognition
from detection.face_detection import detect_faces
from detection.preprocessing import extract_face
from detection.virtual_cam import (
    VirtualCamError,
    VirtualCamPublisher,
    resolve_cam_dims,
    resolve_cam_fps,
)
from detection.gesture_recognition import detect_gestures
from sounds.play_sound import play
from utils.mediapipe_fix import apply_fix
 
apply_fix()
 
 
def start_detection(
    camera_index: int = 2,
    *,
    virtual_cam: bool = False,
    virtual_cam_preview: bool = False,
    virtual_cam_width: Optional[int] = None,
    virtual_cam_height: Optional[int] = None,
    virtual_cam_fps: Optional[int] = None,
):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Kamera-Index {camera_index} konnte nicht geöffnet werden.")
 
    publisher: Optional[VirtualCamPublisher] = None
    thumb_start: Optional[float] = None
    sound_played = False
    er = EmotionRecognition(threshold=10)
    preview_window = "MOODY Virtual Cam Preview"
 
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if virtual_cam and publisher is None:
                width, height = resolve_cam_dims(frame, virtual_cam_width, virtual_cam_height)
                fps = resolve_cam_fps(cap, virtual_cam_fps)
                try:
                    publisher = VirtualCamPublisher(width=width, height=height, fps=fps)
                except VirtualCamError as exc:
                    print(f"[virtual-cam] deaktiviert: {exc}")
                    virtual_cam = False
 
            # Gesten (brauchen das ganze Frame)
            gestures = detect_gestures(frame)
            if "thumbsup" in gestures:
                if thumb_start is None:
                    thumb_start = time.time()
                    sound_played = False
                elif time.time() - thumb_start >= 1.0 and not sound_played:
                    play("sounds/sound_cache/Rammus_Select.mp3")
                    sound_played = True
            else:
                thumb_start = None
                sound_played = False
 
            # Gesichter finden & Emotion berechnen
            boxes = detect_faces(frame)
            emotion = None
            if boxes:
                face_crop = extract_face(frame, boxes[0])
                if face_crop is not None:
                    emotion = er.analyze_frame(face_crop)
 
            if "thumbsup" in gestures:
                cv2.putText(frame, "👍 Daumen hoch erkannt!", (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if emotion:
                cv2.putText(frame, f"Emotion: {emotion}", (40, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
 
            cv2.imshow("MOODY Detection", frame)
            key = cv2.waitKey(1) & 0xFF
 
            if virtual_cam_preview:
                cv2.imshow(preview_window, frame)
            if publisher:
                publisher.send(frame)
            if key == 27:  # ESC
                break
    finally:
        cap.release()
        if publisher:
            publisher.close()
        cv2.destroyAllWindows()