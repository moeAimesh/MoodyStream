"""Webcam loop with simple face detection + ID matching."""

from __future__ import annotations

import queue
import threading
import time
from collections import defaultdict, deque
import math
from itertools import count
from typing import Dict, Optional, Callable

import cv2
import matplotlib.pyplot as plt
import numpy as np

from detection.emotion_recognition import EmotionRecognition
from detection.face_detection import detect_faces
from detection.preprocessing import preprocess_face
from detection.facemesh_features import extract_facemesh_features
from detection.gesture_recognition import detect_gestures
from detection.tracking_types import TrackInfo, random_color
from detection.virtual_cam import (
    VirtualCamError,
    VirtualCamPublisher,
    resolve_cam_dims,
    resolve_cam_fps,
)
from detection.gesture_mapper import get_sound_for_gestures
from detection.emotion_mapper import get_sound_for_emotions
from sounds.play_sound import play
from utils.json_manager import load_json
from utils.mediapipe_fix import apply_fix
from utils.settings import (
    SETUP_CONFIG_PATH,
    TRIGGER_TIME_EMOTION_SEC,
    TRIGGER_TIME_GESTURE_SEC,
)
apply_fix()

FPS_BEFORE_GESTURE: list[float] = []
FPS_BEFORE: list[float] = []
FPS_AFTER: list[float] = []
EMOTION_LAG_MS: deque[float] = deque(maxlen=500)

EMOTION_ENQUEUE_INTERVAL = 0.35
# Run a face detection every N frames to refresh bounding boxes/IDs
FACE_DETECT_INTERVAL_FRAMES = 5


def visualise_avg_fps():
    if FPS_BEFORE:
        plt.plot(FPS_BEFORE, label="FPS_before")
    if FPS_BEFORE_GESTURE:
        plt.plot(FPS_BEFORE_GESTURE, label="FPS_before_gesture")
    if FPS_AFTER:
        plt.plot(FPS_AFTER, label="FPS_after")
    plt.xlabel("Frame")
    plt.ylabel("FPS")
    plt.legend()
    plt.title("FPS Vergleich: vor/nach Emotionserkennung")
    plt.show()

    avg_before = sum(FPS_BEFORE) / len(FPS_BEFORE) if FPS_BEFORE else 0
    avg_before_gesture = (
        sum(FPS_BEFORE_GESTURE) / len(FPS_BEFORE_GESTURE) if FPS_BEFORE_GESTURE else 0
    )
    avg_after = sum(FPS_AFTER) / len(FPS_AFTER) if FPS_AFTER else 0
    avg_lag = sum(EMOTION_LAG_MS) / len(EMOTION_LAG_MS) if EMOTION_LAG_MS else 0

    print("\n==== Durchschnittliche FPS/Lag ====")
    print(f"FPS_BEFORE_GESTURE: {avg_before_gesture:.1f}")
    print(f"FPS_BEFORE:         {avg_before:.1f}")
    print(f"FPS_AFTER:          {avg_after:.1f}")
    print(f"Emotion lag:        {avg_lag:.1f} ms")
    print("===================================\n")


def start_detection(
    camera_index: int = 0,
    *,
    show_window: bool = True,
    window_name: str = "MOODY Detection",
    virtual_cam: bool = True,
    virtual_cam_preview: bool = False,
    virtual_cam_width: Optional[int] = None,
    virtual_cam_height: Optional[int] = None,
    virtual_cam_fps: Optional[int] = None,
    frame_callback: Optional[Callable[[np.ndarray], None]] = None,
    stop_event: Optional[threading.Event] = None,
    show_fps_plot: bool = False,  # only activate when necessary otherwise gui will crash
):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Kamera-Index {camera_index} konnte nicht geÃ¶ffnet werden.")
    gesture_start_time = None
    gesture_sound_played = False
    config = load_json(SETUP_CONFIG_PATH)
    trigger_cfg = config.get("trigger_times", {}) if isinstance(config, dict) else {}
    gesture_trigger_secs = float(trigger_cfg.get("gesture", TRIGGER_TIME_GESTURE_SEC))
    emotion_trigger_secs = float(trigger_cfg.get("emotion", TRIGGER_TIME_EMOTION_SEC))
    gesture_trigger_secs = max(0.0, gesture_trigger_secs)
    emotion_trigger_secs = max(0.0, emotion_trigger_secs)

    er = EmotionRecognition(threshold=10)
    print("Kamera gestartet Gesten- und Emotionserkennung aktiv!")

    emotion_jobs: "queue.Queue[tuple]" = queue.Queue(maxsize=8)
    emotion_results: "queue.Queue[tuple]" = queue.Queue()
    worker_stop = threading.Event()
    job_ids = count()

    def emotion_worker():
        while not worker_stop.is_set():
            try:
                frame_id, track_id, face_crop, features, captured_at = emotion_jobs.get(timeout=0.1)
            except queue.Empty:
                continue
            if frame_id is None:
                emotion_jobs.task_done()
                break
            emotion = er.analyze_frame(face_crop, features=features, track_id=track_id)
            lag_ms = (time.perf_counter() - captured_at) * 1000
            emotion_results.put((track_id, emotion, lag_ms))
            emotion_jobs.task_done()

    worker_thread = threading.Thread(target=emotion_worker, daemon=True)
    worker_thread.start()

    label_overlays = True
    raw_track_ids = count()
    raw_track_cache: Dict[int, TrackInfo] = {}
    raw_detection_state = {"last_frame": -FACE_DETECT_INTERVAL_FRAMES}
    default_emotion = "neutral"
    
    #merkt sich zuletzt gespielte Emotion pro Track (damit wir nicht spammen)
    last_emotions: Dict[int, Optional[str]] = defaultdict(lambda: None)
    emotion_stable: Dict[int, tuple[Optional[str], float]] = {}

    frame_idx = 0
    virtual_cam_publisher: Optional[VirtualCamPublisher] = None
    preview_window_name = "MOODY Virtual Cam Preview"

    try:
        while True:
            # check for stop event
            if stop_event and stop_event.is_set():
                print("🛑 Stop event detected at loop start - breaking")
                break
            
            loop_start = time.perf_counter()
            ret, frame_full = cap.read()
            if not ret:
                break
            now = time.time()
            frame_idx += 1

            if virtual_cam and virtual_cam_publisher is None:
                width, height = resolve_cam_dims(frame_full, virtual_cam_width, virtual_cam_height)
                fps = resolve_cam_fps(cap, virtual_cam_fps)
                try:
                    virtual_cam_publisher = VirtualCamPublisher(width=width, height=height, fps=fps)
                except VirtualCamError as exc:
                    print(f"[virtual-cam] deaktiviert: {exc}")
                    virtual_cam = False

            tracks = update_raw_detections(
                frame_full,
                frame_idx,
                now,
                raw_track_ids,
                raw_track_cache,
                raw_detection_state,
                FACE_DETECT_INTERVAL_FRAMES,
                default_emotion,
            )
            
            # check for stop event
            if stop_event and stop_event.is_set():
                print("🛑 Stop event detected after face detection - breaking")
                break

            elapsed_before_gesture = (time.perf_counter() - loop_start) * 1000
            fps_before_gesture = 1000 / elapsed_before_gesture if elapsed_before_gesture else 0
            FPS_BEFORE_GESTURE.append(fps_before_gesture)
            cv2.putText(
                frame_full,
                f"FPS_before_gesture: {fps_before_gesture:.1f}",
                (15, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
            )
            
            #GESTENERKENNUNG

            gestures = detect_gestures(frame_full)
            current_time = time.time()
            if gestures:
                if gesture_start_time is None:
                    gesture_start_time = current_time
                    gesture_sound_played = False
                elif (current_time - gesture_start_time) >= gesture_trigger_secs and not gesture_sound_played:
                    g_key, g_path = get_sound_for_gestures(gestures)
                    if g_path:
                        play(g_path)
                    gesture_sound_played = True
            else:
                gesture_start_time = None
                gesture_sound_played = False
            
            # check for stop event
            if stop_event and stop_event.is_set():
                print("🛑 Stop event detected after gesture detection - breaking")
                break

            elapsed_before = (time.perf_counter() - loop_start) * 1000
            fps_before = 1000 / elapsed_before if elapsed_before else 0
            FPS_BEFORE.append(fps_before)
            cv2.putText(
                frame_full,
                f"FPS_before: {fps_before:.1f}",
                (15, 210),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
            )
            
            #Emotion-Jobs in Queue schieben
            for track in tracks.values():
                if now - track.last_enqueued < EMOTION_ENQUEUE_INTERVAL:
                    continue
                crop = preprocess_face(frame_full, track.bbox)
                if crop is None:
                    continue
                features = extract_facemesh_features(frame_full, track.bbox)
                if features is None:
                    continue
                capture_ts = time.perf_counter()
                job = (next(job_ids), track.track_id, crop, features, capture_ts)
                try:
                    emotion_jobs.put_nowait(job)
                    track.last_enqueued = now
                except queue.Full:
                    pass
                
            #Emotion-Ergebnisse aus Queue holen
            try:
                while True:
                    track_id, emotion_value, lag_ms = emotion_results.get_nowait()
                    if track_id in tracks:
                        tracks[track_id].emotion = emotion_value
                    elif track_id in raw_track_cache:
                        raw_track_cache[track_id].emotion = emotion_value
                    EMOTION_LAG_MS.append(lag_ms)
            except queue.Empty:
                pass

            #Emotion-Sounds ?ber Mapper (pro Track, bei ?nderung & != neutral, nach Haltedauer)
            for track in tracks.values():
                current_emotion = track.emotion or default_emotion
                last = last_emotions[track.track_id]
                prev_label, start_ts = emotion_stable.get(track.track_id, (current_emotion, now))
                if current_emotion != prev_label:
                    start_ts = now
                    emotion_stable[track.track_id] = (current_emotion, start_ts)
                else:
                    emotion_stable[track.track_id] = (prev_label, start_ts)

                stable_enough = (now - start_ts) >= emotion_trigger_secs
                should_play = current_emotion != "neutral" and current_emotion != last and stable_enough

                if should_play:
                    s_key, s_path = get_sound_for_emotions([current_emotion])
                    if s_path:
                        play(s_path)
                    last_emotions[track.track_id] = current_emotion
                elif last is None:
                    #einmal setzen, damit sp?tere Vergleiche stabil sind
                    last_emotions[track.track_id] = current_emotion
            # remove stale entries for disappeared tracks
            active_ids = {t.track_id for t in tracks.values()}
            for tid in list(emotion_stable.keys()):
                if tid not in active_ids:
                    emotion_stable.pop(tid, None)

            # check for stop event 
            if stop_event and stop_event.is_set():
                print("🛑 Stop event detected after emotion sounds - breaking")
                break
            
            for track in tracks.values():
                x, y, w, h = track.bbox
                cv2.rectangle(frame_full, (x, y), (x + w, y + h), track.color, 2)
                if label_overlays:
                    label = track.emotion or "neutral"
                    cv2.putText(
                        frame_full,
                        label,
                        (x, max(20, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        track.color,
                        2,
                    )

            if "thumbsup" in gestures:
                cv2.putText(
                    frame_full,
                    "ðŸ‘ Daumen hoch erkannt!",
                    (15, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

            elapsed_after = (time.perf_counter() - loop_start) * 1000
            fps_after = 1000 / elapsed_after if elapsed_after else 0
            FPS_AFTER.append(fps_after)
            cv2.putText(
                frame_full,
                f"FPS_after: {fps_after:.1f}",
                (15, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
            )

            # callback for gui
            if frame_callback is not None:
                frame_callback(frame_full)

            key = -1
            if show_window:
                cv2.imshow(window_name, frame_full)
                key = cv2.waitKey(1) & 0xFF
            elif virtual_cam_preview:
                cv2.imshow(preview_window_name, frame_full)
                key = cv2.waitKey(1) & 0xFF

            if virtual_cam_publisher:
                virtual_cam_publisher.send(frame_full)

            # stop event to stop detection
            if stop_event and stop_event.is_set():
                break

            if key == 27:
                break
    finally:
        worker_stop.set()
        try:
            emotion_jobs.put_nowait((None, None, None, None, 0.0))
        except queue.Full:
            pass
        worker_thread.join(timeout=1)
        cap.release()
        if virtual_cam_publisher:
            virtual_cam_publisher.close()
        cv2.destroyAllWindows()
        print("ðŸ›‘ Erkennung gestoppt.")
        
        # plot only when necessary otherwise gui will crash!!!!!!
        if show_fps_plot:
            visualise_avg_fps()


def update_raw_detections(
    frame,
    frame_idx,
    now,
    id_counter,
    cache: Dict[int, TrackInfo],
    state,
    interval,
    default_emotion,
    max_distance=80.0,
):
    """Lightweight centroid matching to keep IDs stable without trackers."""
    if frame_idx - state["last_frame"] < interval:
        for info in cache.values():
            info.last_seen = now
        if cache:
            return cache
        return {}

    detections = detect_faces(frame)
    updated: Dict[int, TrackInfo] = {}

    def centroid(box):
        x, y, w, h = box
        return x + w / 2.0, y + h / 2.0

    for bbox in detections:
        cx, cy = centroid(bbox)
        best_id = None
        best_dist = max_distance
        for tid, info in cache.items():
            if tid in updated:
                continue
            px, py = centroid(info.bbox)
            dist = math.hypot(cx - px, cy - py)
            if dist < best_dist:
                best_dist = dist
                best_id = tid

        if best_id is not None:
            info = cache[best_id]
            info.bbox = tuple(int(v) for v in bbox)
            info.last_seen = now
            updated[best_id] = info
        else:
                    track_id = next(id_counter)
                    updated[track_id] = TrackInfo(
                        track_id=track_id,
                        bbox=tuple(int(v) for v in bbox),
                        color=random_color(),
                        last_seen=now,
                        emotion=default_emotion,
                    )

    cache.clear()
    cache.update(updated)
    state["last_frame"] = frame_idx
    return cache
