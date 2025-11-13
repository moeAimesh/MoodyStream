"""Webcam loop with configurable single or multi-face tracking."""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from itertools import count
from typing import Dict

import cv2
import matplotlib.pyplot as plt
import numpy as np

from detection.emotion_recognition import EmotionRecognition
from detection.face_detection import detect_faces
from detection.gesture_recognition import detect_gestures
from detection.kalman_tracker import Tracker as KalmanTracker
from detection.single_face_tracker import SingleFaceTracker
from detection.tracking_types import TrackInfo, random_color
from sounds.play_sound import play
from utils.json_manager import load_json
from utils.mediapipe_fix import apply_fix
from utils.settings import (
    FACE_DETECT_INTERVAL_MULTI,
    MAX_MISSING_FRAMES,
    SOUND_MAP_PATH,
    TRACKING_MODE,
)

apply_fix()

FPS_BEFORE_GESTURE: list[float] = []
FPS_BEFORE: list[float] = []
FPS_AFTER: list[float] = []
EMOTION_LAG_MS: deque[float] = deque(maxlen=500)

EMOTION_ENQUEUE_INTERVAL = 0.35


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


def extract_face(frame, bbox):
    x, y, w, h = bbox
    h_frame, w_frame, _ = frame.shape
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w_frame, x + w)
    y2 = min(h_frame, y + h)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)


def start_detection():
    cap = cv2.VideoCapture(0)
    thumb_start_time = None
    sound_played = False
    sounds = load_json(SOUND_MAP_PATH)

    er = EmotionRecognition(threshold=10)
    print("🎥 Kamera gestartet – Gesten- und Emotionserkennung aktiv!")

    emotion_jobs: "queue.Queue[tuple]" = queue.Queue(maxsize=8)
    emotion_results: "queue.Queue[tuple]" = queue.Queue()
    worker_stop = threading.Event()
    job_ids = count()

    def emotion_worker():
        while not worker_stop.is_set():
            try:
                frame_id, track_id, face_crop, captured_at = emotion_jobs.get(timeout=0.1)
            except queue.Empty:
                continue
            if frame_id is None:
                emotion_jobs.task_done()
                break
            emotion = er.analyze_frame(face_crop, track_id=track_id)
            lag_ms = (time.perf_counter() - captured_at) * 1000
            emotion_results.put((track_id, emotion, lag_ms))
            emotion_jobs.task_done()

    worker_thread = threading.Thread(target=emotion_worker, daemon=True)
    worker_thread.start()

    tracking_mode = (TRACKING_MODE or "single").lower()
    single_tracker = SingleFaceTracker()
    multi_tracker = KalmanTracker(max_missing_frames=MAX_MISSING_FRAMES)
    multi_tracker.start()
    multi_cache: Dict[int, TrackInfo] = {}

    frame_idx = 0

    try:
        while True:
            loop_start = time.perf_counter()
            ret, frame_full = cap.read()
            if not ret:
                break
            now = time.time()
            frame_idx += 1

            if tracking_mode == "multi":
                tracks = update_multi_face_tracks(
                    frame_full, frame_idx, now, multi_tracker, multi_cache
                )
            else:
                tracks = single_tracker.update(frame_full, frame_idx, now)

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

            gestures = detect_gestures(frame_full)
            current_time = time.time()
            if "thumbsup" in gestures:
                if thumb_start_time is None:
                    thumb_start_time = current_time
                    sound_played = False
                elif (current_time - thumb_start_time) >= 1 and not sound_played:
                    sound_path = sounds.get("thumbsup") or sounds.get("ok")
                    if sound_path:
                        play(sound_path)
                    sound_played = True
            else:
                thumb_start_time = None
                sound_played = False

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

            for track in tracks.values():
                if now - track.last_enqueued < EMOTION_ENQUEUE_INTERVAL:
                    continue
                crop = extract_face(frame_full, track.bbox)
                if crop is None:
                    continue
                capture_ts = time.perf_counter()
                job = (next(job_ids), track.track_id, crop, capture_ts)
                try:
                    emotion_jobs.put_nowait(job)
                    track.last_enqueued = now
                except queue.Full:
                    pass

            try:
                while True:
                    track_id, emotion_value, lag_ms = emotion_results.get_nowait()
                    if track_id in tracks:
                        tracks[track_id].emotion = emotion_value
                    EMOTION_LAG_MS.append(lag_ms)
            except queue.Empty:
                pass

            for track in tracks.values():
                x, y, w, h = track.bbox
                cv2.rectangle(frame_full, (x, y), (x + w, y + h), track.color, 2)
                label = track.emotion or "neutral"
                cv2.putText(
                    frame_full,
                    f"Face {track.track_id}: {label}",
                    (x, max(20, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    track.color,
                    2,
                )

            if "thumbsup" in gestures:
                cv2.putText(
                    frame_full,
                    "👍 Daumen hoch erkannt!",
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

            cv2.imshow("MOODY Detection", frame_full)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        worker_stop.set()
        try:
            emotion_jobs.put_nowait((None, None, None, 0.0))
        except queue.Full:
            pass
        worker_thread.join(timeout=1)
        cap.release()
        cv2.destroyAllWindows()
        print("🛑 Erkennung gestoppt.")
        visualise_avg_fps()


def update_multi_face_tracks(frame, frame_idx, now, tracker: KalmanTracker, cache: Dict[int, TrackInfo]):
    detections = []
    if frame_idx % max(1, FACE_DETECT_INTERVAL_MULTI) == 0:
        detections = detect_faces(frame)

    det_array = (
        np.array(detections, dtype=np.float32)
        if detections
        else np.empty((0, 4), dtype=np.float32)
    )
    classes = np.zeros((len(det_array),), dtype=np.int32)
    result = tracker.step({"detections": det_array, "classes": classes})

    updated: Dict[int, TrackInfo] = {}
    for tid, bbox in zip(result["trackIds"], result["tracks"]):
        bbox = tuple(int(v) for v in bbox)
        info = cache.get(tid)
        if info is None:
            info = TrackInfo(track_id=tid, bbox=bbox, color=random_color(), last_seen=now)
        else:
            info.bbox = bbox
            info.last_seen = now
        updated[tid] = info

    cache.clear()
    cache.update(updated)
    return updated
