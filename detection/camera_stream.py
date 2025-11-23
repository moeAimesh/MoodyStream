"""Webcam loop with configurable single or multi-face tracking."""

from __future__ import annotations

import queue
import threading
import time
from collections import defaultdict, deque
import math
from itertools import count
from typing import Dict, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np

from detection.emotion_recognition import EmotionRecognition
from detection.face_detection import detect_faces
from detection.preprocessing import preprocess_face
from detection.facemesh_features import extract_facemesh_features
from detection.gesture_recognition import detect_gestures
from detection.kalman_tracker import Tracker as KalmanTracker
from detection.single_face_tracker import SingleFaceTracker
from detection.tracking_types import TrackInfo, random_color
from detection.virtual_cam import (
    VirtualCamError,
    VirtualCamPublisher,
    resolve_cam_dims,
    resolve_cam_fps,
)
from sounds.play_sound import play
from utils.json_manager import load_json
from utils.mediapipe_fix import apply_fix
from utils.settings import (
    FACE_DETECT_INTERVAL_MULTI,
    FACE_DETECT_INTERVAL_SINGLE,
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


def start_detection(
    camera_index: int = 0,
    *,
    show_window: bool = True,
    window_name: str = "MOODY Detection",
    virtual_cam: bool = False,
    virtual_cam_preview: bool = False,
    virtual_cam_width: Optional[int] = None,
    virtual_cam_height: Optional[int] = None,
    virtual_cam_fps: Optional[int] = None,
):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Kamera-Index {camera_index} konnte nicht geöffnet werden.")
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

    tracking_mode = (TRACKING_MODE or "single").lower()
    label_overlays = True
    single_tracker = SingleFaceTracker()
    multi_tracker = KalmanTracker(max_missing_frames=MAX_MISSING_FRAMES)
    multi_tracker.start()
    multi_cache: Dict[int, TrackInfo] = {}
    raw_track_ids = count()
    raw_track_cache: Dict[int, TrackInfo] = {}
    raw_detection_state = {"last_frame": -FACE_DETECT_INTERVAL_SINGLE}
    default_emotion = "neutral"

    frame_idx = 0
    virtual_cam_publisher: Optional[VirtualCamPublisher] = None
    preview_window_name = "MOODY Virtual Cam Preview"

    try:
        while True:
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

            if tracking_mode == "multi":
                tracks = update_multi_face_tracks(
                    frame_full, frame_idx, now, multi_tracker, multi_cache, er
                )
            elif tracking_mode == "single":
                tracks = single_tracker.update(frame_full, frame_idx, now)
            else:
                tracks = update_raw_detections(
                    frame_full,
                    frame_idx,
                    now,
                    raw_track_ids,
                    raw_track_cache,
                    raw_detection_state,
                    FACE_DETECT_INTERVAL_SINGLE,
                    default_emotion,
                )

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

            try:
                while True:
                    track_id, emotion_value, lag_ms = emotion_results.get_nowait()
                    if tracking_mode == "none" and track_id not in tracks and track_id in raw_track_cache:
                        raw_track_cache[track_id].emotion = emotion_value
                    if track_id in tracks:
                        tracks[track_id].emotion = emotion_value
                    EMOTION_LAG_MS.append(lag_ms)
            except queue.Empty:
                pass

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

            key = -1
            if show_window:
                cv2.imshow(window_name, frame_full)
                key = cv2.waitKey(1) & 0xFF
            elif virtual_cam_preview:
                cv2.imshow(preview_window_name, frame_full)
                key = cv2.waitKey(1) & 0xFF

            if virtual_cam_publisher:
                virtual_cam_publisher.send(frame_full)

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
        print("🛑 Erkennung gestoppt.")
        visualise_avg_fps()


def update_multi_face_tracks(
    frame,
    frame_idx,
    now,
    tracker: KalmanTracker,
    cache: Dict[int, TrackInfo],
    er: EmotionRecognition | None = None,
):
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
    previous_ids = set(cache.keys())
    for tid, bbox in zip(result["trackIds"], result["tracks"]):
        bbox = tuple(int(v) for v in bbox)
        info = cache.get(tid)
        if info is None:
            info = TrackInfo(track_id=tid, bbox=bbox, color=random_color(), last_seen=now)
        else:
            info.bbox = bbox
            info.last_seen = now
        updated[tid] = info

    removed_ids = previous_ids - set(updated.keys())
    if er:
        for rid in removed_ids:
            er.drop_track_state(rid)

    cache.clear()
    cache.update(updated)
    return updated


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
