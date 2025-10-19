import cv2
from deepface import DeepFace
import json
import time
import numpy as np
import os
from utils.json_manager import save_json, load_json


# -----------------------------------------------------------
# 🔹 Einzelne Kalibrierung
# -----------------------------------------------------------
def single_calibration(duration=5, text="Kalibriere...", analyze_every=5):
    """Eine einzelne Kalibrierung"""
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    emotion_sums = {}
    frame_count = 0
    analyze_count = 0
    start = time.time()

    while time.time() - start < duration:
        ret, frame = cam.read()
        if not ret:
            continue

        # Overlay anzeigen
        cv2.putText(frame, text, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow("calibration", frame)

        # ESC = Abbrechen
        if cv2.waitKey(1) & 0xFF == 27:
            break

        # Nur jedes n-te Frame analysieren
        if frame_count % analyze_every == 0:
            try:
                result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                emotions = result[0]['emotion']
                for k, v in emotions.items():
                    emotion_sums[k] = emotion_sums.get(k, 0) + v
                analyze_count += 1
            except Exception as e:
                print("⚠️ Error while analyzing:", e)

        frame_count += 1

    cam.release()
    cv2.destroyAllWindows()

    if analyze_count == 0:
        print("⚠️ no frames analyzed successfully. maybe no face detected?")
        return None

    # Durchschnitt berechnen
    return {k: float(v / analyze_count) for k, v in emotion_sums.items()}


# -----------------------------------------------------------
# 🔹 Mehrfache Kalibrierung (verschiedene Blickwinkel)
# -----------------------------------------------------------
def multi_perspective_calibration(rounds=5, duration=4, user="default"):
    print("📷 Please look at the camera with a REST FACE from different angles")
    all_results = []

    for i in range(rounds):
        input(f"\n➡️ Position {i+1}/{rounds}: Press ENTER when ready...")
        result = single_calibration(duration, text=f"Calibrating ({i+1}/{rounds})...", analyze_every=5)
        if result:
            all_results.append(result)

    if not all_results:
        print("❌ no valid calibration data collected.")
        return None

    combined = {}
    for emotion in all_results[0].keys():
        combined[emotion] = float(np.mean([r[emotion] for r in all_results]))

    # Ergebnis speichern
    profile_dir = os.path.join("setup", "profiles")
    os.makedirs(profile_dir, exist_ok=True)
    path = os.path.join(profile_dir, f"{user}_face_baseline.json")

    with open(path, "w") as f:
        json.dump(combined, f, indent=2)

    print(f"\n✅ done -> saved under {path}")
    return combined


# -----------------------------------------------------------
# 🔹 Setup-Wrapper
# -----------------------------------------------------------
def run_face_setup(user="default"):
    """Wird vom Setup-Wizard aufgerufen.
       Führt Kalibrierung nur aus, wenn kein Eintrag vorhanden ist."""
    
    config_path = "setup/setup_config.json"

    # Prüfen, ob Datei existiert und bereits Gesichtsdaten enthält
    if os.path.exists(config_path):
        try:
            data = load_json(config_path)
            if "faces" in data and isinstance(data["faces"], dict) and len(data["faces"]) > 0:
                print("ℹ️ Face baseline already exists – skipping calibration.")
                return True
        except Exception as e:
            print(f"⚠️ Warning: Could not read {config_path} ({e}), will redo calibration.")

    # Wenn keine Daten vorhanden -> Kalibrierung starten
    result = multi_perspective_calibration(rounds=5, duration=4, user=user)
    if result:
        save_json(config_path, "faces", result)
        print("✅ Rest-face baseline saved to setup_config.json")
        return True

    return False
