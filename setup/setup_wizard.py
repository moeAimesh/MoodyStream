"""Aufgabe: Geführter Einrichtungs-Ablauf in Schritten (GUI oder simple Popups):

Profil wählen/neu anlegen,

Gesichts-Baseline erfassen,

Sounds je Verhalten/Emotion zuordnen,

alles in Profil-JSON speichern.

Eingaben: Kamera-Frames, Sound-URLs (vom integrierten Browser).

Ausgaben: profiles/<name>.json (vollständiges Nutzer-Profil).

Wichtig: Wizard führt dich weiter, bis alles Notwendige vorhanden ist."""

"""
Aufgabe: Geführter Einrichtungs-Ablauf in Schritten (GUI oder simple Popups):

Profil wählen/neu anlegen,
Gesichts-Baseline erfassen,
Sounds je Verhalten/Emotion zuordnen,
alles in Profil-JSON speichern.
"""
#python -m setup.setup_wizard

from .sound_setup import run_sound_setup
from .face_setup import RestFaceCalibrator
from utils.settings import FACE_SETUP_ENABLED, REST_FACE_MODEL_PATH


def _should_capture_faces() -> bool:
    return FACE_SETUP_ENABLED


def run_rest_face_setup(user="default", force_record=None):
    """Führt die neue Rest-Face-Kalibrierung aus."""
    print("📷 Starte Rest-Face-Kalibrierung ...")
    model_path = REST_FACE_MODEL_PATH

    calibrator = RestFaceCalibrator(model_path=model_path)
    if force_record is None:
        force_record = _should_capture_faces()

    if not force_record:
        if not calibrator.load_snapshot():
            print("⚠️ Kein Snapshot vorhanden – starte neue Aufnahme.")
            force_record = True

    if force_record:
        success = calibrator.record_emotions(duration=12, analyze_every=5)
        if not success:
            print("✖️ Keine Daten erfasst – bitte erneut versuchen.")
            return False
    elif not calibrator.profiles:
        print("✖️ Keine gespeicherten Profile gefunden.")
        return False

    calibrator.train()
    calibrator.save_model()
    calibrator.visualize_space()
    print("✅ Rest-Face-Modell erfolgreich erstellt.")
    return True


def main():
    print("🚀 Starting Moody Setup Wizard...")

    if not run_rest_face_setup(user="default"):
        print("✖️ Rest-Face-Setup abgebrochen.")
        return False

    if not run_sound_setup(user="default"):
        print("✖️ Sound-Setup abgebrochen.")
        return False

    print("✅ Setup vollständig abgeschlossen.")
    return True


if __name__ == "__main__":
    main()
