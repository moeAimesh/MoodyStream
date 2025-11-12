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
from utils.settings import REST_FACE_MODEL_PATH


def run_rest_face_setup(user="default"):
    """
    Führt die neue Rest-Face-Kalibrierung aus.
    """
    print("📷 Starte Rest-Face-Kalibrierung ...")
    model_path = REST_FACE_MODEL_PATH

    calibrator = RestFaceCalibrator(model_path=model_path)
    success = calibrator.record_rest_face(duration=20, analyze_every=5)

    if not success:
        print("❌ Keine Daten erfasst – bitte erneut versuchen.")
        return False

    calibrator.train()
    calibrator.save_model()
    calibrator.visualize_space()
    print("✅ Rest-Face-Modell erfolgreich erstellt.")
    return True


def main():
    print("🚀 Starting Moody Setup Wizard...")
    

    # 🧠 Rest-Face-Kalibrierung (neuer Ansatz)
    if not run_rest_face_setup(user="default"):
        print("❌ Rest-Face-Setup abgebrochen.")
        return False

    # 🔊 Sound-Zuordnung (alter Sound-Setup-Schritt)
    if not run_sound_setup(user="default"):
        print("❌ Sound-Setup abgebrochen.")
        return False

    print("✅ Setup vollständig abgeschlossen.")
    return True



if __name__ == "__main__":
    main()
