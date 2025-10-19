"""Aufgabe: Geführter Einrichtungs-Ablauf in Schritten (GUI oder simple Popups):

Profil wählen/neu anlegen,

Gesichts-Baseline erfassen,

Sounds je Verhalten/Emotion zuordnen,

alles in Profil-JSON speichern.

Eingaben: Kamera-Frames, Sound-URLs (vom integrierten Browser).

Ausgaben: profiles/<name>.json (vollständiges Nutzer-Profil).

Wichtig: Wizard führt dich weiter, bis alles Notwendige vorhanden ist."""




from setup.face_setup import run_face_setup
from setup.sound_setup import run_sound_setup

def main():
    print("🚀 Starting Moody Setup Wizard...")

    if not run_face_setup(user="default"):
        print("❌ Face setup aborted.")
        return False

    if not run_sound_setup(user="default"):
        print("❌ Sound setup aborted.")
        return False

    print("✅ Setup completed.")
    return True
