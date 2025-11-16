"""Aufgabe: Startpunkt deiner App. Orchestriert den Ablauf:

Setup-Assistent starten (Gesicht & Sounds),

danach Live-Erkennung starten.

Eingaben: keine.

Ausgaben: startet Prozesse/Threads.

Kernlogik (Pseudocode):

if not setup_wizard():
    exit()
start_detection()


Fehlerfälle betrachten: keine Kamera; fehlende Profile/Sounds → sauber melden und zum Setup zurückführen."""

from detection.camera_stream import start_detection
from setup.setup_wizard import main as run_setup_wizard


def main():
    print("🚀 Starting Moody Setup Wizard...")

    # Setup einmal starten (Gesicht + Sounds)
    #setup_success = run_setup_wizard()
    #if not setup_success:
    #    print("❌ Setup abgebrochen oder fehlgeschlagen.")
    #    return
    

    print("\n✅ Setup abgeschlossen! Starte Hauptprogramm...\n")

    # Starte Kameraerkennung
    start_detection()


if __name__ == "__main__":
    main()