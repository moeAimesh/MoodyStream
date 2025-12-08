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
# python -m main

import sys
from pathlib import Path
from PyQt5.QtWidgets import QApplication, QMessageBox


def main():
    # Erstelle die Qt-Anwendung früh, damit GUI-Komponenten funktionieren
    app = QApplication(sys.argv)
    
    print("🚀 Starting MoodyStream...")
    
    # Setup-Wizard IMMER zuerst ausführen (mit GUI)
    print("📋 Starte Setup-Wizard...")
    
    try:
        from gui.setup import SetupWizard
        
        # Erstelle Setup-Wizard
        setup_wizard = SetupWizard()
        result = setup_wizard.exec_()  # Zeigt als Dialog
        
        if result != setup_wizard.Accepted:
            print("❌ Setup wurde abgebrochen.")
            reply = QMessageBox.question(
                None,
                "Setup Cancelled",
                "Setup was cancelled. Do you want to continue without setup?\n\n"
                "Note: Detection may not work properly without setup.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.No:
                print("👋 Beende Anwendung...")
                return 0
            else:
                print("⚠️ Fahre ohne Setup fort...")
        else:
            print("✅ Setup erfolgreich abgeschlossen!")
    
    except ImportError as e:
        print(f"⚠️ Setup-Wizard nicht gefunden: {e}")
        print("⚠️ Fahre ohne Setup fort...")
    
    except Exception as e:
        print(f"❌ Fehler beim Setup: {e}")
        import traceback
        traceback.print_exc()
        
        reply = QMessageBox.question(
            None,
            "Setup Error",
            f"An error occurred during setup:\n{e}\n\n"
            "Do you want to continue anyway?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply == QMessageBox.No:
            return 1
    
    print("\n🎭 Starte Main Window...\n")
    
    # Importiere MainWindow
    try:
        from gui.main_window import MainWindow
    except ImportError as e:
        print(f"❌ Fehler beim Laden der GUI: {e}")
        QMessageBox.critical(
            None,
            "Import Error",
            f"Could not load the main window:\n{e}\n\n"
            "Please check your installation."
        )
        return 1
    
    # Erstelle und zeige das Hauptfenster
    try:
        window = MainWindow()
        window.show()
        print("✅ MainWindow geöffnet!")
        print("💡 Klicke auf 'Detection: OFF' um die Kamera zu starten.\n")
    except Exception as e:
        print(f"❌ Fehler beim Erstellen des Hauptfensters: {e}")
        import traceback
        traceback.print_exc()
        QMessageBox.critical(
            None,
            "Startup Error",
            f"Could not create the main window:\n{e}\n\n"
            "Please try running the setup again."
        )
        return 1
    
    # Starte die Event-Loop (hält die App am Laufen)
    print("🔄 Event-Loop läuft...")
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())