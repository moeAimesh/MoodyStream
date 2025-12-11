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
import sys
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import QApplication, QMessageBox

from gui.setup import show_setup_instructions
from .sound_setup import run_sound_setup
from .face_setup import RestFaceCalibrator
from utils.settings import FACE_SETUP_ENABLED, REST_FACE_MODEL_PATH



class SetupController(QObject):
    finished = pyqtSignal(bool)   # True = success, False = cancelled

    def __init__(self, parent=None):
        super().__init__(parent)

    def start(self):
        """Starts the entire setup wizard as a GUI-driven process."""

        # Run the actual setup tasks
        if not run_rest_face_setup(user="default"):
            self.finished.emit(False)
            return

        # Setup successful
        self.finished.emit(True)

def _show_repeat_popup(message: str) -> None:
    """Show a small info popup; fallback to console if GUI fails."""
    try:

        app = QApplication.instance() or QApplication([])
        QMessageBox.information(None, "Moody Setup", message)
        app.processEvents()
    except Exception:
        print(message)


def _should_capture_faces() -> bool:
    return FACE_SETUP_ENABLED


def run_rest_face_setup(user="default", force_record=None):
    """Führt die neue Rest-Face-Kalibrierung aus."""
    print("📷 Starte Rest-Face-Kalibrierung ...")
    model_path = REST_FACE_MODEL_PATH
    calibrator = RestFaceCalibrator(model_path=model_path)
    if force_record is None:
        force_record_local = _should_capture_faces()
    else:
        force_record_local = force_record

    if not force_record_local:
        if not calibrator.load_snapshot():
            print("⚠️ Kein Snapshot vorhanden – starte neue Aufnahme.")
            force_record_local = True

    if force_record_local:
        success = calibrator.record_emotions(duration=12, analyze_every=5)
        if not success:
            print("✖️ Keine Daten erfasst – bitte erneut versuchen.")
            return False
    elif not calibrator.profiles:
        print("✖️ Keine gespeicherten Profile gefunden.")
        return False

    def _run_outlier_filter():
        try:
            from filter_outlier import filter_outliers

            summary = filter_outliers(
                calibrator,
                method="lof",
                contamination=0.05,
                radius_sigma=2.5,
            )
            removed_total = sum(item["removed"] for item in summary.values())
            print("🧹 Ausreißer-Check:")
            for emotion, stats in summary.items():
                print(
                    f"  {emotion:>8s}: behalten {stats['kept']:3d}/{stats['original']:3d} "
                    f"(entfernt {stats['removed']:2d})"
                )
            if removed_total:
                calibrator._save_snapshot()
                print(f"→ Gesamt entfernt: {removed_total} Ausreißer.")
            else:
                print("→ Keine Ausreißer gefunden.")
            return summary
        except Exception as exc:
            print(f"⚠️ Ausreißer-Prüfung übersprungen: {exc}")
            return None

    while True:
        summary = _run_outlier_filter()
        if not summary:
            break

        retry_emotions = [emo for emo, stats in summary.items() if stats.get("removed", 0) > 3]
        if not retry_emotions:
            break

        print("⚠️ Folgende Emotionen werden neu aufgenommen (zu viele Ausreißer):", ", ".join(retry_emotions))
        _show_repeat_popup(
            "Unfortunately, the setup process needs to be repeated because the emotions weren't exaggerated enough. Please try again!"
        )
        success = calibrator.record_emotions(
            emotions=retry_emotions,
            duration=12,
            analyze_every=5,
            selector_emotions=retry_emotions,
            enabled_emotions=retry_emotions,
        )
        if not success:
            print("✖️ Wiederholung abgebrochen.")
            return False

    calibrator.train()
    calibrator.save_model()
    # Skip blocking visualization to allow automatic continuation to next setup steps.
    print("✅ Rest-Face-Modell erfolgreich erstellt.")
    return True

if __name__ == "__main__":
    app = QApplication(sys.argv)

    controller = SetupController()
    controller.finished.connect(lambda success: print("Setup erfolgreich" if success else "Abgebrochen"))

    controller.start()

    sys.exit(app.exec_())