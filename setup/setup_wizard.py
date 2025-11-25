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

import math

from filter_outlier import filter_outliers
from .sound_setup import run_sound_setup
from .face_setup import EMOTION_PROFILES, RestFaceCalibrator
from utils.settings import FACE_SETUP_ENABLED, REST_FACE_MODEL_PATH

def _should_capture_faces() -> bool:
    return FACE_SETUP_ENABLED


def _filter_and_report(calibrator: RestFaceCalibrator):
    """Run the outlier filter across all emotions, log summary, and return (summary, avg_removed)."""
    if not calibrator.profiles:
        return {}, 0.0

    summary = filter_outliers(
        calibrator,
        method="lof",
        contamination="auto",
        radius_sigma=2.5,
    )
    removed_total = sum(stats.get("removed", 0) for stats in summary.values())
    count = len(summary) or 1
    avg_removed = math.ceil(removed_total / count)

    print("🔎 Outlier check per emotion:")
    for emotion, stats in summary.items():
        print(
            f" - {emotion}: kept {stats['kept']}/{stats['original']} "
            f"(removed {stats['removed']})"
        )
    print(f"Avg removed samples per emotion: {avg_removed:.2f}")
    return summary, avg_removed


def _emotions_above_avg(summary, avg_removed):
    """Return a list of emotions whose removed count exceeds the baseline average."""
    return [
        emotion
        for emotion, stats in summary.items()
        if stats.get("removed", 0) > avg_removed
    ]


def _show_outlier_popup(emotions, baseline_avg, summary):
    """Show a dialog listing emotions that need to be redone."""
    names = ", ".join(emotions)
    details = "; ".join(
        f"{emotion}: removed {summary.get(emotion, {}).get('removed', 0)}"
        for emotion in emotions
    )
    try:
        from PyQt5 import QtWidgets
        from gui.setup import ensure_qt_app

        app = ensure_qt_app()
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Warning)
        msg.setWindowTitle("Please record emotions again")
        msg.setText(
            f"Hey, unfortunately, there were too many moments in {names} "
            "where you didn't fully show the emotion."
        )
        msg.setInformativeText("Please repeat these moments.")
        msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
        msg.exec_()
    except Exception:
        print(
            f"⚠️ Hey, unfortunately, there were too many moments in {names} "
            "where you didn't fully show the emotion. Please repeat these moments."
        )


def run_rest_face_setup(user="default", force_record=None):
    """Führt die neue Rest-Face-Kalibrierung aus."""
    print("📷 Starte Rest-Face-Kalibrierung ...")
    model_path = REST_FACE_MODEL_PATH

    calibrator = RestFaceCalibrator(model_path=model_path)
    if force_record is None:
        force_record = _should_capture_faces()

    while True:
        if not force_record:
            if not calibrator.load_snapshot():
                print("⚠️ Kein Snapshot vorhanden – starte neue Aufnahme.")
                force_record = True
                continue

        if force_record:
            success = calibrator.record_emotions(duration=12, analyze_every=5)
            if not success:
                print("✖️ Keine Daten erfasst – bitte erneut versuchen.")
                return False
        elif not calibrator.profiles:
            print("✖️ Keine gespeicherten Profile gefunden.")
            return False

        summary, first_pass_avg = _filter_and_report(calibrator)
        calibrator._save_snapshot()  # persist filtered data so training uses the cleaned snapshot

        to_redo = _emotions_above_avg(summary, first_pass_avg)
        if to_redo:
            _show_outlier_popup(to_redo, baseline_avg=first_pass_avg, summary=summary)

        # Repeat only the problematic emotions until each drops below the baseline average.
        while to_redo:
            print(f"🔁 Re-recording emotions above baseline average: {', '.join(to_redo)}")
            success = calibrator.record_emotions(
                emotions=to_redo,
                duration=12,
                analyze_every=5,
                selector_emotions=[name for name, _ in EMOTION_PROFILES],
                enabled_emotions=to_redo,
            )
            if not success:
                print("✖️ Keine Daten erfasst – bitte erneut versuchen.")
                return False
            summary, _ = _filter_and_report(calibrator)
            calibrator._save_snapshot()
            to_redo = _emotions_above_avg(summary, first_pass_avg)
            if to_redo:
                _show_outlier_popup(to_redo, baseline_avg=first_pass_avg, summary=summary)
        break

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
