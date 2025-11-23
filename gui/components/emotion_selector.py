"""PyQt-basierte Mini-GUI zum Auswählen der nächsten Emotion während des Setups."""

from __future__ import annotations

from typing import Dict, Iterable, Optional

from PyQt5 import QtCore, QtWidgets


def ensure_qt_app() -> QtWidgets.QApplication:
    """Return existing QApplication or create a new one."""
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


class EmotionSelectorWindow(QtWidgets.QWidget):
    """Shows emotion buttons and tracks user selections."""

    def __init__(self, emotions: Iterable[str]):
        super().__init__()
        self.setWindowTitle("Moody – Emotion auswählen")
        self.setFixedWidth(320)
        self.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint, True)
        self._buttons: Dict[str, QtWidgets.QPushButton] = {}
        self._pending_selection: Optional[str] = None
        self._active_emotion: Optional[str] = None
        self._aborted = False

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        self.info_label = QtWidgets.QLabel("Wähle die nächste Emotion.")
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)

        grid = QtWidgets.QGridLayout()
        grid.setSpacing(8)
        row = col = 0
        for emotion in emotions:
            btn = QtWidgets.QPushButton(emotion.title())
            btn.setObjectName(emotion)
            btn.clicked.connect(lambda _=False, name=emotion: self._handle_selection(name))
            self._buttons[emotion] = btn
            grid.addWidget(btn, row, col)
            col += 1
            if col >= 2:
                col = 0
                row += 1
        layout.addLayout(grid)

        self.status_label = QtWidgets.QLabel("Keine Auswahl getroffen.")
        layout.addWidget(self.status_label)

        self.move(100, 100)

    def take_selection(self) -> Optional[str]:
        """Return and clear the last selection."""
        selection = self._pending_selection
        self._pending_selection = None
        return selection

    def _handle_selection(self, emotion: str):
        if self._buttons.get(emotion) and self._buttons[emotion].isEnabled():
            self._pending_selection = emotion
            self.status_label.setText(f"Auswahl: {emotion.title()} (bereit)")

    def set_active_emotion(self, emotion: Optional[str]):
        """Highlight the emotion that is currently being recorded."""
        if self._active_emotion == emotion:
            return
        if self._active_emotion and self._active_emotion in self._buttons:
            self._buttons[self._active_emotion].setStyleSheet("")
        self._active_emotion = emotion
        if emotion and emotion in self._buttons:
            self._buttons[emotion].setStyleSheet("background-color: #ffd966;")
            self.status_label.setText(f"Aufnahme läuft: {emotion.title()}")

    def mark_completed(self, emotion: str):
        """Disable a button once the emotion is captured."""
        btn = self._buttons.get(emotion)
        if btn:
            btn.setEnabled(False)
            btn.setStyleSheet("background-color: #c6efce;")
            btn.setText(f"{emotion.title()} ✓")
        self.status_label.setText(f"{emotion.title()} abgeschlossen.")
        if self._active_emotion == emotion:
            self._active_emotion = None

    def remaining_emotions(self) -> int:
        """Return the count of still enabled emotions."""
        return sum(1 for btn in self._buttons.values() if btn.isEnabled())

    def is_aborted(self) -> bool:
        return self._aborted

    def closeEvent(self, event):  # noqa: N802
        self._aborted = True
        super().closeEvent(event)
