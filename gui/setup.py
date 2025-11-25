"""PyQt-basierte Mini-GUI zum Auswählen der nächsten Emotion während des Setups."""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence

import cv2
import numpy as np

from PyQt5 import QtCore, QtGui, QtWidgets


def ensure_qt_app() -> QtWidgets.QApplication:
    """Return existing QApplication or create a new one."""
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


class EmotionSelectorWindow(QtWidgets.QWidget):
    """Shows emotion buttons and tracks user selections."""

    EMOTION_HINTS = {
        "fear": "Fear: For example, widen your eyes and slightly open your mouth.",
        "happy": "Happy: Make your smile or laugh loudly as if you had heard a joke.",
        "sad": "Sad: Look very sad or disappointed, with downturned corners of the mouth or a jutting lower lip.",
        "surprise": "Surprised: Open your eyes and mouth wide, look extremely surprised, as if you've just learned THE secret.",
        "neutral": "Neutral: Look like you do when you watch a video or game—don't force emotionlessness, just look the way you normally do.",
    }

    def __init__(
        self,
        emotions: Iterable[str],
        position: Optional[tuple[int, int]] = None,
        enabled_emotions: Optional[Iterable[str]] = None,
    ):
        super().__init__()
        self.setWindowTitle("Moody – Emotion Selector")
        self.setFixedWidth(520)
        self.setMinimumHeight(640)
        self.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint, True)
        self._buttons: Dict[str, QtWidgets.QPushButton] = {}
        self._pending_selection: Optional[str] = None
        self._active_emotion: Optional[str] = None
        self._aborted = False
        self._start_requested = False
        self._done_requested = False
        self._completed_emotions: set[str] = set()
        self._disabled_emotions: set[str] = set()
        enabled_set = {e.lower() for e in enabled_emotions} if enabled_emotions else None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        self.header_label = QtWidgets.QLabel(
            "We'll go through the setup process. First, you'll select an emotion. "
            "When you're ready to mime the emotion, press start. It's best to show the emotion beforehand, "
            "because the program will start recording immediately. If you're finished and you don't like the result "
            "because you accidentally laughed, you can record it again."
        )
        self.header_label.setWordWrap(True)
        layout.addWidget(self.header_label)

        self.camera_label = QtWidgets.QLabel()
        self.camera_label.setMinimumHeight(420)
        self.camera_label.setAlignment(QtCore.Qt.AlignCenter)
        self.camera_label.setStyleSheet("background: #111; border: 1px solid #444;")
        layout.addWidget(self.camera_label)

        self.info_label = QtWidgets.QLabel("Choose the next emotion.")
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)

        grid = QtWidgets.QGridLayout()
        grid.setSpacing(8)
        row = col = 0
        for emotion in emotions:
            btn = QtWidgets.QPushButton(emotion.title())
            btn.setObjectName(emotion)
            btn.clicked.connect(lambda _=False, name=emotion: self._handle_selection(name))
            is_enabled = enabled_set is None or emotion.lower() in enabled_set
            btn.setEnabled(is_enabled)
            if not is_enabled:
                btn.setStyleSheet("background-color: #e0e0e0; color: #777;")
                self._disabled_emotions.add(emotion)
            self._buttons[emotion] = btn
            grid.addWidget(btn, row, col)
            col += 1
            if col >= 2:
                col = 0
                row += 1
        layout.addLayout(grid)

        self.status_label = QtWidgets.QLabel("No selection yet.")
        layout.addWidget(self.status_label)

        if position:
            self.move(*position)
        else:
            self.move(100, 100)

        self.detail_label = QtWidgets.QLabel("")
        self.detail_label.setWordWrap(True)
        self.detail_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.detail_label)

        button_row = QtWidgets.QHBoxLayout()
        button_row.setSpacing(10)
        self.start_button = QtWidgets.QPushButton("Start")
        self.start_button.clicked.connect(self._handle_start_clicked)
        self.abort_button = QtWidgets.QPushButton("Stop")
        self.abort_button.clicked.connect(self._handle_abort_clicked)
        self.done_button = QtWidgets.QPushButton("Done")
        self.done_button.setEnabled(False)
        self.done_button.clicked.connect(self._handle_done_clicked)
        button_row.addWidget(self.start_button)
        button_row.addWidget(self.abort_button)
        button_row.addWidget(self.done_button)
        layout.addLayout(button_row)

    def take_selection(self) -> Optional[str]:
        """Return and clear the last selection."""
        selection = self._pending_selection
        self._pending_selection = None
        return selection

    def _handle_selection(self, emotion: str):
        if self._buttons.get(emotion) and emotion not in self._disabled_emotions:
            self._pending_selection = emotion
            self.status_label.setText(f"Selected: {emotion.title()} (ready)")
            self._update_instruction(emotion)

    def set_active_emotion(self, emotion: Optional[str]):
        """Highlight the emotion that is currently being recorded."""
        if self._active_emotion == emotion:
            return
        if self._active_emotion and self._active_emotion in self._buttons:
            self._buttons[self._active_emotion].setStyleSheet("")
        self._active_emotion = emotion
        if emotion and emotion in self._buttons:
            self._buttons[emotion].setStyleSheet("background-color: #ffd966;")
            self.status_label.setText(f"Recording: {emotion.title()}")
            self._update_instruction(emotion)
        else:
            self._update_instruction(None)

    def mark_completed(self, emotion: str):
        """Mark an emotion as captured."""
        btn = self._buttons.get(emotion)
        if btn:
            btn.setStyleSheet("background-color: #c6efce;")
            btn.setText(f"{emotion.title()} ✓")
        self.status_label.setText(f"{emotion.title()} completed.")
        if self._active_emotion == emotion:
            self._active_emotion = None
        self._completed_emotions.add(emotion)
        if self.remaining_emotions() == 0:
            self.done_button.setEnabled(True)

    def remaining_emotions(self) -> int:
        """Return the count of still enabled emotions."""
        enabled_total = len(self._buttons) - len(self._disabled_emotions)
        return max(0, enabled_total - len(self._completed_emotions))

    def consume_start_request(self) -> bool:
        if self._start_requested:
            self._start_requested = False
            return True
        return False

    def consume_done_request(self) -> bool:
        if self._done_requested:
            self._done_requested = False
            return True
        return False

    def is_aborted(self) -> bool:
        return self._aborted

    def closeEvent(self, event):  # noqa: N802
        self._aborted = True
        super().closeEvent(event)

    def _update_instruction(self, emotion: Optional[str]):
        if not emotion:
            self.detail_label.setText("")
            return
        hint = self.EMOTION_HINTS.get(emotion.lower())
        self.detail_label.setText(hint or "")

    def _handle_start_clicked(self):
        self._start_requested = True

    def _handle_abort_clicked(self):
        self._aborted = True
        self.close()

    def _handle_done_clicked(self):
        self._done_requested = True
        self.start_button.setEnabled(False)
        self.done_button.setEnabled(False)

    def update_frame(self, frame: np.ndarray):
        """Display the latest camera frame inside the selector window."""
        if frame is None or frame.size == 0:
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        qimg = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qimg)
        self.camera_label.setPixmap(
            pixmap.scaled(
                self.camera_label.width(),
                self.camera_label.height(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation,
            )
        )


class SetupInstructionDialog(QtWidgets.QDialog):
    """Modal dialog that lists setup instructions before camera usage."""

    def __init__(self, instructions: Sequence[str]):
        super().__init__()
        self.setWindowTitle("Moody – Setup Instructions")
        self.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint, True)
        self.setMinimumWidth(420)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        header = QtWidgets.QLabel("Please note the following points before you start:")
        header.setWordWrap(True)
        layout.addWidget(header)

        for idx, line in enumerate(instructions, start=1):
            label = QtWidgets.QLabel(f"{idx}. {line}")
            label.setWordWrap(True)
            layout.addWidget(label)

        layout.addSpacing(8)
        button = QtWidgets.QPushButton("I understand and am ready to go")
        button.clicked.connect(self.accept)
        layout.addWidget(button)

    @staticmethod
    def show_dialog(instructions: Sequence[str]) -> bool:
        app = ensure_qt_app()
        dialog = SetupInstructionDialog(instructions)
        return dialog.exec_() == QtWidgets.QDialog.Accepted


def show_setup_instructions(instructions: Sequence[str]) -> bool:
    """Convenience helper used by the setup script."""
    return SetupInstructionDialog.show_dialog(instructions)
