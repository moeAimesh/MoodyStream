"""
Styled GUI components for the Moody setup wizard.
Includes emotion selector window and instruction dialogs with modern dark theme.
"""

import sys
from typing import Optional, Sequence
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QDialog, QCheckBox, QScrollArea, QFrame,
    QGraphicsDropShadowEffect, QComboBox
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5 import QtGui, QtCore
from PyQt5.QtGui import QPixmap, QImage, QFont, QColor
import numpy as np
import cv2


# stylesheet to make sure design is the same across different windows
MOODY_STYLESHEET = """
    /* Hauptfenster - Base Layer */
    QMainWindow {
        background-color: #161618;
    }
    
    /* Standard Widgets - Base Layer */
    QWidget {
        background-color: #161618;
        color: #FFFFFF;
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Text', Monaco, monospace;
    }
    
    /* Labels - Text auf Base Layer */
    QLabel {
        color: #FFFFFF;
        background-color: transparent;
    }
    
    /* Buttons - Elevated Layer mit Logo-Farbverlauf beim Hover */
    QPushButton {
        background-color: #212124;
        color: #FFFFFF;
        border: none;
        border-radius: 6px;
        padding: 8px 16px;
        font-size: 13px;
    }
    QPushButton:hover {
        background: qlineargradient(
            x1:0, y1:0, x2:1, y2:1,
            stop:0 rgba(255, 107, 74, 0.3),
            stop:0.5 rgba(255, 59, 143, 0.3),
            stop:1 rgba(255, 105, 180, 0.3)
        );
    }
    QPushButton:pressed {
        background: qlineargradient(
            x1:0, y1:0, x2:1, y2:1,
            stop:0 rgba(255, 107, 74, 0.5),
            stop:0.5 rgba(255, 59, 143, 0.5),
            stop:1 rgba(255, 105, 180, 0.5)
        );
    }
    QPushButton:checked {
        background: qlineargradient(
            x1:0, y1:0, x2:1, y2:1,
            stop:0 rgba(255, 107, 74, 0.5),
            stop:0.5 rgba(255, 59, 143, 0.5),
            stop:1 rgba(255, 105, 180, 0.5)
        );
    }
    
    /* ComboBox - Elevated Layer mit Farbverlauf beim Hover */
    QComboBox {
        background-color: #212124;
        color: #FFFFFF;
        border: none;
        border-radius: 6px;
        padding: 6px 12px;
        font-size: 13px;
    }
    QComboBox:hover {
        background: qlineargradient(
            x1:0, y1:0, x2:1, y2:1,
            stop:0 rgba(255, 107, 74, 0.3),
            stop:0.5 rgba(255, 59, 143, 0.3),
            stop:1 rgba(255, 105, 180, 0.3)
        );
    }
    QComboBox::drop-down {
        border: none;
        width: 20px;
    }
    QComboBox::down-arrow {
        image: none;
        border-left: 4px solid transparent;
        border-right: 4px solid transparent;
        border-top: 5px solid #FFFFFF;
        margin-right: 8px;
    }
    QComboBox QAbstractItemView {
        background-color: #212124;
        color: #FFFFFF;
        selection-background-color: rgba(255, 107, 74, 0.3);
        border: 1px solid #000000;
        border-radius: 6px;
        padding: 4px;
        outline: none;
    }
    
    /* Slider - Custom Apple Style */
    QSlider::groove:vertical {
        background-color: #212124;
        width: 4px;
        border-radius: 2px;
    }
    QSlider::handle:vertical {
        background-color: #FFFFFF;
        width: 18px;
        height: 18px;
        margin: 0 -7px;
        border-radius: 9px;
        border: none;
    }
    QSlider::handle:vertical:hover {
        background-color: #818181;
    }
    QSlider::groove:horizontal {
        background-color: #212124;
        height: 4px;
        border-radius: 2px;
    }
    QSlider::handle:horizontal {
        background-color: #FFFFFF;
        width: 18px;
        height: 18px;
        margin: -7px 0;
        border-radius: 9px;
        border: none;
    }
    QSlider::handle:horizontal:hover {
        background-color: #818181;
    }
    
    /* SpinBox - Elevated Layer */
    QSpinBox {
        background-color: #212124;
        color: #FFFFFF;
        border: none;
        border-radius: 6px;
        padding: 6px 12px;
        font-size: 13px;
    }
    QSpinBox:hover {
        background-color: #818181;
    }
    QSpinBox::up-button, QSpinBox::down-button {
        background-color: transparent;
        border: none;
        width: 16px;
    }
    QSpinBox::up-arrow {
        image: none;
        border-left: 4px solid transparent;
        border-right: 4px solid transparent;
        border-bottom: 5px solid #FFFFFF;
    }
    QSpinBox::down-arrow {
        image: none;
        border-left: 4px solid transparent;
        border-right: 4px solid transparent;
        border-top: 5px solid #FFFFFF;
    }
    
    /* MenuBar - Darkest Layer */
    QMenuBar {
        background-color: #000000;
        color: #FFFFFF;
        border: none;
        padding: 4px;
        font-size: 13px;
    }
    QMenuBar::item {
        background-color: transparent;
        padding: 4px 12px;
        border-radius: 4px;
    }
    QMenuBar::item:selected {
        background-color: #212124;
    }
    QMenuBar::item:pressed {
        background-color: #818181;
    }
    
    /* Menu Dropdown */
    QMenu {
        background-color: #212124;
        color: #FFFFFF;
        border: 1px solid #000000;
        border-radius: 8px;
        padding: 4px;
    }
    QMenu::item {
        padding: 6px 24px 6px 12px;
        border-radius: 4px;
    }
    QMenu::item:selected {
        background-color: #818181;
    }
    QMenu::separator {
        height: 1px;
        background-color: #000000;
        margin: 4px 8px;
    }
    
    /* MessageBox/Dialog */
    QDialog {
        background-color: #161618;
        color: #FFFFFF;
    }
    QMessageBox {
        background-color: #161618;
        color: #FFFFFF;
    }
    QMessageBox QLabel {
        color: #FFFFFF;
    }
    QMessageBox QPushButton {
        min-width: 70px;
    }
    
    /* FileDialog */
    QFileDialog {
        background-color: #161618;
        color: #FFFFFF;
    }
    QFileDialog QWidget {
        background-color: #161618;
        color: #FFFFFF;
    }
    QFileDialog QPushButton {
        background-color: #212124;
        color: #FFFFFF;
    }
    QFileDialog QTreeView {
        background-color: #161618;
        color: #FFFFFF;
        border: 1px solid #212124;
    }
    QFileDialog QListView {
        background-color: #161618;
        color: #FFFFFF;
        border: 1px solid #212124;
    }
    QFileDialog QLineEdit {
        background-color: #212124;
        color: #FFFFFF;
        border: 1px solid #000000;
        border-radius: 4px;
        padding: 4px;
    }
    QFileDialog QComboBox {
        background-color: #212124;
        color: #FFFFFF;
    }
    
    /* ScrollArea */
    QScrollArea {
        background-color: transparent;
        border: none;
    }
    QScrollBar:vertical {
        background-color: #161618;
        width: 12px;
        border-radius: 6px;
    }
    QScrollBar::handle:vertical {
        background-color: #212124;
        border-radius: 6px;
        min-height: 20px;
    }
    QScrollBar::handle:vertical:hover {
        background-color: #818181;
    }
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
        height: 0px;
    }
    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
        background: none;
    }
"""


# button for each emotion
class EmotionButton(QPushButton):
    """Styled emotion button with status indicator"""
    
    def __init__(self, emotion: str, enabled: bool = True, parent=None):
        super().__init__(emotion.capitalize(), parent)
        self.emotion = emotion
        self.is_completed = False
        self.is_active = False
        self._enabled = enabled
        
        self.setMinimumHeight(50)
        self.setCheckable(False)
        self.setCursor(Qt.PointingHandCursor if enabled else Qt.ArrowCursor)
        self.setEnabled(enabled)
        
        self._update_style()
    
    def set_completed(self, completed: bool):
        """Mark this emotion as completed"""
        self.is_completed = completed
        self._update_style()
    
    def set_active(self, active: bool):
        """Mark this emotion as currently active"""
        self.is_active = active
        self._update_style()
    
    def _update_style(self):
        """Update button appearance based on state"""
        if self.is_completed:
            # completed state - transparent green
            self.setStyleSheet("""
                QPushButton {
                    background: qlineargradient(
                        x1:0, y1:0, x2:1, y2:0,
                        stop:0 rgba(76, 175, 80, 0.25),
                        stop:1 rgba(76, 175, 80, 0.35)
                    );
                    color: #FFFFFF;
                    border: none;
                    border-radius: 8px;
                    padding: 10px 20px;
                    font-size: 14px;
                    font-weight: 600;
                }
                QPushButton:hover {
                    background: qlineargradient(
                        x1:0, y1:0, x2:1, y2:0,
                        stop:0 rgba(76, 175, 80, 0.3),
                        stop:1 rgba(76, 175, 80, 0.4)
                    );
                }
                QPushButton:pressed {
                    background: qlineargradient(
                        x1:0, y1:0, x2:1, y2:0,
                        stop:0 rgba(76, 175, 80, 0.4),
                        stop:1 rgba(76, 175, 80, 0.5)
                    );
                }
            """)
        elif self.is_active:
            # active state - same as checked state in main window
            self.setStyleSheet("""
                QPushButton {
                    background: qlineargradient(
                        x1:0, y1:0, x2:1, y2:1,
                        stop:0 rgba(255, 107, 74, 0.5),
                        stop:0.5 rgba(255, 59, 143, 0.5),
                        stop:1 rgba(255, 105, 180, 0.5)
                    );
                    color: #FFFFFF;
                    border: none;
                    border-radius: 8px;
                    padding: 10px 20px;
                    font-size: 14px;
                    font-weight: 600;
                }
                QPushButton:hover {
                    background: qlineargradient(
                        x1:0, y1:0, x2:1, y2:1,
                        stop:0 rgba(255, 107, 74, 0.6),
                        stop:0.5 rgba(255, 59, 143, 0.6),
                        stop:1 rgba(255, 105, 180, 0.6)
                    );
                }
                QPushButton:pressed {
                    background: qlineargradient(
                        x1:0, y1:0, x2:1, y2:1,
                        stop:0 rgba(255, 107, 74, 0.7),
                        stop:0.5 rgba(255, 59, 143, 0.7),
                        stop:1 rgba(255, 105, 180, 0.7)
                    );
                }
            """)
        elif not self._enabled:
            # disabled state
            self.setStyleSheet("""
                QPushButton {
                    background-color: #1a1a1c;
                    color: #666666;
                    border: none;
                    border-radius: 8px;
                    padding: 10px 20px;
                    font-size: 14px;
                }
            """)
        else:
            # default state - plain with transparent hover
            self.setStyleSheet("""
                QPushButton {
                    background-color: #212124;
                    color: #FFFFFF;
                    border: none;
                    border-radius: 8px;
                    padding: 10px 20px;
                    font-size: 14px;
                    font-weight: 500;
                }
                QPushButton:hover {
                    background: qlineargradient(
                        x1:0, y1:0, x2:1, y2:1,
                        stop:0 rgba(255, 107, 74, 0.3),
                        stop:0.5 rgba(255, 59, 143, 0.3),
                        stop:1 rgba(255, 105, 180, 0.3)
                    );
                }
                QPushButton:pressed {
                    background: qlineargradient(
                        x1:0, y1:0, x2:1, y2:1,
                        stop:0 rgba(255, 107, 74, 0.5),
                        stop:0.5 rgba(255, 59, 143, 0.5),
                        stop:1 rgba(255, 105, 180, 0.5)
                    );
                }
            """)


# 
class EmotionSelectorWindow(QMainWindow):
    """
    Modern emotion selector for the setup wizard.
    Shows available emotions with start/stop controls and video preview.
    """
    
    camera_index_changed = pyqtSignal(int)  # Signal when camera index changes
    
    def __init__(self, emotions: Sequence[str], position: Optional[tuple] = None,
                 enabled_emotions: Optional[Sequence[str]] = None, parent=None):
        super().__init__(parent)
        
        self.emotions = list(emotions)
        self.enabled_emotions = set(enabled_emotions) if enabled_emotions else set(emotions)
        self.completed_emotions = set()
        self.active_emotion = None
        self._selection_queue = []
        self._start_requested = False
        self._done_requested = False
        self._aborted = False
        self._camera_index = 0  # Current camera index
        
        self.setWindowTitle("Emotion Profiling - Moody Setup")
        self.setFixedSize(600, 700)
        
        if position:
            self.move(*position)
        
        self.setStyleSheet(MOODY_STYLESHEET)
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the user interface"""
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # top bar with Logo and title (like main window)
        top_bar = QHBoxLayout()
        top_bar.setSpacing(15)
        
        # Logo
        self.logo_label = QLabel()
        logo_pixmap = QPixmap("/Users/juliamoor/Desktop/MoodyStream/gui/moody_logo.jpg")
        if not logo_pixmap.isNull():
            scaled_logo = logo_pixmap.scaled(40, 40, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.logo_label.setPixmap(scaled_logo)
        self.logo_label.setStyleSheet("""
            background-color: #212124;
            border-radius: 20px;
        """)
        self.logo_label.setFixedSize(40, 40)
        self.logo_label.setAlignment(Qt.AlignCenter)
        top_bar.addWidget(self.logo_label)
        
        # title
        title_label = QLabel("MOODYSTREAM")
        title_label.setStyleSheet("""
            font-size: 20px; 
            font-weight: bold; 
            letter-spacing: 1px;
            color: #FFFFFF;
        """)
        top_bar.addWidget(title_label)
        
        top_bar.addStretch()
        
        # Camera Index Dropdown
        camera_label = QLabel("Camera:")
        camera_label.setStyleSheet("""
            font-size: 11px;
            color: #999999;
        """)
        top_bar.addWidget(camera_label)
        
        self.camera_index_combo = QComboBox()
        self.camera_index_combo.addItems(["0", "1", "2"])
        self.camera_index_combo.setCurrentIndex(0)
        self.camera_index_combo.setFixedWidth(60)
        self.camera_index_combo.currentIndexChanged.connect(self._on_camera_index_changed)
        top_bar.addWidget(self.camera_index_combo)
        
        layout.addLayout(top_bar)
        
        # header
        header = QLabel("Emotion Profiling")
        header.setStyleSheet("""
            font-size: 18px;
            font-weight: bold;
            color: #FFFFFF;
            padding: 10px 0;
            letter-spacing: 1px;
        """)
        header.setAlignment(Qt.AlignCenter)
        layout.addWidget(header)
        
        # subtitle
        subtitle = QLabel("Select emotions to record for your profile")
        subtitle.setStyleSheet("""
            font-size: 11px;
            color: #999999;
            padding-bottom: 10px;
        """)
        subtitle.setAlignment(Qt.AlignCenter)
        layout.addWidget(subtitle)
        
        # video preview area
        preview_frame = QFrame()
        preview_frame.setStyleSheet("""
            QFrame {
                background-color: #000000;
                border: none;
                border-radius: 12px;
                min-height: 240px;
                max-height: 240px;
            }
        """)
        preview_layout = QVBoxLayout(preview_frame)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        
        self.video_label = QLabel("Camera Preview")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("""
            color: #666666;
            font-size: 14px;
            background-color: transparent;
        """)
        preview_layout.addWidget(self.video_label)
        
        layout.addWidget(preview_frame)
        
        # status label
        self.status_label = QLabel("Select an emotion to begin")
        self.status_label.setStyleSheet("""
            font-size: 11px;
            color: #FFFFFF;
            padding: 10px;
            background-color: #212124;
            border-radius: 6px;
            border: none;
        """)
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)
        
        # emotion buttons container
        emotions_label = QLabel("EMOTIONS")
        emotions_label.setStyleSheet("""
            font-size: 11px;
            font-weight: 600;
            color: #999999;
            letter-spacing: 1px;
            padding-top: 10px;
        """)
        layout.addWidget(emotions_label)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; }")
        
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setSpacing(8)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        
        self.emotion_buttons = {}
        for emotion in self.emotions:
            enabled = emotion in self.enabled_emotions
            btn = EmotionButton(emotion, enabled=enabled)
            btn.clicked.connect(lambda checked, e=emotion: self._on_emotion_clicked(e))
            self.emotion_buttons[emotion] = btn
            scroll_layout.addWidget(btn)
        
        scroll_layout.addStretch()
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll, 1)
        
        # control buttons
        controls = QHBoxLayout()
        controls.setSpacing(10)
        
        self.start_button = QPushButton("Start Recording")
        self.start_button.setMinimumHeight(45)
        self.start_button.setEnabled(False)
        self.start_button.clicked.connect(self._on_start_clicked)
        
        self.done_button = QPushButton("Done")
        self.done_button.setMinimumHeight(45)
        self.done_button.setEnabled(False)
        self.done_button.clicked.connect(self._on_done_clicked)
        
        controls.addWidget(self.start_button, 2)
        controls.addWidget(self.done_button, 1)
        
        layout.addLayout(controls)
        
        # cancel button
        cancel_btn = QPushButton("Cancel Setup")
        cancel_btn.clicked.connect(self._on_cancel_clicked)
        layout.addWidget(cancel_btn)
    
    def _on_emotion_clicked(self, emotion: str):
        """Handle emotion button click"""
        if emotion not in self.enabled_emotions:
            return
        
        self._selection_queue.append(emotion)
        self.start_button.setEnabled(True)
        
        # update active state
        for e, btn in self.emotion_buttons.items():
            btn.set_active(e == emotion)
        
        self.status_label.setText(f"Ready to record: {emotion.capitalize()}")
    
    def _on_start_clicked(self):
        """Handle start button click"""
        self._start_requested = True
        self.start_button.setEnabled(False)
        self.status_label.setText("Recording in progress...")
    
    def _on_done_clicked(self):
        """Handle done button click"""
        self._done_requested = True
    
    def _on_cancel_clicked(self):
        """Handle cancel button click"""
        self._aborted = True
        self.close()
    
    def _on_camera_index_changed(self, index: int):
        """Handle camera index change"""
        self._camera_index = index
        self.camera_index_changed.emit(index)
    
    def get_camera_index(self) -> int:
        """Get current camera index"""
        return self._camera_index
    
    def set_active_emotion(self, emotion: str):
        """Set the currently active emotion"""
        self.active_emotion = emotion
        for e, btn in self.emotion_buttons.items():
            btn.set_active(e == emotion)
        self.status_label.setText(f"Recording: {emotion.capitalize()}")
    
    def mark_completed(self, emotion: str):
        """Mark an emotion as completed"""
        self.completed_emotions.add(emotion)
        if emotion in self.emotion_buttons:
            self.emotion_buttons[emotion].set_completed(True)
        
        # enable done button if all emotions are complete
        if len(self.completed_emotions) >= len(self.enabled_emotions):
            self.done_button.setEnabled(True)
            self.status_label.setText("All emotions recorded! Click Done to continue.")
    
    def take_selection(self) -> Optional[str]:
        """Get next emotion selection from queue"""
        if self._selection_queue:
            return self._selection_queue.pop(0)
        return None
    
    def consume_start_request(self) -> bool:
        """Check and consume start request flag"""
        if self._start_requested:
            self._start_requested = False
            return True
        return False
    
    def consume_done_request(self) -> bool:
        """Check and consume done request flag"""
        if self._done_requested:
            self._done_requested = False
            return True
        return False
    
    def is_aborted(self) -> bool:
        """Check if setup was aborted"""
        return self._aborted
    
    def remaining_emotions(self) -> int:
        """Get count of remaining emotions"""
        return len(self.enabled_emotions) - len(self.completed_emotions)
    
    def update_frame(self, frame: np.ndarray):
        """Display the latest camera frame inside the selector window."""
        if frame is None or frame.size == 0:
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        qimg = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qimg)
        self.video_label.setPixmap(
            pixmap.scaled(
                self.video_label.width(),
                self.video_label.height(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation,
            )
        )


# instructions in the beginning
class InstructionsDialog(QDialog):
    """Modern dialog showing setup instructions"""
    
    def __init__(self, instructions: list, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Setup Instructions")
        self.setModal(True)
        self.setFixedSize(550, 500)
        self.setStyleSheet(MOODY_STYLESHEET)
        
        self._accepted = False
        self._setup_ui(instructions)
    
    def _setup_ui(self, instructions: list):
        """Setup the user interface"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)
        
        # Logo and Title
        top_bar = QHBoxLayout()
        top_bar.setSpacing(15)
        
        # Logo
        self.logo_label = QLabel()
        logo_pixmap = QPixmap("/Users/juliamoor/Desktop/MoodyStream/gui/moody_logo.jpg")
        if not logo_pixmap.isNull():
            scaled_logo = logo_pixmap.scaled(40, 40, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.logo_label.setPixmap(scaled_logo)
        self.logo_label.setStyleSheet("""
            background-color: #212124;
            border-radius: 20px;
        """)
        self.logo_label.setFixedSize(40, 40)
        self.logo_label.setAlignment(Qt.AlignCenter)
        top_bar.addWidget(self.logo_label)
        
        # Title
        title_label = QLabel("MOODYSTREAM")
        title_label.setStyleSheet("""
            font-size: 20px; 
            font-weight: bold; 
            letter-spacing: 1px;
            color: #FFFFFF;
        """)
        top_bar.addWidget(title_label)
        
        top_bar.addStretch()
        
        layout.addLayout(top_bar)
        
        # header
        header = QLabel("Before We Start")
        header.setStyleSheet("""
            font-size: 18px;
            font-weight: bold;
            color: #FFFFFF;
            letter-spacing: 1px;
            padding-top: 10px;
        """)
        layout.addWidget(header)
        
        # subtitle
        subtitle = QLabel("Please review these important guidelines:")
        subtitle.setStyleSheet("""
            font-size: 11px;
            color: #999999;
            padding-bottom: 10px;
        """)
        layout.addWidget(subtitle)
        
        # instructions container
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; }")
        
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setSpacing(12)
        scroll_layout.setContentsMargins(0, 0, 10, 0)
        
        for i, instruction in enumerate(instructions, 1):
            item = QFrame()
            item.setStyleSheet("""
                QFrame {
                    background-color: #212124;
                    border-radius: 8px;
                    border: none;
                    padding: 15px;
                }
            """)
            
            item_layout = QHBoxLayout(item)
            item_layout.setSpacing(15)
            
            # number badge
            number = QLabel(str(i))
            number.setStyleSheet("""
                background-color: #2a2a2d;
                color: #FFFFFF;
                font-size: 14px;
                font-weight: bold;
                border-radius: 4px;
                border: none;
                min-width: 28px;
                max-width: 28px;
                min-height: 28px;
                max-height: 28px;
                padding: 0px;
            """)
            number.setAlignment(Qt.AlignCenter)
            
            # Instruction text
            text = QLabel(instruction)
            text.setWordWrap(True)
            text.setStyleSheet("""
                font-size: 11px;
                color: #FFFFFF;
                background-color: transparent;
            """)
            
            item_layout.addWidget(number)
            item_layout.addWidget(text, 1)
            
            scroll_layout.addWidget(item)
        
        scroll_layout.addStretch()
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll, 1)
        
        # buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setMinimumHeight(45)
        cancel_btn.clicked.connect(self.reject)
        
        ok_btn = QPushButton("I Understand, Let's Start!")
        ok_btn.setMinimumHeight(45)
        ok_btn.clicked.connect(self._on_accept)
        
        button_layout.addWidget(cancel_btn, 1)
        button_layout.addWidget(ok_btn, 2)
        
        layout.addLayout(button_layout)
    
    def _on_accept(self):
        """Handle accept button"""
        self._accepted = True
        self.accept()
    
    def was_accepted(self) -> bool:
        """Check if instructions were accepted"""
        return self._accepted


# instructions 
def show_setup_instructions(instructions: list) -> bool:
    """
    Show setup instructions dialog.
    Returns True if user accepts, False if cancelled.
    """
    app = ensure_qt_app()
    dialog = InstructionsDialog(instructions)
    result = dialog.exec_()
    return result == QDialog.Accepted and dialog.was_accepted()


def ensure_qt_app():
    """Ensure QApplication exists"""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


# ==================== TESTING ====================
if __name__ == "__main__":
    app = ensure_qt_app()
    
    # Test instructions dialog
    test_instructions = [
        "Keep the camera at eye level.",
        "Avoid top-down lighting so your face has no harsh shadows.",
        "Do not laugh or perform other emotions than the prompted one during the setup.",
        "Act the emotions exactly as you expect them to trigger sounds later while streaming.",
        "Exaggerate the emotions so the AI only reacts when you are really shocked/sad/happy/etc.",
    ]
    
    if show_setup_instructions(test_instructions):
        # Test emotion selector
        emotions = ["happy", "sad", "surprise", "fear", "neutral"]
        window = EmotionSelectorWindow(emotions, enabled_emotions=emotions)
        window.show()
        sys.exit(app.exec_())
    else:
        print("Instructions cancelled")