import sys
import os
import cv2
import pygame
import threading
import json
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QScrollArea, QSlider,
                             QDialog, QSpinBox, QMenu, QFileDialog, QMessageBox)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot, QObject, QMetaObject, Q_ARG
from PyQt5.QtGui import QPixmap, QImage
import numpy as np
from setup.sound_setup import run_sound_setup
# Import camera_stream
try:
    from detection.camera_stream import start_detection
    DETECTION_AVAILABLE = True
except ImportError as e:
    DETECTION_AVAILABLE = False
    print(f"Could not import detection module: {e}")


class HoverBox(QWidget):
    """Custom widget that changes color on hover"""
    clicked = pyqtSignal()
    sound_clicked = pyqtSignal()
    play_clicked = pyqtSignal()
    
    def __init__(self, text, parent=None):
        super().__init__(parent)
        self.text = text
        self.hovered = False
        self.selected_sound_file = None
        
        # Initialize pygame mixer for sound playback
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        
        # Load sound from config if available
        self._load_sound_from_config()
        
        self.setMinimumHeight(38)  # Noch kleiner: 45 → 38
        self.setMaximumHeight(38)  # Noch kleiner: 45 → 38
        self.setCursor(Qt.PointingHandCursor)
        
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet("""
            HoverBox {
                background-color: #161618;
                border: none;
                border-radius: 8px;
            }
        """)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 0, 12, 0)
        
        self.label = QLabel(text)
        self.label.setStyleSheet("color: #FFFFFF; font-size: 11px; background: transparent; border: none;")
        layout.addWidget(self.label)
        
        layout.addStretch(7)
        
        # Play Sound Button
        self.play_button = QPushButton("Play Sound")
        self.play_button.setStyleSheet("""
            QPushButton {
                background-color: #2a2a2d;
                color: #FFFFFF;
                border: none;
                border-radius: 4px;
                padding: 3px 10px;
                font-size: 10px;
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
        self.play_button.setFixedHeight(26)  # Kleiner: 30 → 26
        self.play_button.clicked.connect(self.on_play_button_clicked)
        layout.addWidget(self.play_button)
        
        layout.addSpacing(10)
        
        # Choose Sound Button with Dropdown
        self.sound_button = QPushButton("Choose Sound")
        self.sound_button.setStyleSheet("""
            QPushButton {
                background-color: #2a2a2d;
                color: #FFFFFF;
                border: none;
                border-radius: 4px;
                padding: 3px 10px;
                font-size: 10px;
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
        self.sound_button.setFixedHeight(26)  # Kleiner: 30 → 26
        self.sound_button.clicked.connect(self.show_sound_menu)
        layout.addWidget(self.sound_button)
        
        layout.addSpacing(25)
    
    def _load_sound_from_config(self):
        """Load sound file path from setup_config.json"""
        try:
            config_path = os.path.join(os.path.dirname(__file__), "setup_config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    sounds = config.get("sounds", {})
                    # Match emotion name (case insensitive)
                    emotion_key = self.text.lower()
                    if emotion_key in sounds:
                        sound_path = sounds[emotion_key]
                        # Convert relative path to absolute
                        if not os.path.isabs(sound_path):
                            sound_path = os.path.join(os.path.dirname(__file__), sound_path)
                        if os.path.exists(sound_path):
                            self.selected_sound_file = sound_path
                            print(f"✅ Loaded sound for {self.text}: {sound_path}")
                        else:
                            print(f"⚠️ Sound file not found: {sound_path}")
        except Exception as e:
            print(f"⚠️ Could not load sound from config: {e}")
    
    def show_sound_menu(self):
        """Show dropdown menu for sound selection"""
        menu = QMenu(self)
        menu.setStyleSheet("""
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
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(255, 107, 74, 0.3),
                    stop:0.5 rgba(255, 59, 143, 0.3),
                    stop:1 rgba(255, 105, 180, 0.3)
                );
            }
        """)
        
        browse_web_action = menu.addAction("Browse Web")
        browse_computer_action = menu.addAction("Browse Computer")
        
        browse_web_action.triggered.connect(self.browse_web)
        browse_computer_action.triggered.connect(self.browse_computer)
        
        menu.exec_(self.sound_button.mapToGlobal(self.sound_button.rect().bottomLeft()))
    
    def browse_web(self):
        """Handle Browse Web option"""
        print(f"Browse Web selected for {self.text}")
        self.sound_clicked.emit()
        run_sound_setup()
        # Reload sound after setup
        self._load_sound_from_config()
    
    def browse_computer(self):
        """Handle Browse Computer option - open file dialog"""
        file_dialog = QFileDialog(self)
        file_dialog.setStyleSheet("""
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
                border: none;
                border-radius: 6px;
                padding: 6px 12px;
            }
            QFileDialog QPushButton:hover {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(255, 107, 74, 0.3),
                    stop:0.5 rgba(255, 59, 143, 0.3),
                    stop:1 rgba(255, 105, 180, 0.3)
                );
            }
            QFileDialog QTreeView, QFileDialog QListView {
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
        """)
        
        file_path, _ = file_dialog.getOpenFileName(
            self,
            "Select Sound File",
            "",
            "Audio Files (*.mp3 *.wav *.ogg *.m4a);;All Files (*.*)"
        )
        
        if file_path:
            self.selected_sound_file = file_path
            print(f"Selected sound file for {self.text}: {file_path}")
            filename = os.path.basename(file_path)
            self.sound_button.setToolTip(f"Selected: {filename}")
    
    def on_play_button_clicked(self):
        if self.selected_sound_file:
            try:
                pygame.mixer.music.load(self.selected_sound_file)
                pygame.mixer.music.play()
                print(f"Playing sound: {self.selected_sound_file}")
            except Exception as e:
                print(f"Error playing sound: {e}")
        else:
            print(f"No sound file selected for {self.text}")
        self.play_clicked.emit()
        
    def on_sound_button_clicked(self):
        self.sound_clicked.emit()
        
    def enterEvent(self, event):
        self.setStyleSheet("""
            HoverBox {
                background-color: #3a3a3d;
                border: none;
                border-radius: 8px;
            }
        """)
        self._update_button_hover_style(True)
        super().enterEvent(event)
        
    def leaveEvent(self, event):
        self.setStyleSheet("""
            HoverBox {
                background-color: #161618;
                border: none;
                border-radius: 8px;
            }
        """)
        self._update_button_hover_style(False)
        super().leaveEvent(event)
    
    def _update_button_hover_style(self, hover):
        bg_color = "#3a3a3d" if hover else "#2a2a2d"
        style = f"""
            QPushButton {{
                background-color: {bg_color};
                color: #FFFFFF;
                border: none;
                border-radius: 4px;
                padding: 3px 10px;
                font-size: 10px;
            }}
            QPushButton:hover {{
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(255, 107, 74, 0.3),
                    stop:0.5 rgba(255, 59, 143, 0.3),
                    stop:1 rgba(255, 105, 180, 0.3)
                );
            }}
            QPushButton:pressed {{
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(255, 107, 74, 0.5),
                    stop:0.5 rgba(255, 59, 143, 0.5),
                    stop:1 rgba(255, 105, 180, 0.5)
                );
            }}
        """
        self.play_button.setStyleSheet(style)
        self.sound_button.setStyleSheet(style)
        
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if not self.sound_button.geometry().contains(event.pos()) and \
               not self.play_button.geometry().contains(event.pos()):
                self.clicked.emit()
        super().mousePressEvent(event)


class SettingsDialog(QDialog):
    """Settings dialog for camera configuration"""
    
    restart_setup_signal = pyqtSignal()
    
    def __init__(self, current_camera_index, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setModal(True)
        self.setFixedSize(400, 280)  # Größer: 200 → 280 für Volume Slider
        
        self.setStyleSheet("""
            QDialog {
                background-color: #161618;
                color: #FFFFFF;
            }
            QLabel {
                color: #FFFFFF;
                font-size: 13px;
            }
            QPushButton {
                background-color: #212124;
                color: #FFFFFF;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 13px;
                min-width: 80px;
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
            QSpinBox {
                background-color: #212124;
                color: #FFFFFF;
                border: none;
                border-radius: 6px;
                padding: 6px 12px;
                font-size: 13px;
            }
            QSpinBox:hover {
                background-color: #2a2a2d;
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
            QSlider::groove:horizontal {
                background-color: #2a2a2d;
                height: 10px;
                border-radius: 5px;
            }
            QSlider::handle:horizontal {
                background-color: #FFFFFF;
                width: 40px;
                height: 16px;
                margin: -3px 0;
                border-radius: 8px;
                border: none;
            }
            QSlider::handle:horizontal:hover {
                background-color: #d0d0d0;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)
        
        camera_layout = QHBoxLayout()
        camera_label = QLabel("Camera Index:")
        camera_layout.addWidget(camera_label)
        
        self.camera_spinbox = QSpinBox()
        self.camera_spinbox.setMinimum(0)
        self.camera_spinbox.setMaximum(10)
        self.camera_spinbox.setValue(current_camera_index)
        self.camera_spinbox.setFixedWidth(100)
        camera_layout.addWidget(self.camera_spinbox)
        camera_layout.addStretch()
        
        layout.addLayout(camera_layout)
        
        # Volume Slider
        volume_layout = QHBoxLayout()
        volume_label = QLabel("Volume:")
        volume_layout.addWidget(volume_label)
        
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setMinimum(0)
        self.volume_slider.setMaximum(100)
        self.volume_slider.setValue(50)  # Default: 50%
        self.volume_slider.setFixedWidth(200)
        self.volume_slider.valueChanged.connect(self.update_volume)
        volume_layout.addWidget(self.volume_slider)
        
        self.volume_value_label = QLabel("50%")
        self.volume_value_label.setFixedWidth(40)
        self.volume_value_label.setStyleSheet("color: #FFFFFF; font-size: 13px;")
        volume_layout.addWidget(self.volume_value_label)
        
        volume_layout.addStretch()
        
        layout.addLayout(volume_layout)
        
        restart_layout = QHBoxLayout()
        restart_button = QPushButton("Restart Setup")
        restart_button.clicked.connect(self.restart_setup)
        restart_layout.addWidget(restart_button)
        restart_layout.addStretch()
        
        layout.addLayout(restart_layout)
        
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)
        
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.accept)
        button_layout.addWidget(ok_button)
        
        layout.addLayout(button_layout)
    
    def get_camera_index(self):
        return self.camera_spinbox.value()
    
    def update_volume(self, value):
        """Update volume label and pygame mixer volume"""
        self.volume_value_label.setText(f"{value}%")
        # Set pygame mixer volume (0.0 to 1.0)
        pygame.mixer.music.set_volume(value / 100.0)
    
    def restart_setup(self):
        """Emit signal to restart setup"""
        self.restart_setup_signal.emit()
        print("Restart Setup clicked - Setup wird neu gestartet")


class MainWindow(QMainWindow):
    # Signal for thread-safe frame updates
    frame_ready = pyqtSignal(object)  # Will carry QPixmap
    
    def __init__(self, camera_index=0):
        super().__init__()
        self.setWindowTitle("Moodystream")
        self.setGeometry(100, 100, 1400, 800)  # Zurück auf 800px
        
        # Get the directory where this file is located
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Define image paths relative to this file
        self.logo_path = os.path.join(self.base_dir, "moody_logo.jpg")
        self.ad_mockup_path = os.path.join(self.base_dir, "ad_mockup.jpg")
        
        self.setStyleSheet(self._get_main_stylesheet())
        
        self.camera_index = camera_index
        self.detection_running = False
        self.detection_thread = None
        self.emotion_detection_active = True  # Detection state
        
        # Initialize pygame mixer for sound playback
        if not pygame.mixer.get_init():
            pygame.mixer.init()
            pygame.mixer.music.set_volume(0.5)  # Default 50% volume
        
        # Connect signal for frame updates
        self.frame_ready.connect(self._update_camera_display)
        
        self._setup_ui()
        
        # DON'T start detection automatically - let user click button
        # This prevents crashes on startup
        print("✅ MainWindow initialized. Click 'Detection: ON' to start camera.")
    
    def _get_main_stylesheet(self):
        """Returns the main stylesheet for the window"""
        return """
            QMainWindow {
                background-color: #161618;
            }
            QWidget {
                background-color: #161618;
                color: #FFFFFF;
                font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Text', Monaco, monospace;
            }
            QLabel {
                color: #FFFFFF;
                background-color: transparent;
            }
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
            QSlider::groove:horizontal {
                background-color: #2a2a2d;
                height: 10px;
                border-radius: 5px;
            }
            QSlider::handle:horizontal {
                background-color: #FFFFFF;
                width: 40px;
                height: 16px;
                margin: -3px 0;
                border-radius: 8px;
                border: none;
            }
            QSlider::handle:horizontal:hover {
                background-color: #d0d0d0;
            }
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
                background-color: #FFFFFF;
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
    
    def _setup_ui(self):
        """Setup the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Left ads
        left_ads = self._create_ad_column()
        
        # Center content
        center_layout = self._create_center_layout()
        
        # Right ads
        right_ads = self._create_ad_column()
        
        # Add to main layout
        main_layout.addLayout(left_ads, 1)
        main_layout.addLayout(center_layout, 3)
        main_layout.addLayout(right_ads, 1)
    
    def _create_ad_column(self):
        """Create an ad column"""
        ads_layout = QVBoxLayout()
        
        for _ in range(2):
            ad_label = QLabel()
            ad_label.setStyleSheet("border: 1px solid #212124; border-radius: 8px;")
            ad_label.setAlignment(Qt.AlignCenter)
            ad_label.setMinimumSize(220, 280)
            ad_label.setScaledContents(True)
            
            if os.path.exists(self.ad_mockup_path):
                ad_pixmap = QPixmap(self.ad_mockup_path)
                ad_label.setPixmap(ad_pixmap)
            else:
                ad_label.setText("AD\nMockup")
            
            ads_layout.addWidget(ad_label)
        
        ads_layout.addStretch()
        return ads_layout
    
    def _create_center_layout(self):
        """Create the center layout with camera and controls"""
        center_layout = QVBoxLayout()
        
        # Top controls
        top_controls = self._create_top_controls()
        center_layout.addLayout(top_controls)
        center_layout.addSpacing(5)  # Noch weniger: 8 → 5
        
        # Camera display
        self.camera_label = QLabel()
        self.camera_label.setStyleSheet(
            "background-color: #000000; border: 1px solid #212124; border-radius: 8px;"
        )
        self.camera_label.setFixedSize(520, 260)  # Noch kleiner: 560x280 → 520x260
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setText("Click 'Detection: ON' to start camera")
        center_layout.addWidget(self.camera_label, 0, Qt.AlignHCenter)
        center_layout.addSpacing(10)  # Noch weniger: 15 → 10
        
        # Emotions section
        emotions_section = self._create_emotions_section()
        center_layout.addLayout(emotions_section)
        center_layout.addSpacing(8)  # Noch weniger: 12 → 8
        
        # Gestures section
        gestures_section = self._create_gestures_section()
        center_layout.addLayout(gestures_section)
        center_layout.addStretch()
        
        return center_layout
    
    def _create_top_controls(self):
        """Create top control bar"""
        top_controls = QHBoxLayout()
        
        # Logo
        self.logo_label = QLabel()
        if os.path.exists(self.logo_path):
            logo_pixmap = QPixmap(self.logo_path)
            scaled_logo = logo_pixmap.scaled(40, 40, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.logo_label.setPixmap(scaled_logo)
        else:
            self.logo_label.setText("🎭")
            self.logo_label.setStyleSheet("font-size: 30px;")
        self.logo_label.setFixedSize(40, 40)
        top_controls.addWidget(self.logo_label)
        
        top_controls.addSpacing(15)
        
        # Title
        title_label = QLabel("MOODYSTREAM")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; letter-spacing: 1px;")
        top_controls.addWidget(title_label)
        
        top_controls.addStretch()
        
        # Settings button
        self.settings_button = QPushButton("⚙ Settings")
        self.settings_button.setFixedSize(120, 32)  # Breiter: 100 → 120
        self.settings_button.clicked.connect(self.open_settings)
        top_controls.addWidget(self.settings_button)
        
        top_controls.addSpacing(10)
        
        # Detection toggle button
        self.emotion_detection_button = QPushButton("Detection: OFF")
        self.emotion_detection_button.setFixedSize(140, 32)
        self.emotion_detection_button.setCheckable(True)
        self.emotion_detection_button.setChecked(False)
        self.emotion_detection_button.clicked.connect(self.toggle_emotion_detection)
        self.emotion_detection_button.setStyleSheet("""
            QPushButton {
                background-color: #212124;
                color: #FFFFFF;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 11px;
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
        """)
        top_controls.addWidget(self.emotion_detection_button)
        
        return top_controls
    
    def _create_emotions_section(self):
        """Create emotions section"""
        emotions_layout = QVBoxLayout()
        
        # Header
        emotions_header = QHBoxLayout()
        emotions_label = QLabel("EMOTIONS")
        emotions_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        emotions_header.addWidget(emotions_label)
        emotions_header.addStretch()
        emotions_header.addSpacing(50)
        
        # Trigger Time Slider
        emotion_trigger_time_label = QLabel("Trigger Time")
        emotion_trigger_time_label.setStyleSheet("font-size: 11px;")
        emotions_header.addWidget(emotion_trigger_time_label)
        
        # Info icon for Trigger Time
        trigger_time_info = QLabel("ⓘ")
        trigger_time_info.setStyleSheet("""
            QLabel {
                color: #888888;
                font-size: 14px;
                background: transparent;
                border: none;
            }
            QLabel:hover {
                color: #FFFFFF;
            }
        """)
        trigger_time_info.setToolTip("How long should the bot wait to play a sound")
        trigger_time_info.setCursor(Qt.WhatsThisCursor)
        emotions_header.addWidget(trigger_time_info)
        
        emotions_header.addSpacing(5)
        
        self.emotion_trigger_slider = QSlider(Qt.Horizontal)
        self.emotion_trigger_slider.setMinimum(0)
        self.emotion_trigger_slider.setMaximum(100)
        self.emotion_trigger_slider.setValue(50)
        self.emotion_trigger_slider.setFixedWidth(120)
        emotions_header.addWidget(self.emotion_trigger_slider)
        
        emotions_header.addSpacing(20)
        
        # Sensitivity Slider
        sensitivity_label = QLabel("Sensitivity")
        sensitivity_label.setStyleSheet("font-size: 11px;")
        emotions_header.addWidget(sensitivity_label)
        
        # Info icon for Sensitivity
        sensitivity_info = QLabel("ⓘ")
        sensitivity_info.setStyleSheet("""
            QLabel {
                color: #888888;
                font-size: 14px;
                background: transparent;
                border: none;
            }
            QLabel:hover {
                color: #FFFFFF;
            }
        """)
        sensitivity_info.setToolTip("How confident should the bot be to play a sound")
        sensitivity_info.setCursor(Qt.WhatsThisCursor)
        emotions_header.addWidget(sensitivity_info)
        
        emotions_header.addSpacing(5)
        
        self.emotion_sensitivity_slider = QSlider(Qt.Horizontal)
        self.emotion_sensitivity_slider.setMinimum(0)
        self.emotion_sensitivity_slider.setMaximum(100)
        self.emotion_sensitivity_slider.setValue(50)
        self.emotion_sensitivity_slider.setFixedWidth(120)
        emotions_header.addWidget(self.emotion_sensitivity_slider)
        
        emotions_layout.addLayout(emotions_header)
        emotions_layout.addSpacing(5)
        
        # Emotion boxes
        emotion_names = ["Happy", "Surprise", "Sad", "Fear"]
        for i, name in enumerate(emotion_names):
            box = HoverBox(name)
            box.setMinimumWidth(775)
            emotions_layout.addWidget(box)
            if i < len(emotion_names) - 1: 
                emotions_layout.addSpacing(4)
        
        return emotions_layout
    
    def _create_gestures_section(self):
        """Create gestures section"""
        gestures_layout = QVBoxLayout()
        
        # Header
        gestures_header = QHBoxLayout()
        gestures_label = QLabel("GESTURES")
        gestures_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        gestures_header.addWidget(gestures_label)
        gestures_header.addStretch()
        gestures_header.addSpacing(50)
        
        # Sensitivity Slider
        gesture_sensitivity_label = QLabel("Sensitivity")
        gesture_sensitivity_label.setStyleSheet("font-size: 11px;")
        gestures_header.addWidget(gesture_sensitivity_label)
        
        # Info icon for Sensitivity
        gesture_sensitivity_info = QLabel("ⓘ")
        gesture_sensitivity_info.setStyleSheet("""
            QLabel {
                color: #888888;
                font-size: 14px;
                background: transparent;
                border: none;
            }
            QLabel:hover {
                color: #FFFFFF;
            }
        """)
        gesture_sensitivity_info.setToolTip("How confident should the bot be to play a sound")
        gesture_sensitivity_info.setCursor(Qt.WhatsThisCursor)
        gestures_header.addWidget(gesture_sensitivity_info)
        
        gestures_header.addSpacing(5)
        
        self.gesture_sensitivity_slider = QSlider(Qt.Horizontal)
        self.gesture_sensitivity_slider.setMinimum(0)
        self.gesture_sensitivity_slider.setMaximum(100)
        self.gesture_sensitivity_slider.setValue(50)
        self.gesture_sensitivity_slider.setFixedWidth(120)
        gestures_header.addWidget(self.gesture_sensitivity_slider)
        
        gestures_layout.addLayout(gestures_header)
        gestures_layout.addSpacing(5)
        
        # Gesture boxes
        gesture_names = ["Thumbs up", "Thumbs down", "Peace"]
        for i, name in enumerate(gesture_names):
            box = HoverBox(name)
            box.setMinimumWidth(775)
            gestures_layout.addWidget(box)
            if i < len(gesture_names) - 1:  
                gestures_layout.addSpacing(4)
        
        return gestures_layout
    
    def start_detection_thread(self):
        """Start detection in a separate thread with custom frame handling"""
        if self.detection_running:
            print("⚠ Detection already running")
            return
        
        self.detection_running = True
        
        def detection_worker():
            import time
            try:
                print(f"🎥 Starting camera with index {self.camera_index}")
                
                # Import detection functions
                from detection.emotion_recognition import EmotionRecognition
                from detection.face_detection import detect_faces
                from detection.gesture_recognition import detect_gestures
                
                cap = cv2.VideoCapture(self.camera_index)
                if not cap.isOpened():
                    print(f"❌ ERROR: Could not open camera {self.camera_index}")
                    raise RuntimeError(f"Camera index {self.camera_index} could not be opened.")
                
                print("✅ Camera opened successfully")
                
                # Set camera to 60 FPS if supported
                cap.set(cv2.CAP_PROP_FPS, 60)
                actual_fps = cap.get(cv2.CAP_PROP_FPS)
                print(f"📹 Camera FPS: {actual_fps}")
                
                er = EmotionRecognition(threshold=10)
                frame_count = 0
                
                # Store last known face positions
                last_faces = []
                last_gestures = []
                
                print("🔄 Starting frame capture loop...")
                
                while self.detection_running:
                    ret, frame = cap.read()
                    if not ret:
                        print("❌ Failed to read frame")
                        break
                    
                    frame_count += 1
                    
                    # Debug: Print every 30 frames
                    if frame_count % 30 == 0:
                        print(f"📸 Frame {frame_count} captured, shape: {frame.shape}")
                    
                    # Run face detection every 3rd frame (save performance)
                    if frame_count % 3 == 0:
                        try:
                            last_faces = detect_faces(frame)
                            
                            if last_faces and frame_count % 30 == 0:
                                print(f"😊 Detected {len(last_faces)} face(s)")
                        except Exception as e:
                            print(f"⚠️ Face detection error: {e}")
                    
                    # Run gesture detection every 5th frame
                    if frame_count % 5 == 0:
                        try:
                            last_gestures = detect_gestures(frame)
                        except Exception as e:
                            print(f"⚠️ Gesture detection error: {e}")
                    
                    # Send frame to GUI for display
                    self._queue_frame_display(frame)
                    
                    # No sleep - let it run as fast as possible (60 FPS)
                
                cap.release()
                print("🛑 Camera released")
                
            except Exception as e:
                print(f"❌ Detection thread error: {e}")
                import traceback
                traceback.print_exc()
                self.detection_running = False
        
        self.detection_thread = threading.Thread(target=detection_worker, daemon=True)
        self.detection_thread.start()
        print("✅ Detection thread started")
    
    def _queue_frame_display(self, frame):
        """Queue a frame for display in the GUI thread (thread-safe)"""
        try:
            # Debug counter
            if not hasattr(self, '_frame_display_count'):
                self._frame_display_count = 0
            self._frame_display_count += 1
            
            # Print every 30 frames
            if self._frame_display_count % 30 == 0:
                print(f"🖼️  Queue frame #{self._frame_display_count} for display")
            
            # Convert frame in worker thread
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            
            if self._frame_display_count % 30 == 0:
                print(f"   Frame converted: {w}x{h}x{ch}")
            
            bytes_per_line = ch * w
            qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            # Make a copy so the data doesn't get freed
            qt_image = qt_image.copy()
            pixmap = QPixmap.fromImage(qt_image)
            
            if self._frame_display_count % 30 == 0:
                print(f"   Pixmap created: {pixmap.width()}x{pixmap.height()}")
            
            # Emit signal (automatically thread-safe!)
            self.frame_ready.emit(pixmap)
            
            if self._frame_display_count % 30 == 0:
                print(f"   ✅ Signal emitted")
                
        except Exception as e:
            print(f"❌ Frame display error: {e}")
            import traceback
            traceback.print_exc()
    
    @pyqtSlot(object)
    def _update_camera_display(self, pixmap):
        """Update camera display (called in main GUI thread)"""
        try:
            if not hasattr(self, '_gui_update_count'):
                self._gui_update_count = 0
            self._gui_update_count += 1
            
            if self._gui_update_count % 30 == 0:
                print(f"🖥️  GUI Update #{self._gui_update_count}")
                print(f"   Received pixmap: {pixmap.width()}x{pixmap.height()}")
                print(f"   Label size: {self.camera_label.width()}x{self.camera_label.height()}")
            
            scaled_pixmap = pixmap.scaled(
                self.camera_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            
            if self._gui_update_count % 30 == 0:
                print(f"   Scaled to: {scaled_pixmap.width()}x{scaled_pixmap.height()}")
            
            self.camera_label.setPixmap(scaled_pixmap)
            
            if self._gui_update_count % 30 == 0:
                print(f"   ✅ Pixmap set on label!")
                
        except Exception as e:
            print(f"❌ Display update error: {e}")
            import traceback
            traceback.print_exc()
    
    def stop_detection_internal(self):
        """Internal method to stop detection"""
        self.detection_running = False
        
        # Wait for detection thread to finish
        if self.detection_thread and self.detection_thread.is_alive():
            print("⏳ Waiting for detection thread to stop...")
            self.detection_thread.join(timeout=2)
        
        print("🛑 Detection stopped")
    
    def toggle_emotion_detection(self):
        """Toggle emotion detection on/off"""
        self.emotion_detection_active = self.emotion_detection_button.isChecked()
        if self.emotion_detection_active:
            self.emotion_detection_button.setText("Detection: ON")
            print("✅ Starting detection...")
            # Start detection when button is turned ON
            if DETECTION_AVAILABLE and not self.detection_running:
                try:
                    self.start_detection_thread()
                except Exception as e:
                    print(f"❌ Could not start detection: {e}")
                    import traceback
                    traceback.print_exc()
                    self.emotion_detection_button.setChecked(False)
                    self.emotion_detection_active = False
                    self.emotion_detection_button.setText("Detection: OFF")
        else:
            self.emotion_detection_button.setText("Detection: OFF")
            print("⏸ Stopping detection...")
            # Stop detection
            self.stop_detection_internal()
    
    def open_settings(self):
        """Open settings dialog"""
        dialog = SettingsDialog(self.camera_index, self)
        dialog.restart_setup_signal.connect(self.handle_restart_setup)
        
        if dialog.exec_() == QDialog.Accepted:
            new_camera_index = dialog.get_camera_index()
            if new_camera_index != self.camera_index:
                print(f"Camera index changed to {new_camera_index}")
    
    def handle_restart_setup(self):
        """Handle restart setup signal from settings dialog"""
        print("Restart Setup wird durchgeführt...")
    
    def closeEvent(self, event):
        """Clean shutdown when window closes"""
        print("🛑 Closing MainWindow...")
        self.stop_detection_internal()
        event.accept()
        print("✅ MainWindow closed cleanly")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())