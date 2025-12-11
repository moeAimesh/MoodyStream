import sys
import os
import traceback
import cv2
import pygame
import threading
import json
import time
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QSlider,
                             QDialog, QSpinBox, QMenu, QFileDialog)
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot, QUrl
from PyQt5.QtGui import QPixmap, QImage, QDesktopServices
from pathlib import Path
from sounds.play_sound import play as play_sound 
from utils.json_manager import save_json, update_json

SOUND_CACHE_DIR = Path("sounds/sound_cache")
SETUP_CONFIG_PATH = Path("setup_config.json")
SOUND_MAP_PATH = Path("sounds/sound_map.json")
SOUND_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# import camera_stream
try:
    from detection.camera_stream import start_detection
    DETECTION_AVAILABLE = True
except ImportError as e:
    DETECTION_AVAILABLE = False
    print(f"Could not import detection module: {e}")


# match gesture button text to gesture detection
def normalize_gesture_name(gesture_name: str) -> str:
    """
    Normalize gesture names to match the detection format.
    
    Examples:
        'Thumbs up' -> 'thumbsup'
        'Thumbs down' -> 'thumbsdown'
        'Peace' -> 'peace'
    """
    return gesture_name.lower().replace(" ", "")

class HoverBox(QWidget):
    """Custom widget that changes color on hover"""
    clicked = pyqtSignal()
    sound_clicked = pyqtSignal()
    play_clicked = pyqtSignal()
    
    def __init__(self, text, parent=None):
        super().__init__(parent)
        self.text = text
        self.emotion_key = text.lower()  # Store normalized emotion key
        self.hovered = False
        self.selected_sound_file = None
        
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        
        self._load_sound_from_config()
        
        self.setMinimumHeight(38)
        self.setMaximumHeight(38)  
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
        self.play_button.setFixedHeight(26)
        self.play_button.clicked.connect(self.on_play_button_clicked)
        layout.addWidget(self.play_button)
        
        layout.addSpacing(10)
        
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
        self.sound_button.setFixedHeight(26)
        self.sound_button.clicked.connect(self.show_sound_menu)
        layout.addWidget(self.sound_button)
        
        layout.addSpacing(25)
    
    def _load_sound_from_config(self):
        """Load sound from setup_config.json if available."""
        try:
            if SETUP_CONFIG_PATH.exists():
                with open(SETUP_CONFIG_PATH, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    sounds = config.get("sounds", {})
                    
                    if self.emotion_key in sounds:
                        sound_path = sounds[self.emotion_key]
                        
                        # convert relative path to absolute if needed
                        if not os.path.isabs(sound_path):
                            # try relative to main_window.py directory
                            abs_path = os.path.join(os.path.dirname(__file__), sound_path)
                            if not os.path.exists(abs_path):
                                # try relative to project root
                                abs_path = os.path.join(os.getcwd(), sound_path)
                            sound_path = abs_path
                        
                        if os.path.exists(sound_path):
                            self.selected_sound_file = sound_path
                            filename = os.path.basename(sound_path)
                            self.sound_button.setToolTip(f"Selected: {filename}")
                            print(f"✅ Loaded sound for '{self.emotion_key}': {filename}")
                        else:
                            print(f"⚠️ Sound file not found: {sound_path}")
        except Exception as e:
            print(f"⚠️ Could not load sound config: {e}")
    
    def _save_sound_to_config(self, sound_path: str):
        """
        Save sound mapping to both sound_map.json and setup_config.json.
        Uses utils.json_manager functions - EXACTLY like sound_setup.py!
        """
        try:
            # get just the filename
            filename = os.path.basename(sound_path)
            
            rel_path = "sounds/sound_cache/" + filename
            
            print(f"\n=== Saving Sound Mapping ===")
            print(f"Emotion key: {self.emotion_key}")
            print(f"Filename: {filename}")
            print(f"Relative path: {rel_path}")
            
            # save to sound_map.json using save_json from utils.json_manager
            save_json(str(SOUND_MAP_PATH), self.emotion_key, rel_path)
            print(f"✅ Updated sound_map.json: {self.emotion_key} -> {rel_path}")
            
            # update setup_config.json using update_json from utils.json_manager
            update_json(str(SETUP_CONFIG_PATH), "sounds", {self.emotion_key: rel_path})
            print(f"✅ Updated setup_config.json: sounds.{self.emotion_key} -> {rel_path}")
            
            print(f"✅ Sound successfully mapped: '{filename}' to '{self.emotion_key}'")
            print("=== Mapping Complete ===\n")
            
        except Exception as e:
            print(f"❌ Could not save sound config: {e}")
            traceback.print_exc()
    
    def show_sound_menu(self):
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
        """Open myinstants.com in browser."""
        self.sound_clicked.emit()
        QDesktopServices.openUrl(QUrl("https://www.myinstants.com"))
    
    def browse_computer(self):
        """Browse for sound file and save it to sound_cache - EXACTLY like sound_setup.py"""
        file_dialog = QFileDialog(self)
        file_path, _ = file_dialog.getOpenFileName(
            self,
            "Select Sound File",
            "",
            "Audio Files (*.mp3 *.wav *.ogg *.m4a);;All Files (*.*)"
        )
        
        if not file_path:
            return
        
        try:
            # Get filename
            original_filename = os.path.basename(file_path)
            target_path = SOUND_CACHE_DIR / original_filename
            
            print(f"\n=== Sound Selection ===")
            print(f"Selected file: {file_path}")
            print(f"Target path: {target_path}")
            print(f"Emotion: {self.text} (key: {self.emotion_key})")
            
            # Copy file to sound_cache if not already there
            if not target_path.exists() or os.path.abspath(file_path) != os.path.abspath(target_path):
                import shutil
                shutil.copy2(file_path, target_path)
                file_size_kb = target_path.stat().st_size / 1024
                print(f"✅ Copied sound to cache ({file_size_kb:.1f} KB)")
            else:
                print(f"✅ Sound already in cache")
            
            # Save the ABSOLUTE path for internal use
            self.selected_sound_file = str(target_path.resolve())
            
            # Update tooltip
            self.sound_button.setToolTip(f"Selected: {original_filename}")
            
            # Save to config files (with RELATIVE path like sound_setup.py)
            self._save_sound_to_config(str(target_path))
            
            print(f"✅ Sound assigned to '{self.text}'")
            print("=== Selection Complete ===\n")
            
        except Exception as e:
            print(f"❌ Error saving sound: {e}")
            import traceback
            traceback.print_exc()
    
    def on_play_button_clicked(self):
        """Play the selected sound using sounds.play_sound.play()"""
        if self.selected_sound_file and os.path.exists(self.selected_sound_file):
            try:
                print(f"🎵 Playing sound: {os.path.basename(self.selected_sound_file)}")
                # use sounds.play_sound.play() from sound.play_sound
                play_sound(self.selected_sound_file, volume=1.0)
            except Exception as e:
                print(f"❌ Error playing sound: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"⚠️ No sound file selected for {self.text}")
            if self.selected_sound_file:
                print(f"   Path: {self.selected_sound_file}")
                print(f"   Exists: {os.path.exists(self.selected_sound_file)}")
        
        self.play_clicked.emit()
        
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
    restart_setup_signal = pyqtSignal()
    
    def __init__(self, current_camera_index, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setModal(True)
        self.setFixedSize(400, 280)
        self._restart_setup_requested = False
        
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
        
        volume_layout = QHBoxLayout()
        volume_label = QLabel("Volume:")
        volume_layout.addWidget(volume_label)
        
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setMinimum(0)
        self.volume_slider.setMaximum(100)
        self.volume_slider.setValue(50)
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
        self.restart_button = QPushButton("Restart Setup")
        self.restart_button.clicked.connect(self.restart_setup)
        restart_layout.addWidget(self.restart_button)
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
        self.volume_value_label.setText(f"{value}%")
        pygame.mixer.music.set_volume(value / 100.0)
    
    def restart_setup(self):
        self._restart_setup_requested = True
        self.restart_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(255, 107, 74, 0.5),
                    stop:0.5 rgba(255, 59, 143, 0.5),
                    stop:1 rgba(255, 105, 180, 0.5)
                );
                color: #FFFFFF;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 13px;
                min-width: 80px;
            }
        """)
    
    def is_restart_setup_requested(self):
        return self._restart_setup_requested


class MainWindow(QMainWindow):
    """Main window with full UI controls"""
    
    frame_ready = pyqtSignal(object)
    restart_setup_signal = pyqtSignal()
    
    def __init__(self, camera_index=0):
        super().__init__()
        
        self.setWindowTitle("Moodystream")
        self.setGeometry(100, 100, 1400, 800)
        
        self.setAttribute(Qt.WA_QuitOnClose, True)
        self._is_closing = False
        
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.logo_path = os.path.join(self.base_dir, "moody_logo.jpg")
        self.ad_mockup_path = os.path.join(self.base_dir, "ad_mockup.jpg")
        
        self.setStyleSheet(self._get_main_stylesheet())
        
        # Detection state
        self.camera_index = camera_index
        self.detection_running = False
        self.detection_thread = None
        self.stop_event = None
        self.emotion_detection_active = False
        
        # Initialize pygame
        if not pygame.mixer.get_init():
            pygame.mixer.init()
            pygame.mixer.music.set_volume(0.5)
        
        self.frame_ready.connect(self._update_camera_display)
        
        self._setup_ui()
    
    def _get_main_stylesheet(self):
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
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        left_ads = self._create_ad_column()
        center_layout = self._create_center_layout()
        right_ads = self._create_ad_column()
        
        main_layout.addLayout(left_ads, 1)
        main_layout.addLayout(center_layout, 3)
        main_layout.addLayout(right_ads, 1)
    
    def _create_ad_column(self):
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
        center_layout = QVBoxLayout()
        
        top_controls = self._create_top_controls()
        center_layout.addLayout(top_controls)
        center_layout.addSpacing(5)
        
        self.camera_label = QLabel()
        self.camera_label.setStyleSheet(
            "background-color: #000000; border: 1px solid #212124; border-radius: 8px;"
        )
        self.camera_label.setFixedSize(520, 260)
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setText("Click 'Detection: ON' to start camera")
        center_layout.addWidget(self.camera_label, 0, Qt.AlignHCenter)
        center_layout.addSpacing(10)
        
        emotions_section = self._create_emotions_section()
        center_layout.addLayout(emotions_section)
        center_layout.addSpacing(8)
        
        gestures_section = self._create_gestures_section()
        center_layout.addLayout(gestures_section)
        center_layout.addStretch()
        
        return center_layout
    
    def _create_top_controls(self):
        top_controls = QHBoxLayout()
        
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
        
        title_label = QLabel("MOODYSTREAM")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; letter-spacing: 1px;")
        top_controls.addWidget(title_label)
        
        top_controls.addStretch()
        
        self.settings_button = QPushButton("⚙ Settings")
        self.settings_button.setFixedSize(120, 32)
        self.settings_button.clicked.connect(self.open_settings)
        top_controls.addWidget(self.settings_button)
        
        top_controls.addSpacing(10)
        
        self.emotion_detection_button = QPushButton("Detection: OFF")
        self.emotion_detection_button.setFixedSize(140, 32)
        self.emotion_detection_button.setCheckable(True)
        self.emotion_detection_button.setChecked(False)
        self.emotion_detection_button.clicked.connect(self.toggle_emotion_detection)
        top_controls.addWidget(self.emotion_detection_button)
        
        return top_controls
    
    def _create_emotions_section(self):
        """Create emotions section with sliders and info icons"""
        emotions_layout = QVBoxLayout()
    
        # header
        emotions_header = QHBoxLayout()
        emotions_label = QLabel("EMOTIONS")
        emotions_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        emotions_header.addWidget(emotions_label)
        emotions_header.addStretch()
        emotions_header.addSpacing(50)
    
        emotion_trigger_time_label = QLabel("Trigger Time")
        emotion_trigger_time_label.setStyleSheet("font-size: 11px;")
        emotions_header.addWidget(emotion_trigger_time_label)
    
        trigger_time_info = QLabel("ℹ")
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
    
        sensitivity_label = QLabel("Sensitivity")
        sensitivity_label.setStyleSheet("font-size: 11px;")
        emotions_header.addWidget(sensitivity_label)
    
        sensitivity_info = QLabel("ℹ")
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
    
        # emotion boxes (hoverbox)
        emotion_names = ["Happy", "Surprise", "Sad", "Fear"]
        self.emotion_boxes = {}
    
        for i, name in enumerate(emotion_names):
            box = HoverBox(name)
            box.setMinimumWidth(775)
            emotions_layout.addWidget(box)
        
            # store with lowercase key for easy lookup
            self.emotion_boxes[name.lower()] = box
            
            if i < len(emotion_names) - 1: 
                emotions_layout.addSpacing(4)
    
        return emotions_layout

    def _create_gestures_section(self):
        """Create gestures section with slider and info icon"""
        gestures_layout = QVBoxLayout()
    
        # header
        gestures_header = QHBoxLayout()
        gestures_label = QLabel("GESTURES")
        gestures_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        gestures_header.addWidget(gestures_label)
        gestures_header.addStretch()
        gestures_header.addSpacing(50)
    
        gesture_sensitivity_label = QLabel("Sensitivity")
        gesture_sensitivity_label.setStyleSheet("font-size: 11px;")
        gestures_header.addWidget(gesture_sensitivity_label)
    
        gesture_sensitivity_info = QLabel("ℹ")
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
    
        # gesture boxes 
        gesture_names = ["Thumbs up", "Thumbs down", "Peace"]
        self.gesture_boxes = {} 
    
        for i, name in enumerate(gesture_names):
            box = HoverBox(name)
            box.setMinimumWidth(775)
            gestures_layout.addWidget(box)
        
            # store with normalized key to match what camera_stream is expecting
            normalized_key = normalize_gesture_name(name)
            self.gesture_boxes[normalized_key] = box
        
            if i < len(gesture_names) - 1:  
                gestures_layout.addSpacing(4)
    
        return gestures_layout
    
    # camera detection
    def start_detection_thread(self):
        if self.detection_running:
            return
        
        if not DETECTION_AVAILABLE:
            print("❌ camera_stream.py not available!")
            return
        
        self.detection_running = True
        self.stop_event = threading.Event()
        
        def detection_worker():
            try:
                def frame_callback(frame):
                    if self.detection_running:
                        try:
                            self._queue_frame_display(frame)
                        except Exception:
                            pass
                
                start_detection(
                    camera_index=self.camera_index,
                    show_window=False,
                    virtual_cam=False,
                    frame_callback=frame_callback,
                    stop_event=self.stop_event,
                    show_fps_plot=False,
                )
                
            except Exception as e:
                print(f"❌ Detection error: {e}")
            finally:
                self.detection_running = False
        
        self.detection_thread = threading.Thread(target=detection_worker, daemon=True)
        self.detection_thread.start()
    
    def _queue_frame_display(self, frame):
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            qt_image = qt_image.copy()
            pixmap = QPixmap.fromImage(qt_image)
            self.frame_ready.emit(pixmap)
        except Exception:
            pass
    
    @pyqtSlot(object)
    def _update_camera_display(self, pixmap):
        try:
            scaled_pixmap = pixmap.scaled(
                self.camera_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.camera_label.setPixmap(scaled_pixmap)
        except Exception:
            pass
    
    def stop_detection_internal(self):
        if not self.detection_running:
            return
        
        if self.stop_event:
            self.stop_event.set()
        
        self.detection_running = False
        
        if self.detection_thread and self.detection_thread.is_alive():
            for attempt in range(5):
                timeout = 1.0 + attempt
                self.detection_thread.join(timeout=timeout)
                
                if not self.detection_thread.is_alive():
                    break
        
        self.camera_label.clear()
        self.camera_label.setText("Click 'Detection: ON' to start camera")
        self.detection_thread = None
        self.stop_event = None
    
    def toggle_emotion_detection(self):
        self.emotion_detection_active = self.emotion_detection_button.isChecked()
        
        if self.emotion_detection_active:
            self.emotion_detection_button.setText("Detection: ON")
            
            if DETECTION_AVAILABLE and not self.detection_running:
                try:
                    self.start_detection_thread()
                except Exception as e:
                    print(f"❌ Failed to start detection: {e}")
                    self.emotion_detection_button.setChecked(False)
                    self.emotion_detection_active = False
                    self.emotion_detection_button.setText("Detection: OFF")
        else:
            self.emotion_detection_button.setText("Detection: OFF")
            self.stop_detection_internal()
    
    def open_settings(self):
        dialog = SettingsDialog(self.camera_index, self)
        
        if dialog.exec_() == QDialog.Accepted:
            new_camera_index = dialog.get_camera_index()
            
            # camera index debug
            if new_camera_index != self.camera_index:
                print(f"📷 Camera index changed: {self.camera_index} → {new_camera_index}")
                
                was_running = self.detection_running
                if was_running:
                    print("🔄 Restarting detection with new camera...")
                    self.stop_detection_internal()
                    time.sleep(1.0)
                
                self.camera_index = new_camera_index
                print(f"✅ Camera index updated to {new_camera_index}")
                
                if was_running:
                    QApplication.processEvents()
                    time.sleep(0.5)
                    
                    try:
                        self.start_detection_thread()
                        print("✅ Detection restarted with new camera")
                    except Exception as e:
                        print(f"❌ Failed to restart: {e}")
                        self.emotion_detection_button.setChecked(False)
                        self.emotion_detection_active = False
                        self.emotion_detection_button.setText("Detection: OFF")
            
            # setup restart
            if dialog.is_restart_setup_requested():
                if self.detection_running:
                    self.stop_detection_internal()
                    time.sleep(1.0)
                
                self.restart_setup_signal.emit()
    
    def closeEvent(self, event):
        if self._is_closing:
            event.accept()
            return
        
        self._is_closing = True
        
        if self.detection_running:
            self.stop_detection_internal()
            time.sleep(0.5)
        
        event.accept()
        QApplication.instance().quit()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())