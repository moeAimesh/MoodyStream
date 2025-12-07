import sys
import cv2
import pygame
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QScrollArea, QSlider,
                             QDialog, QSpinBox, QMenu, QFileDialog, QMessageBox)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage

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
        
        self.setMinimumHeight(50)
        self.setMaximumHeight(50)
        self.setCursor(Qt.PointingHandCursor)
        
        # Set transparent background for the main widget
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
                padding: 4px 12px;
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
        """)
        self.play_button.setFixedHeight(30)
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
                padding: 4px 12px;
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
        """)
        self.sound_button.setFixedHeight(30)
        self.sound_button.clicked.connect(self.show_sound_menu)
        layout.addWidget(self.sound_button)
        
        layout.addSpacing(25)
    
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
        
        # Show menu below the button
        menu.exec_(self.sound_button.mapToGlobal(self.sound_button.rect().bottomLeft()))
    
    def browse_web(self):
        """Handle Browse Web option"""
        print(f"Browse Web selected for {self.text}")
        # Hier kann später die Web-Browser-Funktionalität implementiert werden
        self.sound_clicked.emit()
    
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
            # Optional: Update button text to show file is selected
            import os
            filename = os.path.basename(file_path)
            self.sound_button.setToolTip(f"Selected: {filename}")
    
    def on_play_button_clicked(self):
        if self.selected_sound_file:
            try:
                # Load and play the sound
                pygame.mixer.music.load(self.selected_sound_file)
                pygame.mixer.music.play()
                print(f"Playing sound: {self.selected_sound_file}")
            except Exception as e:
                print(f"Error playing sound: {e}")
                # Show error message to user
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setWindowTitle("Sound Playback Error")
                msg.setText(f"Could not play sound file:\n{str(e)}")
                msg.setStyleSheet("""
                    QMessageBox {
                        background-color: #161618;
                        color: #FFFFFF;
                    }
                    QMessageBox QLabel {
                        color: #FFFFFF;
                    }
                    QMessageBox QPushButton {
                        background-color: #212124;
                        color: #FFFFFF;
                        border: none;
                        border-radius: 6px;
                        padding: 6px 12px;
                        min-width: 70px;
                    }
                    QMessageBox QPushButton:hover {
                        background: qlineargradient(
                            x1:0, y1:0, x2:1, y2:1,
                            stop:0 rgba(255, 107, 74, 0.3),
                            stop:0.5 rgba(255, 59, 143, 0.3),
                            stop:1 rgba(255, 105, 180, 0.3)
                        );
                    }
                """)
                msg.exec_()
        else:
            print(f"No sound file selected for {self.text}")
            # Show info message to user
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setWindowTitle("No Sound Selected")
            msg.setText("Please select a sound file first using 'Choose Sound'.")
            msg.setStyleSheet("""
                QMessageBox {
                    background-color: #161618;
                    color: #FFFFFF;
                }
                QMessageBox QLabel {
                    color: #FFFFFF;
                }
                QMessageBox QPushButton {
                    background-color: #212124;
                    color: #FFFFFF;
                    border: none;
                    border-radius: 6px;
                    padding: 6px 12px;
                    min-width: 70px;
                }
                QMessageBox QPushButton:hover {
                    background: qlineargradient(
                        x1:0, y1:0, x2:1, y2:1,
                        stop:0 rgba(255, 107, 74, 0.3),
                        stop:0.5 rgba(255, 59, 143, 0.3),
                        stop:1 rgba(255, 105, 180, 0.3)
                    );
                }
            """)
            msg.exec_()
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
        # Update button colors when hovering over box (but not the button itself)
        self.play_button.setStyleSheet("""
            QPushButton {
                background-color: #3a3a3d;
                color: #FFFFFF;
                border: none;
                border-radius: 4px;
                padding: 4px 12px;
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
        """)
        self.sound_button.setStyleSheet("""
            QPushButton {
                background-color: #3a3a3d;
                color: #FFFFFF;
                border: none;
                border-radius: 4px;
                padding: 4px 12px;
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
        """)
        super().enterEvent(event)
        
    def leaveEvent(self, event):
        self.setStyleSheet("""
            HoverBox {
                background-color: #161618;
                border: none;
                border-radius: 8px;
            }
        """)
        # Reset button colors to default
        self.play_button.setStyleSheet("""
            QPushButton {
                background-color: #2a2a2d;
                color: #FFFFFF;
                border: none;
                border-radius: 4px;
                padding: 4px 12px;
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
        """)
        self.sound_button.setStyleSheet("""
            QPushButton {
                background-color: #2a2a2d;
                color: #FFFFFF;
                border: none;
                border-radius: 4px;
                padding: 4px 12px;
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
        """)
        super().leaveEvent(event)
        
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            # Check if click was on the buttons
            if not self.sound_button.geometry().contains(event.pos()) and not self.play_button.geometry().contains(event.pos()):
                self.clicked.emit()
        super().mousePressEvent(event)

class SettingsDialog(QDialog):
    """Settings dialog for camera configuration"""
    
    restart_setup_signal = pyqtSignal()
    
    def __init__(self, current_camera_index, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setModal(True)
        self.setFixedSize(400, 200)
        
        # Apply the same dark theme
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
        """)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Camera Index Setting
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
        
        # Restart Setup Button
        restart_layout = QHBoxLayout()
        restart_button = QPushButton("Restart Setup")
        restart_button.clicked.connect(self.restart_setup)
        restart_layout.addWidget(restart_button)
        restart_layout.addStretch()
        
        layout.addLayout(restart_layout)
        
        # Buttons
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
    
    def restart_setup(self):
        """Emit signal to restart setup"""
        self.restart_setup_signal.emit()
        print("Restart Setup clicked - Setup wird neu gestartet")
        # Hier kann später die Logik zum Neustarten des Setups implementiert werden

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Moodystream")
        self.setGeometry(100, 100, 1400, 800)
        
        self.setStyleSheet("""
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
        """)
        
        # Initialize camera
        self.camera_index = 0
        self.cap = cv2.VideoCapture(self.camera_index)
        self.emotion_detection_active = True
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
       # left ad mockups (2 stacked)
        left_ads = QVBoxLayout()
        self.left_ad1 = QLabel()
        self.left_ad1.setStyleSheet("border: 1px solid #212124; border-radius: 8px;")
        self.left_ad1.setAlignment(Qt.AlignCenter)
        self.left_ad1.setMinimumSize(220, 280)  
        self.left_ad1.setScaledContents(True)
        ad_pixmap1 = QPixmap("/Users/juliamoor/Desktop/MoodyStream/gui/ad_mockup.jpg")
        self.left_ad1.setPixmap(ad_pixmap1)
        
        self.left_ad2 = QLabel()
        self.left_ad2.setStyleSheet("border: 1px solid #212124; border-radius: 8px;")
        self.left_ad2.setAlignment(Qt.AlignCenter)
        self.left_ad2.setMinimumSize(220, 280) 
        self.left_ad2.setScaledContents(True)
        ad_pixmap2 = QPixmap("/Users/juliamoor/Desktop/MoodyStream/gui/ad_mockup.jpg")
        self.left_ad2.setPixmap(ad_pixmap2)
        
        left_ads.addWidget(self.left_ad1)
        left_ads.addWidget(self.left_ad2)
        left_ads.addStretch()
        
        # --- CENTER CONTENT ---
        center_layout = QVBoxLayout()
        
        # Top controls (Logo, Title, Settings and Emotion Detection toggle)
        top_controls = QHBoxLayout()
        
        # Logo
        self.logo_label = QLabel()
        logo_pixmap = QPixmap("/Users/juliamoor/Desktop/MoodyStream/gui/moody_logo.jpg")
        scaled_logo = logo_pixmap.scaled(40, 40, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.logo_label.setPixmap(scaled_logo)
        self.logo_label.setFixedSize(40, 40)
        top_controls.addWidget(self.logo_label)
        
        top_controls.addSpacing(15)
        
        # Title
        title_label = QLabel("MOODYSTREAM")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; letter-spacing: 1px;")
        top_controls.addWidget(title_label)
        
        top_controls.addStretch()
        
        self.settings_button = QPushButton("⚙ Settings")
        self.settings_button.setFixedSize(100, 32)
        self.settings_button.clicked.connect(self.open_settings)
        self.settings_button.setStyleSheet("""
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
        """)
        
        top_controls.addSpacing(10)
        
        self.emotion_detection_button = QPushButton("Detection: ON")
        self.emotion_detection_button.setFixedSize(140, 32)
        self.emotion_detection_button.setCheckable(True)
        self.emotion_detection_button.setChecked(True)
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
        
        top_controls.addWidget(self.settings_button)
        top_controls.addWidget(self.emotion_detection_button)
        
        center_layout.addLayout(top_controls)
        center_layout.addSpacing(10)
        
        # Camera display (verkleinert auf 320px Höhe)
        self.camera_label = QLabel()
        self.camera_label.setStyleSheet("background-color: #000000; border: 1px solid #212124; border-radius: 8px;")
        self.camera_label.setFixedSize(640, 320)
        self.camera_label.setAlignment(Qt.AlignCenter)
        center_layout.addWidget(self.camera_label, 0, Qt.AlignHCenter)
        center_layout.addSpacing(20)
        
        # --- EMOTIONS SECTION ---
        emotions_header = QHBoxLayout()
        
        emotions_label = QLabel("EMOTIONS")
        emotions_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        emotions_header.addWidget(emotions_label)
        
        emotions_header.addStretch()
        
        # Sensitivity Slider
        sensitivity_label = QLabel("Sensitivity")
        sensitivity_label.setStyleSheet("font-size: 11px;")
        emotions_header.addWidget(sensitivity_label)
        
        self.emotion_sensitivity_slider = QSlider(Qt.Horizontal)
        self.emotion_sensitivity_slider.setMinimum(0)
        self.emotion_sensitivity_slider.setMaximum(100)
        self.emotion_sensitivity_slider.setValue(50)
        self.emotion_sensitivity_slider.setFixedWidth(150)
        emotions_header.addWidget(self.emotion_sensitivity_slider)
        
        emotions_header.addSpacing(325)
        
        center_layout.addLayout(emotions_header)
        
        emotions_layout = QHBoxLayout()
        
        # Container für Emotion Buttons
        emotion_container = QWidget()
        emotion_layout = QVBoxLayout(emotion_container)
        emotion_layout.setSpacing(8)
        emotion_layout.setContentsMargins(0, 0, 0, 0)
        
        # Emotion Boxes
        emotion_names = ["Happy", "Surprise", "Sad", "Fear"]
        for name in emotion_names:
            box = HoverBox(name)
            box.setMinimumWidth(775)
            emotion_layout.addWidget(box)
        
        # Scroll Area
        self.emotion_scroll = QScrollArea()
        self.emotion_scroll.setWidget(emotion_container)
        self.emotion_scroll.setWidgetResizable(False)
        self.emotion_scroll.setFixedHeight(150)
        self.emotion_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.emotion_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        # Slider
        self.emotion_slider = QSlider(Qt.Vertical)
        self.emotion_slider.setMinimum(0)
        self.emotion_slider.setMaximum(100)
        self.emotion_slider.setValue(0)
        self.emotion_slider.setFixedHeight(150)
        self.emotion_slider.setInvertedAppearance(True)
        self.emotion_slider.valueChanged.connect(self.emotion_slider_changed)
        
        emotions_layout.addWidget(self.emotion_scroll)
        emotions_layout.addWidget(self.emotion_slider)
        
        center_layout.addLayout(emotions_layout)
        center_layout.addSpacing(20)
        
        # --- GESTURES SECTION ---
        gestures_header = QHBoxLayout()
        
        gestures_label = QLabel("GESTURES")
        gestures_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        gestures_header.addWidget(gestures_label)
        
        gestures_header.addStretch()
        
        # Sensitivity Slider (jetzt zuerst)
        gesture_sensitivity_label = QLabel("Sensitivity")
        gesture_sensitivity_label.setStyleSheet("font-size: 11px;")
        gestures_header.addWidget(gesture_sensitivity_label)
        
        self.gesture_sensitivity_slider = QSlider(Qt.Horizontal)
        self.gesture_sensitivity_slider.setMinimum(0)
        self.gesture_sensitivity_slider.setMaximum(100)
        self.gesture_sensitivity_slider.setValue(50)
        self.gesture_sensitivity_slider.setFixedWidth(150)
        gestures_header.addWidget(self.gesture_sensitivity_slider)
        
        gestures_header.addSpacing(20)
        
        # Trigger Time Slider (jetzt zweites)
        gesture_trigger_time_label = QLabel("Trigger Time")
        gesture_trigger_time_label.setStyleSheet("font-size: 11px;")
        gestures_header.addWidget(gesture_trigger_time_label)
        
        self.gesture_trigger_slider = QSlider(Qt.Horizontal)
        self.gesture_trigger_slider.setMinimum(0)
        self.gesture_trigger_slider.setMaximum(100)
        self.gesture_trigger_slider.setValue(50)
        self.gesture_trigger_slider.setFixedWidth(150)
        gestures_header.addWidget(self.gesture_trigger_slider)
        
        gestures_header.addSpacing(50)
        
        center_layout.addLayout(gestures_header)
        
        gestures_layout = QHBoxLayout()
        
        # Container für Gesture Buttons
        gesture_container = QWidget()
        gesture_layout = QVBoxLayout(gesture_container)
        gesture_layout.setSpacing(8)
        gesture_layout.setContentsMargins(0, 0, 0, 0)
        
        # Gesture Boxes
        gesture_names = ["Thumbs up", "Thumbs down", "Peace", "Open Hand"]
        for name in gesture_names:
            box = HoverBox(name)
            box.setMinimumWidth(775)
            gesture_layout.addWidget(box)
        
        # Scroll Area
        self.gesture_scroll = QScrollArea()
        self.gesture_scroll.setWidget(gesture_container)
        self.gesture_scroll.setWidgetResizable(False)
        self.gesture_scroll.setFixedHeight(150)
        self.gesture_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.gesture_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        # Slider
        self.gesture_slider = QSlider(Qt.Vertical)
        self.gesture_slider.setMinimum(0)
        self.gesture_slider.setMaximum(100)
        self.gesture_slider.setValue(0)
        self.gesture_slider.setFixedHeight(150)
        self.gesture_slider.setInvertedAppearance(True)
        self.gesture_slider.valueChanged.connect(self.gesture_slider_changed)
        
        gestures_layout.addWidget(self.gesture_scroll)
        gestures_layout.addWidget(self.gesture_slider)
        
        center_layout.addLayout(gestures_layout)
        center_layout.addStretch()
        
        
        # right adds (stacked over each other)
        right_ads = QVBoxLayout()
        self.right_ad1 = QLabel()
        self.right_ad1.setStyleSheet("border: 1px solid #212124; border-radius: 8px;")
        self.right_ad1.setAlignment(Qt.AlignCenter)
        self.right_ad1.setMinimumSize(220, 280)
        self.right_ad1.setScaledContents(True)
        ad_pixmap3 = QPixmap("/Users/juliamoor/Desktop/MoodyStream/gui/ad_mockup.jpg")
        self.right_ad1.setPixmap(ad_pixmap3)
        
        self.right_ad2 = QLabel()
        self.right_ad2.setStyleSheet("border: 1px solid #212124; border-radius: 8px;")
        self.right_ad2.setAlignment(Qt.AlignCenter)
        self.right_ad2.setMinimumSize(220, 280)
        self.right_ad2.setScaledContents(True)
        ad_pixmap4 = QPixmap("/Users/juliamoor/Desktop/MoodyStream/gui/ad_mockup.jpg")
        self.right_ad2.setPixmap(ad_pixmap4)
        
        right_ads.addWidget(self.right_ad1)
        right_ads.addWidget(self.right_ad2)
        right_ads.addStretch()
    
        # adding everything to main layout
        main_layout.addLayout(left_ads, 1)
        main_layout.addLayout(center_layout, 3)
        main_layout.addLayout(right_ads, 1)
        
        # Timer for camera updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)
    
    def toggle_emotion_detection(self):
        """Toggle emotion detection on/off"""
        self.emotion_detection_active = self.emotion_detection_button.isChecked()
        if self.emotion_detection_active:
            self.emotion_detection_button.setText("Detection: ON")
        else:
            self.emotion_detection_button.setText("Detection: OFF")
    
    def open_settings(self):
        """Open settings dialog"""
        dialog = SettingsDialog(self.camera_index, self)
        dialog.restart_setup_signal.connect(self.handle_restart_setup)
        
        if dialog.exec_() == QDialog.Accepted:
            new_camera_index = dialog.get_camera_index()
            if new_camera_index != self.camera_index:
                # Release old camera
                self.cap.release()
                # Set new camera index
                self.camera_index = new_camera_index
                # Initialize new camera
                self.cap = cv2.VideoCapture(self.camera_index)
    
    def handle_restart_setup(self):
        """Handle restart setup signal from settings dialog"""
        print("Restart Setup wird durchgeführt...")
        # Hier kann die Logik zum Neustarten des Setups implementiert werden
        # Zum Beispiel: Alle Einstellungen zurücksetzen, Kamera neu initialisieren, etc.
    
    def emotion_slider_changed(self, value):
        scrollbar = self.emotion_scroll.verticalScrollBar()
        max_scroll = scrollbar.maximum()
        scroll_position = int((value / 100.0) * max_scroll)
        scrollbar.setValue(scroll_position)
    
    def gesture_slider_changed(self, value):
        scrollbar = self.gesture_scroll.verticalScrollBar()
        max_scroll = scrollbar.maximum()
        scroll_position = int((value / 100.0) * max_scroll)
        scrollbar.setValue(scroll_position)
    
    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            scaled_pixmap = pixmap.scaled(self.camera_label.size(), 
                                         Qt.KeepAspectRatio,
                                         Qt.SmoothTransformation)
            self.camera_label.setPixmap(scaled_pixmap)
    
    def closeEvent(self, event):
        self.cap.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())