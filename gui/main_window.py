"""Task: Window with tabs: Setup, Live, Settings.
Inputs: Events from Camera/Audio.
Outputs: Visual feedback, Buttons ("Restart Setup", "Switch Profile").
Tip: GUI in separate thread or async."""
import sys
import cv2
import webbrowser
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QDialog, 
                             QSpinBox, QSlider, QFormLayout, QMenuBar, QMenu,
                             QComboBox, QGridLayout, QLineEdit, QFileDialog,
                             QMessageBox, QScrollArea)
from PyQt5.QtCore import QTimer, Qt, QUrl
from PyQt5.QtGui import QImage, QPixmap, QCursor, QIcon
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent

class SoundButton(QPushButton):
    """Extended Button with Sound Functionality"""
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.sound_path = None
        self.player = QMediaPlayer()
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)
    
    def show_context_menu(self, position):
        """Shows context menu on right click"""
        menu = QMenu(self)
        
        # options in context menu
        search_action = menu.addAction("Search sound on web")
        load_action = menu.addAction("Load sound file")
        menu.addSeparator()
        
        if self.sound_path:
            play_action = menu.addAction("Play sound")
            remove_action = menu.addAction("Remove sound")
            info_action = menu.addAction(f"Current sound: {self.sound_path.split('/')[-1]}")
            info_action.setEnabled(False)
        
        # execute action
        action = menu.exec_(self.mapToGlobal(position))
        
        if action == search_action:
            self.open_sound_search()
        elif action == load_action:
            self.load_sound_file()
        elif self.sound_path and action == play_action:
            self.play_sound()
        elif self.sound_path and action == remove_action:
            self.remove_sound()
    
    def open_sound_search(self):
        """Opens MyInstants in browser"""
        url = "https://www.myinstants.com/en/search/?name=MEME"
        webbrowser.open(url)
        
        # show info dialog
        QMessageBox.information(
            self,
            "Sound Search",
            "Browser has been opened!\n\n"
            "How to link a sound:\n"
            "1. Search for a sound on MyInstants\n"
            "2. Download the sound (e.g., by emailing it to yourself)\n"
            "3. Return to our app and right-click on 'add sound'\n"
            "4. Select your downloaded sound"
        )
    
    def load_sound_file(self):
        """Loads a sound file from computer"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select sound file",
            "",
            "Audio files (*.mp3 *.wav *.ogg);;All files (*.*)"
        )
        
        if file_path:
            self.sound_path = file_path
            url = QUrl.fromLocalFile(file_path)
            self.player.setMedia(QMediaContent(url))
            self.player.setVolume(100)
    
    def play_sound(self):
        """Plays the linked sound"""
        if self.sound_path and self.player:
            self.player.stop()
            self.player.play()
    
    def remove_sound(self):
        """Removes the linked sound"""
        self.sound_path = None
        self.player.stop()
        self.player.setMedia(QMediaContent())
    
    def mousePressEvent(self, event):
        """Handles normal clicks (left) and plays sound"""
        if event.button() == Qt.LeftButton:
            if self.sound_path:
                self.play_sound()
        super().mousePressEvent(event)

class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(300)
        self.parent_window = parent
        
        layout = QFormLayout()
        
        # camera selection dropdown
        self.camera_combo = QComboBox()
        for i in range(4):
            self.camera_combo.addItem(f"Camera {i}", i)
        if parent:
            self.camera_combo.setCurrentIndex(parent.camera_index)
        self.camera_combo.currentIndexChanged.connect(self.change_camera)
        layout.addRow("Camera Index:", self.camera_combo)
        
        # restart setup button
        self.restart_setup_btn = QPushButton("Restart Setup")
        self.restart_setup_btn.clicked.connect(self.restart_setup)
        layout.addRow(self.restart_setup_btn)
        
        self.setLayout(layout)
    
    def change_camera(self, index):
        """Kamera sofort wechseln"""
        if self.parent_window:
            new_camera_index = self.camera_combo.currentData()
            if new_camera_index != self.parent_window.camera_index:
                self.parent_window.cap.release()
                self.parent_window.camera_index = new_camera_index
                self.parent_window.cap = cv2.VideoCapture(new_camera_index)
    
    def restart_setup(self):
        """Setup erneut starten"""
        QMessageBox.information(
            self,
            "Setup",
            "Setup will be resstarted one Julia remembers to implement it."
        )

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
            
            /* Sound Buttons mit Logo-Farbverlauf beim Hover */
            SoundButton {
                background-color: #212124;
                color: #FFFFFF;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 13px;
            }
            
            SoundButton:hover {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(255, 107, 74, 0.3),
                    stop:0.5 rgba(255, 59, 143, 0.3),
                    stop:1 rgba(255, 105, 180, 0.3)
                );
            }
            
            SoundButton:pressed {
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
        
        # initialising camera
        self.camera_index = 1
        self.cap = cv2.VideoCapture(self.camera_index)
        self.streaming = False
        
        # menu
        self.create_menu()
        
        # main widget and layout
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
        
        # middle area (Camera + Controls + Buttons)
        center_layout = QVBoxLayout()
        
        # controls center on top of camera
        controls_layout = QHBoxLayout()
        
        # Logo
        logo_label = QLabel()
        logo_label.setStyleSheet("background-color: transparent; border: none;")
        logo_label.setAlignment(Qt.AlignCenter)
        logo_label.setFixedSize(50, 50) 
        logo_label.setScaledContents(True)
        logo_path = "/Users/juliamoor/Desktop/MoodyStream/gui/moody_logo.jpg"
        if os.path.exists(logo_path):
            logo_pixmap = QPixmap(logo_path)
            logo_label.setPixmap(logo_pixmap)
        controls_layout.addWidget(logo_label)
        controls_layout.addSpacing(20)
        
        # Sliders in vertikalem Layout
        sliders_layout = QVBoxLayout()
        
        # Volume slider row
        volume_row = QHBoxLayout()
        volume_label = QLabel("Volume:")
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setMinimum(0)
        self.volume_slider.setMaximum(100)
        self.volume_slider.setValue(50)
        self.volume_slider.setMaximumWidth(100)
        self.volume_slider.valueChanged.connect(self.volume_changed)
        self.volume_value_label = QLabel("50%")
        
        volume_row.addWidget(volume_label)
        volume_row.addWidget(self.volume_slider)
        volume_row.addWidget(self.volume_value_label)
        
        # Sensitivity slider row
        sensitivity_row = QHBoxLayout()
        sensitivity_label = QLabel("Sensitivity:")
        self.sensitivity_slider = QSlider(Qt.Horizontal)
        self.sensitivity_slider.setMinimum(0)
        self.sensitivity_slider.setMaximum(100)
        self.sensitivity_slider.setValue(50)
        self.sensitivity_slider.setMaximumWidth(100)
        self.sensitivity_slider.valueChanged.connect(self.sensitivity_changed)
        self.sensitivity_value_label = QLabel("50%")
        
        sensitivity_row.addWidget(sensitivity_label)
        sensitivity_row.addWidget(self.sensitivity_slider)
        sensitivity_row.addWidget(self.sensitivity_value_label)
        
        # Beide Slider Rows zum vertikalen Layout hinzufügen
        sliders_layout.addLayout(volume_row)
        sliders_layout.addLayout(sensitivity_row)
        
        controls_layout.addLayout(sliders_layout)
        controls_layout.addSpacing(20)
        
        # start stream button
        self.start_stream_button = QPushButton("Start Stream")
        self.start_stream_button.clicked.connect(self.start_stream)
        self.start_stream_button.setFixedWidth(130)
        controls_layout.addWidget(self.start_stream_button)
        
        # end stream button
        self.end_stream_button = QPushButton("End Stream")
        self.end_stream_button.clicked.connect(self.end_stream)
        self.end_stream_button.setFixedWidth(130)
        self.end_stream_button.setEnabled(False)
        controls_layout.addWidget(self.end_stream_button)
        
        # settings button
        self.settings_button = QPushButton("Settings")
        self.settings_button.clicked.connect(self.open_settings)
        self.settings_button.setFixedWidth(130)
        controls_layout.addWidget(self.settings_button)
        
        controls_layout.addStretch()
        
        # Live Icon
        self.live_icon_label = QLabel()
        self.live_icon_label.setStyleSheet("background-color: transparent; border: none;")
        self.live_icon_label.setAlignment(Qt.AlignCenter)
        self.live_icon_label.setFixedSize(70, 40)
        self.live_icon_label.setScaledContents(True)
        live_icon_path = "/Users/juliamoor/Desktop/MoodyStream/gui/live_icon.png"
        if os.path.exists(live_icon_path):
            live_icon_pixmap = QPixmap(live_icon_path)
            self.live_icon_label.setPixmap(live_icon_pixmap)
        self.live_icon_label.setVisible(False)  # initially hidden
        controls_layout.addWidget(self.live_icon_label)
        
        # camera display
        self.camera_label = QLabel()
        self.camera_label.setStyleSheet("background-color: #000000; border: 1px solid #212124; border-radius: 8px;")
        self.camera_label.setFixedSize(720, 405) #16:9 Format
        self.camera_label.setAlignment(Qt.AlignCenter)
        
        # container widget for buttons
        button_container = QWidget()
        button_grid = QGridLayout(button_container)
        button_grid.setSpacing(4)
        button_grid.setContentsMargins(0, 0, 0, 0)
        
        # creating buttons
        self.btn_happy = SoundButton("Happy")
        self.btn_surprise = SoundButton("Surprise")
        self.btn_sad = SoundButton("Sad")
        self.btn_fear = SoundButton("Fear")
        self.btn_thumbs_up = SoundButton("Thumbs up")
        self.btn_thumbs_down = SoundButton("Thumbs down")
        self.btn_peace = SoundButton("Peace")
        self.btn_middle_finger = SoundButton("Middelfinger")
        self.btn_open_hand = SoundButton("Open Hand")
        
        buttons = [
            self.btn_happy, self.btn_surprise, self.btn_sad,
            self.btn_fear, self.btn_thumbs_up, self.btn_thumbs_down,
            self.btn_peace, self.btn_middle_finger, self.btn_open_hand
        ]
        
        # buttonsize
        for btn in buttons:
            btn.setMinimumHeight(75)
            btn.setMinimumWidth(237)
        
        # buttons in 3x3 grid
        row = 0
        col = 0
        for btn in buttons:
            button_grid.addWidget(btn, row, col)
            col += 1
            if col >= 3:
                col = 0
                row += 1
        
        for i in range(3):
            button_grid.setColumnStretch(i, 1)
        
        # adding everything into the center layout
        center_layout.addLayout(controls_layout)
        center_layout.addSpacing(8)
        center_layout.addWidget(self.camera_label, 0, Qt.AlignHCenter)
        center_layout.addSpacing(8)
        center_layout.addWidget(button_container, 0, Qt.AlignHCenter)
        center_layout.addStretch()
        
        # right ads (stacked over each other)
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
        
        # timer for frame updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)
    
    def create_menu(self):
        menubar = self.menuBar()
        settings_menu = menubar.addMenu("Options")
        
        from PyQt5.QtWidgets import QAction
        settings_action = QAction("Settings", self)
        settings_action.triggered.connect(self.open_settings)
        settings_menu.addAction(settings_action)
    
    def open_settings(self):
        dialog = SettingsDialog(self)
        dialog.exec_()
    
    def start_stream(self):
        """Stream starten"""
        self.streaming = True
        self.start_stream_button.setEnabled(False)
        self.end_stream_button.setEnabled(True)
        self.live_icon_label.setVisible(True)
    
    def end_stream(self):
        """Stream beenden"""
        self.streaming = False
        self.start_stream_button.setEnabled(True)
        self.end_stream_button.setEnabled(False)
        self.live_icon_label.setVisible(False)
    
    def volume_changed(self, value):
        """Lautstärke aktualisieren"""
        self.volume_value_label.setText(f"{value}%")
    
    def sensitivity_changed(self, value):
        """Sensitivity aktualisieren"""
        self.sensitivity_value_label.setText(f"{value}%")
    
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