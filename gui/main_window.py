"""Aufgabe: Fenster mit Tabs: Setup, Live, Einstellungen.

Eingaben: Events aus Kamera/Audio.

Ausgaben: visuelles Feedback, Buttons („Setup erneut starten", „Profil wechseln").

Tipp: GUI in eigenem Thread oder async."""
import sys
import cv2
import webbrowser
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QDialog, 
                             QSpinBox, QSlider, QFormLayout, QMenuBar, QMenu,
                             QComboBox, QGridLayout, QLineEdit, QFileDialog,
                             QMessageBox)
from PyQt5.QtCore import QTimer, Qt, QUrl
from PyQt5.QtGui import QImage, QPixmap, QCursor, QIcon
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent


class SoundButton(QPushButton):
    """Erweiterter Button mit Sound-Funktionalität"""
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.sound_path = None
        self.player = QMediaPlayer()
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)
    
    def show_context_menu(self, position):
        """Zeigt Kontextmenü beim Rechtsklick"""
        menu = QMenu()
        
        # Optionen im Kontextmenü
        search_action = menu.addAction("Sound im Web suchen")
        load_action = menu.addAction("Sound-Datei laden")
        menu.addSeparator()
        
        if self.sound_path:
            play_action = menu.addAction("Sound abspielen")
            remove_action = menu.addAction("Sound entfernen")
            info_action = menu.addAction(f"Aktueller Sound: {self.sound_path.split('/')[-1]}")
            info_action.setEnabled(False)
        
        # Aktion ausführen
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
        """Öffnet MyInstants im Browser"""
        url = "https://www.myinstants.com/de/search/?name=MEME"
        webbrowser.open(url)
        
        # Info-Dialog anzeigen
        QMessageBox.information(
            self,
            "Sound-Suche",
            "Browser wurde geöffnet!\n\n"
            "So verknüpfst du einen Sound:\n"
            "1. Suche einen Sound auf MyInstants\n"
            "2. Lade den Sound runter (z.B. indem du ihn dir selber als Email schickst\n"
            "3. Geh wieder auf unsere App und gehe mit einem Rechtsklick auf sound hinzufügen'\n"
            "4. Wähle deinen eruntergeladenen Sound aus"
        )
    
    def load_sound_file(self):
        """Lädt eine Sound-Datei vom Computer"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Sound-Datei auswählen",
            "",
            "Audio-Dateien (*.mp3 *.wav *.ogg);;Alle Dateien (*.*)"
        )
        
        if file_path:
            self.sound_path = file_path
            url = QUrl.fromLocalFile(file_path)
            self.player.setMedia(QMediaContent(url))
            self.player.setVolume(100)
            print(f"Sound '{file_path}' wurde Button '{self.text()}' zugewiesen")
    
    def play_sound(self):
        """Spielt den verknüpften Sound ab"""
        if self.sound_path and self.player:
            self.player.stop()  # stoppt vorherigen Sound falls noch am Laufen
            self.player.play()
            print(f"Sound wird abgespielt: {self.sound_path}")
    
    def remove_sound(self):
        """Entfernt den verknüpften Sound"""
        self.sound_path = None
        self.player.stop()
        self.player.setMedia(QMediaContent())
        print(f"Sound von Button '{self.text()}' entfernt")
    
    def mousePressEvent(self, event):
        """Behandelt normale Klicks (links) und spielt Sound ab"""
        if event.button() == Qt.LeftButton:
            if self.sound_path:
                self.play_sound()
            else:
                print(f"{self.text()} geklickt (kein Sound verknüpft)")
        super().mousePressEvent(event)


class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Einstellungen")
        self.setMinimumWidth(300)
        
        layout = QFormLayout()
        
        # Kameraindex Einstellung
        self.camera_index = QSpinBox()
        self.camera_index.setMinimum(0)
        self.camera_index.setMaximum(3)
        self.camera_index.setValue(1)
        layout.addRow("Kamera Index:", self.camera_index)
        
        # Lautstärke Einstellung
        self.volume = QSlider(Qt.Horizontal)
        self.volume.setMinimum(0)
        self.volume.setMaximum(100)
        self.volume.setValue(50)
        layout.addRow("Lautstärke:", self.volume)
        
        # Übernehmen Button
        apply_btn = QPushButton("Übernehmen")
        apply_btn.clicked.connect(self.accept)
        layout.addRow(apply_btn)
        
        self.setLayout(layout)
    
    def get_camera_index(self):
        return self.camera_index.value()
    
    def get_volume(self):
        return self.volume.value()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Moodystream")
        self.setGeometry(100, 100, 1400, 800)
        
        # Titelleiste schwarz machen (Plattformabhängig)
        self.setStyleSheet("QMainWindow { background-color: #36454F; }")
        
        # Globales Stylesheet für die gesamte Anwendung
        self.setStyleSheet("""
            QMainWindow {
                background-color: #36454F;
            }
            QWidget {
                background-color: #36454F;
                color: #FFFFFF;
                font-family: Monaco, monospace;
            }
            QLabel {
                color: #FFFFFF;
                font-family: Monaco, monospace;
            }
            QPushButton {
                background-color: #333333;
                color: #FFFFFF;
                font-family: Monaco, monospace;
                border: 1px solid #555555;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #555555;
            }
            QPushButton:pressed {
                background-color: #666666;
            }
            QComboBox {
                background-color: #333333;
                color: #FFFFFF;
                font-family: Monaco, monospace;
                border: 1px solid #555555;
                padding: 5px;
            }
            QComboBox:hover {
                background-color: #555555;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                background-color: #333333;
                color: #FFFFFF;
                selection-background-color: #555555;
            }
            QSlider::groove:horizontal {
                background-color: #333333;
                height: 8px;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background-color: #FFFFFF;
                width: 16px;
                margin: -4px 0;
                border-radius: 8px;
            }
            QSlider::handle:horizontal:hover {
                background-color: #CCCCCC;
            }
            QMenuBar {
                background-color: #000000;
                color: #FFFFFF;
                font-family: Monaco, monospace;
            }
            QMenuBar::item {
                background-color: #000000;
                color: #FFFFFF;
            }
            QMenuBar::item:selected {
                background-color: #333333;
            }
            QMenu {
                background-color: #000000;
                color: #FFFFFF;
                font-family: Monaco, monospace;
                border: 1px solid #555555;
            }
            QMenu::item:selected {
                background-color: #333333;
            }
        """)
        
        # Kamera initialisieren
        self.camera_index = 0
        self.cap = cv2.VideoCapture(self.camera_index)
        
        # Menüleiste erstellen
        self.create_menu()
        
        # Hauptwidget und Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Linke Werbemockups (2 Stück übereinander)
        left_ads = QVBoxLayout()
        self.left_ad1 = QLabel()
        self.left_ad1.setStyleSheet("border: 2px solid #555555;")
        self.left_ad1.setAlignment(Qt.AlignCenter)
        self.left_ad1.setMinimumSize(250, 300)
        self.left_ad1.setScaledContents(True)
        ad_pixmap1 = QPixmap("/Users/juliamoor/Desktop/MoodyStream/gui/ad_mockup.jpg")
        self.left_ad1.setPixmap(ad_pixmap1)
        
        self.left_ad2 = QLabel()
        self.left_ad2.setStyleSheet("border: 2px solid #555555;")
        self.left_ad2.setAlignment(Qt.AlignCenter)
        self.left_ad2.setMinimumSize(250, 300)
        self.left_ad2.setScaledContents(True)
        ad_pixmap2 = QPixmap("/Users/juliamoor/Desktop/MoodyStream/gui/ad_mockup.jpg")
        self.left_ad2.setPixmap(ad_pixmap2)
        
        left_ads.addWidget(self.left_ad1)
        left_ads.addWidget(self.left_ad2)
        left_ads.addStretch()
        
        # Mittlerer Bereich (Kamera + Controls + Buttons)
        center_layout = QVBoxLayout()
        
        # Controls über der Kamera (Kameraauswahl und Lautstärke)
        controls_layout = QHBoxLayout()
        
        # Kamera-Auswahl Dropdown
        camera_select_label = QLabel("Kamera:")
        self.camera_combo = QComboBox()
        # Kameras 0-5 zur Auswahl hinzufügen
        for i in range(6):
            self.camera_combo.addItem(f"Kamera {i}", i)
        self.camera_combo.currentIndexChanged.connect(self.change_camera)
        
        # Lautstärkeregler
        volume_label = QLabel("Lautstärke:")
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setMinimum(0)
        self.volume_slider.setMaximum(100)
        self.volume_slider.setValue(50)
        self.volume_slider.setMaximumWidth(200)
        self.volume_slider.valueChanged.connect(self.volume_changed)
        self.volume_value_label = QLabel("50%")
        
        controls_layout.addWidget(camera_select_label)
        controls_layout.addWidget(self.camera_combo)
        controls_layout.addSpacing(20)
        controls_layout.addWidget(volume_label)
        controls_layout.addWidget(self.volume_slider)
        controls_layout.addWidget(self.volume_value_label)
        controls_layout.addStretch()
        
        # Kamera-Display (breiter gemacht)
        self.camera_label = QLabel()
        self.camera_label.setStyleSheet("background-color: black; border: 2px solid #555555;")
        self.camera_label.setFixedSize(800, 480)
        self.camera_label.setAlignment(Qt.AlignCenter)
        
        # Button Container Widget mit fester Breite (angepasst an neue Kamerabreite)
        button_container = QWidget()
        button_container.setFixedWidth(800)
        button_container.setFixedHeight(280)
        
        # 6 quadratische Buttons unter der Kamera in 2 Reihen
        button_grid = QGridLayout(button_container)
        button_grid.setSpacing(0)  # Keine Abstände zwischen Buttons
        button_grid.setContentsMargins(0, 0, 0, 0)  # Keine Ränder
        
        # Buttons erstellen (jetzt mit SoundButton Klasse)
        self.btn_happy = SoundButton("Happy")
        self.btn_schock = SoundButton("Schock")
        self.btn_sad = SoundButton("Sad")
        self.btn_angry = SoundButton("Angry")
        self.btn_thumbs_up = SoundButton("Thumbs Up")
        self.btn_thumbs_down = SoundButton("Thumbs Down")
        
        # Liste aller Buttons für einheitliche Formatierung
        buttons = [self.btn_happy, self.btn_schock, self.btn_sad, 
                  self.btn_angry, self.btn_thumbs_up, self.btn_thumbs_down]
        
        # Buttons Mindesthöhe setzen
        for btn in buttons:
            btn.setMinimumHeight(140)
        
        # Buttons im Grid anordnen (2 Reihen x 3 Spalten)
        button_grid.addWidget(self.btn_happy, 0, 0)
        button_grid.addWidget(self.btn_schock, 0, 1)
        button_grid.addWidget(self.btn_sad, 0, 2)
        button_grid.addWidget(self.btn_angry, 1, 0)
        button_grid.addWidget(self.btn_thumbs_up, 1, 1)
        button_grid.addWidget(self.btn_thumbs_down, 1, 2)
        
        # Grid-Spalten gleichmäßig verteilen
        for i in range(3):
            button_grid.setColumnStretch(i, 1)
        for i in range(2):
            button_grid.setRowStretch(i, 1)
        
        # Alles zum Center Layout hinzufügen
        center_layout.addLayout(controls_layout)
        center_layout.addWidget(self.camera_label, 0, Qt.AlignHCenter)
        center_layout.addWidget(button_container, 0, Qt.AlignHCenter)
        center_layout.addStretch()

        # Rechte Werbemockups (2 Stück übereinander)
        right_ads = QVBoxLayout()
        self.right_ad1 = QLabel()
        self.right_ad1.setStyleSheet("border: 2px solid #555555;")
        self.right_ad1.setAlignment(Qt.AlignCenter)
        self.right_ad1.setMinimumSize(250, 300)
        self.right_ad1.setScaledContents(True)
        ad_pixmap3 = QPixmap("/Users/juliamoor/Desktop/MoodyStream/gui/ad_mockup.jpg")
        self.right_ad1.setPixmap(ad_pixmap3)
        
        self.right_ad2 = QLabel()
        self.right_ad2.setStyleSheet("border: 2px solid #555555;")
        self.right_ad2.setAlignment(Qt.AlignCenter)
        self.right_ad2.setMinimumSize(250, 300)
        self.right_ad2.setScaledContents(True)
        ad_pixmap4 = QPixmap("/Users/juliamoor/Desktop/MoodyStream/gui/ad_mockup.jpg")
        self.right_ad2.setPixmap(ad_pixmap4)
        
        right_ads.addWidget(self.right_ad1)
        right_ads.addWidget(self.right_ad2)
        right_ads.addStretch()
    
        # Alles zum Hauptlayout hinzufügen
        main_layout.addLayout(left_ads, 1)
        main_layout.addLayout(center_layout, 3)
        main_layout.addLayout(right_ads, 1)
        
        # Timer für Kamera-Updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # ~30 FPS
    
    def create_menu(self):
        menubar = self.menuBar()
        settings_menu = menubar.addMenu("Optionen")
        
        from PyQt5.QtWidgets import QAction
        settings_action = QAction("Einstellungen", self)
        settings_action.triggered.connect(self.open_settings)
        settings_menu.addAction(settings_action)
    
    def open_settings(self):
        dialog = SettingsDialog(self)
        if dialog.exec_():
            new_camera_index = dialog.get_camera_index()
            volume = dialog.get_volume()
            
            # Kamera neu initialisieren falls Index geändert wurde
            if new_camera_index != self.camera_index:
                self.cap.release()
                self.camera_index = new_camera_index
                self.cap = cv2.VideoCapture(self.camera_index)
            
            print(f"Neue Einstellungen: Kamera={new_camera_index}, Lautstärke={volume}")
    
    def change_camera(self, index):
        """Kamera wechseln basierend auf Dropdown-Auswahl"""
        new_camera_index = self.camera_combo.currentData()
        if new_camera_index != self.camera_index:
            self.cap.release()
            self.camera_index = new_camera_index
            self.cap = cv2.VideoCapture(self.camera_index)
            print(f"Kamera gewechselt zu Index: {new_camera_index}")
    
    def volume_changed(self, value):
        """Lautstärke aktualisieren"""
        self.volume_value_label.setText(f"{value}%")
        print(f"Lautstärke: {value}%")
    
    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Frame von BGR (OpenCV) zu RGB konvertieren
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            
            # QImage erstellen und anzeigen
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