"""Aufgabe: robuster Player:

Einreihung in Queue, kein Abbrechen laufender Sounds,

Lautstärke aus Profil,

Rückmeldung an GUI („spielt: ok.mp3“).

API:

def play(path: str, volume: float = 1.0): ...
def stop_all(): ...


Tipp: pygame.mixer einmal initialisieren; Fehler (Datei fehlt) loggen.


alle MP3s sollen in sound_cache landen

Pfad in JSON: relativ speichern (sounds/sound_cache/xyz.mp3), damit portable.

"""




import pygame
# from utils.logger import log_info  # temporarily disabled

pygame.mixer.init()

def play(path: str, volume: float = 1.0):
    """Spielt eine MP3- oder WAV-Datei ab."""
    try:
        sound = pygame.mixer.Sound(path)
        sound.set_volume(volume)
        sound.play()
        # log_info(f"🎵 Sound abgespielt: {path}")  # temporarily disabled
    except Exception as e:
        # log_info(f"❌ Fehler beim Abspielen: {e}")  # temporarily disabled
        pass

def stop_all():
    pygame.mixer.stop()
