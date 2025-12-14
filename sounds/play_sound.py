"""Aufgabe: robuster Player:

Einreihung in Queue, kein Abbrechen laufender Sounds,

Lautstärke aus Profil,

Rückmeldung an GUI („spielt: ok.mp3“).

API:

def play(path: str, volume: float = 1.0): ...
def stop_all(): ...


Tipp: pygame.mixer einmal initialisieren


alle MP3s sollen in sound_cache landen

Pfad in JSON: relativ speichern (sounds/sound_cache/xyz.mp3), damit portable.

"""




import pygame
# from utils.logger import log_info  # temporarily disabled

pygame.mixer.init()

from utils.json_manager import load_json
from utils.settings import SETUP_CONFIG_PATH

_DEFAULT_VOLUME = 0.50


def _load_default_volume() -> float:
    cfg = load_json(SETUP_CONFIG_PATH)
    vol = cfg.get("volume") if isinstance(cfg, dict) else None
    try:
        val = float(vol)
    except (TypeError, ValueError):
        return 1.0
    return max(0.0, min(1.0, val))


_DEFAULT_VOLUME = _load_default_volume()


def set_default_volume(volume: float) -> None:
    """Set global fallback volume for play()."""
    global _DEFAULT_VOLUME
    _DEFAULT_VOLUME = max(0.0, min(1.0, float(volume)))


def play(path: str, volume: float | None = None):
    """Spielt eine MP3- oder WAV-Datei ab."""
    try:
        sound = pygame.mixer.Sound(path)
        vol = _DEFAULT_VOLUME if volume is None else max(0.0, min(1.0, float(volume)))
        sound.set_volume(vol)
        sound.play()
    except Exception as e:
        # log_info(f"❌ Fehler beim Abspielen: {e}")  # temporarily disabled
        pass

def stop_all():
    pygame.mixer.stop()
