"""Aufgabe: aus emotion_recognition.py erkannte Emotion fangen → passende Sound-Key spielen (z.B. "happy" → "laugh").

Eingaben: Emotion, Profil-Mapping.

Ausgaben: Sound-Key oder None.



possible API:
def map_emotion_to_sound(emotion: str, profile: dict) -> Optional[str]:
    return "laugh"  #Beispiel """
    
from typing import Iterable, Optional, Tuple, Dict
from utils.json_manager import load_json
from utils.settings import SOUND_MAP_PATH


def _load_sound_map() -> Dict[str, str]:
    """Lädt die globale Sound-Map (key -> mp3-pfad)."""
    return load_json(SOUND_MAP_PATH)


def map_emotion_to_sound(
    emotions: Iterable[str],
    sound_map: Optional[Dict[str, str]] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Nimmt eine Liste von Emotion-Labels (z. B. ["happy"]) und liefert

      (sound_key, sound_path) oder (None, None).
    """
    if sound_map is None:
        sound_map = _load_sound_map()

    for label in emotions:
        if label in sound_map:
            return label, sound_map[label]

    return None, None