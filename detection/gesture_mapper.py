"""Aufgabe: Gesten-Label → Sound-Key (z. B. "thumbsup" → "ok").

Eingaben: Geste, Profil-Mapping (frei definierbar).

Ausgaben: Sound-Key oder None."""
from typing import Iterable, Optional, Tuple, Dict
from utils.json_manager import load_json
from utils.settings import SOUND_MAP_PATH


def _load_sound_map() -> Dict[str, str]:
    """Lädt die globale Sound-Map (key -> mp3-pfad)."""
    return load_json(SOUND_MAP_PATH)


def map_gesture_to_sound(
    gestures: Iterable[str],
    sound_map: Optional[Dict[str, str]] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Nimmt eine Liste von Gesten-Labels (z. B. ["thumbsup"]) und liefert

      (sound_key, sound_path) oder (None, None).
    """
    if sound_map is None:
        sound_map = _load_sound_map()

    for label in gestures:
        if label in sound_map:
            return label, sound_map[label]

    return None, None