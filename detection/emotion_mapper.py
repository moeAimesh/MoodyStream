"""Aufgabe: aus emotion_recognition.py erkannte Emotion fangen → passende Sound-Key spielen (z.B. "happy" → "laugh").

Eingaben: Emotion, Profil-Mapping.

Ausgaben: Sound-Key oder None.



possible API:
def map_emotion_to_sound(emotion: str, profile: dict) -> Optional[str]:
    return "laugh"  #Beispiel """
    
# detection/emotion_mapper.py

from typing import Iterable, Optional, Tuple, Dict

from utils.json_manager import load_json
from utils.settings import SOUND_MAP_PATH, SETUP_CONFIG_PATH


def _load_sound_map() -> Dict[str, str]:
    """
    Lädt die globale Sound-Map (sound_key -> mp3-pfad).
    """
    return load_json(SOUND_MAP_PATH)


def _load_emotion_profile(profile_name: str) -> Dict[str, str]:
    """
    Lädt optionale Profil-Mappings für Emotionen aus setup_config.json.

    Erwartete (zukünftige) Struktur z. B.:

    {
      "profiles": {
        "default": {
          "emotion_to_sound": {
            "happy": "happy",
            "sad": "ok"
          }
        }
      }
    }

    Wenn nichts davon vorhanden ist, geben wir einfach {} zurück
    und fallen auf "label == sound_key" zurück.
    """
    config = load_json(SETUP_CONFIG_PATH)
    profiles = config.get("profiles") or {}
    profile = profiles.get(profile_name) or {}
    emotion_to_sound = profile.get("emotion_to_sound") or {}
    return emotion_to_sound


def get_sound_for_emotions(
    emotions: Iterable[str],
    profile_name: str = "default",
    sound_map: Optional[Dict[str, str]] = None,
    emotion_to_sound: Optional[Dict[str, str]] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Nimmt eine Liste von Emotion-Labels und liefert (sound_key, sound_path) zurück
    oder (None, None), wenn nichts gemappt werden kann.

    - `profile_name`: welches Profil genutzt werden soll (optional, default 'default')
    - `sound_map` / `emotion_to_sound`: können in Tests injiziert werden,
      dann wird kein File-I/O gemacht.
    """
    if sound_map is None:
        sound_map = _load_sound_map()
    if emotion_to_sound is None:
        emotion_to_sound = _load_emotion_profile(profile_name)

    for label in emotions:
        if not label:
            continue

        # 1) Versuche zuerst Profil-Mapping: Emotion → Sound-Key
        sound_key = emotion_to_sound.get(label) if emotion_to_sound else None

        # 2) Fallback: nimm die Emotion direkt als Sound-Key
        if not sound_key and label in sound_map:
            sound_key = label

        if not sound_key:
            continue

        sound_path = sound_map.get(sound_key)
        if sound_path:
            return sound_key, sound_path

    return None, None
