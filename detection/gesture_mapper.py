"""Aufgabe: Gesten-Label → Sound-Key (z. B. "thumbsup" → "ok").

Eingaben: Geste, Profil-Mapping (frei definierbar).

Ausgaben: Sound-Key oder None."""

# detection/gesture_mapper.py

from typing import Iterable, Optional, Tuple, Dict

from utils.json_manager import load_json
from utils.settings import SOUND_MAP_PATH, SETUP_CONFIG_PATH


def _load_sound_map() -> Dict[str, str]:
    """
    Lädt die globale Sound-Map (sound_key -> mp3-pfad).
    """
    return load_json(SOUND_MAP_PATH)


def _load_gesture_profile(profile_name: str) -> Dict[str, str]:
    """
    Lädt optionale Profil-Mappings für Gesten aus setup_config.json.

    Erwartete (zukünftige) Struktur z. B.:

    {
      "profiles": {
        "default": {
          "gesture_to_sound": {
            "thumbsup": "thumbsup",
            "wave": "wave"
          }
        }
      }
    }

    Wenn nichts davon vorhanden ist, geben wir {} zurück
    und fallen auf "label == sound_key" zurück.
    """
    config = load_json(SETUP_CONFIG_PATH)
    profiles = config.get("profiles") or {}
    profile = profiles.get(profile_name) or {}
    gesture_to_sound = profile.get("gesture_to_sound") or {}
    return gesture_to_sound


def _normalize_label(label: str) -> str:
    """Normalize gesture labels to improve mapping robustness."""
    return (label or "").strip().lower().replace(" ", "")


def get_sound_for_gestures(
    gestures: Iterable[str],
    profile_name: str = "default",
    sound_map: Optional[Dict[str, str]] = None,
    gesture_to_sound: Optional[Dict[str, str]] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Nimmt eine Liste von Gesten-Labels und liefert (sound_key, sound_path) zurück
    oder (None, None), wenn nichts gemappt werden kann.
    """
    if sound_map is None:
        sound_map = _load_sound_map()
    if gesture_to_sound is None:
        gesture_to_sound = _load_gesture_profile(profile_name)

    # Build a normalized lookup for sound_map to tolerate spacing/case differences
    normalized_sound_map = {}
    for k, v in (sound_map or {}).items():
        nk = _normalize_label(k)
        if nk and nk not in normalized_sound_map:
            normalized_sound_map[nk] = v

    for label in gestures:
        if not label:
            continue

        candidates = [label]
        norm = _normalize_label(label)
        if norm and norm not in candidates:
            candidates.append(norm)

        # 1) Profil-Mapping: Geste → Sound-Key (try raw + normalized)
        sound_key = None
        if gesture_to_sound:
            for cand in candidates:
                sound_key = gesture_to_sound.get(cand)
                if sound_key:
                    break

        # 2) Fallback: direktes Label als Sound-Key (try raw + normalized)
        if not sound_key:
            for cand in candidates:
                if cand in sound_map:
                    sound_key = cand
                    break
                nk = _normalize_label(cand)
                if nk and nk in normalized_sound_map:
                    sound_key = nk
                    break

        if not sound_key:
            continue

        sound_path = sound_map.get(sound_key) or normalized_sound_map.get(sound_key)
        if sound_path:
            return sound_key, sound_path

    return None, None
