"""Aufgabe: Zentrale Pfade/Konstanten:
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]
CACHE = BASE/"sounds"/"sound_cache"
PROFILES = BASE/"setup"/"profiles"


SOUND_MAP_PATH = BASE / "sounds" / "sound_map.json"
SETUP_CONFIG_PATH = BASE / "setup" / "setup_config.json"

# Optional: erlaubte Zuordnungs-Keys (Emotionen + Gesten)
ALLOWED_BEHAVIOUR_KEYS = [
    "happy","sad","angry","surprise","neutral",
    "thumbsup","peace","wave","fist","ok","laugh"
]

# Start-URL für myinstants (kannst du anpassen)
MYINSTANTS_URL = "https://www.myinstants.com/en/index/de/"

Nutzen: überall identische Pfade, kein Copy-Paste."""