"""Aufgabe: sicheres Lesen/Schreiben/Updaten (mit Lock, Fallback).

API:

load_json(path) -> dict
save_json(path, data: dict)
update_json(path, key, value)

Damit kann man später in setup_config.json mehrere Setup-Ergebnisse speichern (z. B. Gesicht + Sounds).
"""


import json, os

def load_json(path):
    """Lädt JSON-Datei oder gibt leeres Dict zurück, wenn leer oder fehlerhaft."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                return {}  # ⚙️ leer -> kein Fehler
            return json.loads(content)
    except json.JSONDecodeError:
        print(f"⚠️ Warnung: Datei {path} ist kein gültiges JSON, wird neu erstellt.")
        return {}

def save_json(path, key, value):
    data = load_json(path)
    data[key] = value
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

        
def update_json(path, key, value):
    pass