"""Aufgabe: Sicheres Lesen, Schreiben und Aktualisieren von JSON-Dateien.

Nutzen:
- load_json gibt immer ein Dictionary zurück, auch bei leeren/fehlerhaften Dateien.
- save_json speichert einen einzelnen Key ohne vorhandene Daten zu löschen.
- update_json merged neue Werte (z. B. neue Sound-Zuordnungen) in bestehende Strukturen,
  anstatt sie zu überschreiben – wichtig für setup_config.json.
"""

import json, os
from typing import Any

def load_json(path):
    """Lädt JSON-Datei oder gibt leeres Dict zurück, wenn leer oder fehlerhaft."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                return {}
            return json.loads(content)
    except json.JSONDecodeError:
        print(f"⚠️ Warnung: Datei {path} ist kein gültiges JSON, wird neu erstellt.")
        return {}

def _ensure_parent(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def save_json(path, key, value):
    """
    Speichert EINEN Key in der Datei (bestehende anderen Keys bleiben erhalten).
    Für einfache {key: value}-Zuweisungen.
    """
    data = load_json(path)
    data[key] = value
    _ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def update_json(path, key, value):
    """
    Merged 'value' in data[key], wenn beides dicts sind.
    Sonst verhält es sich wie save_json.
    Beispiel:
      update_json("setup_config.json", "sounds", {"happy": "pfad.mp3"})
    -> hängt das neue Mapping an statt alles zu überschreiben.
    """
    data = load_json(path)
    cur = data.get(key)

    if isinstance(cur, dict) and isinstance(value, dict):
        # Shallow-merge
        cur.update(value)
        data[key] = cur
    else:
        data[key] = value

    _ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
