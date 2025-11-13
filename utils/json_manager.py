"""Utility helpers for safely reading and writing shared JSON files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Union

PathLike = Union[str, "Path"]


def _as_path(path: PathLike) -> Path:
    return Path(path)


def _ensure_parent(path: Path) -> None:
    """Create parent folders if they do not exist yet."""
    path.parent.mkdir(parents=True, exist_ok=True)


def load_json(path: PathLike) -> Dict[str, Any]:
    """Load JSON and always return a dict, even for empty/malformed files."""
    target = _as_path(path)
    if not target.exists():
        return {}
    try:
        content = target.read_text(encoding="utf-8").strip()
        if not content:
            return {}
        return json.loads(content)
    except json.JSONDecodeError:
        print(f" s   ? Warnung: Datei {target} ist kein gültiges JSON, wird neu erstellt.")
        return {}


def save_json(path: PathLike, key: str, value: Any) -> None:
    """
    Persist a single key/value pair in the JSON file without dropping other keys.
    Useful for simple `{key: value}` assignments.
    """
    target = _as_path(path)
    data = load_json(target)
    data[key] = value
    _ensure_parent(target)
    target.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def update_json(path: PathLike, key: str, value: Mapping[str, Any]) -> None:
    """
    Merge dictionaries under `key` instead of overwriting them entirely.
    Falls back to `save_json` semantics if the existing value is not a dict.
    """
    target = _as_path(path)
    data = load_json(target)
    current = data.get(key)

    if isinstance(current, MutableMapping) and isinstance(value, Mapping):
        current.update(value)
        data[key] = current
    else:
        data[key] = dict(value) if isinstance(value, Mapping) else value

    _ensure_parent(target)
    target.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
