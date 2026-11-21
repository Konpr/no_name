"""Only local user-defined presets; no model downloading/executable payloads."""
from __future__ import annotations
import json
import os
import re
from pathlib import Path

MAX_PRESETS = 128


def _valid_name(value: str) -> str:
    value = str(value).strip()
    if not 1 <= len(value) <= 64 or any(c in value for c in '<>/\\\x00\r\n'):
        raise ValueError('Preset name must be 1–64 characters without path or HTML characters')
    return value


def load_presets(path: Path) -> dict:
    if not Path(path).exists():
        return {}
    with Path(path).open(encoding='utf8') as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError('Preset file must contain a JSON object')
    return {k:v for k,v in data.items() if isinstance(k,str) and isinstance(v,dict)}


def save_preset(path: Path, name: str, settings: dict) -> dict:
    name = _valid_name(name)
    if not isinstance(settings, dict):
        raise TypeError('Invalid settings')
    path = Path(path)
    presets = load_presets(path)
    if name not in presets and len(presets) >= MAX_PRESETS:
        raise ValueError('Preset limit reached')
    presets[name] = settings
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf8') as out:
        json.dump(presets, out, indent=2, ensure_ascii=False)
        out.flush()
        os.fsync(out.fileno())
    os.replace(tmp, path)
    return presets
