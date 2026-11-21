"""Keep user presets outside extensions so extension upgrades do not delete them.

Legacy presets in extensions/tripclipskip-studio/presets/user_presets.json are
read on first use; they are not modified or destroyed by migration.
"""
from __future__ import annotations

from pathlib import Path
from studio_core.config import load_presets, save_preset


def preferred_path(extension_root: Path, webui_data_path: Path | None = None) -> Path:
    if webui_data_path is not None:
        return Path(webui_data_path) / 'tripclipskip-studio' / 'presets' / 'user_presets.json'
    return Path(extension_root) / 'presets' / 'user_presets.json'


def load_user_presets(extension_root: Path, webui_data_path: Path | None = None) -> dict:
    legacy = Path(extension_root) / 'presets' / 'user_presets.json'
    stable = preferred_path(extension_root, webui_data_path)
    if stable == legacy:
        return load_presets(legacy)
    # Stable settings win when both exist; old keys can still be migrated.
    return {**load_presets(legacy), **load_presets(stable)}


def save_user_preset(extension_root: Path, webui_data_path: Path | None,
                     name: str, settings: dict) -> tuple[dict, Path]:
    path = preferred_path(extension_root, webui_data_path)
    legacy = Path(extension_root) / 'presets' / 'user_presets.json'
    if path != legacy:
        # Merge *missing* legacy entries on every save, not only on first
        # creation. A user may restore an older extension folder containing
        # additional presets after the stable file has already been created.
        # Existing stable entries always win name collisions.
        stable = load_presets(path)
        for key, data in load_presets(legacy).items():
            if key not in stable:
                save_preset(path, key, data)
                stable[key] = data
    return save_preset(path, name, settings), path
