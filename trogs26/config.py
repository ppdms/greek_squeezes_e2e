"""The run configuration (one JSON file under configs/) and its digest.

A configuration file is either complete, or an experiment of another one:

    {"extends": "configs/pagenet_only_style_neural.json",
     "override": {"pagenet": {"version": "style_mild_15_epochs", "epochs": 15, "drop_path": 0.1}}}

The override is merged into the base recursively. Every key of an override must exist in the base (a misspelt key
would silently change nothing), a dict is only overridden by a dict, and any other value, a list included, is
replaced. New keys (a new sweep grid, say) go under "add", which is merged the same way but whose keys must
not exist in the base:

    {"extends": "configs/pagenet_only_style_neural.json",
     "add": {"decode": {"grids": {"narrow": {"lam": [0.5], "w": [2.0], "beta": [1.0]}}}}}
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


class Config(dict):
    """A loaded configuration (the merged JSON) and `source`, where it came from: the file's path and, for an
    experiment, the base's source and the override."""

    source: dict


def _merge(base: dict, override: dict, where: str = '') -> dict:
    merged = dict(base)
    for key, value in override.items():
        if key not in base:
            raise KeyError(f'override key {where}{key} does not exist in the base configuration')
        if isinstance(base[key], dict):
            if not isinstance(value, dict):
                raise TypeError(f'override {where}{key} must be a dict, as in the base configuration')
            merged[key] = _merge(base[key], value, f'{where}{key}.')
        else:
            merged[key] = value
    return merged


def _add(base: dict, add: dict, where: str = '') -> dict:
    merged = dict(base)
    for key, value in add.items():
        if key not in base:
            merged[key] = value
        elif isinstance(base[key], dict) and isinstance(value, dict):
            merged[key] = _add(base[key], value, f'{where}{key}.')
        else:
            raise KeyError(f'added key {where}{key} already exists in the base configuration')
    return merged


def load(path: str | Path) -> Config:
    """Load a configuration, resolving `extends` and `override`; relative paths resolve against the repository."""
    file = Path(path)
    raw = json.loads((file if file.is_absolute() else REPO / file).read_text(encoding='utf-8'))
    if 'extends' not in raw:
        cfg = Config(raw)
        cfg.source = {'path': str(path)}
        return cfg
    if set(raw) - {'extends', 'override', 'add'}:
        raise KeyError(f'{path}: an extending configuration holds only "extends", "override" and "add"')
    base = load(raw['extends'])
    cfg = Config(_add(_merge(base, raw.get('override', {})), raw.get('add', {})))
    cfg.source = {'path': str(path), 'extends': base.source, 'override': raw.get('override', {})}
    if 'add' in raw:
        cfg.source['add'] = raw['add']
    return cfg


def digest(cfg: dict) -> str:
    """SHA-256 of the canonical JSON form (sorted keys, compact separators); recorded with every output."""
    blob = json.dumps(cfg, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    return hashlib.sha256(blob).hexdigest()
