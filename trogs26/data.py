"""Squeeze splits and cross-validation folds (files under data/)."""
from __future__ import annotations

import json
from functools import lru_cache

from .config import REPO


@lru_cache(maxsize=None)
def splits(path: str = 'data/splits.json') -> dict:
    """train / validation / internal_test / trogs26 squeeze lists and the TROGS-26 image file names."""
    return json.loads((REPO / path).read_text(encoding='utf-8'))


@lru_cache(maxsize=None)
def folds(path: str) -> dict[int, tuple[str, ...]]:
    """fold id -> squeezes, in the order the folds file lists them."""
    raw = json.loads((REPO / path).read_text(encoding='utf-8'))
    return {int(f['id']): tuple(f['bases']) for f in raw['folds']}


def fold_of(fold_map: dict[int, tuple[str, ...]]) -> dict[str, int]:
    return {base: k for k, bases in fold_map.items() for base in bases}


def all_bases(fold_map: dict[int, tuple[str, ...]]) -> list[str]:
    """Every squeeze of the folds file, fold-major."""
    return [base for k in sorted(fold_map) for base in fold_map[k]]
