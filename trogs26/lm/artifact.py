"""Language-model artifacts: a directory holding `lm.json` (provenance and the name of the counts file) and the
pickled `Counts`.

Language-model files are pickles, so loading refuses every class except the few a counts file contains.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

from .ppm import PPM, Counts

_ALLOWED = {('collections', 'Counter'), (Counts.__module__, Counts.__name__)}


class _Unpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if (module, name) in _ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f'language model file contains unexpected {module}.{name}')


def save(counts: Counts, directory: Path, manifest: dict) -> None:
    """Write `counts` and `manifest` (provenance: bases, rows, characters, ...) to `directory`."""
    directory.mkdir(parents=True, exist_ok=True)
    model = f'counts_order{counts.order}.pkl'
    with open(directory / model, 'wb') as f:
        pickle.dump(counts, f, protocol=pickle.HIGHEST_PROTOCOL)
    (directory / 'lm.json').write_text(
        json.dumps({**manifest, 'order': counts.order, 'alphabet': counts.alphabet, 'model': model},
                   indent=2, sort_keys=True), encoding='utf-8')


def load_counts(directory: Path) -> Counts:
    model = json.loads((directory / 'lm.json').read_text(encoding='utf-8'))['model']
    with open(directory / model, 'rb') as f:
        return _Unpickler(f).load()


def load(directory: Path, max_order: int) -> PPM:
    """The PPM-C model of the counts in `directory`, reading contexts of at most `max_order - 1` characters."""
    return PPM(load_counts(directory), max_order)
