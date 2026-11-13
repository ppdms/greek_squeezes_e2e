"""Build the character language models from the public transcripts.

The text of a squeeze is the row transcript of its Rotation1 annotation (the organisers' `getRowTranscript`),
upper-cased, one corpus line per non-empty row. The language model of chain k is built from the squeezes of every
fold except k (so the chain never sees the squeezes it decodes); the model for pages outside the folds uses every
fold's squeezes. Names and the order come from the `lm` section of the configuration.

    TROGS26_WORK=<work> python -m trogs26.lm.build [--config configs/pagenet_only_style_neural.json]
"""
from __future__ import annotations

import argparse

from .. import config, data, official, provenance
from ..layout import Work
from . import artifact
from .ppm import Counts, keep_chars


def transcript_rows(work: Work, base: str) -> list[str]:
    """Non-empty upper-cased rows of the Rotation1 transcript of `base`."""
    text = official.v1.getRowTranscript(str(work.annotation(base, 1))).upper()
    return [line for line in text.split('\n') if line]


def _write(work: Work, cfg: dict, name: str, rows_by_base: dict[str, list[str]]) -> None:
    """Count the n-grams of the squeezes in `rows_by_base` (taken in sorted order) as language model `name`."""
    bases = sorted(rows_by_base)
    rows = [row for base in bases for row in rows_by_base[base]]
    counts = Counts.from_lines(rows, cfg['lm']['order'], cfg['alphabet'])
    artifact.save(counts, work.lm(name), {
        'created_by': 'trogs26.lm.build', **provenance.stamp(cfg), 'train_bases': len(bases), 'train_rows': len(rows),
        'train_chars': sum(len(keep_chars(row, cfg['alphabet'])) for row in rows), 'bases': bases})


def build_all(work: Work, cfg: dict) -> None:
    """The final language model (every fold's squeezes) and one fold-excluded model per chain."""
    folds = data.folds(cfg['data']['folds'])
    rows_by_base = {base: transcript_rows(work, base) for base in data.all_bases(folds)}
    _write(work, cfg, cfg['lm']['final'], rows_by_base)
    for k in sorted(folds):
        held_out = set(folds[k])
        _write(work, cfg, cfg['lm']['oof'].format(fold=k),
               {base: rows for base, rows in rows_by_base.items() if base not in held_out})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--config', default='configs/pagenet_only_style_neural.json')
    build_all(Work.from_env(), config.load(parser.parse_args().config))


if __name__ == '__main__':
    main()
