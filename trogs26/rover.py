"""Combine several transcripts of one squeeze (the chains' decodes) into one.

- `medoid(texts)`: the transcript with the smallest summed edit distance to the others.
- `rover(texts)`: align every transcript to the medoid (Levenshtein opcodes; '\\n' is an ordinary symbol) and take
  a plurality vote per medoid position (letter / other letter / gap) and per insertion slot (nothing / inserted
  string). Ties keep the medoid's choice. Empty rows are dropped, as the scorer would count them as errors.
  `ins_frac` is the share of members that must insert the same string; a gap vote at a medoid position counts
  `gap_weight`.

    TROGS26_WORK=<work> python -m trogs26.rover --run <run> --split trogs26 [--config <file>]

writes `runs/<run>/<split>/rover/<squeeze>_transcript.txt` from the members' transcripts under
`runs/<run>/<split>/f<k>/<variant>/` (`ensemble.members` lists the variants), members ordered variant-major (every
chain of the first variant, then of the next).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

from rapidfuzz.distance import Levenshtein

from . import config, data, provenance
from .layout import Work
from .score import transcript_name


def medoid(texts: list[str]) -> str:
    cost = [sum(Levenshtein.distance(a, b) for b in texts) for a in texts]
    return texts[min(range(len(texts)), key=lambda i: (cost[i], i))]


def _align(pivot: str, text: str) -> tuple[list[str], list[str]]:
    """Per pivot position the aligned symbol of `text` ('' = gap), and per slot (before position i; slot len(pivot)
    is the end) the string `text` inserts there."""
    aligned, inserted = [''] * len(pivot), [''] * (len(pivot) + 1)
    for tag, i1, i2, j1, j2 in Levenshtein.opcodes(pivot, text):
        if tag in ('equal', 'replace'):
            n = min(i2 - i1, j2 - j1)
            for d in range(n):
                aligned[i1 + d] = text[j1 + d]
            if j2 - j1 > n:
                inserted[i1 + n] += text[j1 + n:j2]
        elif tag == 'insert':
            inserted[i1] += text[j1:j2]
    return aligned, inserted


def _drop_empty_rows(text: str) -> str:
    return '\n'.join(row for row in text.split('\n') if row)


def rover(texts: list[str], ins_frac: float, gap_weight: float) -> str:
    """Plurality vote of `texts` around their medoid (fewer than three texts: the medoid)."""
    if len(texts) < 3:
        return medoid(texts)
    pivot = medoid(texts)
    k = texts.index(pivot)
    others = [_align(pivot, t) for i, t in enumerate(texts) if i != k]
    out = []
    for slot in range(len(pivot) + 1):
        votes = Counter([''] + [inserted[slot] for _, inserted in others])
        best, n = votes.most_common(1)[0]
        if best and n > votes[''] and n > len(texts) * ins_frac:
            out.append(best)
        if slot < len(pivot):
            votes = Counter([pivot[slot]] + [aligned[slot] for aligned, _ in others])
            if '' in votes:
                votes[''] *= gap_weight
            top = max(votes.values())
            out.append(pivot[slot] if votes[pivot[slot]] == top else
                       min((c for c, v in votes.items() if v == top), key=lambda c: (c == '', c)))
    return _drop_empty_rows(''.join(out))


def member_texts(work: Work, run: str, split: str, variants: list[str], folds: list[int],
                 bases: list[str]) -> list[dict[str, str]]:
    """Transcripts by base of every member: variant-major, then chains in fold order."""
    texts = []
    for variant in variants:
        for fold in folds:
            directory = work.run(run, split) / f'f{fold}' / variant
            texts.append({base: (directory / transcript_name(base)).read_text(encoding='utf8') for base in bases})
    return texts


def write_ensemble(work: Work, cfg: dict, run: str, split: str) -> None:
    """ROVER of the configured members for every squeeze of `split`."""
    bases = data.splits()[split]
    folds = sorted(data.folds(cfg['data']['folds']))
    members = member_texts(work, run, split, cfg['ensemble']['members'], folds, bases)
    settings = cfg['ensemble']['rover']
    out = work.run(run, split) / 'rover'
    out.mkdir(parents=True, exist_ok=True)
    for base in bases:
        text = rover([m[base] for m in members], settings['ins_frac'], settings['gap_weight'])
        (out / transcript_name(base)).write_text(text, encoding='utf8')
    (out.parent / 'rover_summary.json').write_text(json.dumps(
        {**provenance.stamp(cfg),
         'members': [f'{run}/f{k}/{v}' for v in cfg['ensemble']['members'] for k in folds],
         **settings, 'squeezes': len(bases)}, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--config', default='configs/pagenet_only_style_neural.json')
    parser.add_argument('--run', required=True)
    parser.add_argument('--split', required=True)
    args = parser.parse_args()
    write_ensemble(Work.from_env(), config.load(args.config), args.run, args.split)


if __name__ == '__main__':
    main()
