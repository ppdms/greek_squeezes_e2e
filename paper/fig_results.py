"""Helpers of the results figures of paper/make_figures.py (`sweep`, `squeezes`, `rover`) and of the numbers they quote.

- `read_sweep`: the P decoder's out-of-fold sweep at the configured external weight (`make_results.final_run`,
  `runs/<run>/oof/sweep_P.json`: Core errors and characters of every public squeeze, both views, at every point of the
  grid `decode.variants.P.grid`), the operating point `decode.variants.P.combo` and the totals per grid point. The
  operating point is checked to be the minimum of the grid, as paper/make_results.py does over every external weight.
  `_grid_errors` lays the errors out on the (lam, w, beta) of that grid at the operating point's neural weight; `pct`,
  `tex_name` and `combo_tex` write numbers, names and grid points as LaTeX.
- `Squeeze`, `typical_squeeze`: a public squeeze's Core errors and characters at the operating point, and the typical
  squeeze (of the squeezes with 150-450 characters in view 1, the one whose CER is nearest the median of theirs).
  `WORST_SHARE` is the share of the squeezes, those of highest CER, whose errors `squeezes` quotes.
- `ballot`, `pivot_rows`: the ROVER vote (`trogs26.rover`) over the five chains of a TROGS-26 squeeze with its decisions
  kept: the pivot (the chain with the smallest summed edit distance to the others; ties to the first), every chain
  aligned to it (`trogs26.rover._align`), the voted symbols and the accepted insertions. The replayed vote must give
  the text of `trogs26.rover.rover`. `pivot_rows` marks where each row of the pivot lies in those positions.
No annotation of TROGS-26 is read and nothing is scored on it.
"""
from __future__ import annotations

import json
from collections import Counter
from typing import NamedTuple

import figstyle
import make_results
import numpy as np

from trogs26 import data
from trogs26 import rover as rover_system
from trogs26.decode.search import Combo

VARIANT = 'P'
TYPICAL_CHARS = (150, 450)  # the typical squeeze has this many Core characters in view 1
WORST_SHARE = 0.1  # `squeezes` quotes the errors made on this share of the squeezes, those of highest CER


def pct(errors: int, chars: int, places: int = 2) -> str:
    return f'{100 * errors / chars:.{places}f}\\%'


def tex_name(base: str) -> str:
    return base.replace('_', '\\_')


def read_sweep(work, cfg: dict) -> tuple[dict, Combo, dict[str, list[int]]]:
    """The out-of-fold sweep of the P decoder, its frozen operating point and the point's totals per combo key.
    The operating point must be the minimum of the grid."""
    settings = cfg['decode']['variants'][VARIANT]
    combo = Combo(**settings['combo'])
    sweep = json.loads((work.run(make_results.final_run(cfg), 'oof') / f'sweep_{VARIANT}.json').read_text())
    if sorted(sweep['scores']) != sorted(data.all_bases(figstyle.folds(cfg))):
        raise ValueError('the sweep does not cover the squeezes of the folds file')
    grid_totals = make_results.totals(sweep)
    least = min(errors for errors, _ in grid_totals.values())
    if grid_totals[combo.key][0] != least:
        raise ValueError(f'the operating point {combo.key} makes {grid_totals[combo.key][0]} errors, '
                         f'the grid minimum is {least}')
    return sweep, combo, grid_totals


def combo_tex(key: str) -> str:
    """A combo key ('0.80|2.00|2.00', or with the neural weight '0.80|2.00|2.00|0.70') as TeX: the weights of the
    point."""
    values = key.split('|')
    names = ['\\lambda', 'w', '\\beta', '\\nu'][:len(values)]
    return f'$({", ".join(names)}) = ({", ".join(f"{float(v):g}" for v in values)})$'


def _grid_errors(cfg: dict, grid_totals: dict[str, list[int]]) -> tuple[np.ndarray, dict[str, list[float]]]:
    """Errors at every (lam, w, beta) of the configured grid at the operating point's neural weight, indexed
    [lam, w, beta] in the order of the grid, and that grid."""
    settings = cfg['decode']['variants'][VARIANT]
    grid = cfg['decode']['grids'][settings['grid']]
    nn = settings['combo'].get('nn')
    errors = np.array([[[grid_totals[Combo(lam, w, beta, nn).key][0] for beta in grid['beta']] for w in grid['w']]
                       for lam in grid['lam']])
    return errors, grid


class Squeeze(NamedTuple):
    base: str
    errors: int  # Core errors of both views
    chars: int  # Core characters of both views
    chars1: int  # Core characters of view 1

    @property
    def cer(self) -> float:
        return self.errors / self.chars


def typical_squeeze(squeezes: list[Squeeze]) -> Squeeze:
    """Of the squeezes with TYPICAL_CHARS Core characters in view 1, the one whose CER is nearest the median of theirs
    (ties: the name)."""
    pool = [s for s in squeezes if TYPICAL_CHARS[0] <= s.chars1 <= TYPICAL_CHARS[1]]
    median = float(np.median([s.cer for s in pool]))
    return min(pool, key=lambda s: (abs(s.cer - median), s.base))


class Ballot(NamedTuple):
    """One squeeze's vote: the pivot (member index `k`), every member aligned to it, and the voted text."""
    texts: list[str]
    pivot: str
    k: int
    aligned: list[list[str]]  # per member, per pivot position: its symbol ('' = gap)
    inserted: list[list[str]]  # per member, per slot (before position i; len(pivot) = the end): what it inserts
    symbols: list[str]  # per pivot position: the voted symbol ('' = removed)
    accepted: list[str]  # per slot: the voted insertion ('' = none)
    text: str  # the ROVER output


def ballot(texts: list[str], ins_frac: float, gap_weight: float) -> Ballot:
    """The vote of `trogs26.rover.rover` with its decisions kept. It repeats rover's loop (which returns only the
    joined text) and must give rover's text."""
    pivot = rover_system.medoid(texts)
    k = texts.index(pivot)
    aligned, inserted = [], []
    for i, text in enumerate(texts):
        a, ins = (list(pivot), [''] * (len(pivot) + 1)) if i == k else rover_system._align(pivot, text)
        aligned.append(a)
        inserted.append(ins)
    others = [i for i in range(len(texts)) if i != k]
    symbols, accepted = [], []
    for slot in range(len(pivot) + 1):
        votes = Counter([''] + [inserted[i][slot] for i in others])
        best, n = votes.most_common(1)[0]
        accepted.append(best if best and n > votes[''] and n > len(texts) * ins_frac else '')
        if slot < len(pivot):
            votes = Counter([pivot[slot]] + [aligned[i][slot] for i in others])
            if '' in votes:
                votes[''] *= gap_weight
            top = max(votes.values())
            symbols.append(pivot[slot] if votes[pivot[slot]] == top else
                           min((c for c, v in votes.items() if v == top), key=lambda c: (c == '', c)))
    text = rover_system._drop_empty_rows(''.join(x for pair in zip(accepted, symbols + ['']) for x in pair))
    if text != rover_system.rover(texts, ins_frac, gap_weight):
        raise ValueError('the replayed vote differs from trogs26.rover.rover')
    return Ballot(texts, pivot, k, aligned, inserted, symbols, accepted, text)


class Row(NamedTuple):
    index: int  # of the row in the pivot, from 1
    start: int  # of its first letter in the pivot
    end: int  # one past its last letter (the position of its line break, or the end)


def pivot_rows(b: Ballot) -> list[Row]:
    rows, start = [], 0
    for index, line in enumerate(b.pivot.split('\n'), 1):
        end = start + len(line)
        rows.append(Row(index, start, end))
        start = end + 1
    return rows
