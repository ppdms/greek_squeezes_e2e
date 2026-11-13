"""Joint-emission decoding of one row: which boxes emit a letter, and which letter, are decided together.

A row is a sequence of positions, each with letter log-probabilities `emit` and a keep probability k. A candidate
text takes at every position either one of its `topk` letters or, if the position is skippable (k < k_hi), nothing.
Beam search proposes candidates with a (lam, w, beta) weighting
    sum over emitted positions of (log P(letter) + w log k + beta + lam log P_LM(letter | text so far))
    + sum over skipped positions of w log(1 - k)
under several proposal weightings (`passes`). Every proposed text is then scored once for the final weighting:
    visual + w * keep + lam * language model + beta * length,
so the text of any (lam, w, beta) combination is an argmax over the pool, with no further search.

A row may come with its positions' places along the row (`along`, in units of the spacing that makes a gap). A
squeeze row is no continuous text: where letters are worn away the annotators leave them out, and a row can cross
columns. So a letter emitted more than one unit after the last emitted letter is preceded in the candidate text by
`GAP`, and the language model reads only the letters since the last `GAP`, as it does at the start of a row (it
falls back to its shorter contexts, which average over the text it cannot see). `plain` removes the marks.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

_KEEP_CLIP = 1e-4
GAP = '|'  # in a candidate text: letters are missing here (no letter of any alphabet)


class Combo(NamedTuple):
    """Final weighting: language-model weight, keep weight, per-letter length bonus and, for a variant with a neural
    model, that model's weight in the language model (`rescore`)."""
    lam: float
    w: float
    beta: float
    nn: float | None = None

    @property
    def key(self) -> str:
        """The combo's name in result files, e.g. '1.25|2.00|4.00', with a neural weight '1.25|2.00|4.00|0.70'."""
        key = f'{self.lam:.2f}|{self.w:.2f}|{self.beta:.2f}'
        return key if self.nn is None else f'{key}|{self.nn:.2f}'


def grid_combos(grid: dict) -> list[Combo]:
    """Every combination of a grid's `lam`, `w`, `beta` and, if it has them, neural weights `nn`, lam-major."""
    return [Combo(lam, w, beta, nn) for lam in grid['lam'] for w in grid['w'] for beta in grid['beta']
            for nn in grid.get('nn', [None])]


@dataclass(frozen=True)
class Search:
    """Beam-search settings: alphabet, beam width, letters per position, k_hi and the proposal passes (module
    docstring)."""
    alphabet: str
    beam: int
    topk: int
    k_hi: float
    passes: tuple[Combo, ...]

    @classmethod
    def from_config(cls, cfg: dict) -> 'Search':
        d = cfg['decode']
        return cls(cfg['alphabet'], d['beam'], d['char_topk'], d['k_hi'],
                   tuple(Combo(p['lam'], p['w'], p['beta']) for p in d['passes']))


class Pool(NamedTuple):
    """Candidate texts of one row (sorted) with the sums that score them."""
    texts: list[str]
    visual: np.ndarray
    keep: np.ndarray
    lm: np.ndarray
    length: np.ndarray


def top_letters(emit_row: np.ndarray, alphabet: str, k: int) -> list[tuple[str, float]]:
    """The `k` most probable letters of one position, best first (equal log-probs keep alphabet order)."""
    letters = [(alphabet[j], float(emit_row[j])) for j in range(len(alphabet))]
    letters.sort(key=lambda item: item[1], reverse=True)
    return letters[:k]


def plain(text: str) -> str:
    """A candidate text without its gap marks."""
    return text.replace(GAP, '')


def text_logprob(lm, text: str) -> float:
    """The language model's log-probability of a candidate text: of each run of letters between gap marks."""
    if GAP not in text:
        return lm.text_logprob(text)
    return float(sum(lm.text_logprob(part) for part in text.split(GAP)))


def _beam(positions, skippable, log_keep, log_skip, lm, lam: float, beta: float, w: float, width: int,
          along=None):
    """Beam search under one weighting. Returns (text, score, visual, keep, last) tuples, best first.

    `visual` and `keep` are the sums of letter log-probs and of keep / skip log-probs along the path, `last` the
    position that emitted the text's last letter (-1: none). With `along`, paths are told apart by their text and
    `last` (the next gap depends on both); without, by their text alone.
    """
    beams = [('', 0.0, 0.0, 0.0, -1)]
    for j, letters in enumerate(positions):
        best: dict = {}
        for text, score, visual, keep, last in beams:
            if skippable[j]:
                skipped = score + w * log_skip[j]
                key = text if along is None else (text, last)
                held = best.get(key)
                if held is None or skipped > held[1]:
                    best[key] = (text, skipped, visual, keep + log_skip[j], last)
            base = score + w * log_keep[j] + beta
            context = text
            if along is not None:
                if last >= 0 and along[j] - along[last] > 1.0:
                    text += GAP
                context = text[text.rfind(GAP) + 1:]
            for letter, logp in letters:
                bonus = 0.0 if lam == 0.0 else lam * float(lm.cond_logprob(letter, context))
                extended = base + logp + bonus
                grown = text + letter
                key = grown if along is None else (grown, j)
                held = best.get(key)
                if held is None or extended > held[1]:
                    best[key] = (grown, extended, visual + logp, keep + log_keep[j], j)
        paths = list(best.values())
        beams = (heapq.nlargest(width, paths, key=lambda r: r[1]) if len(paths) > width
                 else sorted(paths, key=lambda r: -r[1]))
    return beams


def candidate_pool(emit: np.ndarray, keep: np.ndarray, lm, search: Search, along=None) -> Pool:
    """Pool of one row: the union of the beams of every proposal pass, each text with its best feature sums.

    `lm` supplies `cond_logprob(letter, context)` and `text_logprob(text)`. With `along` (the positions' places
    along the row, see the module docstring) the texts carry gap marks.
    """
    positions = [top_letters(emit[j], search.alphabet, search.topk) for j in range(len(emit))]
    k = np.clip(np.asarray(keep, dtype=np.float64), _KEEP_CLIP, 1 - _KEEP_CLIP)
    skippable = k < search.k_hi
    log_keep = np.where(skippable, np.log(k), 0.0).tolist()
    log_skip = np.where(skippable, np.log1p(-k), 0.0).tolist()
    places = None if along is None else np.asarray(along, dtype=np.float64).tolist()
    found: dict[str, tuple[float, float]] = {}
    for lam, w, beta in ((p.lam, p.w, p.beta) for p in search.passes):
        if not skippable.any() and (beta != 0.0 or w != 1.0):
            continue  # without skips, beta and w shift every path equally
        for text, _score, visual, kept, _last in _beam(positions, skippable, log_keep, log_skip, lm, lam, beta, w,
                                                       search.beam, places):
            held = found.get(text)
            if held is None or visual + kept > held[0] + held[1]:
                found[text] = (visual, kept)
    texts = sorted(found)
    return Pool(texts,
                np.array([found[t][0] for t in texts]),
                np.array([found[t][1] for t in texts]),
                np.array([text_logprob(lm, t) for t in texts]),
                np.array([len(t) - t.count(GAP) for t in texts], dtype=np.float64))


def select(pool: Pool, combos: list[Combo]) -> list[str]:
    """The best text of the pool under each combo (the first of equal scores in sorted text order), with its gap
    marks."""
    lam = np.array([c.lam for c in combos])[None, :]
    w = np.array([c.w for c in combos])[None, :]
    beta = np.array([c.beta for c in combos])[None, :]
    score = pool.visual[:, None] + w * pool.keep[:, None] + lam * pool.lm[:, None] + beta * pool.length[:, None]
    return [pool.texts[i] for i in np.argmax(score, axis=0)]
