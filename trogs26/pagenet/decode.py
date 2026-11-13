"""The row decoder over the page network's output.

Each token's keep probability is 1 - P(none) and its emission the 27 letter log-probabilities renormalised. The
tokens the network keeps (keep probability at least 0.5) give the page angle and are grouped into rows by the
organisers' v2 grouping; tokens with keep probability in [`drop_below`, 0.5) join the row they sit on
(`decode.rows.attach`, `decode.extras.span_margin`), or are dropped. Each row is then decoded by the joint search of
`decode.search` with the chain's language model and the per-squeeze cache (`decode.decoder`).

With the variant's `gap` (in letter pitches) each row also carries its tokens' places along it, in units of that
many pitches, so that the search restarts the language model's context wherever the letters it emits stand
further apart (`decode.search`). The pitch is the median distance between neighbouring kept tokens of the page's
rows.
"""
from __future__ import annotations

import math

import numpy as np

from .. import artifacts, fallbacks
from ..decode import rows as row_grouping
from ..decode.decoder import Cache, chain_lm
from ..decode.rescore import neural_mix
from ..decode.search import Combo, Search
from ..layout import Chain, Work
from . import pages
from .infer import posteriors

KEEPS = 0.5  # a token the network keeps: it is more probably a letter than none


def log_normalise(x: np.ndarray) -> np.ndarray:
    """Row-wise log-softmax."""
    x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


class PageNetDecoder:
    """Decoder for the pages `chain.bases` of one chain under one variant (`cfg['decode']['variants'][variant]`)."""

    def __init__(self, work: Work, chain: Chain, cfg: dict, variant: str):
        settings = cfg['decode']['variants'][variant]
        self.chain, self.cfg = chain, cfg
        self.search = Search.from_config(cfg)
        self.lm = chain_lm(work, chain, cfg)
        self.drop_below = settings['drop_below']
        if not 0 <= self.drop_below < KEEPS:
            raise ValueError(f'variant {variant}: drop_below must lie in [0, {KEEPS})')
        self.gap = settings.get('gap')
        if self.gap is not None and self.gap <= 0:
            raise ValueError(f'variant {variant}: gap must be positive')
        self.span_margin = cfg['decode']['extras']['span_margin']
        self.posteriors = posteriors(chain, settings['pagenet'], cfg['alphabet'])
        cache = cfg['decode']['cache']
        self.cache = (Cache(cache['gamma'], cache['order'], Combo(**settings['cache_ref']))
                      if cache['enabled'] else None)
        self.neural = neural_mix(work, chain, cfg, variant)

    def page_tokens(self, base: str) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray | None]:
        """(rows, emissions, keep probabilities, places) of `base`: each row the indices of its tokens in the
        detector record, reading order; the emissions (n, 27) and keep probabilities (n,) of every token of the
        record; with `gap`, every row token's place along its row in units of `gap` pitches, else None."""
        det = artifacts.det_page(self.chain, base)
        det_ids, lp = self.posteriors[base]
        if not np.array_equal(det_ids, [int(b['id']) for b in det['boxes']]):
            raise ValueError(f'{base}: the page network read other detections than the record holds')
        keep = 1.0 - np.exp(lp[:, -1].astype(np.float64))
        emit = log_normalise(lp[:, :-1].astype(np.float64)).astype(np.float32)
        candidates = np.flatnonzero(keep >= self.drop_below)
        if not len(candidates):
            fallbacks.note('pagenet_page_empty', 1, base)
            return [], emit, keep, None
        letter_height = float(det['letter_height_px'])
        angle = pages.text_angle(det['boxes'], letter_height, keep, KEEPS)
        quads = np.array([det['boxes'][i]['quad'] for i in candidates], np.float64).reshape(-1, 4, 2)
        centre = quads.mean(1)
        along = centre @ np.array([math.cos(angle), math.sin(angle)])
        across = centre @ np.array([-math.sin(angle), math.cos(angle)])
        strong = np.flatnonzero(keep[candidates] >= KEEPS)
        weak = np.flatnonzero(keep[candidates] < KEEPS)
        rows = [strong[row] for row in row_grouping.group_rows(quads[strong], angle)]
        if not rows:
            fallbacks.note('pagenet_page_ungrouped', 1, base)
            return [], emit, keep, None
        places = None
        if self.gap is not None:
            steps = [np.diff(np.sort(along[row])) for row in rows if len(row) > 1]
            pitch = float(np.median(np.concatenate(steps))) if steps else letter_height
            places = np.full(len(keep), np.nan)
            places[candidates] = along / (self.gap * (pitch if pitch > 0 else letter_height))
        if len(weak):
            joins = row_grouping.attach([along[r] for r in rows], [across[r] for r in rows], along[weak],
                                        across[weak], letter_height, self.span_margin)
            rows = [np.concatenate([row, weak[joins == r]]) for r, row in enumerate(rows)]
        return [candidates[row[np.argsort(along[row], kind='stable')]] for row in rows], emit, keep, places

    def page_rows(self, base: str) -> list[tuple]:
        """The rows of `base` as (emissions (n, 27), keep probabilities (n,)), reading order; with `gap`, also
        the tokens' places along the row (n,), in units of `gap` pitches."""
        rows, emit, keep, places = self.page_tokens(base)
        return [(emit[r], keep[r]) if places is None else (emit[r], keep[r], places[r]) for r in rows]
