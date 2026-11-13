"""Decoding the pages of one chain: language model, joint-emission search, per-squeeze cache.

The decoder of a chain (`pagenet.decode.PageNetDecoder`) holds what its pages share (language model, page-network
reads) and gives any page's rows; `decode_page` returns, for any page and any list of (lam, w, beta[, nn]) combos,
the page's transcript under each combo.
"""
from __future__ import annotations

from typing import NamedTuple

from ..layout import Chain, Work
from ..lm import artifact
from ..lm.mixture import Mixture
from ..lm.ppm import PPM, Backward
from ..lm.twosided import TwoSided
from .cache import CacheLM
from .rescore import NeuralMix
from .search import GAP, Combo, Pool, Search, candidate_pool, plain, select


class Cache(NamedTuple):
    """Per-squeeze cache settings: mixture weight, PPM order, and the combo of the first pass that feeds it."""
    gamma: float
    order: int
    ref: Combo


def chain_lm(work: Work, chain: Chain, cfg: dict) -> PPM | Mixture | TwoSided:
    """The chain's language model: fold-excluded for out-of-fold pages, the final one for all other pages. With
    `lm.open` (`name`, `weight`, `max_order`) it is mixed with the open-corpus model of that artifact. With
    `lm.two_sided` the candidate texts are scored from both sides (`lm.twosided`), the right-to-left model read
    from the same counts and mixed the same way."""
    name = cfg['lm']['oof'].format(fold=chain.fold) if chain.group == 'oof' else cfg['lm']['final']
    counts = artifact.load_counts(work.lm(name))
    open_lm = cfg['lm'].get('open')
    open_counts = None if open_lm is None else artifact.load_counts(work.lm(open_lm['name']))

    def model(reading: type[PPM]) -> PPM | Mixture:
        lm = reading(counts, cfg['lm']['decode_max_order'])
        if open_lm is None:
            return lm
        return Mixture(lm, reading(open_counts, open_lm['max_order']), open_lm['weight'])

    return TwoSided(model(PPM), model(Backward)) if cfg['lm'].get('two_sided') else model(PPM)


def cache_lm(lm, others: list[str], cache: Cache, alphabet: str):
    """`lm` mixed with the cache of the texts `others` (a squeeze's other rows); a two-sided model on both sides."""
    if isinstance(lm, TwoSided):
        return TwoSided(CacheLM(lm.forward, others, cache.gamma, cache.order, alphabet),
                        CacheLM(lm.backward, [text[::-1] for text in others], cache.gamma, cache.order, alphabet))
    return CacheLM(lm, others, cache.gamma, cache.order, alphabet)


def row_pools(inputs: list[tuple], lm, search: Search, cache: Cache | None) -> list[tuple[Pool, object]]:
    """For each row of a squeeze, given as (emissions, keep probabilities) or (emissions, keep probabilities,
    places along the row: `search`'s module docstring), its final candidate pool and the language model that
    scored it.

    With a cache every row is decoded twice: first with `lm`, then, given the first-pass texts (at `cache.ref`)
    of the squeeze's other rows, with the cache mixture. Without one, once with `lm`. The cache counts each run of
    letters between gap marks as a line of its own.
    """
    pools = [candidate_pool(*row[:2], lm, search, *row[2:]) for row in inputs]
    models = [lm] * len(pools)
    if cache is not None:
        first = [select(pool, [cache.ref])[0] for pool in pools]
        for r, row in enumerate(inputs):
            others = [part for j, text in enumerate(first) if j != r for part in text.split(GAP) if part]
            if others:
                models[r] = cache_lm(lm, others, cache, search.alphabet)
                pools[r] = candidate_pool(*row[:2], models[r], search, *row[2:])
    return list(zip(pools, models))


def decode_rows(inputs: list[tuple], combos: list[Combo], lm, search: Search,
                cache: Cache | None, neural: NeuralMix | None = None) -> list[list[str]]:
    """For each row of a squeeze (as `row_pools` takes them), its text under each of `combos`, without gap marks;
    with `neural` chosen from the pool it rescores at the combo's neural weight (`rescore`). A combo has a neural
    weight exactly when there is a neural model."""
    if any((combo.nn is None) != (neural is None) for combo in combos):
        raise ValueError('a combo has a neural weight exactly when the variant has a neural model')
    weights = sorted({combo.nn for combo in combos})
    rows = []
    for pool, model in row_pools(inputs, lm, search, cache):
        pools = [pool] if neural is None else neural.rescore(pool, model, weights)
        texts = [''] * len(combos)
        for weight, scored in zip(weights, pools):
            at = [i for i, combo in enumerate(combos) if combo.nn == weight]
            for i, text in zip(at, select(scored, [combos[i] for i in at])):
                texts[i] = plain(text)
        rows.append(texts)
    return rows


def decode_page(decoder, base: str, combos: list[Combo]) -> dict[Combo, str]:
    """Transcript of `base` under each of `combos` (rows joined by newlines, empty rows dropped) by `decoder` (a
    `PageNetDecoder`). With `decode.cache.enabled` every row is decoded twice (see `decode_rows`); with the
    variant's `neural` the final choice is made from the rescored pools."""
    per_row = decode_rows(decoder.page_rows(base), combos, decoder.lm, decoder.search, decoder.cache, decoder.neural)
    return {combo: '\n'.join(text for text in (row[g] for row in per_row) if text) for g, combo in enumerate(combos)}


def chain_decoder(work: Work, chain: Chain, cfg: dict, variant: str):
    """The decoder of `variant` for `chain`'s pages: the row decoder over the page network's output."""
    from ..pagenet.decode import PageNetDecoder
    return PageNetDecoder(work, chain, cfg, variant)
