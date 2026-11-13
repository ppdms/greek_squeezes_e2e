"""A row's candidate texts rescored with a neural character model mixed into the language model.

With `decode.variants.<variant>.neural` ({"model": <stage>}) the final candidate pool of every row
(`decoder.row_pools`: after the cache pass, if any) keeps its visual, keep and length sums, and in its lm sum the
forward letters' part, the sum over the text's letters of log P_lm, becomes the sum of log((1 - a) P_lm + a P_nn),
for the neural weight a of each combo (`Combo.nn`; the model's scores are computed once per pool).
P_lm is the forward model that scored the pool (the cache mixture in the second pass), reading each run of letters
between gap marks from its start; P_nn is the neural model of `trogs26.lm.neural` reading the run from BOS. The
backward letters' part of a two-sided model, and the letters alone that two-sided scoring subtracts, stay as they
are. Chain k reads its out-of-fold pages with `models/<stage>/f<k>` (the base model fine-tuned on the other folds'
transcripts) and every other page with `models/<stage>/all` (fine-tuned on all of them), both written by `python -m
trogs26.lm.neural finetune`.

The search, and so the pool and the cache's first pass, is unchanged: only the final choice among a pool's texts is.
The neural model runs on cuda if there is one, else on the CPU.
"""
from __future__ import annotations

import numpy as np

from ..layout import Chain, Work
from ..lm.twosided import TwoSided
from .search import GAP, Pool

BATCH = 256  # windows the neural model scores at a time


def runs_of(text: str) -> list[str]:
    """The runs of letters of a candidate text (between gap marks), empty ones dropped."""
    return [part for part in text.split(GAP) if part]


class NeuralMix:
    """The neural model of one chain's pages."""

    def __init__(self, work: Work, chain: Chain, settings: dict, alphabet: str):
        import torch

        from ..lm import neural
        self.neural, self.lut = neural, neural.alphabet_lut(alphabet)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        directory = work.models(settings['model'], chain.fold if chain.group == 'oof' else None)
        model, meta = neural.load_model(directory, device)
        if meta['alphabet'] != alphabet:
            raise ValueError(f'{directory}: the model reads {meta["alphabet"]!r}, the decoder {alphabet!r}')
        self.model = model.eval()

    def rescore(self, pool: Pool, lm, weights: list[float]) -> list[Pool]:
        """`pool` (scored by `lm`) with its lm sums mixed as the module docstring says, at each of `weights`."""
        for a in weights:
            if not 0 < a < 1:
                raise ValueError(f'neural weight {a} is not in (0, 1)')
        runs = sorted({run for text in pool.texts for run in runs_of(text)})
        if not runs:
            return [pool] * len(weights)
        own_lm = lm.forward if isinstance(lm, TwoSided) else lm
        own = np.array([own_lm.cond_logprob(letter, text[:i]) for text in runs for i, letter in enumerate(text)])
        nn = self.neural.score(self.model, [self.neural.encode(text, self.lut) for text in runs], batch=BATCH)
        lengths = np.array([len(run) for run in runs])
        ends = np.cumsum(lengths)
        out = []
        for a in weights:
            cum = np.concatenate([[0.0], np.cumsum(np.logaddexp(np.log1p(-a) + own, np.log(a) + nn) - own)])
            per_run = dict(zip(runs, cum[ends] - cum[ends - lengths]))
            out.append(pool._replace(lm=pool.lm + np.array([sum(per_run[run] for run in runs_of(text))
                                                             for text in pool.texts])))
        return out


def neural_mix(work: Work, chain: Chain, cfg: dict, variant: str) -> NeuralMix | None:
    """The variant's neural mixture for `chain`'s pages, None if it has none."""
    settings = cfg['decode']['variants'][variant].get('neural')
    return None if settings is None else NeuralMix(work, chain, settings, cfg['alphabet'])
