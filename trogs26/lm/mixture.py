"""Mixture of the chain's language model with an open-corpus language model.

The decoder's PPM-C is trained on the 224 public transcripts (62k letters). A PPM-C of a large open Greek corpus
knows far more of the language, but not of the squeezes' rows, so the two are mixed per letter in the probability
domain: P(letter | context) = (1 - weight) P_chain + weight P_open. The open model never contains a public
transcript, so the chain's fold exclusion is unchanged.
"""
from __future__ import annotations

import math
from functools import lru_cache

from .ppm import PPM

_CACHE_SIZE = 32768  # distinct contexts kept (a cache only: it never changes a probability)


class Mixture:
    """(1 - weight) * `base` + weight * `other`, per letter in the probability domain, with `PPM`'s interface
    (`cond_logprob`, `text_logprob`) and a per-context cache like `PPM._distribution`."""

    def __init__(self, base: PPM, other: PPM, weight: float):
        if not 0.0 < weight < 1.0:
            raise ValueError('mixture weight must be in (0, 1)')
        if base.vocab != other.vocab:
            raise ValueError('the mixed language models have different alphabets')
        self.base, self.other, self.weight = base, other, weight
        self.vocab = base.vocab
        self.reach = max(base.order, other.order) - 1  # letters of context that can matter
        self._log_base, self._log_other = math.log1p(-weight), math.log(weight)
        self._distribution = lru_cache(maxsize=_CACHE_SIZE)(self._scores)

    def _scores(self, context: str) -> dict[str, float]:
        """log P(letter | context) for every letter of the alphabet."""
        scores = {}
        for letter in self.vocab:
            a = self._log_base + self.base.cond_logprob(letter, context)
            b = self._log_other + self.other.cond_logprob(letter, context)
            top = max(a, b)
            scores[letter] = top + math.log(math.exp(a - top) + math.exp(b - top))
        return scores

    def cond_logprob(self, letter: str, context: str) -> float:
        """log P(letter | context); `context` and `letter` consist of alphabet letters only."""
        return self._distribution(context[-self.reach:] if self.reach else '')[letter]

    def text_logprob(self, text: str) -> float:
        """Sum of log P(letter | preceding letters of `text`) over its letters."""
        return float(sum(self.cond_logprob(letter, text[:i]) for i, letter in enumerate(text)))
