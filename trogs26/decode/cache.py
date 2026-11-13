"""Per-squeeze cache language model.

A squeeze repeats itself (formulae, names, restored text), so after a first decoding pass the other rows of the
same squeeze are evidence about each row: the cache is a PPM-C trained on them and mixed into the base model.
No label and no text from any other squeeze enters the cache.
"""
from __future__ import annotations

import math

from ..lm.ppm import PPM, Counts


class CacheLM:
    """(1 - gamma) * base + gamma * cache per letter, mixed in the log domain; the cache is a PPM-C of order
    `order` over `rows`, the first-pass transcripts of the squeeze's other rows."""

    def __init__(self, base: PPM, rows: list[str], gamma: float, order: int, alphabet: str):
        if not 0.0 < gamma < 1.0:
            raise ValueError('cache gamma must be in (0, 1)')
        self.base = base
        self.cache = PPM(Counts.from_lines(rows, order, alphabet), order)
        self.log_base, self.log_cache = math.log1p(-gamma), math.log(gamma)

    def cond_logprob(self, letter: str, context: str) -> float:
        a = self.log_base + float(self.base.cond_logprob(letter, context))
        b = self.log_cache + float(self.cache.cond_logprob(letter, context))
        m = max(a, b)
        return m + math.log(math.exp(a - m) + math.exp(b - m))

    def text_logprob(self, text: str) -> float:
        return float(sum(self.cond_logprob(letter, text[:i]) for i, letter in enumerate(text)))
