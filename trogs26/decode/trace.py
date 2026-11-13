"""Which position of a row emits each letter of a decoded text.

A row's text is traced back to the path that emits exactly it with the highest
    sum over emitted positions of (log P(letter) + w log k) + sum over skipped positions of w log(1 - k),
under the search's own constraints (letters among each position's `topk`, positions with k >= k_hi always emit);
for a fixed text the language model and the length bonus are constant, so only the keep weight `w` matters.
"""
from __future__ import annotations

import numpy as np

from .search import Search, top_letters


def row_path(emit: np.ndarray, keep: np.ndarray, text: str, w: float, search: Search) -> list[int]:
    """The position that emits each letter of `text` on the best path through the row (`keep` clipped)."""
    n, m = len(emit), len(text)
    skippable = keep < search.k_hi
    log_keep = np.where(skippable, np.log(keep), 0.0)
    log_skip = np.where(skippable, np.log1p(-keep), 0.0)
    allowed = [dict(top_letters(emit[j], search.alphabet, search.topk)) for j in range(n)]
    score = np.full((n + 1, m + 1), -np.inf)
    score[0, 0] = 0.0
    for j in range(n):
        if skippable[j]:
            score[j + 1] = score[j] + w * log_skip[j]
        for i in range(m):
            logp = allowed[j].get(text[i])
            if logp is not None and score[j, i] > -np.inf:
                score[j + 1, i + 1] = max(score[j + 1, i + 1], score[j, i] + logp + w * log_keep[j])
    if score[n, m] == -np.inf:
        raise ValueError(f'no path of the row emits {text!r}')
    path, i = [], m
    for j in range(n - 1, -1, -1):
        logp = allowed[j].get(text[i - 1]) if i else None
        if logp is not None and score[j, i - 1] + logp + w * log_keep[j] == score[j + 1, i]:
            path.append(j)
            i -= 1
    if i:
        raise ValueError(f'the path of {text!r} does not trace back')
    return path[::-1]
