"""Character PPM-C language model with exclusions.

`Counts` holds the character n-gram counts of a corpus; `PPM` turns them into conditional probabilities. The
decoder asks for `P(letter | text so far)` for thousands of candidate prefixes per row, so the model is a plain
count table queried through a cache, never retrained during decoding.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from functools import lru_cache
from typing import Iterable

_CACHE_SIZE = 32768  # distinct contexts kept per model (a cache only: it never changes a probability)


def keep_chars(text: str, alphabet: str) -> str:
    """Upper-case `text` and drop every character outside `alphabet`."""
    allowed = set(alphabet)
    return ''.join(ch for ch in text.upper() if ch in allowed)


class Counts:
    """Character n-gram counts of a corpus: counts[m][context][ch] for contexts of m - 1 characters, m = 1..order.

    Level 0 is empty (it keeps m equal to the n-gram length). Each corpus line is counted on its own, so contexts
    never cross a row boundary.
    """

    def __init__(self, order: int, alphabet: str, counts: list[dict[str, Counter]]):
        self.order, self.alphabet, self.counts = order, alphabet, counts

    @classmethod
    def from_lines(cls, lines: Iterable[str], order: int, alphabet: str) -> 'Counts':
        """Count the n-grams (up to `order`) of every line after `keep_chars`; empty lines are skipped."""
        counts: list[defaultdict[str, Counter]] = [defaultdict(Counter) for _ in range(order + 1)]
        for raw in lines:
            line = keep_chars(raw, alphabet)
            for i, ch in enumerate(line):
                for m in range(1, min(order, i + 1) + 1):
                    counts[m][line[i - (m - 1):i] if m > 1 else ''][ch] += 1
        return cls(order, alphabet, [dict(level) for level in counts])


class PPM:
    """PPM-C with exclusions over `Counts`, using contexts of at most `max_order - 1` characters.

    At each position the longest context seen in the corpus is tried first. A letter seen there gets
    count / (total + distinct) of the remaining probability mass; the rest escapes, with probability
    distinct / (total + distinct), to the next shorter context, where letters already scored are excluded.
    Whatever mass is left after the empty context is shared by the letters never seen.
    """

    def __init__(self, counts: Counts, max_order: int):
        self.order = min(max_order, counts.order)
        if self.order < 1:
            raise ValueError('PPM max_order must be >= 1')
        self.counts = counts.counts
        self.vocab = set(counts.alphabet)
        self._distribution = lru_cache(maxsize=_CACHE_SIZE)(self._scores)

    def _scores(self, context: str) -> dict[str, float]:
        """log P(letter | context) for every letter of the alphabet."""
        escape = 1.0
        excluded: set[str] = set()
        scores: dict[str, float] = {}
        for m in range(min(self.order, len(context) + 1), 0, -1):
            seen = self.counts[m].get(context[-(m - 1):] if m > 1 else '')
            if not seen:
                continue
            fresh = {letter: count for letter, count in seen.items() if letter not in excluded}
            if not fresh:
                continue
            denominator = sum(fresh.values()) + len(fresh)
            for letter, count in fresh.items():
                scores[letter] = math.log(escape * count / denominator)
            escape *= len(fresh) / denominator
            excluded.update(fresh)
        unseen = math.log(escape / max(len(self.vocab) - len(excluded), 1))
        for letter in self.vocab - excluded:
            scores[letter] = unseen
        return scores

    def cond_logprob(self, letter: str, context: str) -> float:
        """log P(letter | context); `context` and `letter` consist of alphabet letters only."""
        return self._distribution(context[-(self.order - 1):] if self.order > 1 else '')[letter]

    def text_logprob(self, text: str) -> float:
        """Sum of log P(letter | preceding letters of `text`) over its letters; `text` has alphabet letters only.

        Computed as (sum / n) * n: the rounding of every score depends on it.
        """
        if not text:
            return 0.0
        return sum(self.cond_logprob(letter, text[:i]) for i, letter in enumerate(text)) / len(text) * len(text)


class Backward(PPM):
    """PPM-C with exclusions read right to left over the same `Counts`: P(letter | the letters that follow it).

    The counts of the n-grams that end in a letter are also the counts of the n-grams that start with it, so no
    second table is needed: the letters seen before `following` are those c for which the n-gram c + following was
    counted. `cond_logprob(letter, context)` takes as `context` the following letters in reverse order (the
    nearest last), so that a text is scored right to left by `text_logprob` of the reversed text and every wrapper
    of a `PPM` (mixture, cache) wraps this model unchanged.
    """

    def _scores(self, context: str) -> dict[str, float]:
        escape = 1.0
        excluded: set[str] = set()
        scores: dict[str, float] = {}
        for m in range(min(self.order, len(context) + 1), 0, -1):
            following = context[-(m - 1):][::-1] if m > 1 else ''
            if m == 1:
                seen = self.counts[1].get('') or {}
            else:
                level = self.counts[m]
                seen = {letter: count for letter in self.vocab
                        if (count := (level.get(letter + following[:-1]) or {}).get(following[-1], 0))}
            fresh = {letter: count for letter, count in seen.items() if letter not in excluded}
            if not fresh:
                continue
            denominator = sum(fresh.values()) + len(fresh)
            for letter, count in fresh.items():
                scores[letter] = math.log(escape * count / denominator)
            escape *= len(fresh) / denominator
            excluded.update(fresh)
        unseen = math.log(escape / max(len(self.vocab) - len(excluded), 1))
        for letter in self.vocab - excluded:
            scores[letter] = unseen
        return scores
