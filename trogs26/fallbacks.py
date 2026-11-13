"""Counted substitutions: a stage used a documented substitute because an input was absent.

Every substitution goes through `note`, so run summaries can report how often each one happened. Worker
processes return `counts()` to their driver, which merges them.
"""
from __future__ import annotations

import logging
from collections import Counter

log = logging.getLogger('trogs26.fallbacks')
_COUNTS: Counter = Counter()


def note(kind: str, n: int = 1, detail: str = '') -> None:
    if n:
        _COUNTS[kind] += int(n)
        log.info('fallback %s +%d %s', kind, n, detail)


def counts() -> dict[str, int]:
    return dict(sorted(_COUNTS.items()))


def merge(*parts: dict[str, int]) -> dict[str, int]:
    total: Counter = Counter()
    for part in parts:
        total.update(part)
    return dict(sorted(total.items()))


def reset() -> None:
    _COUNTS.clear()
