"""Loader for one chain's detector records, in the format the detector stage writes (`detect.records`)."""
from __future__ import annotations

import json
from functools import lru_cache

from .layout import Chain


@lru_cache(maxsize=None)
def det_page(chain: Chain, base: str) -> dict:
    """Complete calibrated detector record of `base` (every detection, letter height)."""
    return json.loads(chain.det(base).read_text())
