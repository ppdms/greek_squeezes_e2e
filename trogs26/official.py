"""The organisers' evaluation modules (`third_party/`), imported with their diagnostic prints silenced.

Their parsers print on malformed input; stages that call them (row grouping, transcript extraction, scoring) import
the modules from here so that stays quiet without editing the verbatim files.
"""
from __future__ import annotations

from .third_party import contest_evaluation_v1 as v1
from .third_party import contest_evaluation_v2 as v2


def _silent(*_args, **_kwargs) -> None:
    return None


for _module in (v1, v2):
    _module.print = _silent
