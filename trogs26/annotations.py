"""Ground truth of one squeeze as the official scorer sees it, parsed with the organisers' code.

Callers pass the two view files (`work.annotation(base, 1)`, `work.annotation(base, 2)`); only squeezes whose
annotations the run may read are ever loaded.
"""
from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .third_party import contest_evaluation_v2 as CE2


@dataclass
class GTSqueeze:
    """Ground truth of one squeeze as one view's letters file defines it (the official scorer scores the
    transcript against each view's file separately and sums).

    GT ids are transcript (reading-order) indices: `letters[g]` is the letter of `quads[g]`
    (official `orderBoxes` order, which is the order the transcript is written in).
    `ref_rows` are the official all-letters reference rows; `ref_conf[p]` is the confidence the
    official `evaluate` assigns to flattened reference position p (it zips the per-box union
    confidence onto the reference in reading order).
    """
    key: str
    letters: str
    quads: np.ndarray
    ref_rows: list[list[int]]
    ref_conf: list[bool]
    angle: float

    def core_reference(self) -> tuple[str, list[int | None]]:
        """Core reference string ('\\n'-joined confident rows) and its provenance (GT id or None for '\\n')."""
        it = iter(self.ref_conf)
        rows = []
        for row in self.ref_rows:
            kept = [g for g in row if next(it)]
            if kept:
                rows.append(kept)
        text, prov = [], []
        for r, row in enumerate(rows):
            if r:
                text.append('\n')
                prov.append(None)
            text.extend(self.letters[g] for g in row)
            prov.extend(row)
        return ''.join(text), prov

    def confident_ids(self) -> set[int]:
        flat = [g for row in self.ref_rows for g in row]
        return {g for g, c in zip(flat, self.ref_conf) if c}


def _quiet(fn, *args, **kwargs):
    """Call an official-scorer function with its diagnostic prints swallowed."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def squeeze_key(letters_name: str) -> str:
    """Squeeze identity of a letters file, as the official transcript lookup defines it.

    `clip_squeeze` cuts at the first `_Rotation` or `_Merged`; `_Merged` (stitched) may tag
    only one of the two views (`IAS_27119_Merged_Rotation2` pairs with `IAS_27119_Rotation1`),
    while `_Copy1`/`_Copy2` are distinct squeezes.
    """
    if '_Rotation' not in letters_name:
        raise ValueError(f'not a rotation letters file: {letters_name}')
    return CE2.clip_squeeze(letters_name)


def _has_confidence_section(path: Path) -> bool:
    text = path.read_text(encoding='utf-8', errors='replace')
    return sum(1 for line in text.splitlines() if line.startswith('#')) >= 5


def gt_squeeze(view_path: Path | str, other_path: Path | str) -> GTSqueeze:
    """GT of `view_path` (boxes in that view's pixel frame); `other_path` is the squeeze's other view, needed
    for the official union-of-views confidence."""
    view_path, other_path = Path(view_path), Path(other_path)
    gangle, boxes, transcript, lines = _quiet(CE2.readBoxFile, str(view_path), all_conf=True)
    letters = ''.join(transcript)
    boxes2 = _quiet(CE2.orderBoxes, boxes, gangle, lines)
    if not lines:
        rows, start = [], 0
        for line in transcript:
            rows.append(list(range(start, start + len(line))))
            start += len(line)
    else:
        _, ids = _quiet(CE2.getRowBoxes, boxes2, gangle=gangle)
        rows = [[int(i) for i in row] for row in ids]
    n_ref = sum(len(r) for r in rows)
    if not _has_confidence_section(view_path) and not _has_confidence_section(other_path):
        conf = [True] * n_ref  # every letter confident (also mirrors the official_scores fallback)
    else:
        c1 = _quiet(CE2.readBoxFile, str(view_path), all_conf=True, get_conf=True)
        c2 = _quiet(CE2.readBoxFile, str(other_path), all_conf=True, get_conf=True)
        if len(c1) != len(c2):
            raise ValueError(f'confidence vectors differ in length: {view_path}')
        conf = [a or b for a, b in zip(c1, c2)]
    official = _quiet(CE2.getRowTranscript, str(view_path), all_conf=True)
    mine = '\n'.join(''.join(letters[g] for g in row) for row in rows)
    if official != mine:
        raise AssertionError(f'reference reconstruction differs from getRowTranscript: {view_path}')
    quads = np.array([b.bd() for b in boxes2], dtype=np.float64).reshape(-1, 4, 2)
    return GTSqueeze(key=squeeze_key(view_path.name), letters=letters, quads=quads,
                     ref_rows=rows, ref_conf=conf[:n_ref], angle=float(gangle))
