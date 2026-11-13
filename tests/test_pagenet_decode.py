"""The page network's rows: built from the kept tokens, completed by the doubtful ones that sit on a row
(trogs26/pagenet/decode.py)."""
from __future__ import annotations

import numpy as np
import pytest

from trogs26.pagenet import decode

H = 40.0  # letter height, pixels
# (x, y, keep probability) of each token; token i reads letter i
TOKENS = [(0, 0, 0.9), (100, 0, 0.9), (200, 0, 0.9),  # row 1, kept
          (150, 4, 0.3),                              # doubtful, on row 1 between tokens 1 and 2
          (0, 100, 0.9), (100, 100, 0.9),             # row 2, kept
          (100, 400, 0.3),                            # doubtful, on no row
          (300, 0, 0.02)]                             # below drop_below, on row 1


def decoder(monkeypatch, det_ids, gap=None):
    # quads as the detector records hold them: NW, SW, SE, NE
    boxes = [{'id': i, 'quad': [[x, y], [x, y + H], [x + 30, y + H], [x + 30, y]]} for i, (x, y, _) in enumerate(TOKENS)]
    monkeypatch.setattr(decode.artifacts, 'det_page', lambda chain, base: {'boxes': boxes, 'letter_height_px': H})
    lp = np.full((len(TOKENS), 28), np.log(1e-4))
    for i, (_, _, keep) in enumerate(TOKENS):
        lp[i, i], lp[i, 27] = np.log(keep), np.log(1 - keep)
    dec = object.__new__(decode.PageNetDecoder)
    dec.chain, dec.drop_below, dec.span_margin, dec.gap = None, 0.05, 2.5, gap
    dec.posteriors = {'x': (np.array(det_ids), lp)}
    return dec


def test_rows_come_from_the_kept_tokens_and_doubtful_tokens_join_the_row_they_sit_on(monkeypatch):
    rows = decoder(monkeypatch, range(len(TOKENS))).page_rows('x')
    assert [[int(np.argmax(e)) for e in emissions] for emissions, _ in rows] == [[0, 1, 3, 2], [4, 5]]
    assert [keep.round(2).tolist() for _, keep in rows] == [[0.9, 0.9, 0.3, 0.9], [0.9, 0.9]]


def test_with_a_gap_the_rows_carry_their_tokens_places_in_units_of_it(monkeypatch):
    rows = decoder(monkeypatch, range(len(TOKENS)), gap=1.6).page_rows('x')
    # kept neighbours stand 100 px apart, so a unit is 160 px: neighbours 0.625 apart, next but one 1.25 (a gap)
    assert np.allclose(np.diff(rows[0][2]), [0.625, 0.3125, 0.3125])
    assert np.allclose(np.diff(rows[1][2]), [0.625])


def test_a_reading_of_other_detections_is_refused(monkeypatch):
    with pytest.raises(ValueError):
        decoder(monkeypatch, [*range(len(TOKENS) - 1), 99]).page_rows('x')
