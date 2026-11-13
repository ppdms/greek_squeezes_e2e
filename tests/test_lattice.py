"""Rows and page-angle fit of trogs26.lattice on synthetic pages."""
from __future__ import annotations

import math

import numpy as np
import pytest

from trogs26 import lattice


def test_rows_follow_a_tilted_line():
    u = np.tile(np.arange(30) * 1.1, 2)
    v = np.repeat([0.0, 1.6], 30) + 0.02 * u  # a residual tilt of 1.1 degrees would merge rows with a v-only cut
    assert lattice.row_labels(u, v).tolist() == [0] * 30 + [1] * 30


def rotated_det(angle, rows_=8, columns=14, height=40.0):
    """A detector record of text lines running at `angle` radians."""
    boxes = []
    for r in range(rows_):
        for c in range(columns):
            x, y = (c * 1.3 + 0.03 * (-1) ** c) * height, r * 1.8 * height
            cx, cy = x * math.cos(angle) - y * math.sin(angle), x * math.sin(angle) + y * math.cos(angle)
            boxes.append({'id': len(boxes), 'p_letter': 0.9, 'quad': [[cx - 20, cy - 20]] * 4})
    return {'boxes': boxes, 'letter_height_px': height}


@pytest.mark.parametrize('truth', [0.0, 0.2, -0.35])
def test_fit_angle_recovers_the_text_line_direction(truth):
    det = rotated_det(truth)
    wrong = truth + math.radians(1.7)  # the starting estimate is off by more than a degree
    assert lattice.fit_angle(det, wrong) == pytest.approx(truth, abs=math.radians(0.1))


def test_fit_angle_keeps_the_starting_angle_without_enough_letters():
    det = rotated_det(0.1, rows_=2, columns=5)
    assert lattice.fit_angle(det, 0.3) == 0.3
