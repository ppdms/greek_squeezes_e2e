"""Text lines of a page from its detections: the rows and the page angle.

Points are deskewed centres in units of the page's letter height: u along the text lines, v across them
(u = x cos a + y sin a, v = -x sin a + y cos a at page angle a). Everything here is label-free (detector record
only) and NaN-free.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from . import fallbacks

P_MIN = 0.5            # detections at or above this probability are letters
LINK_U = 2.5           # two letters belong to one row if they are within LINK_U along the row ...
LINK_V = 0.4           # ... and LINK_V across it (letter heights)


def row_labels(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Row index of every point: rows are the connected groups of points linked as above, numbered by median v."""
    near = (np.abs(u[:, None] - u[None, :]) <= LINK_U) & (np.abs(v[:, None] - v[None, :]) <= LINK_V)
    count, label = connected_components(csr_matrix(near), directed=False)
    order = np.argsort([np.median(v[label == k]) for k in range(count)], kind='stable')
    rank = np.empty(count, dtype=np.int64)
    rank[order] = np.arange(count)
    return rank[label]


def fit_angle(det: dict, angle: float, window: float = 4.0, step: float = 0.05, bin_size: float = 0.1,
              tolerance: float = 0.35, min_letters: int = 20) -> float:
    """Page angle (radians) from the detections' own text lines: the angle within `window` degrees of `angle` (steps
    of `step` degrees) at which the centres of the letters (p >= P_MIN) project sharpest onto the row
    normal (largest sum of squared counts of a histogram with bins of `bin_size` letter heights), refined by
    a pooled least-squares slope through the members of every row, trimmed at `tolerance` letter heights.
    Falls back to `angle` with fewer than `min_letters` letters."""
    boxes = [b for b in det['boxes'] if b['p_letter'] >= P_MIN]
    if len(boxes) < min_letters:
        fallbacks.note('angle_fit_too_few_letters', 1)
        return angle
    centre = np.array([np.asarray(b['quad'], float).mean(0) for b in boxes])
    height = float(det['letter_height_px'])

    def deskew(a):
        return ((centre[:, 0] * math.cos(a) + centre[:, 1] * math.sin(a)) / height,
                (-centre[:, 0] * math.sin(a) + centre[:, 1] * math.cos(a)) / height)

    candidates = angle + np.deg2rad(np.arange(-window, window + step / 2, step))
    sharpness = []
    for a in candidates:
        bins = np.floor(deskew(a)[1] / bin_size).astype(np.int64)
        sharpness.append(float((np.bincount(bins - bins.min()).astype(np.float64) ** 2).sum()))
    best = float(candidates[int(np.argmax(sharpness))])
    u, v = deskew(best)
    label = row_labels(u, v)
    keep = np.bincount(label)[label] >= 5
    rows = int(label.max()) + 1
    for _ in range(3):
        if keep.sum() < min_letters:
            return best
        size = np.maximum(np.bincount(label[keep], minlength=rows), 1)
        mean_u = (np.bincount(label[keep], u[keep], minlength=rows) / size)[label]
        mean_v = (np.bincount(label[keep], v[keep], minlength=rows) / size)[label]
        slope = ((u - mean_u) * (v - mean_v))[keep].sum() / ((u - mean_u) ** 2)[keep].sum()
        keep &= np.abs((v - mean_v) - slope * (u - mean_u)) <= tolerance
        keep &= np.bincount(label, keep.astype(float))[label] >= 5
    return best + math.atan(slope)
