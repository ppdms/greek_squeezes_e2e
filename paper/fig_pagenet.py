"""Geometry and drawing helpers of the `neighbourhood` figure of paper/make_figures.py (figs/neighbourhood.jpg).

The figure shows the part `REGION` of the showcase squeeze (`figstyle.SHOWCASE`), in letter heights in the deskewed
frame of `pages.build`, over the relit relief lit from azimuth 90 degrees; `QUERY` is the detection id of its query
token. `to_uv` carries points from Rotation1 pixels into that frame and `backdrop` draws the relit relief over a
window of it. `to_uv` and `backdrop` take the page's record (`.record`), its tokens (`.page`) and its relief at
`.scale` working pixels per original pixel (`.background`), as a `SimpleNamespace`. `_pale` lightens the backdrop
and `_poly` draws a token's oriented box; `DUPLICATE` is the colour of a token that is none because another token
claims the letter it overlaps, `ATTENTION` that of the query's neighbourhood, and `HALO` outlines text drawn over the
image.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import cv2
import numpy as np
from matplotlib import patheffects
from matplotlib.patches import Polygon

QUERY = 302                         # detection id of the query token: an H with an ambiguous crop and duplicates
REGION = (22.9, 8.4, 33.1, 13.1)    # u0, v0, u1, v1 (letter heights, deskewed frame): the query, the 31 tokens it
                                    # attends to besides itself and the detections on the stone's edge to its right
DUPLICATE = '#E08214'               # none, because another token claims the letter it overlaps
ATTENTION = '#6A3D9A'
HALO = [patheffects.withStroke(linewidth=1.6, foreground='white')]


def to_uv(run: SimpleNamespace, points) -> np.ndarray:
    """(..., 2) points in Rotation1 pixels -> (..., 2) (u, v) in letter heights, the deskewed frame of the tokens."""
    c, s = math.cos(run.page.angle), math.sin(run.page.angle)
    lh = run.record['letter_height_px']
    p = np.asarray(points, np.float64)
    return np.stack([(p[..., 0] * c + p[..., 1] * s) / lh, (-p[..., 0] * s + p[..., 1] * c) / lh], -1)


def backdrop(run: SimpleNamespace, u0: float, v0: float, u1: float, v1: float, per_lh: float) -> np.ndarray:
    """The relit relief over [u0, u1] x [v0, v1] letter heights in the deskewed frame, `per_lh` pixels per letter
    height, (h, w, 3) uint8; off the page grey."""
    c, s = math.cos(run.page.angle), math.sin(run.page.angle)
    lh = run.record['letter_height_px']
    ratio = per_lh / (lh * run.scale)  # destination pixels per source pixel

    def source(x, y):  # destination pixel -> source pixel
        u, v = u0 + (x + 0.5) / per_lh, v0 + (y + 0.5) / per_lh
        return ((lh * (u * c - v * s) + 0.5) * run.scale - 0.5, (lh * (u * s + v * c) + 0.5) * run.scale - 0.5)

    width, height = round((u1 - u0) * per_lh), round((v1 - v0) * per_lh)
    corners = np.array([source(x, y) for x, y in ((0, 0), (width, 0), (0, height), (width, height))])
    x0, y0 = np.maximum(np.floor(corners.min(0)).astype(int) - 2, 0)
    x1, y1 = np.minimum(np.ceil(corners.max(0)).astype(int) + 3, run.background.shape[1::-1])
    window = cv2.resize(run.background[y0:y1, x0:x1], None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)
    (ax, ay), (bx, by), (cx, cy) = (source(*p) for p in ((0, 0), (1, 0), (0, 1)))
    # destination -> window pixels (the window is the source cropped at (x0, y0) and scaled by `ratio`)
    affine = np.array([[(bx - ax) * ratio, (cx - ax) * ratio, (ax - x0 + 0.5) * ratio - 0.5],
                       [(by - ay) * ratio, (cy - ay) * ratio, (ay - y0 + 0.5) * ratio - 0.5]])
    return cv2.warpAffine(window, affine, (width, height), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(128, 128, 128))


def _pale(image: np.ndarray, strength: float = 0.55) -> np.ndarray:
    """One channel of the relit relief (the light from azimuth 90 degrees) as a pale grey, so that boxes stand out."""
    return (255 - strength * (255 - image[..., 0].astype(np.float32))).clip(0, 255).astype(np.uint8)


def _poly(ax, quad, colour: str, fill: float = 0.15, lw: float = 0.7, ls: str = '-', z: float = 3) -> None:
    ax.add_patch(Polygon(quad, closed=True, fill=False, edgecolor=colour, lw=lw, ls=ls, zorder=z))
    if fill:
        ax.add_patch(Polygon(quad, closed=True, facecolor=colour, alpha=fill, edgecolor='none', zorder=z - 0.5))
