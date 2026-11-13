"""The relief of a squeeze, recovered from its two raking-light views.

A squeeze is paper pressed into the letters of a stone. Each view photographs it under light grazing the paper from
one direction; the two directions are roughly 90 degrees apart, so each view shows the strokes that run across its
light and loses the strokes that run along it. For a matte surface of uniform brightness lit from azimuth a at a
low elevation e, the brightness relative to the flat paper around a point is, to first order in the slopes,

    s = I / I_paper - 1 = -cot(e) * (cos(a) * h_x + sin(a) * h_y),

a directional derivative of the paper's height h. In this collection one view of a squeeze is lit along the image's
x axis and the other along its y axis (which is which differs between squeezes and is read off the shadings: the
x-lit view has most of its gradient energy along x). So, up to scale and sign, the x-lit shading is h_x and the y-lit
one h_y. Both then share the mixed derivative h_xy (d/dy of the x-lit view, d/dx of the y-lit view), an image that
does not depend on the lighting: it fixes the relative sign of the two derivatives and, matched tile by tile with
phase correlation, corrects what the registration of view 2 onto view 1 left (a few pixels, against strokes about a
tenth of a letter high). With equal light elevations both shadings get the same scale. From the gradient (p, q):

    height     h, by Frankot-Chellappa integration (FFT), minus its large-scale shape;
    curvature  div(p, q) = laplacian(h): every stroke is a line, whatever its orientation;
    slope      |(p, q)|;
    relit      the shading under canonical lights (three azimuths 120 degrees apart), the same for every squeeze.

Which side each light came from is not in the images: lighting a groove from one side looks like lighting a ridge
from the other, so the field is known up to its overall sign. That sign is set so that letters look alike on every
squeeze: the curvature inside the letters (given detections) is skewed negative -- a chiselled stroke is a narrow,
strong curvature line flanked by weak curvature of the other sign.

Where the two views do not agree locally (a stitched scan whose parts no registration fits, or paper only view 1
shows), view 2's derivative is faded out: the gradient there is view 1's alone, which is what one view can give.

Every length is a multiple of the letter height, so small and large letters get the same treatment. Pages are
processed in view 1's frame, at a working scale where letters are `letter_px` high (never upscaled).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
import scipy.fft
from scipy.stats import skew

from .registration import Registration

PAPER_FLOOR = 0.08          # smoothed brightness (0-1) below which a pixel is scan background, not paper
FLATTEN_LH = 1.5            # width of the paper-brightness estimate, letter heights
DERIVATIVE_LH = 0.02        # smoothing of the shadings before differentiation, letter heights
HEIGHT_HIGHPASS_LH = 1.0    # large-scale shape taken out of the integrated height, letter heights
CURVATURE_LH = 0.03         # smoothing of the gradient before taking its divergence, letter heights
SLOPE_LH = 0.02             # smoothing of the slope magnitude, letter heights
RELIT_AZIMUTHS = (90.0, 210.0, 330.0)
MIXED_LH = 0.05            # smoothing of the mixed derivatives (alignment, agreement), letter heights
# Alignment passes, coarse to fine: (tile side, largest shift), in letter heights. The coarse pass catches stitched
# (merged) scans whose parts the registration placed letters apart. (Smoother derivatives do not help it: at
# letter scale the two views' shadings of the paper's bending differ, and their correlation can change sign.)
ALIGN_LEVELS = ((8.0, 3.0), (3.0, 0.15), (3.0, 0.15))
ALIGN_ACCEPT = 0.1         # a tile's shift is used only if its mixed derivatives then correlate at least this much
RELIABILITY_TILE_LH = 3.0  # tiles of the local agreement map
RELIABLE = (0.1, 0.3)      # view 2's share of the gradient rises from 0 to 1 as the local agreement crosses these


@dataclass
class Fusion:
    """A squeeze's relief quantities on view 1's frame at `scale` (working pixels per original pixel)."""
    scale: float
    letter_px: float        # letter height in working pixels
    mask: np.ndarray        # bool, view 1's paper: the quantities cover it
    both: np.ndarray        # bool, the paper both views show (view 2 aligned)
    s1: np.ndarray          # relative shadings, float32
    s2: np.ndarray          # (view 2 aligned onto view 1)
    p: np.ndarray           # height gradient, float32 (view 2's component scaled by `share`)
    q: np.ndarray
    x_view: int             # the view lit along x (1 or 2); 0 when both views are lit along the same axis
    agreement: float        # correlation of the two views' mixed derivatives: 1 for a perfectly integrable pair
    reliability: np.ndarray # local agreement (same correlation over tiles of a few letters), float32 per pixel
    share: np.ndarray       # view 2's weight in the gradient (0-1, from `reliability`), float32 per pixel
    shift: np.ndarray       # (2, h, w) residual shift applied to view 2 (sampled at x + shift), working pixels
    shift_px: float         # its median length over the paper
    flipped: bool           # the overall sign was flipped to make the letters' curvature skew negative


def working_scale(letter_height: float, letter_px: float) -> float:
    """Factor that brings letters `letter_height` original pixels high to `letter_px` (at most 1)."""
    return min(1.0, letter_px / letter_height)


def _unit(gray: np.ndarray) -> np.ndarray:
    return gray.astype(np.float32) / 255.0


def _blur(x: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur of width `sigma`; wide ones are computed on a copy reduced so that sigma becomes 4 pixels and
    interpolated back (the result is smooth at that scale, so nothing is lost)."""
    factor = sigma / 4.0
    if factor < 2:
        return cv2.GaussianBlur(x, (0, 0), sigma)
    h, w = x.shape
    small = cv2.resize(x, (max(1, round(w / factor)), max(1, round(h / factor))), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigma / factor)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def paper_mask(gray: np.ndarray, letter_px: float) -> np.ndarray:
    """Pixels on the paper: smoothed brightness above `PAPER_FLOOR` (scans are padded with black)."""
    return _blur(gray, 0.25 * letter_px) > PAPER_FLOOR


def flatten(gray: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    """Relative shading s = I / I_paper - 1, with I_paper a Gaussian average of width `sigma` over paper pixels only
    (normalised convolution, so the black scan border does not darken the paper next to it); 0 off the paper."""
    m = mask.astype(np.float32)
    paper = _blur(gray * m, sigma) / np.maximum(_blur(m, sigma), 1e-6)
    return np.where(mask, gray / np.maximum(paper, 1e-3) - 1.0, 0.0).astype(np.float32)


def second_view_on_first(gray2: np.ndarray, registration: Registration, shape1: tuple[int, int], scale: float,
                         step: int = 16) -> np.ndarray:
    """View 2 (original pixels, uint8 or float) resampled onto view 1's frame at `scale`.

    `registration` maps view 2 to view 1 (original pixels); its inverse is evaluated on a grid every `step` working
    pixels and interpolated in between. View 2 is first reduced to the working scale, so the resampling does not
    alias. Pixels that fall outside view 2 are 0 (scan background).
    """
    h, w = shape1
    small2 = cv2.resize(gray2, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray2
    gy, gx = np.mgrid[0:h + step:step, 0:w + step:step].astype(np.float64)
    original1 = np.stack([(gx.ravel() + .5) / scale - .5, (gy.ravel() + .5) / scale - .5], axis=1)
    original2 = registration.inverse().apply(original1)
    coarse_x = (scale * (original2[:, 0] + .5) - .5).reshape(gx.shape).astype(np.float32)
    coarse_y = (scale * (original2[:, 1] + .5) - .5).reshape(gy.shape).astype(np.float32)
    # pixel (x, y) lies at (x / step, y / step) on the coarse grid (grid points at multiples of `step`)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32) / step
    map_x = cv2.remap(coarse_x, xx, yy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    map_y = cv2.remap(coarse_y, xx, yy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return cv2.remap(small2, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _derivative(s: np.ndarray, sigma: float, dx: int, dy: int) -> np.ndarray:
    smooth = cv2.GaussianBlur(s, (0, 0), sigma) if sigma > 0 else s
    return cv2.Sobel(smooth, cv2.CV_32F, dx, dy, ksize=3, scale=1 / 8)


def x_lit_view(s1: np.ndarray, s2: np.ndarray, mask: np.ndarray, sigma: float) -> int:
    """1 or 2: the view whose shading varies most along x (lit along x); 0 if both or neither do."""
    def along_x(s):
        return float(np.mean(_derivative(s, sigma, 1, 0)[mask] ** 2) / max(np.mean(_derivative(s, sigma, 0, 1)[mask] ** 2), 1e-20))
    r1, r2 = along_x(s1), along_x(s2)
    return 1 if r1 > 1 > r2 else 2 if r2 > 1 > r1 else 0


def mixed_derivatives(s1: np.ndarray, s2: np.ndarray, x_view: int, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Each view's estimate of h_xy (up to scale and sign): d/dy of the x-lit view, d/dx of the y-lit view."""
    one = _derivative(s1, sigma, 0, 1) if x_view == 1 else _derivative(s1, sigma, 1, 0)
    two = _derivative(s2, sigma, 1, 0) if x_view == 1 else _derivative(s2, sigma, 0, 1)
    return one, two


def _correlation(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    x, y = a[mask].astype(np.float64), b[mask].astype(np.float64)
    x, y = x - x.mean(), y - y.mean()
    return float((x * y).sum() / max(math.sqrt((x * x).sum() * (y * y).sum()), 1e-20))


def _tile_side(shape: tuple[int, int], letter_heights: float, letter_px: float) -> int:
    """Even side of square tiles `letter_heights` letters high, at least 32 px and at most the page's smaller side."""
    return max(2, min(max(32, int(letter_heights * letter_px)), *shape)) // 2 * 2


def _tiles(shape: tuple[int, int], tile: int) -> tuple[list[int], list[int]]:
    """Rows and columns of the top-left corners of tiles overlapping by half (`tile` fits the page)."""
    step = tile // 2
    return (list(range(0, shape[0] - tile + 1, step)), list(range(0, shape[1] - tile + 1, step)))


def _upsample(grid: np.ndarray, ys: list[int], xs: list[int], tile: int, shape: tuple[int, int]) -> np.ndarray:
    """Per-pixel bilinear interpolation of per-tile values (constant beyond the outermost tile centres)."""
    h, w = shape
    centres_y = np.array(ys, np.float32) + tile / 2
    centres_x = np.array(xs, np.float32) + tile / 2
    xx, yy = np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32)
    rows = np.stack([np.interp(xx, centres_x, grid[i]) for i in range(len(ys))])
    return np.stack([np.interp(yy, centres_y, rows[:, x]) for x in range(w)], axis=1).astype(np.float32)


def residual_shift(m1: np.ndarray, m2: np.ndarray, mask: np.ndarray, letter_px: float, tile_lh: float = 3.0,
                   max_lh: float = 0.15) -> np.ndarray | None:
    """(2, h, w) field d such that view 2 sampled at x + d(x) aligns with view 1, from phase correlation of the
    mixed derivatives (`m2` already sign-matched to `m1`) over overlapping tiles of `tile_lh` letter heights. A tile
    takes its phase-correlation shift if that is at most `max_lh` letter heights and correlates the derivatives
    better than leaving the tile in place, and at least `ALIGN_ACCEPT` (a letter grid also correlates, less well, a
    letter pitch away); it stays in place if that correlates at least `ALIGN_ACCEPT`. The other tiles take the median
    of the decided ones around them (or of all). None if no tile is decided."""
    tile = _tile_side(m1.shape, tile_lh, letter_px)
    limit = max_lh * letter_px
    window = cv2.createHanningWindow((tile, tile), cv2.CV_32F)
    ys, xs = _tiles(m1.shape, tile)
    grid = np.full((2, len(ys), len(xs)), np.nan, np.float32)
    for i, y0 in enumerate(ys):
        for j, x0 in enumerate(xs):
            if mask[y0:y0 + tile, x0:x0 + tile].mean() < 0.8:
                continue
            a = np.ascontiguousarray(m1[y0:y0 + tile, x0:x0 + tile])
            b = np.ascontiguousarray(m2[y0:y0 + tile, x0:x0 + tile])
            (dx, dy), _ = cv2.phaseCorrelate(a, b, window)
            shifted = math.hypot(dx, dy) <= limit
            margin = int(math.ceil(max(abs(dx), abs(dy)))) + 1 if shifted else 1
            inner = np.zeros((tile, tile), bool)
            inner[margin:tile - margin, margin:tile - margin] = True
            if not inner.any():
                continue
            still = _correlation(a, b, inner)
            if shifted:
                moved = cv2.warpAffine(b, np.float32([[1, 0, dx], [0, 1, dy]]), (tile, tile),
                                       flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
                better = _correlation(a, moved, inner)
                shifted = better >= ALIGN_ACCEPT and better > still
            if shifted:
                grid[:, i, j] = dx, dy
            elif still >= ALIGN_ACCEPT:
                grid[:, i, j] = 0.0, 0.0
    known = ~np.isnan(grid[0])
    if not known.any():
        return None
    for c in range(2):                       # fill undecided tiles from decided neighbours, then smooth
        layer = grid[c]
        filled = layer.copy()
        for i, j in zip(*np.nonzero(~known)):
            near = layer[max(i - 1, 0):i + 2, max(j - 1, 0):j + 2]
            near = near[~np.isnan(near)]
            filled[i, j] = np.median(near) if len(near) else np.median(layer[known])
        layer[:] = filled
        grid[c] = cv2.medianBlur(layer, 3) if min(layer.shape) >= 3 else layer
    return np.stack([_upsample(grid[c], ys, xs, tile, m1.shape) for c in range(2)])


def reliability_map(m1: np.ndarray, m2: np.ndarray, mask: np.ndarray, letter_px: float) -> np.ndarray:
    """Local agreement: correlation of the two views' mixed derivatives over tiles of `RELIABILITY_TILE_LH` letter
    heights, interpolated per pixel; 0 where a tile is mostly off the paper."""
    tile = _tile_side(m1.shape, RELIABILITY_TILE_LH, letter_px)
    ys, xs = _tiles(m1.shape, tile)
    grid = np.zeros((len(ys), len(xs)), np.float32)
    for i, y0 in enumerate(ys):
        for j, x0 in enumerate(xs):
            inside = mask[y0:y0 + tile, x0:x0 + tile]
            if inside.mean() >= 0.5:
                grid[i, j] = _correlation(m1[y0:y0 + tile, x0:x0 + tile], m2[y0:y0 + tile, x0:x0 + tile], inside)
    return _upsample(grid, ys, xs, tile, m1.shape)


def _sample(image: np.ndarray, field: np.ndarray, fill: float) -> np.ndarray:
    h, w = image.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return cv2.remap(image, xx + field[0], yy + field[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                     borderValue=fill)


def align(s1: np.ndarray, s2: np.ndarray, mask1: np.ndarray, mask2: np.ndarray, x_view: int,
          letter_px: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """View 2's shading moved onto view 1's (`ALIGN_LEVELS`), the paper both then show, and the total shift.

    The views' relative sign is unknown until they are aligned: where the registration left them more than a stroke
    apart their mixed derivatives hardly correlate, of either sign. So the coarse pass runs with both signs and the
    one whose result correlates better is kept for the finer passes."""
    zero = np.zeros((2, *s1.shape), np.float32)
    sigma = MIXED_LH * letter_px
    m1, m2 = mixed_derivatives(s1, s2, x_view, sigma)
    best = None
    for sign in (1.0, -1.0):
        field = residual_shift(m1, sign * m2, mask1 & mask2, letter_px, *ALIGN_LEVELS[0])
        field = zero if field is None else field
        moved = _sample(s2, field, 0.0)
        mask = mask1 & (_sample(mask2.astype(np.float32), field, 0.0) > .5)
        score = sign * _correlation(m1, mixed_derivatives(s1, moved, x_view, sigma)[1], mask)
        if best is None or score > best[0]:
            best = score, sign, field, moved, mask
    _, sign, total, s2, mask = best
    for tile_lh, max_lh in ALIGN_LEVELS[1:]:
        m1, m2 = mixed_derivatives(s1, s2, x_view, sigma)
        field = residual_shift(m1, sign * m2, mask, letter_px, tile_lh, max_lh)
        if field is None:
            continue
        s2 = _sample(s2, field, 0.0)
        total = total + field
        mask = mask1 & (_sample(mask2.astype(np.float32), total, 0.0) > .5)
    return s2, mask, total


def integrate(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Height whose gradient is closest to (p, q) in the least-squares sense (Frankot-Chellappa, periodic)."""
    rows, cols = p.shape
    wx = (2 * np.pi * scipy.fft.fftfreq(cols)).astype(np.float32)[None, :]
    wy = (2 * np.pi * scipy.fft.fftfreq(rows)).astype(np.float32)[:, None]
    denominator = wx ** 2 + wy ** 2
    denominator[0, 0] = 1.0
    spectrum = (-1j * wx * scipy.fft.fft2(p, workers=-1) - 1j * wy * scipy.fft.fft2(q, workers=-1)) / denominator
    spectrum[0, 0] = 0.0
    return scipy.fft.ifft2(spectrum, workers=-1).real.astype(np.float32)


def fuse(gray1: np.ndarray, gray2: np.ndarray, registration: Registration, letter_height: float,
         letter_quads=None, letter_px: float = 140.0) -> Fusion:
    """Relief quantities of one squeeze from its two uint8 grayscale views (original pixels), the registration of
    view 2 onto view 1, the letter height (original pixels) and the quads of likely letters (original pixels,
    view 1; the polarity statistic falls back to the whole paper without them)."""
    scale = working_scale(letter_height, letter_px)
    lh = letter_height * scale
    one = cv2.resize(gray1, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray1
    two = second_view_on_first(gray2, registration, one.shape[:2], scale)
    one, two = _unit(one), _unit(two)
    mask1 = paper_mask(one, lh)
    s1 = flatten(one, mask1, FLATTEN_LH * lh)
    mask2 = paper_mask(two, lh)
    s2 = flatten(two, mask2, FLATTEN_LH * lh)
    mask = mask1 & mask2
    x_view = x_lit_view(s1, s2, mask, MIXED_LH * lh)
    shift = np.zeros((2, *mask.shape), np.float32)
    if x_view:
        s2, mask, shift = align(s1, s2, mask1, mask2, x_view, lh)
    m1, m2 = mixed_derivatives(s1, s2, x_view or 1, MIXED_LH * lh)
    agreement = _correlation(m1, m2, mask)
    sign = 1.0 if agreement >= 0 else -1.0
    reliability = reliability_map(m1, sign * m2, mask, lh)
    share = np.where(mask, np.clip((reliability - RELIABLE[0]) / (RELIABLE[1] - RELIABLE[0]), 0, 1), 0)
    share = share.astype(np.float32)
    shift_px = float(np.median(np.hypot(*shift)[mask])) if mask.any() else 0.0
    first = np.where(mask1, s1 / max(float(np.sqrt(np.mean(s1[mask1] ** 2))), 1e-6), 0.0)
    second = sign * share * s2 / max(float(np.sqrt(np.mean(s2[mask] ** 2))) if mask.any() else 1.0, 1e-6)
    p, q = (first, second) if x_view != 2 else (second, first)
    f = Fusion(scale, lh, mask1, mask, s1, s2, p.astype(np.float32), q.astype(np.float32), x_view, abs(agreement),
               reliability, share, shift, shift_px, False)
    letters = mask1
    if letter_quads is not None and len(letter_quads):
        drawn = np.zeros(mask1.shape, np.uint8)
        for quad in letter_quads:
            cv2.fillConvexPoly(drawn, np.round((np.asarray(quad, np.float64) + .5) * scale - .5).astype(np.int32), 1)
        if (drawn.astype(bool) & mask1).any():
            letters = drawn.astype(bool) & mask1
    if skew(curvature(f)[letters]) > 0:
        f.p, f.q, f.flipped = -f.p, -f.q, True
    return f


def height(f: Fusion) -> np.ndarray:
    """Integrated height minus its large-scale shape (the paper's bending), 0 off the paper."""
    h = integrate(f.p, f.q)
    h = h - _blur(h, HEIGHT_HIGHPASS_LH * f.letter_px)
    return np.where(f.mask, h, 0.0).astype(np.float32)


def curvature(f: Fusion) -> np.ndarray:
    """div(p, q) = laplacian of the height, at stroke scale; 0 off the paper."""
    sigma = CURVATURE_LH * f.letter_px
    k = _derivative(f.p, sigma, 1, 0) + _derivative(f.q, sigma, 0, 1)
    return np.where(f.mask, k, 0.0).astype(np.float32)


def slope(f: Fusion) -> np.ndarray:
    sigma = SLOPE_LH * f.letter_px
    return np.where(f.mask, cv2.GaussianBlur(np.hypot(f.p, f.q), (0, 0), sigma), 0.0).astype(np.float32)


def relit(f: Fusion, azimuth_degrees: float) -> np.ndarray:
    """Relative shading under light from `azimuth_degrees` (0 = from +x, 90 = from +y), to first order."""
    a = math.radians(azimuth_degrees)
    return np.where(f.mask, -(math.cos(a) * f.p + math.sin(a) * f.q), 0.0).astype(np.float32)


def to_uint8(x: np.ndarray, mask: np.ndarray, signed: bool, spread: float = 4.0) -> np.ndarray:
    """uint8 image of `x` over the paper. Signed quantities: 128 + 127 x / (spread robust sigmas), the robust sigma
    being 1.4826 times the median absolute value; magnitudes: 255 x / (their 99th percentile). Off the paper: 128
    (signed) or 0."""
    values = np.abs(x[mask])
    if signed:
        unit = float(np.median(values)) * 1.4826 if len(values) else 1.0
        unit = unit if unit > 0 else 1.0
        out = 128.0 + 127.0 * np.clip(x / (spread * unit), -1, 1)
        return np.where(mask, out, 128.0).astype(np.uint8)
    top = float(np.percentile(values, 99)) if len(values) else 1.0
    out = 255.0 * np.clip(x / (top if top > 0 else 1.0), 0, 1)
    return np.where(mask, out, 0.0).astype(np.uint8)
