"""The page the detector sees: scale frames, rendered pages and tiles, letter boxes as oriented rectangles.

A squeeze has two views of the same stone. The detector reads either one view alone (a mono page: the view's
grayscale after local normalisation, replicated into three channels) or both together (a packed page:
view 1 | view 2 registered onto it | their pixel-wise minimum). Pages are scaled so letters are a fixed number of
pixels high and cut into overlapping square tiles; detections are mapped back to original pixels.

All pixel arrays handed to the detector are RGB; OpenCV's BGR order only appears at file boundaries.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# Local normalisation of a page (flatten the background, denoise, equalise contrast).
_FLATTEN_RADIUS_PX = 41
_NLM = {'h': 10, 'templateWindowSize': 7, 'searchWindowSize': 21}
_CLAHE = {'clipLimit': 2., 'tileGridSize': (8, 8)}


def is_merged(*files) -> bool:
    """Whether a squeeze is a stitched scan: `_Merged` is in the name of one of its image files (a property of the
    file name, which the registration needs to know)."""
    return any('_Merged' in Path(file).name for file in files)


def quad_to_obb(quad, reading_angle=0.):
    """Minimum-area rectangle of a quad as (cx, cy, width, height, baseline angle in radians).

    The width edge points along the page's reading direction (within 45 degrees of `reading_angle`), so height
    is the letter height, not the rectangle's shorter side.
    """
    q = np.asarray(quad, np.float32).reshape(4, 2)
    if not np.isfinite(q).all():
        raise ValueError('quad must be finite')
    corners = cv2.boxPoints(cv2.minAreaRect(q)).astype(float)
    edges = np.roll(corners, -1, axis=0) - corners
    lengths = np.linalg.norm(edges, axis=1)
    if lengths.min() <= 0:
        raise ValueError('quad must have positive area')
    right = np.array([math.cos(reading_angle), math.sin(reading_angle)])
    alignment = edges @ right / lengths
    i = int(np.argmax(alignment))
    axis = edges[i] / lengths[i]
    return np.array([*corners.mean(0), lengths[i], lengths[(i + 1) % 4], math.atan2(axis[1], axis[0])])


def obb_to_quad(obb):
    """The rectangle's corners in the official order: NW, SW, SE, NE."""
    cx, cy, width, height, angle = np.asarray(obb, float)
    right = np.array([math.cos(angle), math.sin(angle)]) * width / 2
    down = np.array([-math.sin(angle), math.cos(angle)]) * height / 2
    center = np.array([cx, cy])
    return np.array([center - right - down, center - right + down,
                     center + right + down, center + right - down])


def canonical_quad(quad, reading_angle=0.):
    return obb_to_quad(quad_to_obb(quad, reading_angle))


def median_height(quads, reading_angle=0.):
    heights = [quad_to_obb(q, reading_angle)[3] for q in quads]
    if not heights:
        raise ValueError('cannot estimate letter height from zero boxes')
    return float(np.median(heights))


def estimate_reading_angle(quads):
    """Robust reading direction of detected letters (radians).

    Takes each rectangle's edge closest to horizontal and pools the angles circularly modulo pi/2, which
    resolves minAreaRect's edge ambiguity. Purely geometric: no annotation, no text.
    """
    if not len(quads):
        return 0.
    angles = np.array([quad_to_obb(q)[4] for q in quads])
    return float(math.atan2(np.mean(np.sin(4 * angles)), np.mean(np.cos(4 * angles))) / 4)


@dataclass(frozen=True)
class ScaleFrame:
    """A resize of a page, with OpenCV's half-pixel convention, mapping original pixels <-> scaled pixels."""

    source_size: tuple[int, int]
    size: tuple[int, int]

    @classmethod
    def from_height(cls, size, height, target):
        """The frame that makes letters of `height` pixels `target` pixels high."""
        if not np.isfinite(height) or height <= 0 or target <= 0:
            raise ValueError('letter heights must be positive and finite')
        factor = target / height
        return cls(tuple(size), tuple(max(1, round(x * factor)) for x in size))

    @classmethod
    def from_dict(cls, record):
        return cls(tuple(record['source_size']), tuple(record['size']))

    def apply(self, points):
        factors = np.array(self.size) / self.source_size
        return (np.asarray(points, float) + .5) * factors - .5

    def inverse(self, points):
        factors = np.array(self.size) / self.source_size
        return (np.asarray(points, float) + .5) / factors - .5

    def resize(self, image):
        return cv2.resize(image, self.size, interpolation=cv2.INTER_AREA if self.size[0] < image.shape[1]
                          else cv2.INTER_CUBIC)

    def to_dict(self):
        return {'source_size': list(self.source_size), 'size': list(self.size)}


def preprocess(gray, scale):
    """Flatten the background, denoise and equalise a grayscale page that was reduced by `scale`."""
    bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=max(1., _FLATTEN_RADIUS_PX * scale))
    flat = cv2.divide(gray, bg, scale=255)
    den = cv2.fastNlMeansDenoising(flat, None, **_NLM)
    return cv2.createCLAHE(**_CLAHE).apply(den)


def warp_to_reference(image2, registration, frame):
    """Sample view 2 into the scaled view-1 frame through the inverse registration."""
    inverse = registration.inverse()
    width, height = frame.size
    result = np.empty((height, width), np.uint8)
    # Bound memory on giant scans: the warp is evaluated only at scaled pixels, in stripes of 128 rows.
    for start in range(0, height, 128):
        yy, xx = np.mgrid[start:min(height, start + 128), :width]
        reference = frame.inverse(np.column_stack([xx.ravel(), yy.ravel()]))
        # A piecewise inverse allocates a points x controls basis several times per Newton step: chunk it.
        source = np.empty(reference.shape, np.float32)
        chunk = 4096 if registration.model == 'piecewise' else len(reference)
        for offset in range(0, len(reference), chunk):
            source[offset:offset + chunk] = inverse.apply(reference[offset:offset + chunk])
        source = source.reshape(*xx.shape, 2)
        result[start:start + len(xx)] = cv2.remap(image2, source[:, :, 0], source[:, :, 1], cv2.INTER_LINEAR,
                                                borderMode=cv2.BORDER_REPLICATE)
    return result


def make_page(gray1, gray2, registration, frame, *, packed):
    """RGB page of `frame.size`: mono (view 1 in all channels) or packed (view 1, registered view 2, minimum)."""
    scale = frame.size[0] / frame.source_size[0]
    red = preprocess(frame.resize(gray1), scale)
    if not packed:
        return np.repeat(red[:, :, None], 3, axis=2)
    if gray2 is None or registration is None:
        raise ValueError('packed pages need a second view and registration')
    green = preprocess(warp_to_reference(gray2, registration, frame), scale)
    return np.stack([red, green, np.minimum(red, green)], axis=2)


def tile_windows(size, side, overlap):
    """(x0, y0, x1, y1) windows covering a page of `size`; axes not longer than `side` get one unpadded tile."""
    if side <= 0 or not 0 <= overlap < 1:
        raise ValueError('invalid tile settings')

    def starts(length):
        if length <= side:
            return [0]
        step = max(1, round(side * (1 - overlap)))
        positions = list(range(0, length - side + 1, step))
        if positions[-1] != length - side:
            positions.append(length - side)
        return positions

    width, height = size
    return [(x, y, min(width, x + side), min(height, y + side)) for y in starts(height) for x in starts(width)]


def tile_owner(center, windows):
    """Index of the covering window whose centre is closest to `center` (ties: first), or -1 if none covers it.

    A detection is kept only from its owning tile, which removes cross-tile duplicates without any box NMS.
    """
    point = np.asarray(center)
    distances = []
    for x0, y0, x1, y1 in windows:
        inside = x0 <= point[0] <= x1 and y0 <= point[1] <= y1
        distances.append(np.sum((point - [(x0 + x1) / 2, (y0 + y1) / 2]) ** 2) if inside else np.inf)
    return int(np.argmin(distances)) if np.isfinite(distances).any() else -1
