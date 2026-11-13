"""Tokens of a page: one per detection of the chain's calibrated detector record, in record order.

Geometry is label-free. The page angle comes from the detections themselves: the circular mean of the confident
boxes' top edges, refined on their text lines (`lattice.fit_angle`). Positions and sizes are measured in the deskewed
frame in units of the page's letter height: u along the rows, v across them. Overlaps are those of the boxes as
rectangles in that frame.

Targets read the annotations of the squeeze's Rotation1 view. A token's class is the letter of the annotated letter
it wins in the one-to-one match at rotated IoU >= `label_iou` (maximum cardinality first), or none: each annotated
letter is claimed by exactly one detection, and duplicates are none. Among the detections that could claim a letter,
`claim` decides which does: 'iou' the one that overlaps the annotated box most (maximum total IoU), 'p' the one the
detector is surest of (maximum total calibrated probability, then IoU). The annotated box is not seen at test time,
the probability is, so under 'p' the network can tell which of two overlapping detections is the letter. The reader
class, which supervises the crop encoding alone, follows `dino.labels.page_labels` instead (matched at IoU >=
`reader_iou`: the letter; best IoU below `blank_iou`: none; otherwise ignored), since a crop by itself cannot tell a
duplicate from the detection that claims the letter.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from .. import lattice
from ..annotations import gt_squeeze
from ..dino.labels import page_labels
from ..geometry import match_box_ious, rotated_iou_matrix
from ..layout import Work

NODE = ('lp', 'score', 's_mono1', 's_mono2', 's_pack', 'lh', 'lw', 'laspect', 'sin2', 'cos2', 'ov_max', 'ov_higher',
        'lov_n', 'page_lh', 'page_ln', 'page_hi', 'single_view', 'inliers')
PAIR = ('du', 'dv', 'ldist', 'iou', 'dlh', 'dlw', 'dlp', 'self')
IGNORE = -1
OVERLAP_NEIGHBOURS = 16  # node overlap features look at this many nearest detections
QUALITY_USABLE = 0.9     # registration inlier share below which a page is single-view


def logit(p, eps: float = 1e-4) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)
    return np.log(p) - np.log1p(-p)


def estimated_angle(quads) -> float:
    """Baseline angle (radians) of boxes: the circular mean of their top-edge directions (NW to NE), refined by the
    median deviation from it."""
    q = np.asarray(quads, dtype=float).reshape(-1, 4, 2)
    if not len(q):
        return 0.0
    edges = q[:, 3] - q[:, 0]
    angles = np.arctan2(edges[:, 1], edges[:, 0])
    centre = math.atan2(np.sin(angles).sum(), np.cos(angles).sum())
    return float(centre + np.median((angles - centre + np.pi) % (2 * np.pi) - np.pi))


def text_angle(boxes: list[dict], letter_height: float, p: np.ndarray, threshold: float = 0.5) -> float:
    """Page angle from the boxes whose probability `p` is at least `threshold`: their top edges' mean direction,
    refined on their text lines."""
    letters = [b for b, pi in zip(boxes, p) if pi >= threshold]
    start = estimated_angle([b['quad'] for b in letters])
    record = {'boxes': [{'quad': b['quad'], 'p_letter': 1.0} for b in letters], 'letter_height_px': letter_height}
    return lattice.fit_angle(record, start)


def single_view(registration: dict) -> tuple[bool, float]:
    """(whether the page is read from view 1 alone, the share of registered letter centres) of a detector record's
    registration: a fit that used too few letters, or one that puts fewer than `QUALITY_USABLE` of the letter centres
    within a quarter letter height of their partner, leaves the page to view 1."""
    if (registration.get('diagnostics') or {}).get('refinement') == 'insufficient_mono_detections':
        return True, 0.0
    fraction = registration.get('inlier_fraction')
    if fraction is None:
        return registration.get('quality') == 'late_fusion', float(registration.get('quality') == 'pass')
    return float(fraction) < QUALITY_USABLE, float(fraction)


def rect_iou(u1, v1, w1, h1, u2, v2, w2, h2) -> np.ndarray:
    """IoU of axis-aligned rectangles (centre, width along u, height along v), broadcasting."""
    ix = np.clip(np.minimum(u1 + w1 / 2, u2 + w2 / 2) - np.maximum(u1 - w1 / 2, u2 - w2 / 2), 0, None)
    iy = np.clip(np.minimum(v1 + h1 / 2, v2 + h2 / 2) - np.maximum(v1 - h1 / 2, v2 - h2 / 2), 0, None)
    inter = ix * iy
    return inter / np.maximum(w1 * h1 + w2 * h2 - inter, 1e-9)


@dataclass(frozen=True)
class Page:
    base: str
    det_ids: np.ndarray        # (n,) int64, record order
    quads: np.ndarray          # (n, 4, 2) float64, Rotation1 pixels
    p: np.ndarray              # (n,) calibrated detector probability
    u: np.ndarray              # (n,) deskewed centre along the rows, letter heights
    v: np.ndarray              # (n,) across the rows
    w: np.ndarray              # (n,) box width (along the rows), letter heights
    h: np.ndarray              # (n,) box height
    node: np.ndarray           # (n, len(NODE)) float32, raw (unstandardised)
    angle: float
    target: np.ndarray | None          # (n,) int64: letter index, or len(alphabet) for none
    reader_target: np.ndarray | None   # (n,) int64: letter index, none, or IGNORE

    def __len__(self) -> int:
        return len(self.det_ids)


def build(det: dict, base: str) -> Page:
    """The label-free tokens of a calibrated detector record `det`."""
    boxes = det['boxes']
    if not boxes:
        raise ValueError(f'{base}: the detector found no letters')
    n = len(boxes)
    quads = np.array([b['quad'] for b in boxes], dtype=np.float64).reshape(-1, 4, 2)
    p = np.array([float(b['p_letter']) for b in boxes])
    letter_h = float(det['letter_height_px'])
    angle = text_angle(boxes, letter_h, p)
    cos, sin = math.cos(angle), math.sin(angle)
    centre = quads.mean(1)
    u = (centre[:, 0] * cos + centre[:, 1] * sin) / letter_h
    v = (-centre[:, 0] * sin + centre[:, 1] * cos) / letter_h
    w = np.maximum(np.linalg.norm(quads[:, 3] - quads[:, 0], axis=1) / letter_h, 1e-3)
    h = np.maximum(np.linalg.norm(quads[:, 1] - quads[:, 0], axis=1) / letter_h, 1e-3)
    edge = quads[:, 2] - quads[:, 1]
    tilt = np.arctan2(edge[:, 1], edge[:, 0]) - angle

    k = min(OVERLAP_NEIGHBOURS, n - 1)
    ov_max, ov_higher, ov_n = np.zeros(n), np.zeros(n), np.zeros(n)
    if k > 0:
        _, idx = cKDTree(np.stack([u, v], 1)).query(np.stack([u, v], 1), k=k + 1)
        idx = idx[:, 1:]
        iou = rect_iou(u[:, None], v[:, None], w[:, None], h[:, None], u[idx], v[idx], w[idx], h[idx])
        ov_max = iou.max(1)
        ov_higher = (iou * (p[idx] > p[:, None])).max(1)
        ov_n = (iou > 0.1).sum(1)
    support = [b.get('view_support') or {} for b in boxes]
    single, inliers = single_view(det.get('registration') or {})
    lp = logit(p)
    columns = {
        'lp': lp, 'score': [float(b['score']) for b in boxes],
        's_mono1': [float(s.get('mono1') or 0.0) for s in support],
        's_mono2': [float(s.get('mono2') or 0.0) for s in support],
        's_pack': [float(s.get('pack') or 0.0) for s in support],
        'lh': np.log(h), 'lw': np.log(w), 'laspect': np.log(w / h), 'sin2': np.sin(2 * tilt), 'cos2': np.cos(2 * tilt),
        'ov_max': ov_max, 'ov_higher': ov_higher, 'lov_n': np.log1p(ov_n),
        'page_lh': np.full(n, math.log(letter_h)), 'page_ln': np.full(n, math.log(n)),
        'page_hi': np.full(n, float((p >= 0.5).mean())), 'single_view': np.full(n, float(single)),
        'inliers': np.full(n, inliers),
    }
    node = np.stack([np.asarray(columns[c], dtype=np.float64) for c in NODE], 1).astype(np.float32)
    return Page(base, np.array([int(b['id']) for b in boxes], np.int64), quads, p, u, v, w, h, node, angle, None, None)


def claims(ious: np.ndarray, p: np.ndarray, threshold: float, claim: str) -> dict[int, int]:
    """detection -> annotated letter of the one-to-one match at IoU >= `threshold` under rule `claim` (module
    docstring)."""
    if claim == 'iou':
        return match_box_ious(ious, threshold)['match']
    if claim != 'p':
        raise ValueError(f'claim must be iou or p, not {claim!r}')
    valid = ious >= threshold
    # every valid pair outweighs any sum of tie-breaks, so cardinality comes first; then probability, then IoU
    reward = valid * (min(ious.shape, default=0) * 3 + 3 + np.asarray(p, np.float64)[:, None] + 1e-3 * ious)
    rows, cols = linear_sum_assignment(reward, maximize=True)
    return {int(r): int(c) for r, c in zip(rows, cols) if valid[r, c]}


def with_targets(page: Page, det: dict, work: Work, alphabet: str, cfg: dict) -> Page:
    """`page` with its targets from the squeeze's annotations (module docstring), under the page network
    configuration `cfg` (`label_iou`, `claim`, `reader_iou`, `blank_iou`)."""
    gt = gt_squeeze(work.annotation(page.base, 1), work.annotation(page.base, 2))
    if len(gt.quads) != len(gt.letters):
        raise ValueError(f'{page.base}: annotation boxes and letters do not align')
    none = len(alphabet)
    target = np.full(len(page), none, np.int64)
    ious = rotated_iou_matrix(page.quads, gt.quads)
    for i, g in claims(ious, page.p, cfg['label_iou'], cfg['claim']).items():
        target[i] = alphabet.index(gt.letters[g])
    reader = page_labels(det['boxes'], gt, alphabet, cfg['reader_iou'], cfg['blank_iou'])
    return Page(**{**page.__dict__, 'target': target, 'reader_target': reader.labels})


def neighbours(u: np.ndarray, v: np.ndarray, k: int, v_weight: float) -> tuple[np.ndarray, np.ndarray]:
    """Each token's `k` nearest tokens, itself first, by du^2 + (v_weight dv)^2: indices (n, k) and a validity mask
    (pages with fewer than k tokens pad with the token itself, masked)."""
    n = len(u)
    pts = np.stack([u, v_weight * v], 1)
    kk = min(k, n)
    _, idx = cKDTree(pts).query(pts, k=kk)
    idx = np.asarray(idx, np.int64).reshape(n, kk)
    for r in np.flatnonzero(idx[:, 0] != np.arange(n)):  # a token at the same point may come first
        row = [j for j in idx[r] if j != r][:kk - 1]
        idx[r] = [r] + row
    out = np.repeat(np.arange(n)[:, None], k, 1)
    mask = np.zeros((n, k), bool)
    out[:, :kk], mask[:, :kk] = idx, True
    return out, mask


def pair_features(u, v, w, h, lp, idx: np.ndarray) -> np.ndarray:
    """(n, k, len(PAIR)) float32 features of neighbour idx[i, j] relative to token i."""
    i = np.arange(len(u))[:, None]
    du, dv = u[idx] - u[i], v[idx] - v[i]
    iou = rect_iou(u[i], v[i], w[i], h[i], u[idx], v[idx], w[idx], h[idx])
    cols = [du, dv, np.log1p(np.hypot(du, dv)), iou, np.log(h[idx] / h[i]), np.log(w[idx] / w[i]), lp[idx] - lp[i],
            (idx == i).astype(np.float64)]
    return np.stack(cols, -1).astype(np.float32)
