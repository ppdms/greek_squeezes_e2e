"""Rotated-box overlap and one-to-one matching, shared by every stage that scores detections against annotations.

Quads are (4, 2) corner arrays ordered NW, SW, SE, NE in one view's pixel frame.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


def rotated_iou_matrix(pred_quads, gt_quads) -> np.ndarray:
    """Exact convex-quad IoU matrix (len(pred), len(gt)), float64, with an AABB broad phase.

    Degenerate polygons have zero overlap. No centre-distance approximation is used. Corners are cast to float32
    because that is the precision OpenCV intersects in.
    """
    a = np.asarray(pred_quads, dtype=np.float32).reshape(-1, 4, 2)
    b = np.asarray(gt_quads, dtype=np.float32).reshape(-1, 4, 2)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('box coordinates must be finite')
    out = np.zeros((len(a), len(b)), dtype=np.float64)
    if not len(a) or not len(b):
        return out
    alo, ahi, blo, bhi = a.min(1), a.max(1), b.min(1), b.max(1)
    area_a = np.array([cv2.contourArea(q) for q in a])
    area_b = np.array([cv2.contourArea(q) for q in b])
    for i, q in enumerate(a):
        candidates = np.flatnonzero(np.all(ahi[i] > blo, axis=1) & np.all(bhi > alo[i], axis=1))
        for j in candidates:
            inter = float(cv2.intersectConvexConvex(q, b[j])[0])
            union = area_a[i] + area_b[j] - inter
            if union > 0:
                out[i, j] = np.clip(inter / union, 0., 1.)
    return out


def match_box_ious(ious, threshold, pred_ids=None) -> dict:
    """One-to-one assignment on an IoU matrix: maximum valid cardinality, then maximum total IoU.

    Pairs below `threshold` (inclusive bound) are invalid before the assignment, so a valid match is never lost to
    a below-threshold edge. `pred_ids` names the rows (default 0..n-1). Returns
    {'match': {pred id: gt index}, 'best_iou': {pred id: best IoU over all gt, matched or not}, 'matched',
    'predicted', 'reference', 'threshold'}.
    """
    if not 0 < threshold <= 1:
        raise ValueError('IoU threshold must be in (0, 1]')
    ids = list(range(ious.shape[0])) if pred_ids is None else list(pred_ids)
    if len(ids) != ious.shape[0] or len(set(ids)) != len(ids):
        raise ValueError('pred_ids must identify each detection exactly once')
    valid = ious >= threshold
    reward = valid * (min(ious.shape, default=0) + 1 + ious)
    pi, gi = linear_sum_assignment(reward, maximize=True)
    match = {int(ids[p]): int(g) for p, g in zip(pi, gi) if valid[p, g]}
    best = ious.max(1) if ious.shape[1] else np.zeros(len(ids))
    return {'match': match, 'best_iou': {int(p): float(v) for p, v in zip(ids, best)},
            'matched': len(match), 'predicted': len(ids), 'reference': ious.shape[1], 'threshold': threshold}
