"""Fusing the detections of one squeeze's three views: pack, mono1 and mono2 (the latter mapped into Rotation1).

Cross-view weighted box fusion followed by Gaussian soft-NMS. A cluster holds at most one detection per view, so
overlapping alternatives of the same view stay distinct; its box is the score-weighted mean of its members' corners
and its starting score their mean. Soft-NMS orders the clusters by score and decays the score of every cluster that
overlaps a better one (IoU above `soft_nms_iou`) by exp(-IoU^2 / sigma), but never drops one: the later stages
(the calibration, the page network) judge every candidate. Each view's own score stays in `view_support`.
"""
from __future__ import annotations

import numpy as np

from ..geometry import rotated_iou_matrix
from .records import MIN_SCORE, VIEWS


def fuse_views(branches: dict[str, list[dict]], cfg: dict) -> list[dict]:
    """Fused detections {quad (4, 2), score, view_support {view: score or None}}, best first.

    `branches` maps a view to its detections [{quad, score}] (quads in one common frame); `cfg` is the 'fusion'
    configuration section.
    """
    if set(branches) - set(VIEWS):
        raise ValueError('unknown detector view')
    clusters: list[list[tuple[str, np.ndarray, float]]] = []
    for view in VIEWS:
        source = branches.get(view, [])
        quads = np.asarray([box['quad'] for box in source], dtype=np.float32).reshape(-1, 4, 2)
        scores = np.clip(np.asarray([box['score'] for box in source], dtype=np.float64), 0.0, 1.0)
        if len(scores) and np.any(scores < MIN_SCORE):
            raise ValueError('raw detector confidence is below the inference threshold')
        if not len(quads):
            continue
        representatives = np.asarray([np.average(np.stack([m[1] for m in cluster]), axis=0,
                                                 weights=[m[2] for m in cluster]) for cluster in clusters],
                                     dtype=np.float32).reshape(-1, 4, 2)
        ious = rotated_iou_matrix(quads, representatives)
        for index in np.argsort(-scores, kind='stable'):
            candidates = [column for column, cluster in enumerate(clusters)
                          if view not in {m[0] for m in cluster} and ious[index, column] >= cfg['wbf_iou']]
            if candidates:
                best = max(candidates, key=lambda column: (ious[index, column], -column))
                clusters[best].append((view, quads[index], float(scores[index])))
            else:
                clusters.append([(view, quads[index], float(scores[index]))])

    fused = []
    for cluster in clusters:
        weights = np.asarray([m[2] for m in cluster], dtype=np.float64)
        support = dict.fromkeys(VIEWS)
        for view, _, score in cluster:
            support[view] = score
        fused.append({'quad': np.average(np.stack([m[1] for m in cluster]), axis=0, weights=weights),
                      'score': float(weights.mean()), 'view_support': support})
    return _soft_nms(fused, cfg['soft_nms_iou'], cfg['soft_nms_sigma'])


def _soft_nms(fused: list[dict], threshold: float, sigma: float) -> list[dict]:
    pending = list(range(len(fused)))
    scores = np.asarray([box['score'] for box in fused], dtype=np.float64)
    output = []
    while pending:
        best = max(pending, key=lambda index: (scores[index], -index))
        pending.remove(best)
        output.append({**fused[best], 'score': float(scores[best])})
        if pending:
            rest = np.asarray(pending, dtype=np.intp)
            overlaps = rotated_iou_matrix(np.asarray([fused[best]['quad']], dtype=np.float32),
                                          np.asarray([fused[index]['quad'] for index in rest], dtype=np.float32))[0]
            affected = overlaps > threshold
            scores[rest[affected]] *= np.exp(-(overlaps[affected] ** 2) / sigma)
    return output
