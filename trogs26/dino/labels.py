"""Crop labels: which letter, or no letter, does a detection show?

Detections are matched one-to-one to the annotated letters of the squeeze's Rotation1 view (maximum cardinality,
then maximum total rotated IoU, at IoU >= `match_iou`); a matched detection carries its letter's index. A detection
whose best IoU with any letter is below `blank_iou` is blank (class `len(alphabet)`). Every other detection - a
duplicate of a matched letter, or a partial overlap - is ignored (-1) and neither trained on nor scored against.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from ..annotations import GTSqueeze
from ..geometry import match_box_ious, rotated_iou_matrix


class PageLabels(NamedTuple):
    det_ids: np.ndarray  # (n,) int64, detector ids in record order
    labels: np.ndarray   # (n,) int64: letter index, blank index, or -1


def page_labels(boxes: list[dict], gt: GTSqueeze, alphabet: str, match_iou: float, blank_iou: float) -> PageLabels:
    if len(gt.quads) != len(gt.letters):
        raise ValueError('annotation boxes and letters do not align')
    ious = rotated_iou_matrix([b['quad'] for b in boxes], gt.quads)
    best = ious.max(1) if ious.shape[1] else np.zeros(len(boxes))
    labels = np.full(len(boxes), -1, np.int64)
    labels[best < blank_iou] = len(alphabet)
    for i, g in match_box_ious(ious, match_iou)['match'].items():
        labels[i] = alphabet.index(gt.letters[g])
    return PageLabels(np.array([int(b['id']) for b in boxes], np.int64), labels)
