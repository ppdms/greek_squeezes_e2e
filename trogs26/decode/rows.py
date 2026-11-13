"""The rows of a page: the organisers' row grouping of boxes at the page angle, and the rule by which a doubtful box
joins the row it sits on."""
from __future__ import annotations

import numpy as np

from .. import official


def first_occurrence_rows(rows: list[list[int]]) -> list[list[int]]:
    """Drop repeated box ids (the grouping can list a box in two neighbouring clusters) and empty rows."""
    seen: set[int] = set()
    result = []
    for row in rows:
        kept = []
        for box in row:
            if box not in seen:
                seen.add(box)
                kept.append(box)
        if kept:
            result.append(kept)
    return result


def group_rows(quads, angle: float) -> list[list[int]]:
    """Rows of box indices (reading order) by the organisers' v2 grouping at page angle `angle`."""
    if not len(quads):
        return []
    boxes = [official.v2.Box(*np.asarray(q, dtype=float).ravel()) for q in quads]
    index = {id(box): i for i, box in enumerate(boxes)}
    rows, _ = official.v2.getRowBoxes(official.v2.orderBoxes(boxes, angle, []), gangle=angle)
    return first_occurrence_rows([[index[id(box)] for box in row] for row in rows])


def attach(row_u: list[np.ndarray], row_v: list[np.ndarray], u: np.ndarray, v: np.ndarray, letter_height: float,
           span_margin: float) -> np.ndarray:
    """For points with deskewed centres (u along the rows, v across), the index of the row each joins, or -1: of the
    rows (given by their members' u and v) whose along-row span, widened by `span_margin` letter heights on both
    sides, contains the point, the one whose median v is nearest, if within 0.5 letter heights."""
    level = np.array([np.median(r) for r in row_v])
    lo = np.array([r.min() for r in row_u]) - span_margin * letter_height
    hi = np.array([r.max() for r in row_u]) + span_margin * letter_height
    joins = np.full(len(u), -1)
    for j in range(len(u)):
        distance = np.abs(v[j] - level) / letter_height
        distance[(u[j] < lo) | (u[j] > hi)] = np.inf
        r = int(np.argmin(distance))
        if distance[r] < 0.5:
            joins[j] = r
    return joins
