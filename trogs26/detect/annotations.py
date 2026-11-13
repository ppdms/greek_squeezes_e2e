"""Reading a public letter-box annotation file ({base}_Rotation{1,2}_300dpi_letters.txt).

Layout: `# Baseline angle`, the angle, `# Boxes: x1 y1 x2 y2 x3 y3 x4 y4`, one box per line (corners NW, SW, SE,
NE), `# Transcript`, ... The organisers' parser silently drops boxes smaller than `MIN_AREA` pixels; so do we.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

MIN_AREA = 9


def read_letters(path: str | Path) -> tuple[np.ndarray, float]:
    """(N, 4, 2) float64 letter quads and the baseline angle (radians) of one annotation file."""
    lines = Path(path).read_text().splitlines()
    angle = float(lines[1])
    boxes, section = [], 0
    for line in lines:
        if line.startswith('#'):
            section += 1
        elif section == 2:
            fields = line.split()
            if len(fields) == 8:
                box = np.asarray([float(x) for x in fields]).reshape(4, 2)
                if np.linalg.norm(box[3] - box[0]) * np.linalg.norm(box[1] - box[0]) >= MIN_AREA:
                    boxes.append(box)
    return np.asarray(boxes).reshape(-1, 4, 2), angle
