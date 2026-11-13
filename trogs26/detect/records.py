"""The detector record of one squeeze: `chains/f{k}/{group}/det_raw/{base}.json` (before calibration, `p_letter` null)
and `chains/f{k}/{group}/det/{base}.json` (calibrated). Both have the same layout, which every later stage reads:

  version, base, fold (the chain), detector, frame ('Rotation1'), image_size {Rotation1, Rotation2: [W, H]},
  registration (`Registration.to_dict()`: Rotation2 -> Rotation1), letter_height_px,
  boxes: [{id, quad (NW, SW, SE, NE corners in Rotation1 pixels), score (fused, after soft-NMS), p_letter,
           view_support {pack, mono1, mono2: the view's own score, or null}}],
  provenance: where the record came from.

Box ids are the list positions, and the boxes are in descending order of fused score at the time of fusion.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

VERSION = 'det_v1'
FUSION_VERSION = 'det_multiview_wbf_softnms_v1'
VIEWS = ('pack', 'mono1', 'mono2')
MIN_SCORE = 0.05  # no detection below the inference confidence survives


def validate(record: dict) -> None:
    """Raise ValueError unless `record` has the layout above and every box is a plausible detection."""
    if record['version'] != VERSION or record['frame'] != 'Rotation1' or not record['base'] or not record['detector']:
        raise ValueError('record needs the version, frame, base and detector')
    if not isinstance(record['fold'], int) or record['fold'] < 0:
        raise ValueError('record needs the chain index')
    sizes = record['image_size']
    if set(sizes) != {'Rotation1', 'Rotation2'}:
        raise ValueError('both image sizes are required')
    for size in sizes.values():
        if len(size) != 2 or any(not isinstance(v, int) or v <= 0 for v in size):
            raise ValueError('image sizes must be positive integers [W, H]')
    height = record['letter_height_px']
    if height is None or not np.isfinite(height) or height <= 0:
        raise ValueError('record needs a measured positive letter height')
    for index, box in enumerate(record['boxes']):
        quad = np.asarray(box['quad'])
        if box['id'] != index or quad.shape != (4, 2) or not np.isfinite(quad).all():
            raise ValueError('invalid box id or quad')
        if not 0 <= box['score'] <= 1 or (box['p_letter'] is not None and not 0 <= box['p_letter'] <= 1):
            raise ValueError('invalid detection probability')
        if set(box['view_support']) != set(VIEWS):
            raise ValueError('invalid view support')
        supports = [v for v in box['view_support'].values() if v is not None]
        if not supports or any(not MIN_SCORE <= v <= 1 for v in supports):
            raise ValueError('every box needs a view score of at least the inference confidence')
        edges = np.roll(quad, -1, axis=0) - quad
        if edges[0, 0] * edges[1, 1] - edges[0, 1] * edges[1, 0] >= 0:
            raise ValueError('quad is not in NW, SW, SE, NE order')


def write(path: str | Path, record: dict) -> None:
    """Validate and atomically write `record` (sorted keys, indent 2)."""
    validate(record)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + '.partial')
    partial.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False))
    os.replace(partial, path)


def read(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def digest(value) -> str:
    """SHA-256 of the canonical JSON form of `value`."""
    text = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def raw_digest(record: dict) -> str:
    """Identity of the detections of a record, whatever its calibration or provenance."""
    body = {key: value for key, value in record.items() if key != 'provenance'}
    body['boxes'] = [{**box, 'p_letter': None} for box in record['boxes']]
    return digest(body)
