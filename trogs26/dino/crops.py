"""The page network's crops: the relit relief around every detection.

The squeeze's relief is recovered from both views (`trogs26.relief.fuse`, at the scale it chooses) and shown as the
input 'relit': the relief lit from three azimuths, one per channel. Every detection's crop is the oriented square of
`scale` times its longer side around it, cut from that page and warped to `size` x `size`.

Crops are cached per chain, squeeze and input (`Chain.dino_crops`), keyed on the bytes of the detector record and the
crop parameters, so a changed detection or setting is never served from a stale file.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from .. import relief
from ..artifacts import det_page
from ..layout import Chain, Work
from ..registration import Registration

INPUTS = ('relit',)
RELIEF_LETTERS = 0.9  # detections at least this probable give the relief's letter positions


def cache_key(det_path: Path, params: dict) -> str:
    """Digest of a detector record's bytes and the crop parameters that produced a cache file."""
    return hashlib.sha256(det_path.read_bytes() + json.dumps(params, sort_keys=True).encode()).hexdigest()


def is_current(path: Path, key: str) -> bool:
    if not path.exists():
        return False
    with np.load(path) as z:
        return 'key' in z and str(z['key']) == key


def save_cache(path: Path, key: str, **arrays: np.ndarray) -> None:
    """Write a crop cache atomically (a partially written file is never read)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix('.partial.npz')
    np.savez(partial, key=key, **arrays)
    partial.replace(path)


def crop_square(quad, scale: float, crop_scale: float) -> np.ndarray:
    """Three corners (top-left, bottom-left, bottom-right) of a detection's crop square in a page `scale` times the
    original resolution."""
    q = (np.asarray(quad, np.float64) + .5) * scale - .5
    edge = q[3] - q[0]
    width = np.linalg.norm(edge)
    side = crop_scale * max(width, np.linalg.norm(q[1] - q[0]))
    right = edge / width * side / 2
    down = np.array([-right[1], right[0]])
    centre = q.mean(0)
    return np.array([centre - right - down, centre - right + down, centre + right + down], np.float32)


def relief_page(f: relief.Fusion, input: str) -> np.ndarray:
    """(h, w, 3) uint8 page of the relief input `input` (see `INPUTS`) at the fusion's working scale."""
    if input not in INPUTS:
        raise ValueError(f'unknown crop input {input!r}')
    return np.ascontiguousarray(np.dstack([relief.to_uint8(relief.relit(f, a), f.mask, True)
                                           for a in relief.RELIT_AZIMUTHS]))


def record_fusion(record: dict, gray1: np.ndarray, gray2: np.ndarray) -> relief.Fusion:
    """The relief of a squeeze from its grey views under the registration, letter height and likely letters of its
    detector `record`."""
    letters = [b['quad'] for b in record['boxes'] if b['p_letter'] >= RELIEF_LETTERS]
    return relief.fuse(gray1, gray2, Registration.from_dict(record['registration']), record['letter_height_px'],
                       letters)


def fusion_crops(record: dict, f: relief.Fusion, size: int, scale: float, inputs: Sequence[str]
                 ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """(detection ids, input -> (n, 3, size, size) uint8 crops) of every detection of `record` for each relief input
    in `inputs`, cut from the fusion `f` of its squeeze."""
    destination = np.array([[0, 0], [0, size - 1], [size - 1, size - 1]], np.float32)
    boxes = record['boxes']
    transforms = [cv2.getAffineTransform(crop_square(b['quad'], f.scale, scale), destination) for b in boxes]
    out = {}
    for input in inputs:
        page = relief_page(f, input)
        crops = np.empty((len(boxes), 3, size, size), np.uint8)
        for i, m in enumerate(transforms):
            crops[i] = cv2.warpAffine(page, m, (size, size), flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REPLICATE).transpose(2, 0, 1)
        out[input] = crops
    return np.array([int(b['id']) for b in boxes], np.int64), out


def relief_page_crops(record: dict, gray1: np.ndarray, gray2: np.ndarray, size: int, scale: float,
                      inputs: Sequence[str]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """`fusion_crops` of one fusion of the grey views `gray1`, `gray2` (`record_fusion`)."""
    return fusion_crops(record, record_fusion(record, gray1, gray2), size, scale, inputs)


def write_page_crops(work: Work, chains: Sequence[Chain], base: str, crop: dict,
                     inputs: Sequence[str] = INPUTS) -> int:
    """Cache the crops of squeeze `base` in each input of `inputs` for every chain in `chains` that lacks current
    ones; returns how many caches were written. The relief depends on each chain's detections and is fused once per
    chain for all its inputs. `crop` is config 'dino.crop'."""
    unknown = sorted(set(inputs) - set(INPUTS))
    if unknown:
        raise ValueError(f'unknown crop inputs {unknown}')
    todo = []
    for chain in chains:
        stale = {}
        for input in inputs:
            key = cache_key(chain.det(base), {**crop, 'input': input})
            if not is_current(chain.dino_crops(base, input), key):
                stale[input] = key
        if stale:
            todo.append((chain, stale))
    if todo:
        gray1, gray2 = (cv2.imread(str(work.image(base, v)), cv2.IMREAD_GRAYSCALE) for v in (1, 2))
    for chain, stale in todo:
        ids, crops = relief_page_crops(det_page(chain, base), gray1, gray2, crop['size'], crop['scale'], list(stale))
        for input in stale:
            save_cache(chain.dino_crops(base, input), stale[input], det_ids=ids, crops=crops[input])
    return sum(len(stale) for _, stale in todo)
