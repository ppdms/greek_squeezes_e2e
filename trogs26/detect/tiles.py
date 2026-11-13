"""Training tiles: the detector's training images and labels, rendered from the public annotations.

For every training squeeze three page views are rendered, scaled so letters are `letter_px` high and cut into
square, white-padded tiles:

  pack   view 1 | view 2 registered onto view 1 | their minimum (if the registration fitted on the annotated letter
         centres is usable, else view 1 alone: the packed page then degenerates to a mono page)
  mono1  view 1 alone
  mono2  view 2 alone

A tile `{base}__{view}__t{index:03d}` is `images/{name}.png` (RGB) with `labels/{name}.txt` (YOLO polygon format:
class 0 and the four corners of every letter, normalised by the tile side; letters clipped by the tile edge are kept
clipped). Everything is written under `tiles_dir(work, cfg)`, `squeezes/{base}.json` listing a squeeze's tiles last.
Chain k's detector trains on the pack, mono1 and mono2 tiles of the squeezes outside fold k and monitors the pack
tiles of fold k.

The labels depend only on the annotation and the image sizes (`tile_plan`), never on pixels.
"""
from __future__ import annotations

import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from ..layout import Work
from ..registration import fit_centres
from . import page
from .annotations import MIN_AREA, read_letters

VIEWS = ('pack', 'mono1', 'mono2')


def tiles_dir(work: Work, cfg: dict) -> Path:
    """Directory of the tiles rendered with the page settings of `cfg` (the 'detector' section); every setting
    that changes a pixel or a label is part of the name, so differently configured tiles never mix."""
    p = cfg['page']
    return work.detector / 'tiles' / f"t{p['tile']}_l{p['letter_px']:g}_o{p['overlap']:g}"


def tile_name(base: str, view: str, index: int) -> str:
    return f'{base}__{view}__t{index:03d}'


def label_lines(quads, angle, window, side) -> list[str]:
    """Label lines of the letters inside one tile window of a scaled page (quads in scaled-page pixels)."""
    x0, y0, x1, y1 = window
    lines = []
    for quad in quads:
        canonical = page.canonical_quad(quad, angle) - [x0, y0]
        canonical[:, 0] = np.clip(canonical[:, 0], 0, x1 - x0)
        canonical[:, 1] = np.clip(canonical[:, 1], 0, y1 - y0)
        if cv2.contourArea(canonical.astype(np.float32)) < MIN_AREA:
            continue
        canonical /= side
        lines.append('0 ' + ' '.join(f'{v:.8f}' for v in canonical.ravel()))
    return lines


def tile_plan(quads, angle, size, cfg):
    """Scale frame, tile windows and label lines of one annotated view of image `size` (W, H).

    `cfg` is the 'detector' configuration section.
    """
    frame = page.ScaleFrame.from_height(size, page.median_height(quads, angle), cfg['page']['letter_px'])
    scaled = frame.apply(quads)
    side = cfg['page']['tile']
    windows = page.tile_windows(frame.size, side, cfg['page']['overlap'])
    return frame, windows, [label_lines(scaled, angle, window, side) for window in windows]


def _write_tiles(rendered, view, base, windows, labels, out: Path, side: int) -> list[str]:
    names = []
    (out / 'images').mkdir(parents=True, exist_ok=True)
    (out / 'labels').mkdir(parents=True, exist_ok=True)
    for index, ((x0, y0, x1, y1), lines) in enumerate(zip(windows, labels)):
        name = tile_name(base, view, index)
        tile = cv2.copyMakeBorder(rendered[y0:y1, x0:x1], 0, side - (y1 - y0), 0, side - (x1 - x0),
                                  cv2.BORDER_CONSTANT, value=(255, 255, 255))
        if not cv2.imwrite(str(out / 'images' / f'{name}.png'), tile[:, :, ::-1]):
            raise OSError(f'could not write tile {name}')
        (out / 'labels' / f'{name}.txt').write_text('\n'.join(lines) + '\n')
        names.append(name)
    return names


def render_squeeze(work: Work, base: str, cfg: dict) -> dict:
    """Render and write the pack, mono1 and mono2 tiles of one training squeeze; returns its record.

    Skips a squeeze whose record exists. Reads the squeeze's two annotations and two images.
    """
    out = tiles_dir(work, cfg)
    record_path = out / 'squeezes' / f'{base}.json'
    if record_path.exists():
        return json.loads(record_path.read_text())
    side = cfg['page']['tile']
    (one, angle1), (two, angle2) = (read_letters(work.annotation(base, view)) for view in (1, 2))
    gray1, gray2 = (cv2.imread(str(work.image(base, view)), cv2.IMREAD_GRAYSCALE) for view in (1, 2))
    if gray1 is None or gray2 is None:
        raise FileNotFoundError(f'missing image pair for {base}')
    size1, size2 = (gray1.shape[1], gray1.shape[0]), (gray2.shape[1], gray2.shape[0])
    frame1, windows1, labels1 = tile_plan(one, angle1, size1, cfg)
    frame2, windows2, labels2 = tile_plan(two, angle2, size2, cfg)

    # The packed page needs view 2 registered onto view 1; fit it on the annotated letter centres.
    registration = fit_centres(two.mean(1), one.mean(1), page.median_height(one, angle1),
                               merged=page.is_merged(work.image(base, 1), work.image(base, 2)))
    packed = registration.quality in ('pass', 'usable')
    pages = {'pack': (page.make_page(gray1, gray2, registration, frame1, packed=packed), windows1, labels1),
             'mono1': (page.make_page(gray1, None, None, frame1, packed=False), windows1, labels1),
             'mono2': (page.make_page(gray2, None, None, frame2, packed=False), windows2, labels2)}
    record = {'packed': packed, 'tiles': {view: _write_tiles(pages[view][0], view, base, *pages[view][1:], out, side)
                                          for view in VIEWS}}
    record_path.parent.mkdir(parents=True, exist_ok=True)
    partial = record_path.with_name(record_path.name + '.partial')
    partial.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
    os.replace(partial, record_path)
    return record


def _single_thread() -> None:
    cv2.setNumThreads(1)


def render_all(work: Work, bases: list[str], cfg: dict, jobs: int = 1) -> int:
    """Render every squeeze of `bases` that has no record yet, `jobs` at a time; returns how many were rendered."""
    out = tiles_dir(work, cfg)
    todo = [base for base in bases if not (out / 'squeezes' / f'{base}.json').exists()]
    if jobs <= 1:
        for index, base in enumerate(todo, 1):
            render_squeeze(work, base, cfg)
            print(f'tiles {index}/{len(todo)} {base}', flush=True)
        return len(todo)
    with ProcessPoolExecutor(jobs, initializer=_single_thread) as pool:
        for index, base in enumerate(pool.map(render_squeeze, [work] * len(todo), todo, [cfg] * len(todo),
                                              chunksize=1), 1):
            print(f'tiles {index}/{len(todo)}', flush=True)
    return len(todo)


def split_tiles(root: Path, folds: dict[int, tuple[str, ...]], fold: int) -> tuple[list[str], list[str]]:
    """(train, held) tile names of chain `fold`'s detector, sorted.

    Train: every tile of the squeezes outside the fold. Held: the pack tiles of the fold's squeezes, used only to
    monitor training. `root` is `tiles_dir(work, cfg)`.
    """
    held_bases = set(folds[fold])
    train, held = [], []
    for bases in folds.values():
        for base in bases:
            tiles = json.loads((root / 'squeezes' / f'{base}.json').read_text())['tiles']
            if base in held_bases:
                held += tiles['pack']
            else:
                train += [name for view in VIEWS for name in tiles[view]]
    return sorted(train), sorted(held)


def _canonical_obb(quad) -> np.ndarray:
    """Oriented box (cx, cy, w, h, angle) of a quad with w >= h and the angle in [-pi/4, 3pi/4)."""
    (cx, cy), (width, height), angle_degrees = cv2.minAreaRect(np.asarray(quad, np.float32).reshape(4, 2))
    angle = math.radians(angle_degrees)
    if width < height:
        width, height = height, width
        angle += math.pi / 2
    while angle >= 3 * math.pi / 4:
        angle -= math.pi
    while angle < -math.pi / 4:
        angle += math.pi
    return np.asarray([cx, cy, width, height, angle], np.float32)


def obb_targets(label_text: str, side: int) -> tuple[np.ndarray, np.ndarray]:
    """Training targets of one tile: classes (n, 1) and boxes (n, 5) = cx, cy, w, h (normalised) and angle."""
    classes, boxes = [], []
    for line in label_text.splitlines():
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 9 or fields[0] != '0':
            raise ValueError(f'expected one-class letter quad labels, got {line!r}')
        box = _canonical_obb(np.asarray([float(v) for v in fields[1:]], np.float32).reshape(4, 2) * side)
        box[:4] /= side
        classes.append([0.0])
        boxes.append(box)
    return (np.asarray(classes, dtype=np.float32).reshape(-1, 1), np.asarray(boxes, dtype=np.float32).reshape(-1, 5))


def load_tiles(root: Path, names: list[str], side: int) -> tuple[np.ndarray, dict[str, tuple]]:
    """Pixels (n, side, side, 3 BGR uint8, one contiguous array) and OBB targets of the named tiles; `root` is
    `tiles_dir(work, cfg)`. The targets are keyed by file name (`{name}.png`)."""
    pixels = np.empty((len(names), side, side, 3), np.uint8)
    targets = {}
    for index, name in enumerate(names):
        image = cv2.imread(str(root / 'images' / f'{name}.png'), cv2.IMREAD_COLOR)
        if image is None or image.shape != (side, side, 3):
            raise ValueError(f'tile {name} is missing or not {side} px square')
        pixels[index] = image
        targets[f'{name}.png'] = obb_targets((root / 'labels' / f'{name}.txt').read_text(), side)
    return pixels, targets
