"""A registration of the two views from the images alone, which replaces the detector record's map before the relief is
fused.

    python -m trogs26.registration_images --group oof|new [--splits trogs26] [--jobs 4] [--config F]

The detector registers view 2 onto view 1 on the letter centres it finds in both (`trogs26.registration`); a worn
page gives the two views different detections. The images hold a better signal. Each view is lit along one image
axis and shows the height's derivative along it; its derivative across the light is then the mixed derivative h_xy
(`trogs26.relief`), which does not depend on the lighting. So the size of each view's own across-light derivative
is the same picture of the squeeze in both views, and they can be registered like two photographs of one thing:

    coarse   phase correlation of the two pictures over quarter turns +-5 degrees (letters 12 px high)
    fine     ECC, Euclidean then affine (letters 24 px high)
    local    three times, coarse to fine (`LEVELS`: tiles of 12, 6 and 6 letter heights, shifts up to 5, 1.5 and
             0.5): phase correlation over the tiles, the accepted shifts fitted by a thin-plate spline on the
             affine map (the model of `trogs26.registration`, `registration._tps_parameters`), smoothed until it
             keeps its orientation. The first level reaches the parts of a stitched scan that the affine map leaves
             letters apart.

Both views are scaled by the letter height of view 1's first detection pass, nothing else comes from a detector. The
record's own map (`fitted`) is also refined by the last two local levels (`refined`). Each of the three candidates
is measured on the images alone, by the tiles' median correlation of the pictures once it carries view 2 onto view
1, and the best one is `chosen`.

The stage rewrites every chain's records of the pages of its group (`chains/f{k}/{group}/det/{base}.json`; group 'new':
the pages of `--splits`, group 'oof': the squeezes of fold k): the chosen map replaces `model` and `parameters` of
`registration`, whose inlier fraction, quality and diagnostics stay, and `registration.diagnostics.images` keeps the
choice, the candidates' measures and the map replaced. A record that holds `images` already is left alone. The relit
crop caches are keyed on the record's bytes, so the next crop stage cuts them again
(`python -m trogs26.dino.stage crops --input relit --group oof|new [--splits trogs26]`).
Before that, `python -m trogs26.detect.redetect` detects again the pages whose reading the chosen map changes.

Inputs: the chains' detector records, the images of the pages.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from . import config, data, relief
from .layout import Work
from .registration import (Registration, _apply, _homogeneous, _minimum_jacobian, _scale_frame, _source_probes,
                           _tps_parameters)

COARSE_PX, FINE_PX, LOCAL_PX = 12.0, 24.0, 32.0   # letter height at each stage, pixels
PICTURE_LH = 0.08          # smoothing of the across-light derivative's size, letter heights
LEVELS = ((12.0, 5.0), (6.0, 1.5), (6.0, 0.5))   # local stages: (tile side, largest shift), letter heights
TILE_LH = 6.0              # tiles of the agreement
ACCEPT = 0.2               # a tile's shift counts if its pictures then correlate at least this much
AGREE = 0.3
REGULARISATIONS = (4., 16., 64., 256., 1024.)


def picture(gray: np.ndarray, letter_height: float, letter_px: float) -> tuple[np.ndarray, np.ndarray, float]:
    """(the size of the view's across-light derivative, its paper mask, the scale) with letters `letter_px` high:
    0 off the paper, divided by its root mean square on it."""
    scale = min(1.0, letter_px / letter_height)
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray
    lh = letter_height * scale
    unit = small.astype(np.float32) / 255.0
    mask = relief.paper_mask(unit, lh)
    shade = relief.flatten(unit, mask, relief.FLATTEN_LH * lh)
    sigma = max(relief.MIXED_LH * lh, 0.6)
    dx, dy = relief._derivative(shade, sigma, 1, 0), relief._derivative(shade, sigma, 0, 1)
    inside = mask & (cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
    if not inside.any():
        return np.zeros_like(shade), mask, scale
    across = dy if np.mean(dx[inside] ** 2) > np.mean(dy[inside] ** 2) else dx   # lit along x: its y derivative
    size = cv2.GaussianBlur(np.abs(across), (0, 0), max(PICTURE_LH * lh, 0.8))
    size = np.where(inside, size, 0.0).astype(np.float32)
    return size / max(float(np.sqrt(np.mean(size[inside] ** 2))), 1e-9), mask, scale


def to_original(matrix: np.ndarray, scale: float) -> np.ndarray:
    """A 2x3 map between pictures at `scale` as a map between original pixels."""
    return (np.linalg.inv(_scale_frame(scale)) @ _homogeneous(matrix) @ _scale_frame(scale))[:2]


def to_scale(matrix: np.ndarray, scale: float) -> np.ndarray:
    return (_scale_frame(scale) @ _homogeneous(matrix) @ np.linalg.inv(_scale_frame(scale)))[:2]


def coarse(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, float]:
    """(2x3 map of picture `a` onto picture `b`, the phase correlation's peak): the best quarter turn +-5 degrees
    and shift, as `registration._fit_images` searches them."""
    side = cv2.getOptimalDFTSize(math.ceil(max(*a.shape, *b.shape) * math.sqrt(2)))
    target = np.zeros((side, side), np.float32)
    target[:b.shape[0], :b.shape[1]] = b
    window = cv2.createHanningWindow((side, side), cv2.CV_32F)
    best = None
    for quarter in (0, 90, 180, 270):
        for offset in np.arange(-5., 5.01, .5):
            matrix = cv2.getRotationMatrix2D((a.shape[1] / 2, a.shape[0] / 2), quarter + offset, 1.)
            matrix[:, 2] += [(b.shape[1] - a.shape[1]) / 2, (b.shape[0] - a.shape[0]) / 2]
            moving = cv2.warpAffine(a, matrix, (side, side))
            shift, peak = cv2.phaseCorrelate(moving, target.copy(), window)
            if best is None or peak > best[0]:
                matrix[:, 2] += shift
                best = (peak, matrix)
    return best[1], float(best[0])


def fine(a: np.ndarray, b: np.ndarray, matrix: np.ndarray) -> tuple[np.ndarray, list]:
    """`matrix` (picture `a` onto `b`, both at the fine scale) corrected by ECC, Euclidean then affine."""
    size = (b.shape[1], b.shape[0])
    aligned = cv2.warpAffine(a, matrix, size)
    valid = cv2.erode(cv2.warpAffine(np.ones(a.shape, np.uint8), matrix, size), np.ones((7, 7), np.uint8))
    correction = np.eye(3, dtype=np.float32)[:2]
    scores = []
    for motion in (cv2.MOTION_EUCLIDEAN, cv2.MOTION_AFFINE):
        previous = correction.copy()
        try:
            score, correction = cv2.findTransformECC(b, aligned, correction, motion,
                                                     (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-5),
                                                     inputMask=valid, gaussFiltSize=5)
            scores.append(float(score))
        except cv2.error:
            correction = previous
    return (np.linalg.inv(_homogeneous(correction)) @ _homogeneous(matrix))[:2], scores


def warp(image: np.ndarray, registration: Registration, shape: tuple[int, int], scale: float) -> np.ndarray:
    """A view-2 picture at `scale` carried onto view 1's frame at the same scale by `registration` (original
    pixels), as `relief.second_view_on_first` resamples."""
    h, w = shape
    step = 16
    gy, gx = np.mgrid[0:h + step:step, 0:w + step:step].astype(np.float64)
    original1 = np.stack([(gx.ravel() + .5) / scale - .5, (gy.ravel() + .5) / scale - .5], axis=1)
    original2 = registration.inverse().apply(original1)
    coarse_x = (scale * (original2[:, 0] + .5) - .5).reshape(gx.shape).astype(np.float32)
    coarse_y = (scale * (original2[:, 1] + .5) - .5).reshape(gy.shape).astype(np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32) / step
    map_x = cv2.remap(coarse_x, xx, yy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    map_y = cv2.remap(coarse_y, xx, yy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    x, y = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    x, y = x - x.mean(), y - y.mean()
    return float((x * y).sum() / max(math.sqrt((x * x).sum() * (y * y).sum()), 1e-20))


def tile_shifts(one: np.ndarray, two: np.ndarray, mask: np.ndarray, lh: float, tile_lh: float, reach_lh: float
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(tile centres (k, 2) as x, y; shifts (k, 2) with `two` sampled at x + shift matching `one`; correlations
    left in place of every tile on the paper) over half-overlapping tiles of `tile_lh` letter heights: a tile gives
    a shift if it is at most `reach_lh` and its pictures then correlate at least `ACCEPT` and better than before."""
    tile = max(2, min(max(32, int(tile_lh * lh)), *one.shape)) // 2 * 2
    window = cv2.createHanningWindow((tile, tile), cv2.CV_32F)
    centres, shifts, still = [], [], []
    for y0 in range(0, one.shape[0] - tile + 1, tile // 2):
        for x0 in range(0, one.shape[1] - tile + 1, tile // 2):
            if mask[y0:y0 + tile, x0:x0 + tile].mean() < 0.6:
                continue
            a = np.ascontiguousarray(one[y0:y0 + tile, x0:x0 + tile])
            b = np.ascontiguousarray(two[y0:y0 + tile, x0:x0 + tile])
            before = _ncc(a, b)
            still.append(before)
            (dx, dy), _ = cv2.phaseCorrelate(a, b, window)
            if math.hypot(dx, dy) > reach_lh * lh:
                continue
            moved = cv2.warpAffine(b, np.float32([[1, 0, dx], [0, 1, dy]]), (tile, tile),
                                   flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
            margin = int(math.ceil(max(abs(dx), abs(dy)))) + 1
            if 2 * margin >= tile:
                continue
            after = _ncc(a[margin:-margin, margin:-margin], moved[margin:-margin, margin:-margin])
            if after >= ACCEPT and after >= before:
                centres.append((x0 + tile / 2, y0 + tile / 2))
                shifts.append((dx, dy))
    return np.array(centres, np.float64).reshape(-1, 2), np.array(shifts, np.float64).reshape(-1, 2), np.array(still)


def local(registration: Registration, one: np.ndarray, picture2: np.ndarray, mask1: np.ndarray, scale: float,
          letter_height: float, source_size, tile_lh: float, reach_lh: float) -> tuple[Registration, dict]:
    """`registration` with a thin-plate spline through the tiles' shifts (module docstring), and what was found."""
    lh = letter_height * scale
    two = warp(picture2, registration, one.shape, scale)
    centres, shifts, still = tile_shifts(one, two, mask1 & (two > 0), lh, tile_lh, reach_lh)
    found = {'tiles': int(len(still)), 'shifted': int(len(centres)),
             'median_shift_lh': float(np.median(np.hypot(*shifts.T)) / lh) if len(shifts) else 0.0}
    if len(centres) < 6:
        return registration, found
    # the content now at x + shift belongs at x: the view-2 point that lands on x + shift must land on x
    target = (centres + .5) / scale - .5
    source = registration.inverse().apply((centres + shifts + .5) / scale - .5)
    matrix = np.asarray(registration.parameters['matrix'])
    probes = _source_probes(source, letter_height, source_size)
    for regularisation in REGULARISATIONS:
        tps = _tps_parameters(source, target - _apply(matrix, source), letter_height, regularisation)
        candidate = Registration('piecewise', {'matrix': matrix.tolist(), 'tps': tps})
        if _minimum_jacobian(candidate, probes) > 0:
            return candidate, {**found, 'regularisation': regularisation}
    return registration, found


def measure(registration: Registration, one: np.ndarray, two: np.ndarray, mask1: np.ndarray, scale: float,
            letter_height: float) -> dict:
    """How well view 2's picture carried by `registration` matches view 1's, over the tiles of the paper."""
    moved = warp(two, registration, one.shape, scale)
    _, _, still = tile_shifts(one, moved, mask1 & (moved > 0), letter_height * scale, TILE_LH, 0.0)
    return {'agreement': float(np.mean(still >= AGREE)) if len(still) else 0.0,
            'median_correlation': float(np.median(still)) if len(still) else 0.0}


def refine(registration: Registration, levels, one: np.ndarray, two: np.ndarray, mask1: np.ndarray, scale: float,
           letter_height: float, size) -> tuple[list[Registration], list[dict]]:
    """`registration` after each of the local `levels` in turn, and what each found."""
    stages, found = [], []
    for tile_lh, reach_lh in levels:
        try:
            registration, level = local(registration, one, two, mask1, scale, letter_height, size, tile_lh, reach_lh)
        except ValueError as error:   # a spline whose inverse does not converge: keep the stage before it
            level = {'error': str(error)}
        stages.append(registration)
        found.append(level)
    return stages, found


def register(gray1: np.ndarray, gray2: np.ndarray, letter_height: float, fitted: Registration
             ) -> tuple[dict[str, Registration], dict]:
    """The registration after each stage ('affine', 'local1', ... one per level), the fitted one refined, the
    chosen one, and the diagnostics."""
    one, _, scale = picture(gray1, letter_height, COARSE_PX)
    two, _, _ = picture(gray2, letter_height, COARSE_PX)
    matrix, peak = coarse(two, one)
    matrix = to_original(matrix, scale)
    one, _, scale = picture(gray1, letter_height, FINE_PX)
    two, _, _ = picture(gray2, letter_height, FINE_PX)
    corrected, ecc = fine(two, one, to_scale(matrix, scale))
    affine = Registration('similarity', {'matrix': to_original(corrected, scale).tolist(), 'kind': 'affine'})
    one, mask1, scale = picture(gray1, letter_height, LOCAL_PX)
    two, _, _ = picture(gray2, letter_height, LOCAL_PX)
    size = [gray2.shape[1], gray2.shape[0]]
    stages, found = {'affine': affine}, {'peak': peak, 'ecc': ecc}
    images, levels = refine(affine, LEVELS, one, two, mask1, scale, letter_height, size)
    for level, (registration, what) in enumerate(zip(images, levels), 1):
        stages[f'local{level}'], found[f'local{level}'] = registration, what
    refined, levels = refine(fitted, LEVELS[1:], one, two, mask1, scale, letter_height, size)
    stages['refined'], found['refined_levels'] = refined[-1], levels
    candidates = {'fitted': fitted, 'refined': stages['refined'], 'images': images[-1]}
    found['measures'] = {name: measure(reg, one, two, mask1, scale, letter_height) for name, reg in candidates.items()}
    found['chosen'] = max(candidates, key=lambda name: found['measures'][name]['median_correlation'])
    stages['chosen'] = candidates[found['chosen']]
    return stages, {**found, **found['measures']['images']}


def rewrite(record: dict, gray1: np.ndarray, gray2: np.ndarray) -> dict:
    """`record` with its map replaced by the one chosen on the images (module docstring)."""
    registration = record['registration']
    height = float(registration['diagnostics']['first_pass_letter_height_px'])
    stages, found = register(gray1, gray2, height, Registration.from_dict(registration))
    chosen = stages['chosen']
    images = {'chosen': found['chosen'], 'measures': found['measures'],
              'replaced': {'model': registration['model'], 'parameters': registration['parameters']}}
    return {**record, 'registration': {**registration, 'model': chosen.model, 'parameters': chosen.parameters,
                                       'diagnostics': {**registration['diagnostics'], 'images': images}}}


def _rewrite_page(job: tuple) -> dict:
    root, base, paths = job
    work, started, chosen = Work(Path(root)), time.perf_counter(), []
    gray1 = gray2 = None
    for path in map(Path, paths):
        record = json.loads(path.read_text())
        if 'images' not in record['registration']['diagnostics']:
            if gray1 is None:
                gray1, gray2 = (cv2.imread(str(work.image(base, v)), cv2.IMREAD_GRAYSCALE) for v in (1, 2))
                if gray1 is None or gray2 is None:
                    raise FileNotFoundError(f'missing image pair for {base}')
            record = rewrite(record, gray1, gray2)
            partial = path.with_name(path.name + '.partial')
            partial.write_text(json.dumps(record))
            partial.replace(path)
        chosen.append(record['registration']['diagnostics']['images']['chosen'])
    return {'base': base, 'chosen': chosen, 'seconds': round(time.perf_counter() - started, 1)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--group', choices=('oof', 'new'), default='new')
    parser.add_argument('--splits', default='trogs26', help='comma-separated splits of the new pages')
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--config', default='configs/pagenet_only_style_neural.json')
    args = parser.parse_args()
    work, cfg = Work.from_env(), config.load(args.config)
    cv2.setNumThreads(1)
    folds = data.folds(cfg['data']['folds'])
    if args.group == 'oof':
        chains = {base: [work.chain(k, 'oof', bases).det(base)] for k, bases in folds.items() for base in bases}
    else:
        bases = sorted(b for split in args.splits.split(',') for b in data.splits()[split])
        chains = {base: [work.chain(k, 'new', bases).det(base) for k in sorted(folds)] for base in bases}
    jobs = [(str(work.root), base, [str(p) for p in paths]) for base, paths in sorted(chains.items())]
    with ProcessPoolExecutor(args.jobs) as pool:
        for row in pool.map(_rewrite_page, jobs):
            print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
