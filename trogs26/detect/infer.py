"""Run chain k's detector on pages and write their raw (uncalibrated) records `chains/f{k}/{group}/det_raw/{base}.json`.

    TROGS26_WORK=<work> python -m trogs26.detect.infer --fold K --group oof|new [--bases a,b | --splits s,t]
                                                        [--folds-file data/folds_train181.json] [--device cuda:0]
                                                        [--force]

Group 'oof' is the squeezes of fold K of the folds file (the chain never trained on them; `--bases` restricts them),
group 'new' pages named by `--bases` or `--splits` (names in data/splits.json). The reported detectors' folds are those
of data/folds_train181.json. Needs `models/detector/f{K}/{weights.pt,receipt.json}`
and both images of every page; reads no annotation. A page already having a raw record is skipped (`--force`
rewrites it).

The method, per page, two passes over both views:
  1. First pass. Each view alone, 1/coarse_divisor of its size, cut into tiles and detected. The Rotation1
     detections give the reading angle and the letter height; with at least min_mono_detections letters in each view
     their centres refine the registration of Rotation2 onto Rotation1 (otherwise only the image fit is used).
  2. Second pass. Pages with letters 40 px high: the packed page (view 1 | view 2 registered | their minimum, only if
     the registration is usable), and each view alone at the height its own first pass measured. A detection belongs
     to the tile whose centre is nearest to its own, so tiles overlap without duplicates and no box NMS crosses a
     tile. Rotation2 detections are mapped into Rotation1 through the registration.
  3. The three views' detections are fused (`fusion.py`) in the page's reading frame; box corners are canonical
     (NW, SW, SE, NE) rectangles and the letter height is their median.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time

import cv2
import numpy as np

from .. import config, data, provenance
from ..layout import Work
from ..registration import register_pair
from . import fusion, page, records, yolo

_CENTRE_QUALITY = ('pass', 'usable')


def model_id(work: Work, cfg: dict, fold: int) -> dict:
    """Identity of chain `fold`'s detector: its file, hash and selected epoch (from its training receipt)."""
    directory = work.models('detector', fold)
    receipt = json.loads((directory / 'receipt.json').read_text())
    digest = hashlib.sha256((directory / 'weights.pt').read_bytes()).hexdigest()
    if receipt.get('weights_sha256') not in (None, digest):
        raise ValueError(f'{directory}: weights.pt differs from the file its receipt describes')
    return {'checkpoint': f'models/detector/f{fold}/weights.pt', 'fold': fold, 'mode': 'pack',
            'run': cfg['detector']['run'], 'selected_epoch': receipt['selection']['epoch'], 'sha256': digest}


def read_pair(work: Work, base: str) -> tuple[np.ndarray, np.ndarray]:
    grays = [cv2.imread(str(work.image(base, view)), cv2.IMREAD_GRAYSCALE) for view in (1, 2)]
    if any(gray is None for gray in grays):
        raise FileNotFoundError(f'missing image pair for {base}')
    return grays[0], grays[1]


def map_detections(per_tile: list[tuple[np.ndarray, np.ndarray]], windows: list[tuple[int, int, int, int]],
                   frame: page.ScaleFrame) -> list[dict]:
    """Detections [{quad, score}] in the pixels of the view a page was rendered from, given each window's
    detections `per_tile` (quads (k, 4, 2) and scores (k,) in tile pixels).

    A detection is kept only from the window that owns its centre (the covering window whose centre is nearest), so
    overlapping windows add no duplicates; it is then mapped back through `frame`. Order: window by window.
    """
    found = []
    for index, (quads, scores) in enumerate(per_tile):
        for quad, score in zip(quads, scores):
            page_quad = np.asarray(quad, float) + np.asarray(windows[index][:2])
            if page.tile_owner(page_quad.mean(0), windows) == index:
                found.append({'quad': frame.inverse(page_quad), 'score': float(score)})
    return found


def detect_page(model, device: str, cfg: dict, rendered: np.ndarray, frame: page.ScaleFrame) -> list[dict]:
    """Detections of a rendered page (see `map_detections`): the page is cut into windows of the network size,
    white-padded, and detected `infer.batch` windows at a time."""
    d = cfg['detector']
    side = d['page']['tile']
    windows = page.tile_windows(frame.size, side, d['page']['overlap'])
    per_tile = []
    for start in range(0, len(windows), d['infer']['batch']):
        chunk = windows[start:start + d['infer']['batch']]
        tiles = np.full((len(chunk), side, side, 3), 255, np.uint8)
        for index, (x0, y0, x1, y1) in enumerate(chunk):
            tiles[index, :y1 - y0, :x1 - x0] = rendered[y0:y1, x0:x1]
        per_tile += yolo.detect(model, tiles, d['infer'], device)
    return map_detections(per_tile, windows, frame)


def _view_height(detections: list[dict]) -> tuple[float, float]:
    """(reading angle, median letter height) of a view's detections."""
    quads = [box['quad'] for box in detections]
    angle = page.estimate_reading_angle(quads)
    return angle, page.median_height(quads, angle)


def first_pass(model, device: str, cfg: dict, gray1: np.ndarray, gray2: np.ndarray) -> dict[str, list[dict]]:
    """Step 1's detections of each view alone, in the pixels of its view."""
    divisor = cfg['detector']['page']['coarse_divisor']
    coarse = {}
    for view, gray in (('mono1', gray1), ('mono2', gray2)):
        size = (gray.shape[1], gray.shape[0])
        frame = page.ScaleFrame(size, tuple(max(1, round(v / divisor)) for v in size))
        coarse[view] = detect_page(model, device, cfg, page.make_page(gray, None, None, frame, packed=False), frame)
    return coarse


def second_pass(model, device: str, cfg: dict, gray1: np.ndarray, gray2: np.ndarray, coarse: dict[str, list[dict]],
                registration, first_height: float, packed: bool) -> dict[str, list[dict]]:
    """Step 2's detections per view, in the pixels of their own view: the packed page through `registration` if
    `packed`, and each view alone at the height its own first pass measured."""
    target = cfg['detector']['page']['letter_px']
    branches: dict[str, list[dict]] = {view: [] for view in records.VIEWS}
    if packed:
        frame = page.ScaleFrame.from_height((gray1.shape[1], gray1.shape[0]), first_height, target)
        branches['pack'] = detect_page(model, device, cfg,
                                       page.make_page(gray1, gray2, registration, frame, packed=True), frame)
    for view, gray in (('mono1', gray1), ('mono2', gray2)):
        # Rotation2 enters the Rotation1 record only through a usable registration.
        if (view == 'mono2' and not packed) or not coarse[view]:
            continue
        _, height = _view_height(coarse[view])
        frame = page.ScaleFrame.from_height((gray.shape[1], gray.shape[0]), height, target)
        branches[view] = detect_page(model, device, cfg, page.make_page(gray, None, None, frame, packed=False), frame)
    return branches


def infer_page(model, device: str, cfg: dict, work: Work, fold: int, base: str, identity: dict) -> dict:
    """The raw record of `base`: every pass above, fused and canonicalised."""
    d = cfg['detector']
    gray1, gray2 = read_pair(work, base)
    sizes = {'Rotation1': [gray1.shape[1], gray1.shape[0]], 'Rotation2': [gray2.shape[1], gray2.shape[0]]}
    merged = page.is_merged(work.image(base, 1).name, work.image(base, 2).name)

    coarse = first_pass(model, device, cfg, gray1, gray2)
    if not coarse['mono1']:
        raise ValueError(f'{base}: zero Rotation1 detections; no measured letter height')
    _, first_height = _view_height(coarse['mono1'])
    refined = min(len(coarse['mono1']), len(coarse['mono2'])) >= d['infer']['min_mono_detections']
    if refined:
        centres = [np.asarray([box['quad'] for box in coarse[view]]).mean(1) for view in ('mono1', 'mono2')]
        registration = register_pair(gray1, gray2, *centres, letter_height=first_height, merged=merged)
    else:
        registration = register_pair(gray1, gray2, merged=merged)
    registration.diagnostics.update(
        refinement='fold_excluded_mono_detections' if refined else 'insufficient_mono_detections',
        mono_detection_counts=[len(coarse['mono1']), len(coarse['mono2'])], first_pass_letter_height_px=first_height)
    packed = refined and registration.quality in _CENTRE_QUALITY
    branches = second_pass(model, device, cfg, gray1, gray2, coarse, registration, first_height, packed)
    return assemble(cfg, fold, base, sizes, registration, first_height, branches, identity)


def assemble(cfg: dict, fold: int, base: str, sizes: dict, registration, first_height: float,
             branches: dict[str, list[dict]], identity: dict) -> dict:
    """The raw record from the three views' detections, each in the pixels of its own view (`branches`).

    Rotation2 detections are mapped into Rotation1 through `registration`, every box is made a canonical rectangle
    along the page's reading angle, the views are fused and the record's letter height is the boxes' median (the first
    pass's if there are none). Adds the fusion's counts to the registration's diagnostics.
    """
    d = cfg['detector']
    branches = dict(branches)
    if branches['mono2']:
        corners = np.asarray([box['quad'] for box in branches['mono2']], dtype=np.float64).reshape(-1, 2)
        branches['mono2'] = [{'quad': quad, 'score': box['score']} for box, quad
                             in zip(branches['mono2'], registration.apply(corners).reshape(-1, 4, 2))]
    reading_angle = page.estimate_reading_angle([box['quad'] for view in records.VIEWS for box in branches[view]])
    canonical = {view: [{'quad': page.canonical_quad(box['quad'], reading_angle), 'score': box['score']}
                        for box in branches[view]] for view in records.VIEWS}
    fused = fusion.fuse_views(canonical, d['fusion'])
    boxes = [{'id': index, 'quad': page.canonical_quad(box['quad'], reading_angle).tolist(), 'score': box['score'],
              'p_letter': None, 'view_support': box['view_support']} for index, box in enumerate(fused)]
    height = page.median_height([box['quad'] for box in boxes], reading_angle) if boxes else first_height
    counts = {view: len(branches[view]) for view in records.VIEWS}
    registration.diagnostics['detection_fusion'] = {
        'version': records.FUSION_VERSION, 'branch_detection_counts': counts,
        'cross_view_fused_count': sum(sum(v is not None for v in box['view_support'].values()) > 1 for box in boxes),
        'final_detection_count': len(boxes)}
    registration.diagnostics['final_letter_height_source'] = (
        'fused_detections' if boxes else 'first_pass_mono1_detections')
    return {'version': records.VERSION, 'base': base, 'fold': fold, 'detector': f"{d['run']}/f{fold}_pack",
            'frame': 'Rotation1', 'image_size': sizes, 'registration': registration.to_dict(),
            'letter_height_px': float(height), 'boxes': boxes,
            'provenance': {'status': 'uncalibrated', 'run': d['run'], 'model_id': identity,
                           'postprocess_version': records.FUSION_VERSION}}


def infer_pages(work: Work, cfg: dict, fold: int, group: str, bases: list[str], device: str,
                force: bool = False) -> dict:
    """Write the raw records of `bases` (a page whose record exists is skipped unless `force`)."""
    chain = work.chain(fold, group, bases)
    todo = [base for base in bases if force or not chain.raw_det(base).exists()]
    summary = {'pages': len(bases), 'skipped': len(bases) - len(todo), 'written': {}, 'failed': {}}
    if not todo:
        return summary
    yolo.require(cfg['detector']['ultralytics'])
    identity = model_id(work, cfg, fold)
    model = yolo.load(work.models('detector', fold) / 'weights.pt', cfg['detector']['infer']['max_det'], device)
    for base in todo:
        started = time.perf_counter()
        try:
            record = infer_page(model, device, cfg, work, fold, base, identity)
        except (ValueError, FileNotFoundError) as error:
            summary['failed'][base] = str(error)
            print(f'{base}: FAILED {error}', flush=True)
            continue
        records.write(chain.raw_det(base), record)
        summary['written'][base] = {'boxes': len(record['boxes']), 'seconds': round(time.perf_counter() - started, 1),
                                    'registration': record['registration']['quality']}
        print(f"{base}: {len(record['boxes'])} boxes, {summary['written'][base]['seconds']} s", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--config', default='configs/pagenet_only_style_neural.json')
    parser.add_argument('--folds-file', help="default: the configuration's data.folds; the reported detectors are "
                                              "trained on data/folds_train181.json")
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--group', choices=('oof', 'new'), required=True)
    parser.add_argument('--bases', help='comma-separated squeezes')
    parser.add_argument('--splits', help='comma-separated names in data/splits.json, e.g. trogs26')
    parser.add_argument('--device', help='default: cuda:0 if available, else cpu')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    cfg, work, started = config.load(args.config), Work.from_env(), time.perf_counter()
    folds = data.folds(args.folds_file or cfg['data']['folds'])
    if args.fold not in folds:
        raise SystemExit(f'fold {args.fold} is not in the folds file')
    named = args.bases.split(',') if args.bases else []
    if args.splits:
        named += [base for split in args.splits.split(',') for base in data.splits()[split]]
    if args.group == 'oof':
        outside = set(named) - set(folds[args.fold])
        if outside:
            raise SystemExit(f'not in fold {args.fold}: {sorted(outside)}')
        bases = named or list(folds[args.fold])
    elif named:
        bases = named
    else:
        raise SystemExit("group 'new' needs --bases or --splits")
    device = args.device
    if device is None:
        import torch
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    summary = infer_pages(work, cfg, args.fold, args.group, bases, device, args.force)
    print(json.dumps(provenance.summary(cfg, time.perf_counter() - started, fold=args.fold, group=args.group,
                                        device=device, **summary)))
    if summary['failed']:
        raise SystemExit(f"{len(summary['failed'])} pages failed: {sorted(summary['failed'])}")


if __name__ == '__main__':
    main()
