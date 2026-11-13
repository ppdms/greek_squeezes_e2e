"""Detect again, through the registration chosen on the images, the pages whose reading that registration changes.

    TROGS26_WORK=<work> python -m trogs26.detect.redetect --group oof|new [--splits trogs26] [--device cuda:0]

Runs after `python -m trogs26.registration_images` has replaced the map of every chain's calibrated records
`chains/f{k}/{group}/det/{base}.json` (keeping the fitted map it replaced). A record is detected again when its
fitted registration kept view 2 out of the second pass (quality below 'usable'), or when the fitted and the chosen
map send the centres of view 2's first-pass detections apart, their 0.9 quantile at least `APART` letter heights.
Chain k's detector runs the first pass again and the second with the packed page through the chosen map, which the
record carries as 'pass' with inlier fraction 0.99, so that every reader of the record reads both views; the raw
record is calibrated with the map that calibrated the old one (detector/calibration.json, whose map digest must
match the record's). The chains' folds are the configuration's (data.folds): chain k's records, detector and map are
those of the detector work tree's chain k (`detect.frozen`).

Records whose first pass had too few detections in a view to refine the registration stay as they are. Each record
examined keeps the measure and the decision in `registration.diagnostics.images` ('apart', 'redetected'); a record
that holds them is left alone.

Inputs: the chains' rewritten records, models/detector/f{k}, detector/calibration.json, the pages' images.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from .. import config, data
from ..layout import Work
from ..registration import Registration
from . import calibrate, infer, records, yolo

CALIBRATION = 'calibration.json'  # holds the maps that calibrated the chains' records
APART = 0.25  # letter heights


def apart(fitted: Registration, chosen: Registration, detections: list[dict], height: float) -> float | None:
    """The 0.9 quantile, in letter heights, of how far apart the two maps send the centres of `detections` (None
    without any)."""
    if not detections:
        return None
    centres = np.asarray([box['quad'] for box in detections], dtype=np.float64).reshape(-1, 4, 2).mean(axis=1)
    return float(np.quantile(np.linalg.norm(fitted.apply(centres) - chosen.apply(centres), axis=1), .9) / height)


def redetect(model, device: str, cfg: dict, work: Work, record: dict, identity: dict, mapping: dict,
             map_digest: str, group: str) -> dict:
    """`record` with the measure and the decision, detected again if the decision says so (module docstring)."""
    base, registration = record['base'], record['registration']
    images = registration['diagnostics']['images']
    fitted = Registration.from_dict({**registration, **images['replaced']})
    chosen = Registration.from_dict(registration)
    gray1, gray2 = infer.read_pair(work, base)
    coarse = infer.first_pass(model, device, cfg, gray1, gray2)
    if not coarse['mono1']:
        raise ValueError(f'{base}: zero Rotation1 detections; no measured letter height')
    _, first_height = infer._view_height(coarse['mono1'])
    distance = apart(fitted, chosen, coarse['mono2'], first_height)
    again = fitted.quality not in infer._CENTRE_QUALITY or (distance is not None and distance >= APART)
    chosen.diagnostics = {**chosen.diagnostics, 'images': {**images, 'apart': distance, 'redetected': again},
                          'first_pass_letter_height_px': first_height}
    if not again:
        return {**record, 'registration': {**registration, 'diagnostics': {
            **registration['diagnostics'], 'images': chosen.diagnostics['images']}}}
    chosen.diagnostics.update(fit_inlier_fraction=fitted.inlier_fraction, fit_quality=fitted.quality,
                              mono_detection_counts=[len(coarse['mono1']), len(coarse['mono2'])])
    chosen.inlier_fraction, chosen.quality = 0.99, 'pass'
    branches = infer.second_pass(model, device, cfg, gray1, gray2, coarse, chosen, first_height, packed=True)
    sizes = {'Rotation1': [gray1.shape[1], gray1.shape[0]], 'Rotation2': [gray2.shape[1], gray2.shape[0]]}
    raw = infer.assemble(cfg, record['fold'], base, sizes, chosen, first_height, branches, identity)
    return calibrate.calibrated(raw, mapping, map_digest, record['provenance']['calibration_sha256'], group)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--group', choices=('oof', 'new'), default='new')
    parser.add_argument('--splits', default='trogs26', help='comma-separated splits of the new pages')
    parser.add_argument('--config', default='configs/pagenet_only_style_neural.json')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    work, cfg = Work.from_env(), config.load(args.config)
    yolo.require(cfg['detector']['ultralytics'])
    folds = data.folds(cfg['data']['folds'])
    calibration = json.loads((work.detector / CALIBRATION).read_text())
    for k in sorted(folds):
        if args.group == 'oof':
            bases = list(folds[k])
        else:
            bases = sorted(b for split in args.splits.split(',') for b in data.splits()[split])
        chain = work.chain(k, args.group, bases)
        todo = []
        for base in bases:
            record = records.read(chain.det(base))
            diagnostics = record['registration']['diagnostics']
            if 'images' not in diagnostics:
                raise ValueError(f'{chain.det(base)}: no map chosen on the images '
                                 '(python -m trogs26.registration_images)')
            if 'redetected' not in diagnostics['images'] and \
                    diagnostics.get('refinement') == 'fold_excluded_mono_detections':
                todo.append((base, record))
        if not todo:
            continue
        mapping = {key: v for key, v in calibration['maps'][str(k)].items() if key != 'map_sha256'}
        map_digest = calibration['maps'][str(k)]['map_sha256']
        if records.digest(mapping) != map_digest:
            raise ValueError(f'chain {k}: the map does not match its digest')
        identity = infer.model_id(work, cfg, k)
        model = yolo.load(work.models('detector', k) / 'weights.pt', cfg['detector']['infer']['max_det'], args.device)
        for base, record in todo:
            if record['provenance']['map_sha256'] != map_digest:
                raise ValueError(f'{chain.det(base)}: calibrated with another map')
            started = time.perf_counter()
            out = redetect(model, args.device, cfg, work, record, identity, mapping, map_digest, args.group)
            records.write(chain.det(base), out)
            images = out['registration']['diagnostics']['images']
            print(json.dumps({'fold': k, 'base': base, 'apart': images['apart'], 'redetected': images['redetected'],
                              'boxes': [len(record['boxes']), len(out['boxes'])],
                              'seconds': round(time.perf_counter() - started, 1)}), flush=True)


if __name__ == '__main__':
    main()
