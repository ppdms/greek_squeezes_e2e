"""Calibrate the detector: turn every detection's fused score into P(letter) and publish the final records.

    TROGS26_WORK=<work> python -m trogs26.detect.calibrate --folds-file data/folds_train181.json [--config C]

Inputs: the raw records of every fold's squeezes (`chains/f{k}/oof/det_raw`, from `infer --group oof`), the
Rotation1 annotations of those squeezes, and any raw records of new pages (`chains/f{k}/new/det_raw`).
Outputs: `detector/calibration.json` (the maps and what they were fitted on) and `chains/f{k}/{oof,new}/det/{base}.json`
for every raw record found.

A detection's label is 1 if it wins the one-to-one match (maximum cardinality, then maximum total IoU, IoU >=
match_iou) to an annotated Rotation1 letter of its page. The map is the isotonic fit (pool-adjacent-violators on the
distinct scores, linear between them, clipped at the ends) of the label on the fused score. It is cross-fitted:
chain k's map is fitted on the out-of-fold detections of the other folds, never on fold k, and chain k applies it to
its own squeezes and to the new pages it detects. The reported detectors' folds are those of
data/folds_train181.json (the 181 TRAIN squeezes), which `--folds-file` names.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from .. import config, data, provenance
from ..geometry import match_box_ious, rotated_iou_matrix
from ..layout import Work
from . import records
from .annotations import read_letters


def page_labels(quads, annotated, match_iou: float) -> np.ndarray:
    """1.0 for every detection that is matched to an annotated letter (see the module docstring), else 0.0."""
    quads = np.asarray(quads, np.float32).reshape(-1, 4, 2)
    labels = np.zeros(len(quads))
    if len(quads) and len(annotated):
        matched = match_box_ious(rotated_iou_matrix(quads, annotated), match_iou)['match']
        labels[list(matched)] = 1.0
    return labels


def fit_isotonic(scores: np.ndarray, labels: np.ndarray) -> tuple[list[float], list[float]]:
    """Weighted pool-adjacent-violators isotonic fit of `labels` on `scores`: (distinct scores, fitted values)."""
    if not len(scores):
        raise ValueError('isotonic calibration needs detections')
    order = np.argsort(scores, kind='stable')
    sorted_scores, sorted_labels = scores[order], labels[order]
    knots, starts, weights = np.unique(sorted_scores, return_index=True, return_counts=True)
    sums = np.add.reduceat(sorted_labels, starts)
    blocks: list[list] = []  # [first knot, one past the last knot, samples, positives]
    for index, (weight, total) in enumerate(zip(weights, sums)):
        blocks.append([index, index + 1, int(weight), float(total)])
        while len(blocks) > 1 and blocks[-2][3] / blocks[-2][2] > blocks[-1][3] / blocks[-1][2]:
            right = blocks.pop()
            left = blocks[-1]
            left[1] = right[1]
            left[2] += right[2]
            left[3] += right[3]
    values = np.empty(len(knots))
    for first, last, weight, total in blocks:
        values[first:last] = np.clip(total / weight, 0.0, 1.0)
    return knots.tolist(), values.tolist()


def fit_maps(pages: list[tuple[int, np.ndarray, np.ndarray]]) -> dict[int, dict]:
    """Chain k's map for each fold k of `pages` [(fold, scores, labels)]: isotonic, fitted on the other folds."""
    folds = sorted({fold for fold, _, _ in pages})
    maps = {}
    for fold in folds:
        others = [(scores, labels) for other, scores, labels in pages if other != fold]
        scores = np.concatenate([x for x, _ in others]) if others else np.empty(0)
        if not len(scores):
            raise ValueError(f'fold {fold} has no other-fold detections to calibrate on')
        knots, values = fit_isotonic(scores, np.concatenate([y for _, y in others]))
        maps[fold] = {'scores': knots, 'values': values, 'interpolation': 'linear', 'out_of_bounds': 'clip',
                      'held_out_fold': fold, 'training_folds': [k for k in folds if k != fold],
                      'n_samples': len(scores)}
    return maps


def apply_map(mapping: dict, scores) -> np.ndarray:
    """P(letter) of `scores`: linear between the map's knots, clipped to [0, 1]."""
    return np.clip(np.interp(np.asarray(scores, np.float64), mapping['scores'], mapping['values']), 0.0, 1.0)


def calibrated(record: dict, mapping: dict, map_digest: str, calibration_digest: str, group: str) -> dict:
    """`record` with `p_letter` filled from `mapping` and the provenance of a calibrated record."""
    probabilities = apply_map(mapping, [box['score'] for box in record['boxes']])
    boxes = [{**box, 'p_letter': float(p)} for box, p in zip(record['boxes'], probabilities)]
    provenance_ = {'status': 'calibrated', 'run': record['provenance']['run'],
                   'model_id': record['provenance']['model_id'], 'held_out_fold': mapping['held_out_fold'],
                   'training_folds': mapping['training_folds'], 'calibration_sha256': calibration_digest,
                   'map_sha256': map_digest, 'raw_record_sha256': records.raw_digest(record)}
    if group == 'new':
        provenance_['model_fold'] = record['fold']
    return {**record, 'boxes': boxes, 'provenance': provenance_}


def calibrate(work: Work, cfg: dict, fold_map: dict[int, tuple[str, ...]]) -> dict:
    """Fit the cross-fitted maps on the out-of-fold raw records and publish every chain's final records."""
    match_iou = cfg['detector']['calibration']['match_iou']
    pages, inputs = [], {}
    for fold, bases in fold_map.items():
        chain = work.chain(fold, 'oof', bases)
        for base in bases:
            record = records.read(chain.raw_det(base))
            quads = [box['quad'] for box in record['boxes']]
            labels = page_labels(quads, read_letters(work.annotation(base, 1))[0].astype(np.float32), match_iou)
            pages.append((fold, np.asarray([box['score'] for box in record['boxes']]), labels))
            inputs.setdefault(fold, []).append({'base': base, 'raw_record_sha256': records.raw_digest(record),
                                                'detections': len(quads), 'matched': int(labels.sum())})
    maps = fit_maps(pages)
    digests = {fold: records.digest(mapping) for fold, mapping in maps.items()}
    calibration = {'schema': 'det_calibration_v1', 'run': cfg['detector']['run'], 'match_iou': match_iou,
                   'maps': {str(fold): {**maps[fold], 'map_sha256': digests[fold]} for fold in sorted(maps)},
                   'inputs': {str(fold): inputs[fold] for fold in sorted(inputs)}}
    path = work.detector / 'calibration.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(calibration, indent=1, sort_keys=True, allow_nan=False))
    calibration_digest = records.digest(calibration)
    counts = {'oof': 0, 'new': 0}
    for fold in sorted(maps):
        for group in ('oof', 'new'):
            chain = work.chain(fold, group, fold_map[fold] if group == 'oof' else ())
            raw = sorted((chain.dir / 'det_raw').glob('*.json')) if group == 'new' else \
                [chain.raw_det(base) for base in fold_map[fold]]
            for source in raw:
                record = records.read(source)
                records.write(chain.det(record['base']),
                              calibrated(record, maps[fold], digests[fold], calibration_digest, group))
                counts[group] += 1
    return {'calibration': str(path), 'records': counts,
            'detections': {str(k): sum(v['detections'] for v in inputs[k]) for k in sorted(inputs)},
            'matched': {str(k): sum(v['matched'] for v in inputs[k]) for k in sorted(inputs)},
            'map_knots': {str(k): len(maps[k]['scores']) for k in sorted(maps)}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--config', default='configs/pagenet_only_style_neural.json')
    parser.add_argument('--folds-file', help="default: the configuration's data.folds; the reported detectors are "
                                              "trained on data/folds_train181.json")
    args = parser.parse_args()
    cfg, work, started = config.load(args.config), Work.from_env(), time.perf_counter()
    summary = calibrate(work, cfg, data.folds(args.folds_file or cfg['data']['folds']))
    print(json.dumps(provenance.summary(cfg, time.perf_counter() - started, **summary)))


if __name__ == '__main__':
    main()
