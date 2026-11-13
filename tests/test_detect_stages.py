"""Detector stages on synthetic data: view fusion, record assembly, calibration and the epoch rule."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest

from trogs26 import config
from trogs26.detect import calibrate, fusion, infer, page, records, train
from trogs26.geometry import rotated_iou_matrix
from trogs26.registration import Registration

CFG = config.load('configs/pagenet_only_style_neural.json')
FUSION = CFG['detector']['fusion']


def letter(cx, cy, width=30., height=60., angle=0.):
    return page.obb_to_quad([cx, cy, width, height, angle])


def detections(*boxes):
    return [{'quad': quad, 'score': score} for quad, score in boxes]


def test_overlapping_boxes_of_different_views_fuse_to_their_score_weighted_mean():
    a, b = letter(100., 100.), letter(102., 100.)
    fused = fusion.fuse_views({'pack': detections((a, .9)), 'mono1': detections((b, .7)),
                               'mono2': detections((a, .5))}, FUSION)
    assert len(fused) == 1
    assert fused[0]['view_support'] == {'pack': pytest.approx(.9), 'mono1': pytest.approx(.7),
                                        'mono2': pytest.approx(.5)}
    assert fused[0]['score'] == pytest.approx(np.mean([.9, .7, .5]))
    assert fused[0]['quad'] == pytest.approx((1.4 * a + .7 * b) / 2.1, abs=1e-4)


def test_a_second_box_of_the_same_view_never_joins_a_cluster_that_has_its_view():
    a, b = letter(100., 100.), letter(102., 100.)
    fused = fusion.fuse_views({'pack': detections((a, .9)), 'mono1': detections((b, .7), (a, .6))}, FUSION)
    assert len(fused) == 2 and fused[0]['view_support']['mono1'] == pytest.approx(.7)
    assert fused[1]['view_support'] == {'pack': None, 'mono1': pytest.approx(.6), 'mono2': None}
    iou = rotated_iou_matrix(fused[0]['quad'][None], a[None])[0, 0]
    assert fused[1]['score'] == pytest.approx(.6 * math.exp(-iou ** 2 / FUSION['soft_nms_sigma']), rel=1e-4)


def test_soft_nms_decays_the_scores_of_overlaps_and_never_drops_a_candidate():
    a, b = letter(100., 100.), letter(104., 100.)  # same view, IoU 26/34 = 0.76 > soft-NMS threshold
    fused = fusion.fuse_views({'pack': detections((a, .9), (b, .8))}, FUSION)
    assert fused[0]['score'] == pytest.approx(.9)
    iou = 26 / 34
    assert fused[1]['score'] == pytest.approx(.8 * math.exp(-iou ** 2 / FUSION['soft_nms_sigma']), rel=1e-3)
    assert fused[1]['view_support']['pack'] == pytest.approx(.8)  # the view's own score is not decayed
    far = fusion.fuse_views({'pack': detections((a, .9), (letter(300., 100.), .8))}, FUSION)
    assert [box['score'] for box in far] == pytest.approx([.9, .8])


def test_fusion_refuses_scores_below_the_inference_confidence_and_unknown_views():
    with pytest.raises(ValueError):
        fusion.fuse_views({'pack': detections((letter(1., 1.), .01))}, FUSION)
    with pytest.raises(ValueError):
        fusion.fuse_views({'pack2': []}, FUSION)
    assert fusion.fuse_views({}, FUSION) == []


def test_a_detection_is_kept_only_from_the_window_that_owns_it():
    frame = page.ScaleFrame((2000, 1000), (2000, 1000))
    windows = [(0, 0, 1000, 1000), (500, 0, 1500, 1000), (1000, 0, 2000, 1000)]
    quad = letter(700., 400., 30., 60.) - [500., 0.]  # as the second window sees it
    twin = letter(700., 400., 30., 60.)  # and as the first window sees it (it covers x = 700 too)
    scores = np.array([.9], np.float32)
    kept = infer.map_detections([(twin[None], scores), (quad[None], scores), (np.empty((0, 4, 2)), scores[:0])],
                                windows, frame)
    assert len(kept) == 1 and kept[0]['quad'] == pytest.approx(letter(700., 400.), abs=1e-4)


def synthetic_record():
    """A page whose three views see the same two letters (Rotation2 related to Rotation1 by the identity)."""
    letters = [letter(100., 100.), letter(300., 110.)]
    branches = {view: detections(*[(q, s) for q, s in zip(letters, scores)])
                for view, scores in (('pack', (.9, .8)), ('mono1', (.85, .75)), ('mono2', (.88, .7)))}
    registration = Registration('similarity', {'matrix': [[1., 0., 0.], [0., 1., 0.]]}, 1.0, 'pass')
    sizes = {'Rotation1': [500, 400], 'Rotation2': [500, 400]}
    identity = {'checkpoint': 'models/detector/f0/weights.pt', 'fold': 0, 'sha256': 'x', 'selected_epoch': 59,
                'run': 'r', 'mode': 'pack'}
    return infer.assemble(CFG, 0, 'x', sizes, registration, 60., branches, identity), letters


def test_assembled_record_has_the_layout_every_later_stage_reads():
    record, letters = synthetic_record()
    records.validate(record)
    assert [box['id'] for box in record['boxes']] == [0, 1]
    assert record['letter_height_px'] == pytest.approx(60., abs=1e-3)
    assert all(box['p_letter'] is None for box in record['boxes'])
    assert record['boxes'][0]['quad'] == pytest.approx(letters[0], abs=1e-3)
    assert set(record['boxes'][0]['view_support']) == {'pack', 'mono1', 'mono2'}
    assert record['registration']['diagnostics']['detection_fusion']['cross_view_fused_count'] == 2
    assert Registration.from_dict(record['registration']).quality == 'pass'


def test_the_raw_identity_of_a_record_ignores_calibration_and_provenance():
    record, _ = synthetic_record()
    calibrated = calibrate.calibrated(record, {'scores': [0., 1.], 'values': [0., 1.], 'held_out_fold': 0,
                                               'training_folds': [1]}, 'm', 'c', 'oof')
    assert [box['p_letter'] for box in calibrated['boxes']] == pytest.approx([box['score'] for box in record['boxes']])
    assert records.raw_digest(calibrated) == records.raw_digest(record)
    assert calibrated['provenance']['status'] == 'calibrated'
    records.validate(calibrated)


def test_a_record_with_a_wrongly_ordered_quad_or_a_weak_view_is_rejected():
    record, _ = synthetic_record()
    bad = copy.deepcopy(record)
    bad['boxes'][0]['quad'] = bad['boxes'][0]['quad'][::-1]
    with pytest.raises(ValueError):
        records.validate(bad)
    bad = copy.deepcopy(record)
    bad['boxes'][0]['view_support'] = {'pack': .01, 'mono1': None, 'mono2': None}
    with pytest.raises(ValueError):
        records.validate(bad)


def test_isotonic_fit_pools_violators_and_groups_equal_scores():
    scores = np.array([.1, .2, .3, .4, .5, .5])
    labels = np.array([0., 1., 0., 1., 0., 1.])
    knots, values = calibrate.fit_isotonic(scores, labels)
    # .2 (1) and .3 (0) pool to 1/2; .4 (1) and .5 (0 and 1) pool to 2/3
    assert knots == pytest.approx([.1, .2, .3, .4, .5])
    assert values == pytest.approx([0., .5, .5, 2 / 3, 2 / 3])


def test_a_folds_map_never_sees_its_own_fold():
    rng = np.random.default_rng(0)
    pages = [(fold, rng.uniform(.05, 1, 50), (rng.uniform(size=50) < .5).astype(float)) for fold in range(3)]
    maps = calibrate.fit_maps(pages)
    flipped = [(fold, scores, 1 - labels if fold == 1 else labels) for fold, scores, labels in pages]
    changed = calibrate.fit_maps(flipped)
    assert maps[1] == changed[1]  # fold 1's map does not depend on fold 1's labels
    assert maps[0] != changed[0] and maps[2] != changed[2]
    assert maps[0]['training_folds'] == [1, 2] and maps[0]['held_out_fold'] == 0
    with pytest.raises(ValueError):
        calibrate.fit_maps([(0, np.array([.5]), np.array([1.]))])


def test_a_detection_is_labelled_only_if_it_wins_the_one_to_one_match():
    gt = np.array([letter(100., 100.)], np.float32)
    better, worse, elsewhere = letter(101., 100.), letter(105., 100.), letter(300., 100.)
    labels = calibrate.page_labels([worse, better, elsewhere], gt, .5)
    assert labels.tolist() == [0., 1., 0.]
    assert calibrate.page_labels([], gt, .5).shape == (0,)
    assert calibrate.page_labels([better], np.empty((0, 4, 2), np.float32), .5).tolist() == [0.]


def test_the_map_interpolates_between_knots_and_clips_outside_them():
    mapping = {'scores': [.2, .6], 'values': [.1, .9]}
    assert calibrate.apply_map(mapping, [0., .2, .4, .6, 1.]).tolist() == pytest.approx([.1, .1, .5, .9, .9])


def test_the_chosen_epoch_is_the_best_held_out_one_and_ties_go_to_the_earliest():
    metric = train.METRIC
    curve = [{'epoch': e, metric: v} for e, v in zip(range(1, 6), [.5, .7, .7, .6, .1])]
    assert train.best_epoch(curve) == 2
    assert train.select_epoch(curve, 5, 'best') == 2 and train.select_epoch(curve, 5, 4) == 4
    for bad in (0, 6, 2.5, 'worst'):
        with pytest.raises(ValueError):
            train.select_epoch(curve, 5, bad)
    with pytest.raises(ValueError):
        train.select_epoch(curve[:3], 5, 'best')  # an incomplete run has no best epoch
    curve[2][metric] = float('nan')
    with pytest.raises(ValueError):
        train.best_epoch(curve)


def test_the_recipe_overrides_switch_ultralytics_own_augmentation_off():
    d = CFG['detector']
    run = Path('/tmp/x/run')
    ov = train.overrides(d, Path('w.pt'), run, epochs=100, batch=.85, val=True, device='0')
    assert ov['mosaic'] == ov['hsv_h'] == ov['fliplr'] == ov['scale'] == 0. and ov['close_mosaic'] == 0
    assert ov['imgsz'] == 1280 and ov['seed'] == 42 and ov['deterministic'] and ov['max_det'] == 3000
    assert json.dumps(ov) and ov['name'] == run.name and ov['project'] == str(run.parent)
