"""Which of the detector work tree's records each chain of the final work tree gets (`trogs26.detect.frozen`)."""
from __future__ import annotations

import hashlib
import json

import pytest

from trogs26 import data
from trogs26.detect import frozen
from trogs26.layout import Work

SPLITS = {'train': ['t0', 't1'], 'validation': ['v0'], 'internal_test': ['i0'], 'trogs26': ['n0']}
FOLDS = {0: ('t0', 'v0'), 1: ('t1', 'i0')}
CALIBRATION = '{"maps": {"0": {"map_sha256": "a"}, "1": {"map_sha256": "b"}}}'


def record(base: str, detector_chain: int) -> dict:
    box = {'id': 0, 'quad': [[0.0, 0.0], [0.0, 10.0], [10.0, 10.0], [10.0, 0.0]], 'score': 0.9, 'p_letter': 0.8,
           'view_support': {'pack': 0.9, 'mono1': None, 'mono2': None}}
    return {'version': 'det_v1', 'base': base, 'fold': detector_chain, 'detector': f'det/f{detector_chain}_pack',
            'frame': 'Rotation1', 'image_size': {'Rotation1': [20, 20], 'Rotation2': [20, 20]}, 'registration': {},
            'letter_height_px': 10.0, 'boxes': [box], 'provenance': {'status': 'calibrated'}}


def put(source: Work, chain: int, group: str, base: str) -> None:
    path = source.chain(chain, group, ()).det(base)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record(base, chain)))


@pytest.fixture
def source(tmp_path, monkeypatch):
    """The detector work tree: TRAIN squeezes out of fold in its chains 3 and 0, image-only squeezes in every chain,
    and its calibration."""
    monkeypatch.setattr(data, 'splits', lambda: SPLITS)
    tree = Work(tmp_path / 'detectors')
    put(tree, 3, 'oof', 't0')
    put(tree, 0, 'oof', 't1')
    for chain in (0, 1):
        for base in ('v0', 'i0', 'n0'):
            put(tree, chain, 'new', base)
    tree.detector.mkdir(parents=True)
    (tree.detector / 'calibration.json').write_text(CALIBRATION)
    return tree


def chain_of(work: Work, fold: int, group: str, base: str) -> int:
    return json.loads(work.chain(fold, group, ()).det(base).read_text())['fold']


def test_out_of_fold_records_come_from_detectors_that_never_saw_the_page(source, tmp_path):
    work = Work(tmp_path / 'final')
    summary = frozen.place(work, source.root, FOLDS, 'oof', [])
    assert summary['records'] == 4
    # TRAIN squeezes: the detector chain that held them out, whatever the final fold
    assert chain_of(work, 0, 'oof', 't0') == 3 and chain_of(work, 1, 'oof', 't1') == 0
    # validation / internal_test squeezes of final fold k: detector chain k read them image-only
    assert chain_of(work, 0, 'oof', 'v0') == 0 and chain_of(work, 1, 'oof', 'i0') == 1
    manifest = json.loads((work.detector / 'frozen_oof.json').read_text())
    copied = work.chain(1, 'oof', ()).det('i0').read_bytes()
    assert manifest['records']['f1/i0'] == {'source': 'chains/f1/new/det/i0.json', 'detections': 1,
                                            'sha256': hashlib.sha256(copied).hexdigest()}
    assert copied == source.chain(1, 'new', ()).det('i0').read_bytes()


def test_the_detector_work_trees_calibration_is_copied_and_listed_in_the_manifest(source, tmp_path):
    work = Work(tmp_path / 'final')
    frozen.place(work, source.root, FOLDS, 'new', ['trogs26'])
    assert (work.detector / 'calibration.json').read_text() == CALIBRATION
    manifest = json.loads((work.detector / 'frozen_new.json').read_text())
    assert manifest['calibration'] == {'source': 'detector/calibration.json',
                                       'sha256': hashlib.sha256(CALIBRATION.encode()).hexdigest()}


def test_new_pages_of_each_chain_come_from_the_same_detector_chain(source, tmp_path):
    work = Work(tmp_path / 'final')
    frozen.place(work, source.root, FOLDS, 'new', ['trogs26'])
    assert [chain_of(work, k, 'new', 'n0') for k in FOLDS] == [0, 1]
    assert not work.chain(0, 'new', ()).det('v0').exists()


def test_refuses_ambiguous_or_uncalibrated_records(source, tmp_path):
    put(source, 1, 'oof', 't0')
    with pytest.raises(FileNotFoundError, match='t0: 2 out-of-fold records'):
        frozen.place(Work(tmp_path / 'a'), source.root, FOLDS, 'oof', [])
    raw = record('n0', 0)
    raw['boxes'][0]['p_letter'] = None
    source.chain(0, 'new', ()).det('n0').write_text(json.dumps(raw))
    with pytest.raises(ValueError, match='not a calibrated record'):
        frozen.place(Work(tmp_path / 'b'), source.root, FOLDS, 'new', ['trogs26'])
