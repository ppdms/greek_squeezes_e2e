"""The page network's relit crops and their cache (`trogs26.dino.crops`)."""
import cv2
import numpy as np
import pytest

from trogs26.dino import crops
from trogs26.registration import Registration

LETTER = 40.0


def views(shape=(400, 520), seed=0):
    """Two grey views lit along x and along y from the same grooved height field."""
    rng = np.random.default_rng(seed)
    canvas = np.zeros(shape, np.float32)
    for _ in range(90):
        x0, y0, angle = rng.uniform(40, shape[1] - 40), rng.uniform(40, shape[0] - 40), rng.uniform(0, np.pi)
        dx, dy = np.cos(angle) * LETTER / 2, np.sin(angle) * LETTER / 2
        cv2.line(canvas, (int(x0 - dx), int(y0 - dy)), (int(x0 + dx), int(y0 + dy)), 1.0, thickness=4)
    height = -cv2.GaussianBlur(canvas, (0, 0), 1.5)
    return [np.clip(255 * 0.5 * (1 + 1.5 * cv2.Sobel(height, cv2.CV_32F, dx, dy, ksize=3, scale=1 / 8)), 0, 255)
            .astype(np.uint8) for dx, dy in ((1, 0), (0, 1))]


def record() -> dict:
    boxes = [{'id': 7 + i, 'quad': [[x - 15, 150.0], [x - 15, 190.0], [x + 15, 190.0], [x + 15, 150.0]],
              'p_letter': 0.95} for i, x in enumerate((150.0, 220.0, 290.0))]
    registration = Registration('similarity', {'matrix': [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]}, None, 'pass', 'test', {})
    return {'registration': registration.to_dict(), 'letter_height_px': LETTER, 'boxes': boxes}


def test_every_detection_gets_a_relit_crop_of_the_requested_size():
    one, two = views()
    ids, found = crops.relief_page_crops(record(), one, two, 32, 1.6, ['relit'])
    assert ids.tolist() == [7, 8, 9]
    assert list(found) == ['relit'] and found['relit'].shape == (3, 3, 32, 32) and found['relit'].dtype == np.uint8
    assert found['relit'].std() > 0


def test_only_the_relit_input_is_cut():
    f = crops.record_fusion(record(), *views())
    with pytest.raises(ValueError):
        crops.relief_page(f, 'ink')
    assert crops.relief_page(f, 'relit').shape == (*f.mask.shape, 3)


def test_a_cache_is_current_only_for_the_record_and_parameters_it_was_cut_for(tmp_path):
    det = tmp_path / 'det.json'
    det.write_text('{"boxes": []}')
    cache = tmp_path / 'dino_crops_relit' / 'x.npz'
    params = {'size': 224, 'scale': 1.6, 'input': 'relit'}
    key = crops.cache_key(det, params)
    assert not crops.is_current(cache, key)
    crops.save_cache(cache, key, det_ids=np.arange(2), crops=np.zeros((2, 3, 4, 4), np.uint8))
    assert crops.is_current(cache, key)
    assert not crops.is_current(cache, crops.cache_key(det, {**params, 'scale': 1.5}))
    det.write_text('{"boxes": [1]}')
    assert not crops.is_current(cache, crops.cache_key(det, params))
    assert not list(cache.parent.glob('*.partial.npz'))
