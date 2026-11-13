"""The Ultralytics side of the detector (skipped where Ultralytics is not installed: outside the GPU environment)."""
from __future__ import annotations

import cv2
import numpy as np
import pytest
import torch

try:
    from ultralytics.cfg import get_cfg
    from ultralytics.nn.tasks import OBBModel

    from trogs26.detect import tiles, trainer, yolo
except ImportError:
    pytest.skip('Ultralytics is not installed', allow_module_level=True)


def test_batches_keep_images_as_views_and_offset_the_batch_index_of_each_sample():
    pixels = np.zeros((2, 64, 64, 3), np.uint8)
    targets = {'a.png': (np.zeros((2, 1), np.float32), np.ones((2, 5), np.float32)),
               'b.png': (np.zeros((1, 1), np.float32), np.ones((1, 5), np.float32))}
    dataset = trainer.TileDataset(['a', 'b'], pixels, {'a': 0, 'b': 1}, targets)
    batch = dataset.collate_fn([dataset[0], dataset[1]])
    assert batch['batch_idx'].tolist() == [0., 0., 1.]
    assert batch['cls'].shape == (3, 1) and batch['bboxes'].shape == (3, 5)
    assert isinstance(batch['img'], list) and np.shares_memory(batch['img'][1], pixels[1])
    assert dataset.labels[0]['cls'].shape == (2, 1) and len(dataset) == 2


def test_tiles_are_converted_from_bgr_arrays_to_rgb_tensors():
    tile = np.zeros((4, 4, 3), np.uint8)
    tile[..., 0], tile[..., 1], tile[..., 2] = 10, 20, 30  # B, G, R
    converted = trainer.images_to_device([tile], 'cpu')
    assert converted.shape == (1, 3, 4, 4) and converted[0, :, 0, 0].tolist() == [30, 20, 10]


def test_dense_routing_keeps_the_one_to_one_training_branch_and_returns_every_anchor():
    model = OBBModel('yolo26n-obb.yaml', nc=1, verbose=False)
    model.args = get_cfg(overrides={'task': 'obb'})
    yolo.configure_dense(model, 3000)
    head = model.model[-1]
    assert model.end2end is False and head.max_det == 3000 and head.one2one_cv2 is not None
    assert type(model.init_criterion()).__name__ == 'E2ELoss'
    model.eval()
    predictions = model(torch.zeros(1, 3, 320, 320))[0]
    assert predictions.shape[1] == 6 and predictions.shape[2] == (40 ** 2 + 20 ** 2 + 10 ** 2) > 300
    with pytest.raises(TypeError):
        yolo.configure_dense(torch.nn.Identity(), 3000)


def test_targets_of_a_tile_survive_the_dataset(tmp_path):
    root = tmp_path / 'tiles'
    (root / 'images').mkdir(parents=True)
    (root / 'labels').mkdir()
    cv2.imwrite(str(root / 'images' / 't.png'), np.full((64, 64, 3), 255, np.uint8))
    (root / 'labels' / 't.txt').write_text('0 0.25 0.25 0.25 0.5 0.5 0.5 0.5 0.25\n')
    pixels, targets = tiles.load_tiles(root, ['t'], 64)
    dataset = trainer.TileDataset(['t'], pixels, {'t': 0}, targets)
    sample = dataset[0]
    assert sample['bboxes'].shape == (1, 5) and sample['bboxes'][0, :2].tolist() == pytest.approx([.375, .375])
