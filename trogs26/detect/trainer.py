"""The Ultralytics trainer of the detector (imports Ultralytics; `train.py` is its driver).

What differs from Ultralytics' own oriented-box training:
  * the tiles are rendered once (`tiles.py`) and served as views of one RAM array with their oriented-box targets
    precomputed, so no label file is parsed and Ultralytics' dataset transforms (mosaic, HSV, flips, ...) do not exist;
  * the training batch is augmented on the device by `augment.py`;
  * validation runs on the held-out fold's pack tiles, dense one-to-many inference with rotated NMS;
  * every epoch's EMA weights are kept (`epoch_dir/epoch_NNNN.pt`, stripped like a finished Ultralytics weights file,
    the form `infer.py` loads) so the epoch can be chosen after training; Ultralytics' own best-fitness weights
    and its final validation of them are not used;
  * the AMP self-check, which downloads a second model, is replaced by "CUDA yes, CPU no".
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from ultralytics.engine import trainer as engine
from ultralytics.models.yolo.obb import OBBTrainer
from ultralytics.utils.torch_utils import strip_optimizer

from . import augment, yolo


class TileDataset(Dataset):
    """Tiles `names` as zero-copy views of `pixels` (rows `rows[name]`, BGR) with their OBB targets (`targets`,
    keyed `{name}.png`, from `tiles.load_tiles`)."""

    def __init__(self, names: list[str], pixels: np.ndarray, rows: dict[str, int], targets: dict[str, tuple]):
        self.pixels = pixels
        self.rows = [rows[name] for name in names]
        self.im_files = [f'{name}.png' for name in names]
        self.labels = [{'cls': targets[file][0], 'bboxes': targets[file][1]} for file in self.im_files]
        self.shape = pixels.shape[1:3]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        label = self.labels[index]
        return {'img': self.pixels[self.rows[index]], 'im_file': self.im_files[index], 'ori_shape': self.shape,
                'resized_shape': self.shape, 'ratio_pad': ((1., 1.), (0., 0.)),
                'cls': torch.from_numpy(label['cls']), 'bboxes': torch.from_numpy(label['bboxes']),
                'batch_idx': torch.zeros(len(label['cls']))}

    @staticmethod
    def collate_fn(samples: list[dict]) -> dict:
        """Images stay a list of numpy views until the device; targets are concatenated with batch indices."""
        batch = {}
        for key in samples[0]:
            values = [sample[key] for sample in samples]
            if key in ('cls', 'bboxes'):
                batch[key] = torch.cat(values, 0)
            elif key == 'batch_idx':
                batch[key] = torch.cat([value + index for index, value in enumerate(values)], 0)
            else:
                batch[key] = values
        return batch


def images_to_device(images: list[np.ndarray], device) -> torch.Tensor:
    """(n, 3, H, W) uint8 RGB tensor on `device` from BGR (H, W, 3) arrays."""
    stacked = torch.stack([torch.from_numpy(image).to(device, non_blocking=True) for image in images])
    return stacked.permute(0, 3, 1, 2)[:, [2, 1, 0]].contiguous()


class TileTrainer(OBBTrainer):
    """OBBTrainer on `tiles` ({'train': TileDataset, 'val': TileDataset}); epoch weights go to `epoch_dir`. With
    `stop_after`, training ends once that epoch is saved; the schedule stays the full run's, so the weights of every
    epoch up to it are those the full run would have had."""

    def __init__(self, tiles: dict[str, TileDataset], epoch_dir: Path, max_det: int, overrides: dict,
                 stop_after: int | None = None):
        self.tiles, self.epoch_dir, self.max_det, self.stop_after = tiles, Path(epoch_dir), max_det, stop_after
        super().__init__(overrides=overrides)

    def get_dataset(self) -> dict:
        return {'train': 'train', 'val': 'val', 'names': {0: 'letter'}, 'nc': 1, 'channels': 3}

    def build_dataset(self, img_path, mode='train', batch=None):
        return self.tiles['train' if mode == 'train' else 'val']

    def get_model(self, cfg=None, weights=None, verbose=True):
        return yolo.configure_dense(super().get_model(cfg, weights, verbose), self.max_det)

    def _setup_train(self):
        original = engine.check_amp
        engine.check_amp = lambda model: next(model.parameters()).device.type == 'cuda'
        try:
            return super()._setup_train()
        finally:
            engine.check_amp = original

    def preprocess_batch(self, batch):
        batch['img'] = images_to_device(batch['img'], self.device)
        batch = super().preprocess_batch(batch)
        with torch.autocast(device_type=self.device.type, enabled=False):
            return augment.augment_batch(batch)

    def get_validator(self):
        validator = super().get_validator()
        preprocess = validator.preprocess

        def to_device_first(batch):
            batch['img'] = images_to_device(batch['img'], self.device)
            return preprocess(batch)

        validator.preprocess = to_device_first
        return validator

    def save_model(self):
        saved = super().save_model()
        self.epoch_dir.mkdir(parents=True, exist_ok=True)
        target = self.epoch_dir / f'epoch_{self.epoch + 1:04d}.pt'
        partial = target.with_name(target.name + '.partial')
        strip_optimizer(self.last, str(partial), updates={'training_epoch': self.epoch + 1})
        os.replace(partial, target)
        if self.stop_after is not None and self.epoch + 1 >= self.stop_after:
            self.stop = True
        return saved

    def final_eval(self):
        return {}
