"""Working-directory layout shared by every stage.

Everything a stage reads or writes lives under one root, the environment variable `TROGS26_WORK`:

  annotations/{base}_Rotation{1,2}_300dpi_letters.txt   public annotations this run may read
  images/{file}                                          squeeze images (TROGS-26 names: data/splits.json)
  lm/{name}/lm.json                                      character language models
  pretrained/{name}/                                     third-party pretrained weights (DINOv3 ViT-S/16, YOLO26-L)
  detector/                                              detector files shared by all chains (base weights, training
                                                         tiles, cross-fitted calibration.json)
  models/{stage}/f{k}/...                                chain k's trained models (never saw fold k): detector,
                                                         pagenet_{version}, neural_lm
  models/pagenet_{version}/all/                          the page network trained on every fold (reads new pages)
  chains/f{k}/{group}/...                                chain k's stage outputs for one page group:
      group 'oof': the squeezes of fold k; group 'new': image-only squeezes (validation, internal_test, trogs26)
      det_raw/{base}.json                                uncalibrated fused detections (calibration input)
      det/{base}.json                                    calibrated detections
      dino_crops_{input}/{base}.npz                      per-squeeze crop caches (`trogs26.dino.crops`)
      pagenet_{version}.npz                              page network reads
      pagenet_{version}_all.npz                          new pages read by the page network trained on every fold
  runs/{run}/{split}/f{k}/{variant}/{squeeze}_transcript.txt   decoded transcripts per chain and variant
  runs/{run}/{split}/rover/{squeeze}_transcript.txt             ensemble transcripts
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .data import splits

GROUPS = ('oof', 'new')


@dataclass(frozen=True)
class Chain:
    """Chain `fold`'s stage outputs for the squeezes `bases` of page group `group` ('oof' or 'new')."""
    root: Path
    fold: int
    group: str
    bases: tuple[str, ...]

    def __post_init__(self):
        if self.group not in GROUPS:
            raise ValueError(f'unknown page group {self.group!r}')
        object.__setattr__(self, 'root', Path(self.root))
        object.__setattr__(self, 'bases', tuple(self.bases))

    @property
    def dir(self) -> Path:
        return self.root / 'chains' / f'f{self.fold}' / self.group

    def raw_det(self, base: str) -> Path:
        return self.dir / 'det_raw' / f'{base}.json'

    def det(self, base: str) -> Path:
        return self.dir / 'det' / f'{base}.json'

    def dino_crops(self, base: str, input: str) -> Path:
        """Crop cache of one squeeze for a crop input (`trogs26.dino.crops.INPUTS`)."""
        return self.dir / f'dino_crops_{input}' / f'{base}.npz'


@dataclass(frozen=True)
class Work:
    root: Path

    def __post_init__(self):
        object.__setattr__(self, 'root', Path(self.root))

    @classmethod
    def from_env(cls) -> 'Work':
        value = os.environ.get('TROGS26_WORK')
        if not value:
            raise RuntimeError('set TROGS26_WORK to the working directory')
        return cls(Path(value))

    @property
    def annotations(self) -> Path:
        return self.root / 'annotations'

    def annotation(self, base: str, view: int) -> Path:
        return self.annotations / f'{base}_Rotation{view}_300dpi_letters.txt'

    @property
    def images(self) -> Path:
        return self.root / 'images'

    def image(self, base: str, view: int) -> Path:
        """Image of `view` (1 or 2); TROGS-26 file names follow the release (data/splits.json)."""
        files = splits()['trogs26_files'].get(base)
        name = Path(files[f'Rotation{view}']).name if files else f'{base}_Rotation{view}_300dpi.png'
        return self.images / name

    def lm(self, name: str) -> Path:
        return self.root / 'lm' / name

    def pretrained(self, name: str) -> Path:
        return self.root / 'pretrained' / name

    @property
    def detector(self) -> Path:
        return self.root / 'detector'

    def models(self, stage: str, fold: int | None) -> Path:
        """Chain `fold`'s models of `stage`; `fold` None: the model trained on every fold."""
        return self.root / 'models' / stage / ('all' if fold is None else f'f{fold}')

    def chain(self, fold: int, group: str, bases) -> Chain:
        return Chain(self.root, fold, group, tuple(bases))

    def run(self, run: str, split: str) -> Path:
        return self.root / 'runs' / run / split
