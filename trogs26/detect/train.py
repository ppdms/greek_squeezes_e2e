"""Train chain k's detector: YOLO26-L oriented boxes on the other folds' tiles, the held-out fold's tiles monitored.

    TROGS26_WORK=<work> python -m trogs26.detect.train --fold K [--folds-file data/folds_train181.json]
                                                        [--epochs 100] [--imgsz 1280] [--batch 0.85]
                                                        [--epoch N|best] [--stop] [--no-val] [--jobs J]
                                                        [--keep-epochs] [--force] [--device 0]

Reads the annotations and images of every squeeze of the folds file (`tiles.py` renders them, `--jobs` at a time, once
per squeeze; the reported detectors use data/folds_train181.json, the 181 TRAIN squeezes) and
`pretrained/yolo26l-obb.pt`. Writes `models/detector/f{K}/weights.pt` (the selected epoch's weights, the form `infer.py`
loads) and `receipt.json` (recipe, tile counts, selected epoch, per-epoch metrics, seconds); `run/` holds the
Ultralytics run while training. An interrupted run resumes from its last epoch when started again with the same
command; a finished one is left alone (`--force` retrains).

The recipe (see `trainer.py`, `augment.py` and the `detector` section of the configuration): 100 epochs from the
pretrained weights, seed 42, deterministic, AMP, Ultralytics' auto batch (0.85 of the GPU memory) with its default
AdamW schedule, Ultralytics' augmentation off and ours on.

The epoch: `--epoch N` (default the configuration's use_epoch, 59: the epoch with the highest held-out mAP50-95 on
fold 0, used for every chain) takes epoch N; `--epoch best` takes the epoch with the highest held-out mAP50-95 over all
epochs of this run, ties to the earliest. Either way the receipt names the best epoch of the run. With `--epoch N
--stop` training ends after epoch N: the learning-rate schedule is still that of the full run, so epoch N's weights
are those the full run would have reached (later epochs cannot change them), at N/epochs of its cost.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import shutil
import time
from pathlib import Path

from .. import config, data, provenance
from ..layout import Work
from . import augment, tiles, yolo

METRIC = 'metrics/mAP50-95(B)'


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 22), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload) -> None:
    partial = path.with_name(path.name + '.partial')
    partial.write_text(json.dumps(payload, indent=1, sort_keys=True, allow_nan=False))
    os.replace(partial, path)


def overrides(d: dict, weights: Path, run: Path, *, epochs: int, batch: float, val: bool, device: str) -> dict:
    """Ultralytics' training arguments of the recipe; everything not named here is Ultralytics' default."""
    zero = dict.fromkeys(('degrees', 'translate', 'scale', 'shear', 'perspective', 'fliplr', 'flipud', 'hsv_h',
                          'hsv_s', 'hsv_v', 'mosaic', 'mixup', 'cutmix', 'copy_paste', 'multi_scale'), 0.)
    return {'model': str(weights), 'epochs': epochs, 'imgsz': d['page']['tile'], 'batch': batch, 'device': device,
            'workers': 0, 'cache': False, 'project': str(run.parent), 'name': run.name, 'exist_ok': True,
            'resume': False, 'save_period': 0, 'patience': 0, 'val': val, 'seed': d['train']['seed'],
            'deterministic': True, 'close_mosaic': 0, 'amp': True, 'plots': False,
            'max_det': d['train']['max_det'], **zero}


def read_curve(path: Path) -> list[dict[str, float]]:
    """Ultralytics' per-epoch results (epoch, time, losses, held-out metrics)."""
    with path.open(newline='') as stream:
        return [{key.strip(): float(value) for key, value in row.items()} for row in csv.DictReader(stream)]


def best_epoch(curve: list[dict[str, float]]) -> int:
    """The epoch with the highest held-out mAP50-95; ties choose the earliest."""
    if not curve or any(not math.isfinite(row[METRIC]) for row in curve):
        raise ValueError('the held-out mAP50-95 is missing or not finite in some epoch')
    return int(max(curve, key=lambda row: (row[METRIC], -row['epoch']))['epoch'])


def select_epoch(curve: list[dict[str, float]], epochs: int, choice: int | str, ran: int | None = None) -> int:
    """Epoch `choice` (1..epochs) or, for 'best', the best epoch of the complete curve; `ran` is the number of epochs
    trained (default `epochs`; fewer when training stopped after `choice`)."""
    ran = epochs if ran is None else ran
    if [int(row['epoch']) for row in curve] != list(range(1, ran + 1)):
        raise ValueError(f'expected the results of epochs 1..{ran}')
    if choice == 'best':
        if ran != epochs:
            raise ValueError('the best epoch needs the complete run')
        return best_epoch(curve)
    if not isinstance(choice, int) or not 1 <= choice <= ran:
        raise ValueError(f'epoch must be best or in 1..{ran}, not {choice!r}')
    return choice


def _ensure_epoch_weights(run: Path) -> None:
    """The weights of the epoch `last.pt` holds (written a moment later than it, so an interruption can lose them)."""
    import torch
    from ultralytics.utils.torch_utils import strip_optimizer

    last = run / 'weights' / 'last.pt'
    epoch = int(torch.load(last, map_location='cpu', weights_only=False)['epoch']) + 1
    target = run / 'epochs' / f'epoch_{epoch:04d}.pt'
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        strip_optimizer(last, str(target) + '.partial', updates={'training_epoch': epoch})
        os.replace(str(target) + '.partial', target)


def train_fold(work: Work, cfg: config.Config, fold_map: dict[int, tuple[str, ...]], fold: int, *,
               epochs: int | None = None, imgsz: int | None = None, batch: float | None = None,
               choice: int | str | None = None, stop: bool = False, val: bool = True, jobs: int = 1,
               keep_epochs: bool = False, force: bool = False, device: str | None = None) -> dict:
    """Train chain `fold`'s detector (see the module docstring); returns the receipt."""
    d = copy.deepcopy(cfg['detector'])
    if imgsz:
        d['page']['tile'] = imgsz
    epochs = epochs or d['train']['epochs']
    batch = d['train']['batch'] if batch is None else batch
    choice = d['train']['use_epoch'] if choice is None else choice
    if choice == 'best' and not val:
        raise ValueError("--epoch best needs the held-out metrics of every epoch (no --no-val)")
    if choice != 'best' and not 1 <= choice <= epochs:
        raise ValueError(f'--epoch {choice} is outside the {epochs} epochs of the run')
    if stop and choice == 'best':
        raise ValueError('--stop needs a fixed --epoch')
    ran = choice if stop else epochs
    out = work.models('detector', fold)
    if force and out.exists():
        shutil.rmtree(out)
    receipt_path = out / 'receipt.json'
    if receipt_path.exists():
        return json.loads(receipt_path.read_text())

    yolo.require(d['ultralytics'])
    import torch
    device = device or ('0' if torch.cuda.is_available() else 'cpu')
    pretrained = work.pretrained(d['base_weights'])
    if sha256(pretrained) != d['base_weights_sha256']:
        raise ValueError(f'{pretrained} is not the pretrained weights the recipe names (sha256 differs)')

    started = time.perf_counter()
    tiles.render_all(work, data.all_bases(fold_map), d, jobs)
    root = tiles.tiles_dir(work, d)
    train_names, held_names = tiles.split_tiles(root, fold_map, fold)
    pixels, targets = tiles.load_tiles(root, train_names + held_names, d['page']['tile'])
    prepare_seconds = time.perf_counter() - started

    from .trainer import TileDataset, TileTrainer  # imports Ultralytics
    rows = {name: index for index, name in enumerate(train_names + held_names)}
    datasets = {'train': TileDataset(train_names, pixels, rows, targets),
                'val': TileDataset(held_names, pixels, rows, targets)}
    run = out / 'run'
    results, attempts = run / 'results.csv', run / 'attempts.json'
    curve = read_curve(results) if results.exists() else []
    if len(curve) < ran:
        last = run / 'weights' / 'last.pt'
        resumed = last.exists()
        if resumed:
            _ensure_epoch_weights(run)
            args = {'model': str(last), 'resume': str(last), 'device': device, 'save_dir': str(run), 'val': val,
                    'workers': 0, 'plots': False}
        else:
            run.mkdir(parents=True, exist_ok=True)
            args = overrides(d, pretrained, run, epochs=epochs, batch=batch, val=val, device=device)
        trainer = TileTrainer(datasets, run / 'epochs', d['train']['max_det'], args, choice if stop else None)
        attempt = time.perf_counter()
        try:
            trainer.train()
        finally:
            history = json.loads(attempts.read_text()) if attempts.exists() else []
            write_json(attempts, history + [{'seconds': round(time.perf_counter() - attempt, 1),
                                             'resumed': resumed}])
        curve = read_curve(results)
    chosen = select_epoch(curve, epochs, choice, ran)
    selected = run / 'epochs' / f'epoch_{chosen:04d}.pt'
    partial = out / 'weights.pt.partial'
    shutil.copyfile(selected, partial)
    os.replace(partial, out / 'weights.pt')

    import yaml
    held_best = best_epoch(curve) if val else None
    receipt = {
        'fold': fold, 'epochs': epochs, 'stopped_after': choice if stop else None,
        'tiles': {'train': len(train_names), 'held': len(held_names),
                  'squeezes_train': sum(len(bases) for k, bases in fold_map.items() if k != fold),
                  'squeezes_held': len(fold_map[fold])},
        'recipe': {'ultralytics': yaml.safe_load((run / 'args.yaml').read_text()),
                   'ultralytics_version': d['ultralytics'], 'torch': torch.__version__,
                   'base_weights': {'file': d['base_weights'], 'sha256': d['base_weights_sha256']},
                   'page': d['page'], 'augmentation': {'version': augment.VERSION, 'policy': augment.POLICY},
                   'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
        'selection': {'rule': f'epoch {choice}' if choice != 'best' else 'highest held-out mAP50-95, ties earliest',
                      'epoch': chosen, 'value': curve[chosen - 1][METRIC] if val else None,
                      'best_epoch_of_run': held_best, 'best_value_of_run': curve[held_best - 1][METRIC]
                      if held_best else None},
        'curve': [{key: value if math.isfinite(value) else None for key, value in row.items()} for row in curve],
        'weights': 'weights.pt', 'weights_sha256': sha256(out / 'weights.pt'),
        'seconds': {'prepare_tiles': round(prepare_seconds, 1),
                    'training': round(sum(a['seconds'] for a in (json.loads(attempts.read_text())
                                                                 if attempts.exists() else [])), 1)},
        **provenance.stamp(cfg)}
    write_json(receipt_path, receipt)
    shutil.rmtree(run / 'weights', ignore_errors=True)  # Ultralytics' last.pt and best.pt are not needed after training
    if not keep_epochs:
        shutil.rmtree(run / 'epochs', ignore_errors=True)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--config', default='configs/pagenet_only_style_neural.json')
    parser.add_argument('--folds-file', help="default: the configuration's data.folds; the reported detectors are "
                                              "trained on data/folds_train181.json")
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--epochs', type=int, help="default: the configuration's")
    parser.add_argument('--imgsz', type=int, help='tile side and network size (default: the configuration\'s 1280)')
    parser.add_argument('--batch', type=float, help="Ultralytics batch: a count, or a fraction (<1) of the GPU memory")
    parser.add_argument('--epoch', help="N, or best (default: the configuration's use_epoch)")
    parser.add_argument('--stop', action='store_true',
                        help='end training after epoch N (the schedule stays the full run\'s)')
    parser.add_argument('--no-val', action='store_true',
                        help='skip the per-epoch validation (the last epoch is still validated)')
    parser.add_argument('--jobs', type=int, default=1, help='squeezes rendered at a time')
    parser.add_argument('--keep-epochs', action='store_true', help="keep every epoch's weights in run/epochs")
    parser.add_argument('--force', action='store_true', help='discard models/detector/f{K} and train again')
    parser.add_argument('--device', help="Ultralytics device (default: 0 if CUDA is available, else cpu)")
    args = parser.parse_args()
    cfg, work, started = config.load(args.config), Work.from_env(), time.perf_counter()
    folds = data.folds(args.folds_file or cfg['data']['folds'])
    if args.fold not in folds:
        raise SystemExit(f'fold {args.fold} is not in the folds file')
    choice = args.epoch if args.epoch in (None, 'best') else int(args.epoch)
    batch = None if args.batch is None else (int(args.batch) if args.batch >= 1 else args.batch)
    receipt = train_fold(work, cfg, folds, args.fold, epochs=args.epochs, imgsz=args.imgsz, batch=batch, choice=choice,
                         stop=args.stop, val=not args.no_val, jobs=args.jobs, keep_epochs=args.keep_epochs,
                         force=args.force, device=args.device)
    weights = work.models('detector', args.fold) / 'weights.pt'
    print(json.dumps(provenance.summary(
        cfg, time.perf_counter() - started, fold=args.fold, weights=str(weights), selection=receipt['selection'],
        tiles=receipt['tiles'], batch=receipt['recipe']['ultralytics']['batch'], stage_seconds=receipt['seconds'])))


if __name__ == '__main__':
    main()
