"""The relit crop caches of the page network, on one machine.

The page network reads the crops of the relit relief (input 'relit', `crops.py`). Out-of-fold pages are cut from the
detections of their own chain; new pages (TROGS-26) from each chain's detections. A cache is cut again when its
chain's detector record or the crop parameters (config 'dino.crop') changed.

    python -m trogs26.dino.stage crops --input relit --group oof [--jobs 4] [--config F]
    python -m trogs26.dino.stage crops --input relit --group new --splits trogs26 [--jobs 4] [--config F]

The configuration defaults to `configs/pagenet_only_style_neural.json`; the crops read its 'dino.crop' (size, scale).
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor

from .. import config, data
from ..layout import Chain, Work


def _chains(work: Work, cfg: dict, group: str, splits: list[str]) -> dict[str, list[Chain]]:
    """base -> the chains whose detections of it are cut: its own fold's chain out of fold, every chain for new
    pages."""
    folds = data.folds(cfg['data']['folds'])
    if group == 'oof':
        return {base: [work.chain(k, 'oof', bases)] for k, bases in folds.items() for base in bases}
    bases = sorted(b for split in splits for b in data.splits()[split])
    chains = [work.chain(k, 'new', bases) for k in sorted(folds)]
    return {base: chains for base in bases}


def _crop_one(args: tuple) -> tuple[str, int]:
    from .crops import write_page_crops
    root, cfg, base, chains, inputs = args
    return base, write_page_crops(Work(root), chains, base, cfg['dino']['crop'], inputs)


def crops(work: Work, cfg: dict, inputs: list[str], group: str, splits: list[str], jobs: int) -> None:
    pages = sorted(_chains(work, cfg, group, splits).items())
    todo = [(work.root, cfg, base, chains, inputs) for base, chains in pages]
    started = time.perf_counter()
    with ProcessPoolExecutor(jobs) as pool:
        written = sum(n for _, n in pool.map(_crop_one, todo))
    print(json.dumps({'inputs': inputs, 'group': group, 'squeezes': len(todo), 'caches_written': written,
                      'seconds': round(time.perf_counter() - started, 1)}))


def main() -> None:
    from .crops import INPUTS
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('action', choices=('crops',))
    parser.add_argument('--config', default='configs/pagenet_only_style_neural.json')
    parser.add_argument('--input', required=True, help=f'comma-separated, from {INPUTS}')
    parser.add_argument('--group', choices=('oof', 'new'), default='oof')
    parser.add_argument('--splits', default='trogs26', help='comma-separated splits of the new pages')
    parser.add_argument('--jobs', type=int, default=4)
    args = parser.parse_args()
    inputs = args.input.split(',')
    if set(inputs) - set(INPUTS):
        parser.error(f'--input: {args.input!r}')
    work, cfg = Work.from_env(), config.load(args.config)
    crops(work, cfg, inputs, args.group, [s for s in args.splits.split(',') if s], args.jobs)


if __name__ == '__main__':
    main()
