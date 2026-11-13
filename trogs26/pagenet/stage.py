"""The page network stage on one machine: training a chain's network and reading pages with it.

    python -m trogs26.pagenet.stage train --fold K [--config F]     trains, then reads fold K out of fold
    python -m trogs26.pagenet.stage train --all [--config F]        trains on every fold (no page is out of fold)
    python -m trogs26.pagenet.stage read --fold K --group oof [--config F]                    chain K's fold K
    python -m trogs26.pagenet.stage read --fold K --group new --splits trogs26 [--all] [--config F]

`read --group oof` reads the squeezes of fold K with chain K's network, which never saw them; `read --all` reads chain
K's new pages (its detector records and crops) with the network trained on every fold. The configuration defaults to
`configs/pagenet_only_style_neural.json`.

Inputs: the chains' calibrated detector records and relit crop caches (`chains/f{k}/{oof|new}/dino_crops_relit/`),
the annotations of the training squeezes, the pretrained DINOv3 weights. Writes `models/pagenet_{version}/f{K}/`
(`all/`) with `receipt.json`, and `chains/f{K}/{group}/pagenet_{version}.npz` (`pagenet_{version}_all.npz`).
"""
from __future__ import annotations

import argparse
import json
import os
import time

from .. import config, data, provenance
from ..layout import Work


def device():
    import torch
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(work: Work, cfg: config.Config, fold: int | None) -> dict:
    """Train chain `fold`'s network and read its fold, or with `fold` None train the network on every fold."""
    from .infer import read_chain
    from .train import train_fold
    started = time.perf_counter()
    folds = data.folds(cfg['data']['folds'])
    recipe = cfg['pagenet']
    summary = train_fold(work, folds, fold, recipe, cfg['alphabet'], device())
    if fold is not None:
        summary['read'] = read_chain(work, work.chain(fold, 'oof', folds[fold]), recipe['version'],
                                     recipe['read_views'], device())
    receipt = provenance.summary(cfg, time.perf_counter() - started, **summary)
    path = work.models(f"pagenet_{recipe['version']}", fold) / 'receipt.json'
    path.write_text(json.dumps(receipt, indent=1))
    return receipt


def read(work: Work, cfg: config.Config, fold: int, group: str, splits: list[str], every_fold: bool) -> dict:
    from .infer import read_chain
    folds = data.folds(cfg['data']['folds'])
    bases = folds[fold] if group == 'oof' else sorted(b for split in splits for b in data.splits()[split])
    recipe = cfg['pagenet']
    return read_chain(work, work.chain(fold, group, bases), recipe['version'], recipe['read_views'], device(),
                      every_fold=every_fold)


def main() -> None:
    # windows differ in size, so the caching allocator fragments; expandable segments keep a window's peak in memory
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('action', choices=('train', 'read'))
    parser.add_argument('--fold', type=int, help="train: the chain whose fold is held out; read: the chain's pages")
    parser.add_argument('--all', action='store_true', help='the network trained on every fold')
    parser.add_argument('--group', choices=('oof', 'new'), default='oof')
    parser.add_argument('--splits', default='trogs26', help='comma-separated splits of the new pages')
    parser.add_argument('--config', default='configs/pagenet_only_style_neural.json')
    args = parser.parse_args()
    if args.action == 'train' and (args.fold is None) == (not args.all):
        parser.error('train takes --fold K or --all')
    if args.action == 'read' and args.fold is None:
        parser.error('read takes --fold K (the chain whose pages are read)')
    work, cfg = Work.from_env(), config.load(args.config)
    if args.action == 'train':
        result = train(work, cfg, None if args.all else args.fold)
        print(json.dumps({k: v for k, v in result.items() if k != 'history'}))
    else:
        print(json.dumps(read(work, cfg, args.fold, args.group, args.splits.split(','), args.all)))


if __name__ == '__main__':
    main()
