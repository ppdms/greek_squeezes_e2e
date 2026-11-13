"""Out-of-fold sweep: decode every squeeze of the folds (`data.folds`) with its own chain's models under every
(lam, w, beta) combo of the variant's grid and score both annotation views. The result keeps the errors of every
squeeze at every combo, because changes are judged on per-squeeze wins and losses, not on totals alone.

    TROGS26_WORK=<work> python -m trogs26.decode.sweep --variant P [--run sweep] [--folds 0,1] [--chunks 4]
                                                       [--jobs 1] [--config <file>]

runs locally, `--jobs` chunks at a time; `launchers/azure.py sweep` runs it on Azure VMs.
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch

from .. import artifacts, config, data, fallbacks, provenance, score
from ..layout import Work
from .decoder import chain_decoder, decode_page
from .search import grid_combos


def grid_of(cfg: dict, variant: str) -> list:
    """The combos of the variant's grid."""
    return grid_combos(cfg['decode']['grids'][cfg['decode']['variants'][variant]['grid']])


def chunks(work: Work, cfg: dict, fold: int, count: int) -> list[list[str]]:
    """The squeezes of `fold` in at most `count` groups of similar detection count (largest squeeze first, each
    to the lightest group), so groups take about equally long."""
    bases = data.folds(cfg['data']['folds'])[fold]
    chain = work.chain(fold, 'oof', bases)
    sizes = sorted(((len(artifacts.det_page(chain, b)['boxes']), b) for b in bases), reverse=True)
    groups: list[list] = [[0, []] for _ in range(count)]
    for size, base in sizes:
        lightest = min(groups, key=lambda g: g[0])
        lightest[0] += size
        lightest[1].append(base)
    return [sorted(g[1]) for g in groups if g[1]]


def sweep_pages(work: Work, cfg: dict, variant: str, fold: int, bases: list[str]) -> dict:
    """Scores of the out-of-fold squeezes `bases` of chain `fold`: scores[base][combo key] = [err1, chars1,
    err2, chars2] (Core errors and characters of Rotation1 and Rotation2)."""
    started = time.perf_counter()
    fallbacks.reset()
    decoder = chain_decoder(work, work.chain(fold, 'oof', bases), cfg, variant)
    combos = grid_of(cfg, variant)
    scores = {}
    for base in bases:
        texts = decode_page(decoder, base, combos)
        scores[base] = {combo.key: score.squeeze_errors(work, base, texts[combo]) for combo in combos}
    return {'fold': fold, 'scores': scores, 'fallbacks': fallbacks.counts(),
            'seconds': round(time.perf_counter() - started, 1)}


def sweep_path(work: Work, run: str, variant: str):
    return work.run(run, 'oof') / f'sweep_{variant}.json'


def combine(cfg: dict, variant: str, folds: list[int], parts: list[dict], seconds: float) -> dict:
    """The sweep result from chunk results: every squeeze of `folds` exactly once, with the run summary."""
    scores = {base: s for part in parts for base, s in part['scores'].items()}
    fold_bases = data.folds(cfg['data']['folds'])
    expected = [base for k in folds for base in fold_bases[k]]
    if len(scores) != len(expected) or set(scores) != set(expected):
        raise ValueError('the chunks do not cover every squeeze of the folds exactly once')
    return provenance.summary(
        cfg, seconds, fallbacks.merge(*(p['fallbacks'] for p in parts)), variant=variant, folds=folds,
        grid=cfg['decode']['variants'][variant]['grid'], chunk_seconds=sum(p['seconds'] for p in parts),
        scores=dict(sorted(scores.items())))


def totals(scores: dict) -> dict[str, list[int]]:
    """combo key -> [errors, characters] over every squeeze and both views."""
    out: dict[str, list[int]] = {}
    for per_combo in scores.values():
        for key, (e1, c1, e2, c2) in per_combo.items():
            total = out.setdefault(key, [0, 0])
            total[0] += e1 + e2
            total[1] += c1 + c2
    return out


def frozen_combo(scores: dict) -> dict:
    """The sweep's global minimum: the combo with the fewest total errors (ties: the smaller key)."""
    table = totals(scores)
    key = min(table, key=lambda k: (table[k][0], k))
    errors, chars = table[key]
    return {'combo': key, 'errors': errors, 'chars': chars, 'cer': errors / chars}


def _single_thread() -> None:
    torch.set_num_threads(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--config', default='configs/pagenet_only_style_neural.json')
    parser.add_argument('--variant', required=True)
    parser.add_argument('--run', default='sweep')
    parser.add_argument('--folds', help='comma-separated folds (default: all)')
    parser.add_argument('--chunks', type=int, default=4, help='chunks per fold')
    parser.add_argument('--jobs', type=int, default=1, help='chunks decoded at a time (processes)')
    args = parser.parse_args()
    _single_thread()
    cfg, work, started = config.load(args.config), Work.from_env(), time.perf_counter()
    folds = [int(k) for k in args.folds.split(',')] if args.folds else sorted(data.folds(cfg['data']['folds']))
    tasks = [(fold, bases) for fold in folds for bases in chunks(work, cfg, fold, args.chunks)]
    parts = []
    with ProcessPoolExecutor(args.jobs, initializer=_single_thread) as pool:
        pending = [pool.submit(sweep_pages, work, cfg, args.variant, fold, bases) for fold, bases in tasks]
        for done in as_completed(pending):
            parts.append(done.result())
            print(f'fold {parts[-1]["fold"]}: {len(parts[-1]["scores"])} squeezes, {parts[-1]["seconds"]} s',
                  flush=True)
    result = combine(cfg, args.variant, folds, parts, time.perf_counter() - started)
    path = sweep_path(work, args.run, args.variant)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result))
    print(path, frozen_combo(result['scores']))


if __name__ == '__main__':
    main()
