"""Run a per-squeeze stage command on this machine in parallel: the chain's squeezes split over `--jobs` processes.

The command must accept `--bases a,b,...` (detect.infer does); squeezes are dealt round-robin, largest images first,
so the slow squeezes start early. Every process's output goes to logs/<name>/<i>.log under TROGS26_WORK; the exit
status is non-zero if any process fails, whose log's last lines are printed.

    python launchers/local.py --name det-oof-f0 --folds-file data/folds_train181.json --fold 0 --group oof --jobs 8 -- \
        python -m trogs26.detect.infer --config configs/pagenet_only_style_neural.json \
            --folds-file data/folds_train181.json --fold 0 --group oof
    python launchers/local.py --name det-new-f0 --fold 0 --group new --splits trogs26 --jobs 8 -- \
        python -m trogs26.detect.infer --config configs/pagenet_only_style_neural.json --fold 0 --group new \
            --splits trogs26
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

from trogs26 import config, data
from trogs26.layout import Work


def bases_of(cfg: dict, folds_file: str | None, fold: int, group: str, splits: list[str]) -> list[str]:
    if group == 'oof':
        return list(data.folds(folds_file or cfg['data']['folds'])[fold])
    return sorted(b for split in splits for b in data.splits()[split])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--name', required=True)
    parser.add_argument('--config', default='configs/pagenet_only_style_neural.json')
    parser.add_argument('--folds-file', help="default: the configuration's data.folds")
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--group', choices=('oof', 'new'), required=True)
    parser.add_argument('--splits', default='trogs26')
    parser.add_argument('--jobs', type=int, default=8)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    work = Work.from_env()
    bases = bases_of(config.load(args.config), args.folds_file, args.fold, args.group, args.splits.split(','))
    size = {b: sum(work.image(b, v).stat().st_size for v in (1, 2)) for b in bases}
    order = sorted(bases, key=lambda b: -size[b])
    shares = [order[i::args.jobs] for i in range(args.jobs) if order[i::args.jobs]]
    logs = work.root / 'logs' / args.name
    logs.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    procs = []
    for i, share in enumerate(shares):
        log = (logs / f'{i}.log').open('w')
        env = {**os.environ, 'OMP_NUM_THREADS': '1'}
        procs.append((subprocess.Popen([*command, '--bases', ','.join(share)], stdout=log, stderr=subprocess.STDOUT,
                                       env=env), share))
    bad = [i for i, (p, _) in enumerate(procs) if p.wait()]
    for i in bad:
        print(f'--- process {i} failed; last lines of {logs / f"{i}.log"}:')
        print('\n'.join((logs / f'{i}.log').read_text(errors='replace').splitlines()[-20:]))
    print(f'{args.name}: {len(bases)} squeezes over {len(shares)} processes, {time.perf_counter() - started:.0f} s; '
          f'failed processes: {len(bad)} (logs in {logs})')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
