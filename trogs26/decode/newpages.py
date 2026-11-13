"""Decode image-only pages (validation, internal_test, trogs26) with every chain.

Chain k decodes the pages of its 'new' group with its own models and the final language model, at the variant's
`combo`. Transcripts go to `runs/<run>/<split>/f<k>/<variant>/<squeeze>_transcript.txt`, one chain's decode of a
variant per directory.

    TROGS26_WORK=<work> python -m trogs26.decode.newpages --variant P --run <run> --splits trogs26 [--parts 1]
                                                          [--jobs 1] [--config <file>]

runs locally (each chain's pages in `--parts` parts, `--jobs` at a time); `launchers/azure.py newpages` runs them on
Azure VMs.
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

from .. import config, data, fallbacks, provenance
from ..layout import Work
from ..score import transcript_name
from .decoder import chain_decoder, decode_page
from .search import Combo


def variant_combos(cfg: dict, variant: str) -> list[Combo]:
    """The combos to decode: the variant's operating point."""
    return [Combo(**cfg['decode']['variants'][variant]['combo'])]


def decode_chain(work: Work, cfg: dict, variant: str, fold: int, bases: list[str], combos: list[Combo]) -> dict:
    """Chain `fold`'s decode of new pages `bases`: {'texts': {base: {combo key: text}}, 'fallbacks', 'seconds'}."""
    started = time.perf_counter()
    fallbacks.reset()
    decoder = chain_decoder(work, work.chain(fold, 'new', bases), cfg, variant)
    texts = {}
    for base in bases:
        decoded = decode_page(decoder, base, combos)
        texts[base] = {combo.key: decoded[combo] for combo in combos}
    return {'fold': fold, 'texts': texts, 'fallbacks': fallbacks.counts(),
            'seconds': round(time.perf_counter() - started, 1)}


def write_chain(directory: Path, texts: dict[str, dict[str, str]]) -> None:
    """Transcripts of one chain (each base's only text)."""
    directory.mkdir(parents=True, exist_ok=True)
    names = [transcript_name(base) for base in texts]
    if len(set(names)) != len(names):
        raise ValueError('two squeezes share an official transcript name')
    for base, per_combo in texts.items():
        (directory / transcript_name(base)).write_text(next(iter(per_combo.values())), encoding='utf8')


def merge(folds: list[int], parts: list[dict]) -> list[dict]:
    """One part per chain from `decode_chain`'s parts of its pages, in any order."""
    chains = {k: {'fold': k, 'texts': {}, 'fallbacks': {}, 'seconds': 0.0} for k in folds}
    for part in parts:
        chain = chains[part['fold']]
        chain['texts'].update(part['texts'])
        chain['fallbacks'] = fallbacks.merge(chain['fallbacks'], part['fallbacks'])
        chain['seconds'] = round(chain['seconds'] + part['seconds'], 1)
    return list(chains.values())


def write_run(work: Work, cfg: dict, run: str, variant: str, splits: list[str], parts: list[dict],
              seconds: float, **fields) -> None:
    """Write every chain's output per split and one summary of the run (`runs/<run>/decode_<variant>_<splits>.json`,
    with `fields`); `parts` holds one part per chain."""
    for split in splits:
        wanted = data.splits()[split]
        for part in parts:
            write_chain(work.run(run, split) / f'f{part["fold"]}' / variant,
                        {base: part['texts'][base] for base in wanted})
    summary = provenance.summary(
        cfg, seconds, fallbacks.merge(*(p['fallbacks'] for p in parts)), variant=variant,
        squeezes={split: len(data.splits()[split]) for split in splits},
        chain_seconds={p['fold']: p['seconds'] for p in parts}, **fields)
    path = work.run(run, splits[0]).parent / f'decode_{variant}_{"+".join(splits)}.json'
    path.write_text(json.dumps(summary, indent=1))


def _single_thread() -> None:
    torch.set_num_threads(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--config', default='configs/pagenet_only_style_neural.json')
    parser.add_argument('--variant', required=True)
    parser.add_argument('--splits', required=True, help='comma-separated: validation, internal_test, trogs26')
    parser.add_argument('--run', required=True)
    parser.add_argument('--parts', type=int, default=1, help="parts each chain's pages are decoded in")
    parser.add_argument('--jobs', type=int, default=1, help='parts decoded at a time (processes)')
    args = parser.parse_args()
    _single_thread()
    cfg, work, started = config.load(args.config), Work.from_env(), time.perf_counter()
    splits = args.splits.split(',')
    bases = [b for split in splits for b in data.splits()[split]]
    combos = variant_combos(cfg, args.variant)
    folds = sorted(data.folds(cfg['data']['folds']))
    parts = []
    with ProcessPoolExecutor(args.jobs, initializer=_single_thread) as pool:
        pending = [pool.submit(decode_chain, work, cfg, args.variant, k, bases[i::args.parts], combos)
                   for k in folds for i in range(args.parts) if bases[i::args.parts]]
        for done in as_completed(pending):
            parts.append(done.result())
            print(f'chain {parts[-1]["fold"]}: {len(parts[-1]["texts"])} squeezes, {parts[-1]["seconds"]} s',
                  flush=True)
    write_run(work, cfg, args.run, args.variant, splits, merge(folds, parts), time.perf_counter() - started)


if __name__ == '__main__':
    main()
