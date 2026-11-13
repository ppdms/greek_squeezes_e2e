"""Train one chain's page network on the other folds' detections, then read its own fold.

Chain `fold`'s network sees the squeezes outside fold `fold` through the detections their own chains produced out of
fold, so it learns from detections with the errors detections have at test time, and never sees fold `fold`.
The network on every fold (`fold` None) learns from the squeezes of every fold, each through its own chain's
out-of-fold detections; it has no fold to read out of fold, and reads only new pages (`infer.read_chain`).

A training example is a window of a page: the `window` tokens nearest a random seed token (by du^2 + (v_weight dv)^2),
whose `core` nearest carry the loss; the rest are there to be attended to. Each epoch draws about one window per
`core` tokens of every page. Every window token's crop goes through the encoder with gradients, so the whole network
is trained end to end on one loss: cross-entropy of the token classes on the core tokens, a share `mask_prob` of which
see a learned mask vector instead of their crop encoding and must be classified from their neighbours, plus
`reader_weight` times the cross-entropy of the encoder's own reader head against the reader classes. AdamW with
separate encoder and network rates, linear warmup then cosine decay, bf16 autocast, gradient clipping and brightness
and affine augmentation of the crops; fixed length, no checkpoint selection. All randomness derives from
`seed_base + fold` (the network on every fold: `seed_base` plus the number of folds).
`augment` holds the brightness and affine augmentation and may hold more: `relief` (each window's crops as another
squeeze might show the same relief: other lights, a faded gradient component, depth, blur, noise), `shift` and
`erase` (`dino.model.prepare`).

An optimizer step takes `windows_per_step` windows. Their gradients are accumulated one window at a time, with each
loss normalised by the step's token counts, which is the same gradient as one batch of all of them, but needs only one
window's activations in memory. The encoder is the pretrained DINOv3 `pretrained` (pinned by `pretrained_sha256`).
Each encoder layer's residual branches are dropped with probability `drop_path` per crop while training (stochastic
depth, the usual regulariser of ViT fine-tuning).
"""
from __future__ import annotations

import math
import shutil
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from .. import artifacts
from ..dino import pretrained
from ..dino.model import normalisation, prepare
from ..layout import Work
from . import model as net
from . import pages


class Corpus:
    """Tokens of many pages: their crops in one uint8 array, node features, geometry and (with targets) targets."""

    def __init__(self, work: Work, fold_map: dict[int, tuple[str, ...]], cfg: dict, alphabet: str, targets: bool):
        self.pages: list[pages.Page] = []
        dets = {}
        for fold, bases in sorted(fold_map.items()):
            chain = work.chain(fold, 'oof', bases)
            for base in bases:
                dets[base] = (chain, artifacts.det_page(chain, base))
        sizes = [len(d['boxes']) for _, d in dets.values()]
        self.offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
        crop = cfg['crop_size']
        self.crops = np.empty((int(self.offsets[-1]), 3, crop, crop), np.uint8)
        for i, (base, (chain, det)) in enumerate(dets.items()):
            page = pages.build(det, base)
            if targets:
                page = pages.with_targets(page, det, work, alphabet, cfg)
            with np.load(chain.dino_crops(base, cfg['input'])) as z:
                if not np.array_equal(z['det_ids'], page.det_ids):
                    raise ValueError(f'{base}: crop cache and detector record cover different detections')
                self.crops[self.offsets[i]:self.offsets[i + 1]] = z['crops']
            self.pages.append(page)

    def node_stats(self) -> dict:
        x = np.concatenate([p.node for p in self.pages]).astype(np.float64)
        return {'names': list(pages.NODE), 'mean': x.mean(0).tolist(), 'std': (x.std(0) + 1e-6).tolist()}


def standardise(node: np.ndarray, stats: dict) -> np.ndarray:
    return ((node - np.asarray(stats['mean'])) / np.asarray(stats['std'])).astype(np.float32)


def window_batch(corpus: Corpus, window: tuple[int, np.ndarray, int], cfg: dict, stats: dict, pin: bool) -> dict:
    """Tensors of one window (page index, its tokens nearest the seed first, core count); `pin`: page-locked."""
    pi, tokens, core = window
    page = corpus.pages[pi]
    u, v, w, h = page.u[tokens], page.v[tokens], page.w[tokens], page.h[tokens]
    idx, mask = pages.neighbours(u, v, cfg['neighbours'], cfg['v_weight'])
    is_core = np.zeros(len(tokens), bool)
    is_core[:core] = True
    arrays = {'pixels': corpus.crops[corpus.offsets[pi] + tokens], 'node': standardise(page.node[tokens], stats),
              'idx': idx, 'mask': mask, 'pair': pages.pair_features(u, v, w, h, pages.logit(page.p[tokens]), idx),
              'target': page.target[tokens], 'reader_target': page.reader_target[tokens], 'core': is_core}
    out = {key: torch.from_numpy(value) for key, value in arrays.items()}
    return {key: value.pin_memory() for key, value in out.items()} if pin else out


def epoch_windows(corpus: Corpus, trees: list[cKDTree], cfg: dict, rng: np.random.Generator
                  ) -> list[tuple[int, np.ndarray, int]]:
    """About one window per `core` tokens of every page, shuffled."""
    windows = []
    for pi, page in enumerate(corpus.pages):
        n = len(page)
        size = min(cfg['window'], n)
        core = min(cfg['core'], size)
        for seed in rng.choice(n, max(1, round(n / core)), replace=n < max(1, round(n / core))):
            _, tokens = trees[pi].query([page.u[seed], cfg['v_weight'] * page.v[seed]], k=size)
            windows.append((pi, np.atleast_1d(np.asarray(tokens, np.int64)), core))
    order = rng.permutation(len(windows))
    return [windows[i] for i in order]


def fit(corpus: Corpus, cfg: dict, arch: dict, stats: dict, seed: int, pretrained_dir: Path,
        device: torch.device, log=print) -> tuple[net.PageNet, list[dict]]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    model = net.from_pretrained(pretrained_dir, arch, cfg['drop_path']).to(device)
    normalise = normalisation(pretrained_dir, device)
    backbone = set(id(p) for p in model.backbone.parameters())
    optimizer = torch.optim.AdamW([
        {'params': list(model.backbone.parameters()), 'lr': cfg['lr_backbone']},
        {'params': [p for p in model.parameters() if id(p) not in backbone], 'lr': cfg['lr_head']},
    ], weight_decay=cfg['weight_decay'])
    trees = [cKDTree(np.stack([p.u, cfg['v_weight'] * p.v], 1)) for p in corpus.pages]
    per_step = cfg['windows_per_step']
    plans = [epoch_windows(corpus, trees, cfg, rng) for _ in range(cfg['epochs'])]
    steps = sum(math.ceil(len(plan) / per_step) for plan in plans)
    warmup = max(1, int(steps * cfg['warmup']))

    def rate(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, rate)
    autocast = torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda')
    pin = device.type == 'cuda'
    history = []
    with ThreadPoolExecutor(3) as pool:
        for epoch, plan in enumerate(plans, 1):
            started = time.perf_counter()
            model.train()
            ahead = 2 * per_step
            pending = deque(pool.submit(window_batch, corpus, w, cfg, stats, pin) for w in plan[:ahead])
            sums = torch.zeros(5, dtype=torch.float64, device=device)
            for first in range(0, len(plan), per_step):
                step_windows = plan[first:first + per_step]
                batches = []
                for i in range(len(step_windows)):
                    batches.append(pending.popleft().result())
                    if first + i + ahead < len(plan):
                        pending.append(pool.submit(window_batch, corpus, plan[first + i + ahead], cfg, stats, pin))
                n_core = sum(int(b['core'].sum()) for b in batches)
                n_valid = sum(int((b['core'] & (b['reader_target'] >= 0)).sum()) for b in batches)
                optimizer.zero_grad(set_to_none=True)
                for batch in batches:
                    batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
                    core = batch['core']
                    masked = core & (torch.rand(len(core), device=device) < cfg['mask_prob'])
                    with autocast:
                        visual = model.encode(prepare(batch['pixels'], device, normalise, cfg['augment']))
                        logits = model.context(visual, batch['node'], batch['idx'], batch['mask'], batch['pair'],
                                               masked)
                        reader_logits = model.reader(visual)
                    target, reader_target = batch['target'], batch['reader_target']
                    loss_net = torch.nn.functional.cross_entropy(logits.float()[core], target[core], reduction='sum')
                    valid = core & (reader_target >= 0)
                    loss_reader = torch.nn.functional.cross_entropy(reader_logits.float()[valid], reader_target[valid],
                                                                    reduction='sum')
                    (loss_net / n_core + cfg['reader_weight'] * loss_reader / max(n_valid, 1)).backward()
                    with torch.no_grad():
                        right = (logits.argmax(1) == target)[core]
                        sums += torch.stack([x.double() for x in (loss_net.detach(), loss_reader.detach(),
                                                                      right.sum(), core.sum(), valid.sum())])
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
            sums = sums.cpu().numpy()
            row = {'epoch': epoch, 'loss': sums[0] / sums[3], 'reader_loss': sums[1] / max(sums[4], 1),
                   'accuracy': sums[2] / sums[3], 'windows': len(plan), 'seconds': round(time.perf_counter() - started, 1)}
            history.append(row)
            log(f"epoch {epoch}: loss {row['loss']:.4f} reader {row['reader_loss']:.4f} "
                f"accuracy {row['accuracy']:.4f} ({row['windows']} windows, {row['seconds']} s)")
    return model.eval(), history


def train_fold(work: Work, fold_map: dict[int, tuple[str, ...]], fold: int | None, cfg: dict, alphabet: str,
               device: torch.device, log=print) -> dict:
    """Train chain `fold`'s network, or with `fold` None the network on every fold, and save
    `models/pagenet_{version}/f{fold}/` (`all/`; see `model.py`); returns the training summary."""
    started = time.perf_counter()
    pretrained_dir = work.pretrained(cfg['pretrained'].split('/')[-1])
    pretrained.check(pretrained_dir, cfg['pretrained_sha256'])
    train_folds = {k: bases for k, bases in fold_map.items() if k != fold}
    corpus = Corpus(work, train_folds, cfg, alphabet, targets=True)
    stats = corpus.node_stats()
    arch = {'node_dim': len(pages.NODE), 'pair_dim': len(pages.PAIR), 'classes': len(alphabet) + 1,
            **{key: cfg[key] for key in ('d', 'layers', 'heads', 'dropout')}}
    prepared = time.perf_counter() - started
    name = 'every fold' if fold is None else f'fold {fold}'
    log(f'{name}: {len(corpus.pages)} training pages, {len(corpus.crops)} tokens, '
        f'{int(sum((p.target < len(alphabet)).sum() for p in corpus.pages))} letters ({prepared:.0f} s)')
    seed = cfg['seed_base'] + (len(fold_map) if fold is None else fold)
    model, history = fit(corpus, cfg, arch, stats, seed, pretrained_dir, device, log)
    directory = work.models(f"pagenet_{cfg['version']}", fold)
    directory.mkdir(parents=True, exist_ok=True)
    torch.save({'model': {k: v.detach().cpu() for k, v in model.state_dict().items()}, 'arch': arch,
                'node_stats': stats, 'fold': fold, 'alphabet': list(alphabet) + ['none'], 'recipe': cfg,
                'pretrained': cfg['pretrained'], 'pretrained_sha256': cfg['pretrained_sha256']},
               directory / 'pagenet.pt')
    for file in ('config.json', 'preprocessor_config.json', 'LICENSE.md'):
        shutil.copyfile(pretrained_dir / file, directory / file)
    return {'fold': fold, 'version': cfg['version'], 'pages': len(corpus.pages), 'tokens': len(corpus.crops),
            'history': history, 'stage_seconds': {'prepare': round(prepared, 1),
                                                  'total': round(time.perf_counter() - started, 1)},
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
