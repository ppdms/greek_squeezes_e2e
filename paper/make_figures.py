"""Figures of paper/report.tex, drawn from the PageNet-only work tree's real outputs.

    PYTHONPATH=paper:. TROGS26_WORK=<work tree> PAGENET_ONLY_WORK=<PageNet-only work tree> \\
        FIGURE_ASSET_DIR=<figure assets for IG_II2_493_Copy1> \\
        python paper/make_figures.py [FIGURE ...]

Writes paper/figs/<figure>.{pdf,jpg} and paper/figs/<figure>.tex.
The figure assets' decode.json and lm.{npz,json} are those of the final system: paper/decode_assets.py writes them.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from itertools import combinations
from pathlib import Path

PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent
for p in (str(PAPER), str(ROOT), str(ROOT / '.venv' / 'lib' / 'python3.11' / 'site-packages')):
    if p not in sys.path:
        sys.path.append(p)

import cv2  # noqa: E402
import fig_decoder  # noqa: E402
import fig_pagenet  # noqa: E402
import fig_relief  # noqa: E402
import fig_results  # noqa: E402
import fig_tokens  # noqa: E402
import figstyle  # noqa: E402
import make_results  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.patches import Polygon, Rectangle  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa: E402
from rapidfuzz.distance import Levenshtein  # noqa: E402

from trogs26 import config, data, score  # noqa: E402
from trogs26 import relief as relief_lib  # noqa: E402
from trogs26 import rover as rover_system  # noqa: E402
from trogs26.decode import decoder as dec_mod  # noqa: E402
from trogs26.decode.search import Combo, plain, select  # noqa: E402
from trogs26.decode.trace import row_path  # noqa: E402
from trogs26.layout import Work  # noqa: E402
from trogs26.pagenet.infer import posteriors  # noqa: E402
from trogs26.registration import Registration  # noqa: E402
from trogs26.score import transcript_name  # noqa: E402

FIGS_DIR = PAPER / 'figs'
PAGENET_ONLY_WORK = Path(os.environ.get('PAGENET_ONLY_WORK', 'pagenet_only_work'))
ASSET_DIR = Path(os.environ.get('FIGURE_ASSET_DIR', 'figure_assets/IG_II2_493_Copy1'))

_POST_CACHE: dict[int, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
_CHANGED_BASES: set[str] | None = None


def changed_bases(work: Work) -> set[str]:
    """The squeezes whose registration map changed under image registration; they are detected again."""
    global _CHANGED_BASES
    if _CHANGED_BASES is None:
        # image-registered records of every public squeeze: probe/relief_images_style/images; the detected-again
        # records of the 18 squeezes whose map changed: gate/arms_images_style/new, listed in squeezes.json
        gate = json.loads((work.root / 'gate' / 'arms_images_style' / 'squeezes.json').read_text())
        _CHANGED_BASES = {s['base'] for s in gate['squeezes']}
    return _CHANGED_BASES


def system_record(work: Work, cfg: dict, base: str) -> dict:
    """The detector record the system reads out of fold: the detected-again record of a squeeze whose map changed,
    the image-registered record of every other public squeeze."""
    fold = figstyle.fold_of(cfg, base)
    sub = ('gate/arms_images_style/new' if base in changed_bases(work)
           else 'probe/relief_images_style/images')
    src = Work(work.root / sub)
    return json.loads(src.chain(fold, 'oof', (base,)).det(base).read_text())


def system_page_posteriors(work: Work, cfg: dict, fold: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Combined posteriors of the style_mild 5- and 15-epoch page networks under image registration; the squeezes
    whose map changed are read from the work tree that holds their detected-again records."""
    if fold not in _POST_CACHE:
        bases = figstyle.folds(cfg)[fold]
        ch_main = work.chain(fold, 'oof', bases)
        out = dict(posteriors(ch_main, ['style_mild_5_epochs_images', 'style_mild_15_epochs_images'],
                              cfg['alphabet']))
        g_bases = tuple(b for b in bases if b in changed_bases(work))
        if g_bases:
            gate_w = Work(work.root / 'gate' / 'arms_images_style' / 'new')
            ch_gate = gate_w.chain(fold, 'oof', g_bases)
            out.update(posteriors(ch_gate, ['style_mild_5_epochs', 'style_mild_15_epochs'], cfg['alphabet']))
        _POST_CACHE[fold] = out
    return _POST_CACHE[fold]


def configure_hooks() -> None:
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    figstyle.FIGS = FIGS_DIR
    figstyle.record = system_record
    figstyle.page_posteriors = system_page_posteriors


def load_cfg() -> dict:
    cfg = config.load(str(ROOT / 'configs' / 'pagenet_only_style_neural_oof.json'))
    cfg['name'] = make_results.final_run(cfg)
    return cfg


# ---------------------------------------------------------------------------------------- registration figure
def registration(work: Work, cfg: dict) -> list[Path]:
    """figs/registration.{jpg,tex}: mixed-derivative image registration on IAS33163 and over all 224 squeezes."""
    base = 'IAS33163'
    fold = figstyle.fold_of(cfg, base)
    old_det = json.loads(work.chain(fold, 'oof', (base,)).det(base).read_text())
    new_det = system_record(work, cfg, base)
    gray1, gray2 = figstyle.views(work, base)
    reg_old = Registration.from_dict(old_det['registration'])
    reg_new = Registration.from_dict(new_det['registration'])
    lh = float(new_det['letter_height_px'])
    scale = min(1.0, 140.0 / lh)
    one = cv2.resize(gray1, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray1
    two_old = relief_lib.second_view_on_first(gray2, reg_old, one.shape[:2], scale)
    two_new = relief_lib.second_view_on_first(gray2, reg_new, one.shape[:2], scale)

    quads = np.array([b['quad'] for b in new_det['boxes'] if b['p_letter'] >= 0.5], np.float64)
    w_quads = (quads + 0.5) * scale - 0.5
    centres = w_quads.mean(1)
    original_centres = quads.mean(1)
    difference = np.linalg.norm(reg_old.inverse().apply(original_centres)
                                - reg_new.inverse().apply(original_centres), axis=1)
    cx, cy = centres[int(np.argmax(difference))]
    wlh = lh * scale
    win_w, win_h = int(round(7.6 * wlh)), int(round(5.2 * wlh))
    x0 = int(np.clip(round(cx - win_w / 2), 0, one.shape[1] - win_w))
    y0 = int(np.clip(round(cy - win_h / 2), 0, one.shape[0] - win_h))
    x1, y1 = x0 + win_w, y0 + win_h

    fig = plt.figure(figsize=(figstyle.TEXTWIDTH, 2.05))
    pw, ph, bottom = 1.82, 1.24, 0.36
    ax_a = fig.add_axes([0.01, bottom / 2.05, pw / figstyle.TEXTWIDTH, ph / 2.05])
    ax_b = fig.add_axes([(0.01 + pw + 0.08) / figstyle.TEXTWIDTH, bottom / 2.05,
                         pw / figstyle.TEXTWIDTH, ph / 2.05])
    ax_c = fig.add_axes([(0.01 + 2 * pw + 0.54) / figstyle.TEXTWIDTH, bottom / 2.05,
                         (figstyle.TEXTWIDTH - 2 * pw - 0.85) / figstyle.TEXTWIDTH, ph / 2.05])

    inside = ((w_quads[:, :, 0].max(1) >= x0) & (w_quads[:, :, 0].min(1) <= x1)
              & (w_quads[:, :, 1].max(1) >= y0) & (w_quads[:, :, 1].min(1) <= y1))
    for ax, img, title, sub in (
        (ax_a, two_old[y0:y1, x0:x1], 'Letter-centre alignment',
         '63.3% within a quarter letter height'),
        (ax_b, two_new[y0:y1, x0:x1], 'Image-based alignment',
         '100% within a quarter letter height'),
    ):
        lo, hi = np.percentile(img, (1, 99))
        ax.imshow(img, cmap='gray', vmin=lo - 0.1 * (hi - lo), vmax=hi + 0.1 * (hi - lo),
                  extent=(x0, x1, y1, y0), interpolation='bilinear')
        for q in w_quads[inside]:
            ax.add_patch(Polygon(q, closed=True, fill=False, ec=figstyle.LETTER, lw=0.75))
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)
        ax.set_axis_off()
        ax.set_title(title, fontsize=8.5, pad=3)
        ax.text(0.5, -0.06, sub, transform=ax.transAxes, ha='center', va='top', fontsize=8,
                color='black')

    # Panel (c): across all 224 squeezes, median residual in letter heights
    fits = json.loads((work.root / 'probe' / 'relief' / 'fits.json').read_text())
    med_fit, med_img, med_ora = [], [], []
    in98_fit, in98_img = 0, 0
    for row in fits:
        b = row['base']
        ri = json.loads((work.root / 'probe' / 'reg_images' / f'{b}.json').read_text())
        med_fit.append(ri['fitted']['median'])
        med_img.append(ri['chosen']['median'])
        if row['residual_oracle']:
            med_ora.append(row['residual_oracle']['median'])
        in98_fit += int(ri['fitted']['within_0.25'] >= 0.98)
        in98_img += int(ri['chosen']['within_0.25'] >= 0.98)

    for vals, label, col, ls in (
        (med_fit, 'centre fit (168 $\\geq$ 98%)', '#7F7F7F', '--'),
        (med_img, f'image choice ({in98_img} $\\geq$ 98%)', figstyle.P_COLOUR, '-'),
        (med_ora, 'fit to annotations', '#1B9E77', ':'),
    ):
        xs = np.sort(vals)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        ax_c.step(xs, ys, where='post', color=col, ls=ls, lw=1.1, label=label)
    ax_c.set_xscale('log')
    ax_c.set_xlim(0.012, 1.0)
    ax_c.set_ylim(0, 1.02)
    ax_c.set_xticks([0.02, 0.05, 0.1, 0.25, 0.5], ['0.02', '0.05', '0.1', '0.25', '0.5'])
    ax_c.xaxis.set_minor_locator(NullLocator())
    ax_c.set_xlabel('median letter-pair residual (letter heights)')
    ax_c.set_ylabel('share of public squeezes')
    ax_c.legend(loc='lower right', bbox_to_anchor=(1.14, -0.01), handlelength=1.1,
                borderaxespad=0.15, fontsize=7.2, frameon=False)

    figstyle.panel_label(ax_a, 'a', x=0.0, y=1.03)
    figstyle.panel_label(ax_b, 'b', x=0.0, y=1.03)
    figstyle.panel_label(ax_c, 'c', x=-0.18, y=1.03)

    path = figstyle.save_raster(fig, 'registration', 300, 92)
    macros = {
        'RegExampleBase': base.replace('_', '\\_'),
        'RegPassOld': str(in98_fit),
        'RegPassNew': str(in98_img),
        'RegMedianOld': f'{float(np.median(med_fit)):.3f}',
        'RegMedianNew': f'{float(np.median(med_img)):.3f}',
    }
    return [path, figstyle.write_macros('registration', macros)]


# ---------------------------------------------------------------------------------------- augmentation figure
def augmentation(work: Work, cfg: dict) -> list[Path]:
    """figs/augmentation.{jpg,tex}: relief-style training renders of real crops."""
    aug_npz = np.load(ASSET_DIR / 'augment.npz')
    aug_meta = json.loads((ASSET_DIR / 'augment.json').read_text())
    source = aug_npz['source']        # (4, 3, 224, 224)
    variants = aug_npz['variants']    # (16, 4, 3, 224, 224)
    draws = [1, 4, 6, 9, 14]

    width, height = figstyle.TEXTWIDTH, 2.05
    fig = plt.figure(figsize=(width, height))
    n_rows, n_cols = 3, 5
    side = 0.53
    gap = 0.045
    left, bottom = (width - (n_cols * side + (n_cols - 1) * gap)) / 2, 0.10
    for r in range(n_rows):
        for c in range(n_cols):
            x = left + c * (side + gap)
            y = bottom + (n_rows - 1 - r) * (side + gap)
            ax = fig.add_axes([x / width, y / height, side / width, side / height])
            img = source[r] if c == 0 else variants[draws[c - 1], r]
            ax.imshow(img.transpose(1, 2, 0))
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_linewidth(0.6 if c == 0 else 0.4)
                sp.set_edgecolor(figstyle.ACCENT if c == 0 else '#777777')
            if r == 0:
                if c == 0:
                    ax.set_title('Original', fontsize=8, pad=2, fontweight='bold')
                else:
                    d = aug_meta['draws'][draws[c - 1]]
                    turn = d['turn_degrees']
                    depth = d['depth']
                    ax.set_title(f'${turn:+.0f}^\\circ$\n$d={depth:.2f}$', fontsize=8, pad=2)
    path = figstyle.save_raster(fig, 'augmentation', 300, 92)
    return [path, figstyle.write_macros('augmentation', {})]


# ---------------------------------------------------------------------------------------- lm_prior figure
def lm_prior(work: Work, cfg: dict) -> list[Path]:
    """figs/lm_prior.pdf: real next-letter distributions."""
    lm_npz = np.load(ASSET_DIR / 'lm.npz')
    lm_meta = json.loads((ASSET_DIR / 'lm.json').read_text())
    probs = lm_npz['probs']  # (N, 9, 27)
    alphabet = [str(x) for x in lm_npz['alphabet']]
    comps = [str(x) for x in lm_npz['components']]

    fig = plt.figure(figsize=(figstyle.TEXTWIDTH, 2.9))

    # Panels (a, b, c): 3 real positions on IG_II2_493_Copy1
    # Find indices in lm_meta['positions']
    pos_list = lm_meta['positions']
    idx_y = next(i for i, p in enumerate(pos_list) if p['row'] == 1 and p['index'] == 21)
    idx_t = next(i for i, p in enumerate(pos_list) if p['row'] == 19 and p['index'] == 17)
    idx_a = next(i for i, p in enumerate(pos_list) if p['row'] == 0 and p['index'] == 13)

    stage_keys = [
        ('chain', 'Public counts'),
        ('open', 'External counts'),
        ('cache', 'With page cache'),
        ('neural', 'Neural model'),
        ('backward', 'Right context'),
        ('two_sided', 'Both directions'),
    ]
    stage_idx = [comps.index(k) for k, _ in stage_keys]

    examples = [
        ('a', idx_y, '...ATHSPR[Y]TANE...', ('Y', 'O')),
        ('b', idx_t, '...WNPOLI[T]WN...', ('T', 'O')),
        ('c', idx_a, '', ('A', 'E')),
    ]
    for p_idx, (label, pos_i, title, (t_let, r_let)) in enumerate(examples):
        x0, y0 = ((0.08, 0.16), (0.40, 0.16), (0.72, 0.16))[p_idx]
        ax = fig.add_axes([x0, y0, 0.26, 0.58])
        ti, ri = alphabet.index(t_let), alphabet.index(r_let)
        yy = np.arange(len(stage_keys))[::-1]
        p_true = [float(probs[pos_i, si, ti]) for si in stage_idx]
        p_riv = [float(probs[pos_i, si, ri]) for si in stage_idx]
        ax.barh(yy + 0.15, p_true, 0.28, color=figstyle.LETTER, label=f'chosen {t_let}')
        ax.barh(yy - 0.15, p_riv, 0.28, color=figstyle.NONE, alpha=0.75, label=f'rival {r_let}')
        for yk, v in zip(yy, p_true):
            ax.text(v + 0.03, yk + 0.15, f'{v:.2f}', va='center', fontsize=8)
        ax.set_xlim(0, 1.18)
        ax.set_xticks([0, 0.5, 1])
        pos = pos_list[pos_i]
        context = f"...{pos['before'][-7:]}[{pos['letter']}]{pos['after'][:5]}..."
        ax.set_title(context, fontsize=8, family='DejaVu Sans', pad=22)
        ax.legend(loc='lower left', bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=8,
                  handlelength=0.9, columnspacing=0.6, borderaxespad=0.05)
        if p_idx == 0:
            ax.set_yticks(yy, [name for _, name in stage_keys], fontsize=8)
        else:
            ax.set_yticks(yy)
            ax.tick_params(labelleft=False)
        ax.tick_params(axis='y', length=0)
        ax.set_xlabel('probability')
        figstyle.panel_label(ax, label, x=-0.035, y=1.03)

    return [figstyle.save_vector(fig, 'lm_prior')]


# ---------------------------------------------------------------------------------------- reader inputs and outputs
def neighbourhood(work: Work, cfg: dict) -> list[Path]:
    """figs/neighbourhood.{jpg,tex}: a region of the showcase squeeze by training target, the query's 32-token
    neighbourhood by geometry, and the saved outputs for it."""
    from types import SimpleNamespace
    from trogs26.annotations import gt_squeeze
    from trogs26.geometry import rotated_iou_matrix
    from trogs26.pagenet import pages
    from trogs26.dino import crops

    base = figstyle.SHOWCASE
    det = system_record(work, cfg, base)
    page = pages.with_targets(pages.build(det, base), det, work, cfg['alphabet'], cfg['pagenet'])
    ids, lp = system_page_posteriors(work, cfg, figstyle.fold_of(cfg, base))[base]
    if not np.array_equal(ids, page.det_ids):
        raise ValueError('saved outputs and drawn boxes differ')
    keep = figstyle.keep_of(lp)
    q = int(np.flatnonzero(ids == fig_pagenet.QUERY)[0])
    gt = gt_squeeze(work.annotation(base, 1), work.annotation(base, 2))
    iou = rotated_iou_matrix(page.quads, gt.quads)
    claimed = page.target < len(cfg['alphabet'])
    duplicate = ~claimed & (iou.max(1) >= cfg['pagenet']['label_iou'])
    fusion = fig_relief._page(work, cfg, base)['fusion']
    r = SimpleNamespace(page=page, record=det, scale=fusion.scale,
                        background=crops.relief_page(fusion, 'relit'))
    quads = fig_pagenet.to_uv(r, page.quads)
    centres = quads.mean(1)
    u0, v0, u1, v1 = fig_pagenet.REGION
    inside = ((centres[:, 0] >= u0) & (centres[:, 0] <= u1)
              & (centres[:, 1] >= v0) & (centres[:, 1] <= v1))
    shown = ((quads[:, :, 0].max(1) > u0) & (quads[:, :, 0].min(1) < u1)
             & (quads[:, :, 1].max(1) > v0) & (quads[:, :, 1].min(1) < v1))
    neighbours, mask = pages.neighbours(page.u, page.v, 32, 2.0)
    near = neighbours[q][mask[q]]
    image = fig_pagenet._pale(fig_pagenet.backdrop(r, u0, v0, u1, v1, 90))
    fig = plt.figure(figsize=(figstyle.TEXTWIDTH, 3.15))
    axes = [fig.add_axes(pos) for pos in ([.015,.55,.46,.37], [.525,.55,.46,.37], [.015,.075,.46,.37])]
    for ax in axes:
        ax.imshow(image, extent=(u0,u1,v1,v0), cmap='gray', vmin=0, vmax=255)
        ax.set(xlim=(u0,u1), ylim=(v1,v0))
        ax.set_axis_off()
    kind = np.where(claimed, 0, np.where(duplicate, 1, 2))
    styles = [(figstyle.LETTER, '-', 'letter'), (fig_pagenet.DUPLICATE, '--', 'duplicate'),
              (figstyle.NONE, ':', 'no matched letter')]
    for i in np.flatnonzero(shown):
        col, ls, _ = styles[kind[i]]
        fig_pagenet._poly(axes[0], quads[i], col, ls=ls)
        fig_pagenet._poly(axes[1], quads[i], '#999999', fill=0, lw=.4)
        fig_pagenet._poly(axes[2], quads[i], plt.get_cmap(figstyle.KEEP_CMAP)(keep[i]), fill=.2)
    for ax in axes:
        fig_pagenet._poly(ax, quads[q], 'black', fill=0, lw=1.4, z=8)
    for i in near:
        fig_pagenet._poly(axes[1], quads[i], fig_pagenet.ATTENTION, fill=.12, lw=.7)
        if i != q:
            axes[1].plot([page.u[q],page.u[i]], [page.v[q],page.v[i]],
                         color=fig_pagenet.ATTENTION, lw=.45, alpha=.5)
    radius = np.hypot(page.u[near]-page.u[q], 2*(page.v[near]-page.v[q])).max()
    theta = np.linspace(0,2*np.pi,150)
    axes[1].plot(page.u[q]+radius*np.cos(theta), page.v[q]+radius/2*np.sin(theta),
                 color=fig_pagenet.ATTENTION, ls='--', lw=.7)
    for i in np.flatnonzero(inside & (keep >= .5)):
        letter = cfg['alphabet'][int(lp[i,:-1].argmax())]
        colour = 'black' if lp[i,:-1].argmax() == page.target[i] else figstyle.NONE
        axes[2].text(*centres[i], letter, ha='center', va='center', fontsize=7.5,
                     family='monospace', color=colour, path_effects=fig_pagenet.HALO)
    axes[0].set_title('Letter / duplicate / unmatched mark', fontsize=9, pad=4)
    axes[1].set_title('The 32-token neighbourhood', fontsize=9, pad=4)
    axes[2].set_title('Keep probability and first-choice letter', fontsize=9, pad=4)
    crop_ids, found = crops.fusion_crops(det, fusion, 224, cfg['dino']['crop']['scale'], ['relit'])
    if not np.array_equal(crop_ids, ids):
        raise ValueError('crops differ from saved outputs')
    axcrop = fig.add_axes([.54,.13,.17,.28])
    axcrop.imshow(found['relit'][q].transpose(1,2,0))
    axcrop.set_axis_off()
    axcrop.set_title('Outlined crop', fontsize=9)
    bars = fig.add_axes([.78,.12,.19,.28])
    order = np.argsort(-lp[q])[:4]
    bars.barh(np.arange(4)[::-1], np.exp(lp[q,order]), color=figstyle.LETTER)
    bars.set_yticks(np.arange(4)[::-1], [cfg['alphabet'][i] if i<27 else 'none' for i in order])
    bars.set(xlim=(0,1.05), xlabel='class probability')
    bars.set_title(f'keep = {keep[q]:.3f}', fontsize=9)
    for i,(ax,label) in enumerate(zip(axes+[axcrop], 'abcd')):
        figstyle.panel_label(ax,label,x=0,y=1.03)
    path = figstyle.save_raster(fig, 'neighbourhood', 300, 92)
    return [path, figstyle.write_macros('neighbourhood', {
        'NeighbourhoodQuery': cfg['alphabet'][int(page.target[q])],
        'NeighbourhoodTokens': str(inside.sum()), 'NeighbourhoodAttended': str(len(near)),
        'NeighbourhoodQueryKeep': f'{keep[q]:.3f}',
    })]


def tta(work: Work, cfg: dict) -> list[Path]:
    """figs/tta.{pdf,tex}: the saved 5-epoch and 15-epoch distributions of one token, each already averaged over its
    five crop views, and the pair combined."""
    base = figstyle.SHOWCASE
    fold = figstyle.fold_of(cfg, base)
    chain = figstyle.oof_chain(work, cfg, fold)
    alphabet = cfg['alphabet']
    ids, lp = system_page_posteriors(work, cfg, fold)[base]
    _, a = posteriors(chain, ['style_mild_5_epochs_images'], alphabet)[base]
    _, b = posteriors(chain, ['style_mild_15_epochs_images'], alphabet)[base]
    keep = figstyle.keep_of(lp)
    # Largest disagreement among tokens both models call a letter.
    ok = (keep >= .5) & (a[:,:-1].argmax(1) != b[:,:-1].argmax(1))
    candidates = np.flatnonzero(ok)
    t = int(candidates[np.argmax(np.abs(np.exp(a[candidates])-np.exp(b[candidates])).sum(1))])
    letters = np.argsort(-np.exp(a[t])-np.exp(b[t]))[:4]
    fig, axes = plt.subplots(1,3,figsize=(figstyle.TEXTWIDTH,1.55),sharex=True,sharey=True)
    fig.subplots_adjust(left=.07,right=.98,bottom=.25,top=.78,wspace=.35)
    for ax,probs,title in zip(axes,(np.exp(a[t]),np.exp(b[t]),np.exp(lp[t])),
                              ('5-epoch network','15-epoch network','combined (geometric mean)')):
        ys = np.arange(len(letters))[::-1]
        ax.barh(ys,probs[letters],color=figstyle.LETTER,height=.65)
        for y,i in zip(ys,letters):
            ax.text(probs[i]+.02,y,f'{probs[i]:.3f}',va='center',fontsize=7)
        ax.set_yticks(ys,[alphabet[i] if i<27 else 'none' for i in letters])
        ax.set(xlim=(0,1.16),xlabel='class probability',title=title)
        ax.tick_params(axis='y', labelleft=True)
    return [figstyle.save_vector(fig,'tta'),figstyle.write_macros('tta',{
        'TtaToken':str(ids[t]), 'TtaFirstFive':alphabet[int(a[t,:-1].argmax())],
        'TtaFirstFifteen':alphabet[int(b[t,:-1].argmax())],
        'TtaFirstCombined':alphabet[int(lp[t,:-1].argmax())],
    })]


# ---------------------------------------------------------------------------------------- lattice and rows
def choose_row(work: Work, cfg: dict, base: str) -> fig_decoder.RowDecode:
    """Choose a showcase row under the full PageNet-only decoder (gaps, cache, two-sided + neural)."""
    fold = figstyle.fold_of(cfg, base)
    decoder = dec_mod.chain_decoder(work, figstyle.oof_chain(work, cfg, fold), cfg, 'P')
    det = figstyle.record(work, cfg, base)
    layout = fig_decoder.page_rows(cfg, det, decoder.posteriors[base][1])
    op = Combo(**cfg['decode']['variants']['P']['combo'])
    combos = {'none': Combo(0.0, 2.0, 2.0, op.nn), 'op': op}
    inputs = decoder.page_rows(base)
    pools = dec_mod.row_pools(inputs, decoder.lm, decoder.search, decoder.cache)
    rescored = [pool if decoder.neural is None else decoder.neural.rescore(pool, model, [op.nn])[0]
                for pool, model in pools]
    texts = [[plain(t) for t in select(pool, list(combos.values()))] for pool in rescored]
    reference = score.core_text(work, base, 1).split('\n')
    found = []
    for r, ((emit, *_), (none, chosen)) in enumerate(zip(inputs, texts)):
        near = min(reference, key=lambda row: Levenshtein.distance(chosen, row))
        errors = {'none': Levenshtein.distance(none, near), 'op': Levenshtein.distance(chosen, near)}
        if (fig_decoder.POSITIONS[0] <= len(emit) <= fig_decoder.POSITIONS[1]
                and none != chosen and errors['op'] < errors['none']):
            found.append((errors['op'] - errors['none'], errors['op'], r, near, errors))
    if not found:
        raise ValueError(f'{base}: no row meets the rule')
    _, _, r, near, errors = min(found)
    emit, keep = inputs[r][0], inputs[r][1]
    row_texts = dict(zip(combos, texts[r]))
    pool = rescored[r]
    plain_texts = [plain(t) for t in pool.texts]
    terms = {}
    for name, text in row_texts.items():
        # Pick the highest-scoring candidate in `pool` whose plain text equals `text`
        cands = [i for i, pt in enumerate(plain_texts) if pt == text]
        i = max(cands, key=lambda idx: (
            pool.visual[idx] + op.w * pool.keep[idx] + op.lam * pool.lm[idx] + op.beta * pool.length[idx]
        ))
        parts = {'letters': float(pool.visual[i]), 'keep': op.w * float(pool.keep[i]),
                 'lm': op.lam * float(pool.lm[i]), 'length': op.beta * float(pool.length[i])}
        terms[name] = {**parts, 'total': sum(parts.values())}
    return fig_decoder.RowDecode(
        base, r, len(inputs), len(found), layout.rows[r], emit, keep, row_texts,
        {name: row_path(emit, np.clip(keep, 1e-4, 1-1e-4), text, combos[name].w, decoder.search)
         for name, text in row_texts.items()},
        combos, near, errors, terms
    )


def lattice(work: Work, cfg: dict) -> list[Path]:
    orig_choose = fig_decoder.choose_row
    fig_decoder.choose_row = choose_row
    try:
        return fig_decoder.lattice(work, cfg)
    finally:
        fig_decoder.choose_row = orig_choose


def check_rows(decoder, base: str, layout) -> None:
    rows = decoder.page_rows(base)
    if len(rows) != len(layout.rows) or any(
        not np.array_equal(row[1], layout.keep[idx]) for row, idx in zip(rows, layout.rows)
    ):
        raise AssertionError(f'{base}: drawn rows differ from the decoder')


def rows(work: Work, cfg: dict) -> list[Path]:
    orig_check = fig_decoder.check_rows
    fig_decoder.check_rows = check_rows
    try:
        return fig_decoder.rows(work, cfg)
    finally:
        fig_decoder.check_rows = orig_check


# ---------------------------------------------------------------------------------------- squeezes, rover, sweep
def squeezes(work: Work, cfg: dict) -> list[Path]:
    """figs/squeezes.{pdf,tex}: (a,b) 224 public + 50 TROGS-26 squeezes."""
    sweep_data, combo, grid_totals = fig_results.read_sweep(work, cfg)
    rows = []
    for base, per_combo in sorted(sweep_data['scores'].items()):
        e1, c1, e2, c2 = per_combo[combo.key]
        rows.append(fig_results.Squeeze(base, e1 + e2, c1 + c2, c1))
    cer = np.array([s.cer * 100 for s in rows])
    chars1 = np.array([s.chars / 2 for s in rows])
    median = float(np.median([s.cer * 100 for s in rows]))
    typical = fig_results.typical_squeeze(rows)
    showcase = next(s for s in rows if s.base == figstyle.SHOWCASE)

    trogs_data = json.loads((PAPER / 'data' / 'trogs26_score.json').read_text())['per_squeeze']
    t_chars1 = np.array([r['char'] / 2 for r in trogs_data])
    t_cer = np.array([r['cer'] * 100 for r in trogs_data])
    t_median = float(np.median([r['cer'] * 100 for r in trogs_data]))

    fig = plt.figure(figsize=(figstyle.TEXTWIDTH, 2.35))
    left_grid = fig.add_gridspec(1, 2, width_ratios=[4.6, 1.0], wspace=0.05,
                                 left=0.075, right=0.985, bottom=0.19, top=0.88)
    ax = fig.add_subplot(left_grid[0, 0])
    hist = fig.add_subplot(left_grid[0, 1])

    is_merged = np.array(['Merged' in s.base for s in rows])
    ax.scatter(chars1[~is_merged], cer[~is_merged], s=10, c='#0072B2', marker='o',
               edgecolors='white', linewidths=0.3, alpha=0.85, label=f'public, not Merged ({(~is_merged).sum()})')
    ax.scatter(chars1[is_merged], cer[is_merged], s=14, c='#CC79A7', marker='D',
               edgecolors='white', linewidths=0.3, alpha=0.9, label=f'public, Merged ({is_merged.sum()})')
    ax.scatter(t_chars1, t_cer, s=13, c='#E69F00', marker='s',
               edgecolors='white', linewidths=0.3, alpha=0.9, label=f'TROGS-26 test ({len(trogs_data)})')

    ax.axhline(median, color='#0072B2', lw=0.6, ls='--')
    ax.axhline(t_median, color='#E69F00', lw=0.6, ls=':')
    for squeeze, name, text_at in ((showcase, 'showcase', (0.98, 0.14)), (typical, 'typical', (0.98, 0.88))):
        x, y = squeeze.chars / 2, 100 * squeeze.cer
        ax.scatter([x], [y], s=44, facecolors='none', edgecolors='black', linewidths=0.8, zorder=6)
        ax.annotate(f'{name}\n{squeeze.base}', (x, y), xytext=text_at, textcoords='axes fraction',
                    fontsize=6.2, ha='right', va='center', zorder=7, linespacing=1.05,
                    arrowprops={'arrowstyle': '-', 'lw': 0.5, 'shrinkA': 1, 'shrinkB': 3})
    ax.set_xscale('log')
    ax.set_yscale('symlog', linthresh=1)
    ax.set_xlim(15, 4500)
    ax.set_ylim(-0.2, 100)
    ax.xaxis.set_major_locator(FixedLocator([20, 50, 100, 200, 500, 1000, 2000]))
    ax.yaxis.set_major_locator(FixedLocator([0, 1, 2, 5, 10, 20, 50]))
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
        axis.set_minor_locator(NullLocator())
    ax.set_xlabel('Core characters (mean per view)')
    ax.set_ylabel('Core CER (%)')
    ax.legend(loc='lower left', bbox_to_anchor=(-0.02, 1.01), ncol=2, columnspacing=0.7,
              handletextpad=0.2, borderaxespad=0.1, fontsize=6.2)
    figstyle.panel_label(ax, 'a', x=-0.16, y=1.02)

    edges = np.r_[0, 0.5, np.geomspace(1, 100, 17)]
    hist.hist([cer[~is_merged], cer[is_merged], t_cer], bins=edges, stacked=True, orientation='horizontal',
              color=['#0072B2', '#CC79A7', '#E69F00'], edgecolor='white', linewidth=0.3)
    hist.axhline(median, color='#0072B2', lw=0.6, ls='--')
    hist.set_yscale('symlog', linthresh=1)
    hist.set_ylim(*ax.get_ylim())
    hist.yaxis.set_major_locator(NullLocator())
    hist.yaxis.set_minor_locator(NullLocator())
    hist.set_xlabel('squeezes')
    hist.spines['left'].set_visible(False)
    figstyle.panel_label(hist, 'b', x=0.6, y=1.02)

    paths = [figstyle.save_vector(fig, 'squeezes')]
    n_worst = round(fig_results.WORST_SHARE * len(rows))
    worst_sq = sorted(rows, key=lambda s: (-s.cer, s.base))[:n_worst]
    total_errors, total_chars = sum(s.errors for s in rows), sum(s.chars for s in rows)
    macros = {
        'SqueezesCount': str(len(rows)),
        'SqueezesMedianCer': f'{median:.2f}\\%',
        'SqueezesLowestCer': f'{min(s.cer * 100 for s in rows):.2f}\\%',
        'SqueezesHighestCer': f'{max(s.cer * 100 for s in rows):.1f}\\%',
        'SqueezesWorstCount': str(n_worst),
        'SqueezesWorstErrorShare': f'{100 * sum(s.errors for s in worst_sq) / total_errors:.1f}\\%',
        'SqueezesWorstCharShare': f'{100 * sum(s.chars for s in worst_sq) / total_chars:.1f}\\%',
        'SqueezesShowcaseCer': f'{100 * showcase.cer:.2f}\\%',
        'SqueezesTypical': fig_results.tex_name(typical.base),
        'SqueezesTypicalCer': f'{100 * typical.cer:.2f}\\%',
        'SqueezesTypicalChars': figstyle.num(typical.chars1),
        'TrogsMedianCer': f'{t_median:.2f}\\%',
        'TrogsLowestCer': f'{min(r["cer"] * 100 for r in trogs_data):.2f}\\%',
        'TrogsHighestCer': f'{max(r["cer"] * 100 for r in trogs_data):.1f}\\%',
    }
    paths.append(figstyle.write_macros('squeezes', macros))
    return paths


def rover(work: Work, cfg: dict) -> list[Path]:
    """figs/rover.tex: five-chain PageNet-only ROVER vote on TROGS-26."""
    p_work = Work(PAGENET_ONLY_WORK)
    run_name = 'pagenet_only_style_neural'
    bases = data.splits()['trogs26']
    folds = sorted(figstyle.folds(cfg))
    members = rover_system.member_texts(p_work, run_name, 'trogs26', ['P'], folds, bases)
    if len(members) != 5 or not all(len(m) == len(bases) for m in members):
        raise ValueError('expected five P-chain members for every TROGS-26 squeeze')

    ins_frac, gap_weight = 0.4, 1.0
    ballots = {}
    for base in bases:
        ballots[base] = fig_results.ballot([m[base] for m in members], ins_frac, gap_weight)
        shipped = (p_work.run(run_name, 'trogs26') / 'rover' / transcript_name(base)).read_text(encoding='utf8')
        if ballots[base].text != shipped:
            raise ValueError(f'{base}: recomputed 5-chain ROVER differs from shipped transcript')

    pp_dists = [np.mean([Levenshtein.normalized_distance(x, y) for x, y in combinations(b.texts, 2)])
                for b in ballots.values()]
    substituted = sum(s != b.pivot[p] and s != '' for b in ballots.values() for p, s in enumerate(b.symbols))
    removed = sum(s == '' for b in ballots.values() for s in b.symbols)
    inserted_chars = sum(len(s) for b in ballots.values() for s in b.accepted)
    inserted = sum(bool(s) for b in ballots.values() for s in b.accepted)
    proposed = sum(any(b.inserted[i][s] for i in range(5) if i != b.k)
                   for b in ballots.values() for s in range(len(b.pivot) + 1))
    output_chars = sum(len(b.text) for b in ballots.values())

    macros = {
        'RoverSqueezes': str(len(bases)),
        'RoverDistPP': f'{100 * np.mean(pp_dists):.1f}\\%',
        'RoverChanged': figstyle.num(substituted + inserted_chars),
        'RoverChangedShare': f'{100 * (substituted + inserted_chars) / output_chars:.1f}\\%',
        'RoverSubstituted': figstyle.num(substituted),
        'RoverRemoved': figstyle.num(removed),
        'RoverInsertions': str(inserted),
        'RoverInsertionChars': str(inserted_chars),
        'RoverInsertionSlots': str(proposed),
        'RoverOutputChars': figstyle.num(output_chars),
    }
    return [figstyle.write_macros('rover', macros)]


def sweep(work: Work, cfg: dict) -> list[Path]:
    """figs/sweep.{pdf,tex}: three slices through the operating point on the grid `decode.variants.P.grid`."""
    _, combo, grid_totals = fig_results.read_sweep(work, cfg)
    errors, grid = fig_results._grid_errors(cfg, grid_totals)
    chars = {c for _, c in grid_totals.values()}
    cer = 100 * errors / chars.pop()
    i, j, k = (grid[name].index(value) for name, value in zip(('lam', 'w', 'beta'), combo))
    labels = {'lam': '$\\lambda$ (language model)', 'w': '$w$ (keep weight)',
              'beta': '$\\beta$ (letter bonus)'}
    slices = [
        (cer[:, j, :].T, grid['lam'], grid['beta'], 'lam', 'beta', f'$w={combo.w:g}$', (i, k)),
        (cer[:, :, k].T, grid['lam'], grid['w'], 'lam', 'w', f'$\\beta={combo.beta:g}$', (i, j)),
        (cer[i], grid['beta'], grid['w'], 'beta', 'w', f'$\\lambda={combo.lam:g}$', (k, j)),
    ]
    norm = Normalize(vmin=min(m.min() for m, *_ in slices), vmax=max(m.max() for m, *_ in slices))
    cmap = plt.get_cmap('YlGnBu')
    cell_w, height, below = 0.2535, 1.25, 0.4
    left, gaps = 0.5, (0.52, 0.52)
    fig_h = below + height + 0.2
    fig = plt.figure(figsize=(figstyle.TEXTWIDTH, fig_h))
    x0 = left
    for n, (matrix, xs, ys, xname, yname, title, mark) in enumerate(slices):
        width = cell_w * len(xs)
        ax = fig.add_axes([x0 / figstyle.TEXTWIDTH, below / fig_h,
                           width / figstyle.TEXTWIDTH, height / fig_h])
        ax.imshow(matrix, cmap=cmap, norm=norm, origin='lower', aspect='auto', interpolation='nearest')
        for (r, c), value in np.ndenumerate(matrix):
            ax.text(c, r, f'{value:.1f}', ha='center', va='center', fontsize=8,
                    color='white' if norm(value) > 0.55 else 'black')
        ax.add_patch(Rectangle((mark[0] - 0.5, mark[1] - 0.5), 1, 1, fill=False, edgecolor='black', lw=1.3,
                               zorder=3))
        ax.set_xticks(range(len(xs)), [f'{v:g}' for v in xs])
        ax.set_yticks(range(len(ys)), [f'{v:g}' for v in ys])
        ax.set_xticks(np.arange(-0.5, len(xs), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(ys), 1), minor=True)
        ax.grid(which='minor', color='white', lw=0.7)
        ax.tick_params(which='both', length=0)
        ax.tick_params(which='major', length=2, pad=2)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xlabel(labels[xname], labelpad=2)
        ax.set_ylabel(labels[yname], labelpad=2)
        ax.set_title(title, fontsize=9, pad=4)
        figstyle.panel_label(ax, 'abc'[n], x=-0.17 if n == 0 else -0.2)
        x0 += width + gaps[n] if n < 2 else width
    cax = fig.add_axes([(x0 + 0.15) / figstyle.TEXTWIDTH, below / fig_h,
                        0.09 / figstyle.TEXTWIDTH, height / fig_h])
    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    colorbar.set_label('Core CER (%)', labelpad=3)
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(length=2, pad=2)
    paths = [figstyle.save_vector(fig, 'sweep')]
    worst = max(grid_totals, key=lambda key: (grid_totals[key][0], key))
    total_chars = grid_totals[combo.key][1]
    shown = max(m.max() for m, *_ in slices)
    macros = {
        'SweepShownMaxCer': f'{shown:.2f}\\%',
        'SweepWorstCer': fig_results.pct(grid_totals[worst][0], total_chars),
        'SweepWorstCombo': fig_results.combo_tex(worst),
    }
    paths.append(figstyle.write_macros('sweep', macros))
    return paths


def _overview_assets() -> dict:
    """The IG_II2_493_Copy1 figure bundle shared by the two overview pages."""
    from PIL import Image

    z = np.load(ASSET_DIR / 'tokens.npz')

    def page(name: str) -> np.ndarray:
        return np.asarray(Image.open(ASSET_DIR / f'{name}.png').convert('L'))

    view1 = page('pages_view1')
    return {
        'z': z, 'dec': json.loads((ASSET_DIR / 'decode.json').read_text()),
        'alphabet': [str(a) for a in z['alphabet']],
        'view1': view1, 'view2': page('pages_view2'),
        'view2n': page('pages_view2_naive'), 'view2r': page('pages_view2_registered'),
        'thumb_h': view1.shape[0], 'thumb_w': view1.shape[1],
        'sx': view1.shape[1] / 4762.0, 'sy': view1.shape[0] / 5540.0,
    }


class _OverviewPage:
    """Overview stages with a full-width heading above each set of images."""

    def __init__(self, n_rows: int, height: float):
        self.fig = plt.figure(figsize=(figstyle.TEXTWIDTH, height))
        self.band = (0.985 - 0.015) / n_rows

    def ytop(self, i: int) -> float:
        return 0.985 - i * self.band

    def rail(self, i: int, num: int, title: str, sub: str) -> None:
        self.fig.text(0.008, self.ytop(i) - 0.018, str(num), ha='left', va='baseline',
                      fontsize=12, fontweight='bold', color=figstyle.ACCENT)
        self.fig.text(0.050, self.ytop(i) - 0.018, title, ha='left', va='baseline',
                      fontsize=12, fontweight='bold')
        self.fig.text(0.008, self.ytop(i) - 0.033, sub, ha='left', va='top', fontsize=10,
                      color='black')

    def imgax(self, x: float, w: float, i: int, h: float = 0.15, dy: float = 0.064) -> plt.Axes:
        ax = self.fig.add_axes([x, self.ytop(i) - dy - h, w, h])
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        return ax

    def lab(self, x: float, i: int, text: str, dy: float = 0.222) -> None:
        self.fig.text(x, self.ytop(i) - dy, text, ha='center', va='top', fontsize=9, color='black')


def _draw_boxes(ax: plt.Axes, a: dict, only_kept: bool = False) -> None:
    for q, k in zip(a['z']['quads'], a['z']['keep']):
        if only_kept and k < 0.5:
            continue
        col = figstyle.LETTER if k >= 0.5 else (figstyle.MUTED if k >= 0.05 else figstyle.NONE)
        ax.add_patch(Polygon(q * [a['sx'], a['sy']], closed=True, fill=False,
                             edgecolor=col, lw=0.4, alpha=0.85))


def overview_front(work: Work, cfg: dict) -> list[Path]:
    """figs/overview_front.{pdf,tex}: overview page 1 — photographs, alignment, candidates, relighting."""
    a = _overview_assets()
    z, dec = a['z'], a['dec']
    p = _OverviewPage(4, 8.5)
    fig = p.fig

    p.rail(0, 1, 'Photograph under two lights', 'The same squeeze, with different strokes revealed by each light.')
    fax = fig.add_axes([0, 0, 1, 1], zorder=30)
    fax.axis('off'); fax.set_xlim(0, 1); fax.set_ylim(0, 1)
    inv = fig.transFigure.inverted()

    def _expand(ax_from: plt.Axes, corner: tuple[float, float], zx: float, zy: float) -> None:
        ax_from.apply_aspect()
        fx, fy = inv.transform(ax_from.transData.transform(corner))
        fax.plot([fx, zx], [fy, zy], color=figstyle.ACCENT, lw=0.9)

    def _drawn(ax_from: plt.Axes, data_w: float, data_h: float) -> tuple[float, float, float, float]:
        """Figure-fraction rect actually painted by imshow inside an axes (letterbox-aware)."""
        pos = ax_from.get_position()
        fw, fh = fig.get_size_inches()
        if data_w / data_h > (pos.width * fw) / (pos.height * fh):
            w = pos.width
            h = pos.width * fw / (data_w / data_h) / fh
        else:
            h = pos.height
            w = pos.height * fh * (data_w / data_h) / fw
        return pos.x0 + (pos.width - w) / 2, pos.y0 + (pos.height - h) / 2, w, h

    # photos outside, their enlargements side by side in the middle
    for j, (img, name) in enumerate(((a['view1'], 'R$_1$: lit along $y$'),
                                     (a['view2'], 'R$_2$: lit along $x$'))):
        px = 0.0 if j == 0 else 0.77
        ax = p.imgax(px, 0.23, 0)
        ax.imshow(img, cmap='gray', vmin=0, vmax=255)
        ax.set_xlim(0, a['thumb_w']); ax.set_ylim(a['thumb_h'], 0)
        ax.add_patch(Polygon([[430, 420], [760, 420], [760, 750], [430, 750]], closed=True,
                             fill=False, edgecolor=figstyle.ACCENT, lw=1.2))
        p.lab(px + 0.115, 0, name)
        zx = 0.284 if j == 0 else 0.496
        zax = p.imgax(zx, 0.22, 0)
        zax.imshow(img, cmap='gray', vmin=0, vmax=255)
        zax.set_xlim(430, 760); zax.set_ylim(750, 420)
        for s in zax.spines.values():
            s.set_visible(True); s.set_color(figstyle.ACCENT); s.set_linewidth(1.0)
        ix, iy, iw, ih = _drawn(zax, 330, 330)
        if j == 0:
            _expand(ax, (760, 420), ix, iy + ih)
            _expand(ax, (760, 750), ix, iy)
        else:
            _expand(ax, (430, 420), ix + iw, iy + ih)
            _expand(ax, (430, 750), ix + iw, iy)
        p.lab(zx + 0.11, 0, f'R$_{j + 1}$, enlarged')

    p.rail(1, 2, 'Align the paper', 'Match the same locations in both photographs, including local distortions.')
    for j, (img, name) in enumerate(((a['view1'], 'R$_1$: fixed coordinates'),
                                     (a['view2n'], 'R$_2$: global alignment'),
                                     (a['view2r'], 'R$_2$: local alignment'))):
        ax = p.imgax(0.02 + j * 0.33, 0.30, 1)
        ax.imshow(img, cmap='gray', vmin=40, vmax=255)
        _draw_boxes(ax, a, only_kept=True)
        ax.set_xlim(300, 720); ax.set_ylim(760, 340)
        p.lab(0.17 + j * 0.33, 1, name)

    p.rail(2, 3, 'Locate possible letters', 'Keep uncertain boxes so that the reader can judge them in context.')
    page_ax = p.imgax(0.07, 0.30, 2)
    page_ax.imshow(a['view1'], cmap='gray', vmin=60, vmax=255)
    _draw_boxes(page_ax, a)
    page_ax.set_xlim(0, a['thumb_w']); page_ax.set_ylim(a['thumb_h'], 0)
    page_ax.add_patch(Polygon([[300, 340], [720, 340], [720, 760], [300, 760]], closed=True,
                              fill=False, edgecolor=figstyle.ACCENT, lw=1.3))
    p.lab(0.22, 2, f'{len(z["quads"])} candidate boxes')
    zoom_ax = p.imgax(0.61, 0.30, 2)
    zoom_ax.imshow(a['view1'], cmap='gray', vmin=60, vmax=255)
    _draw_boxes(zoom_ax, a)
    zoom_ax.set_xlim(300, 720); zoom_ax.set_ylim(760, 340)
    for s in zoom_ax.spines.values():
        s.set_visible(True); s.set_color(figstyle.ACCENT); s.set_linewidth(1.0)
    p.lab(0.76, 2, 'Detail: a crop and position for each box')

    p.rail(3, 4, 'Show the strokes under fixed lights', 'Combine the two views into a consistent colour image for the reader.')
    hero = int(dec['rows'][22]['tokens'][0])
    crop = np.transpose(z['crops'][hero], (1, 2, 0))
    for j, az in enumerate(('light from 90$^\\circ$', 'light from 210$^\\circ$', 'light from 330$^\\circ$')):
        ax = p.imgax(0.02 + j * 0.245, 0.22, 3)
        ax.imshow(crop[:, :, j], cmap='gray', vmin=0, vmax=255)
        p.lab(0.13 + j * 0.245, 3, az)
    ax = p.imgax(0.755, 0.22, 3)
    ax.imshow(crop)
    p.lab(0.865, 3, 'Combined crop')

    with plt.rc_context({'savefig.bbox': None, 'savefig.pad_inches': 0}):
        paths = [figstyle.save_vector(fig, 'overview_front')]
    paths.append(figstyle.write_macros('overview_front',
                                        {'OverviewSqueeze': 'IG\\_II2\\_493\\_Copy1'}))
    return paths


def overview_back(work: Work, cfg: dict) -> list[Path]:
    """figs/overview_back.pdf: public reading/reference pairs and an actual aligned five-chain vote."""
    from collections import Counter

    a = _overview_assets()
    z, dec, alphabet = a['z'], a['dec'], a['alphabet']
    lp = z['lp']
    width, height = figstyle.TEXTWIDTH * 72, 602.0
    fig = plt.figure(figsize=(width / 72, height / 72))
    substitution, missing, extra = '#C0392B', '#8E44AD', '#D35400'
    glyph_font = 'DejaVu Sans'

    def text(x, y, value, size=10, *, colour='black', weight='normal', align='left', family=None, va='top'):
        return fig.text(
            x / width, 1 - y / height, value, ha=align, va=va, fontsize=size,
            color=colour, fontweight=weight, fontfamily=family,
        )

    def heading(number, y, title, subtitle):
        text(3, y, str(number), 14, colour=figstyle.ACCENT, weight='bold')
        text(22, y, title, 14, weight='bold')
        text(3, y + 23, subtitle, 10)

    def axes(x, y, w, h):
        ax = fig.add_axes([x / width, 1 - (y + h) / height, w / width, h / height])
        ax.set_axis_off()
        return ax

    def crop(token, x, y, side, *, outlined=False):
        ax = axes(x, y, side, side)
        ax.imshow(np.transpose(z['crops'][token], (1, 2, 0)))
        if outlined:
            ax.set_axis_on()
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(figstyle.ACCENT)
                spine.set_linewidth(1.3)
        return ax

    def cell(x, y, w, h, *, highlighted=False):
        fig.add_artist(plt.Rectangle(
            (x / width, 1 - (y + h) / height), w / width, h / height,
            transform=fig.transFigure, fill=False,
            edgecolor=figstyle.ACCENT if highlighted else '#C4C8CC',
            linewidth=1.0 if highlighted else 0.4,
        ))

    row = dec['rows'][22]
    row_tokens = [int(t) for t in dec['rows'][22]['tokens']]
    if row['path'] != list(range(len(row_tokens))) or len(row['text']) != len(row_tokens):
        raise ValueError('the overview specimen must emit one letter per shown crop')
    focus = next(i for i, t in enumerate(row_tokens) if alphabet[int(lp[t, :-1].argmax())] != row['text'][i])
    token = row_tokens[focus]
    neighbours = [
        int(n) for n, valid in zip(z['neighbours'][token], z['neighbour_mask'][token])
        if valid and n != token
    ][:4]

    heading(5, 6, 'Read in context', 'Nearby crops and page geometry help identify the outlined mark.')
    for n, x in zip(neighbours, (8, 64, 216, 272)):
        crop(n, x, 65, 50)
    crop(token, 128, 55, 76, outlined=True)
    text(166, 41, 'Target', 10, align='center', colour=figstyle.ACCENT)
    text(8, 138, 'Four of the target’s 32 neighbours are shown.', 10)
    text(352, 38, 'Probability', 10)
    probs = np.exp(lp[token])
    top = np.argsort(-probs)[:4]
    bars = fig.add_axes([354 / width, 1 - 132 / height, 88 / width, 82 / height])
    bars.barh(
        np.arange(4), probs[top], height=0.55,
        color=[figstyle.NONE if alphabet[i] == 'none' else figstyle.LETTER for i in top],
    )
    bars.set_yticks(np.arange(4), [alphabet[i] for i in top], fontsize=11, fontfamily=glyph_font)
    for j, i in enumerate(top):
        bars.text(float(probs[i]) + 0.04, j, f'{probs[i]:.2f}', va='center', fontsize=10)
    bars.set_xlim(0, 1.05)
    bars.set_ylim(3.5, -0.5)
    bars.set_xticks([])
    bars.tick_params(axis='y', length=0, pad=6)
    for spine in bars.spines.values():
        spine.set_visible(False)

    heading(6, 160, 'Choose a reading', 'Public row 23: compare the system’s letters with the reference.')
    ref_line = dec['reference']['1'].split('\n')[22]
    reference, insertions = rover_system._align(row['text'], ref_line)
    if any(insertions):
        raise ValueError('the overview specimen reference must align to its crop positions')
    for line, y in enumerate((210, 294)):
        text(3, y + 41, 'System', 10)
        text(3, y + 56, 'Reference', 10)
        for col in range(10):
            i = line * 10 + col
            x, side = 59 + col * 39, 36
            ch, ref = row['text'][i], reference[i]
            text(
                x + side / 2, y - 12, str(i + 1), 8.5, align='center',
                colour=figstyle.ACCENT if i == focus else 'black',
            )
            crop(row_tokens[i], x, y, side)
            cell(x, y, side, 71, highlighted=i == focus)
            colour = extra if not ref else (substitution if ch != ref else 'black')
            text(x + side / 2, y + 40, ch, 12, align='center', colour=colour, family=glyph_font)
            text(
                x + side / 2, y + 55, ref or '·', 12, align='center',
                colour='black' if ref else missing, family=glyph_font,
            )
    text(3, 376, 'Blue outline: F corrected to I. Orange: an extra O remains.', 10)

    base, run = 'IAS_27146', 'pagenet_only_style_neural'
    p_work = Work(PAGENET_ONLY_WORK)
    settings = config.load(str(ROOT / 'configs' / 'pagenet_only_style_neural.json'))['ensemble']['rover']
    members = rover_system.member_texts(
        p_work, run, 'trogs26', ['P'], sorted(figstyle.folds(cfg)), [base],
    )
    ballot = fig_results.ballot([m[base] for m in members], settings['ins_frac'], settings['gap_weight'])
    shown = next(r for r in fig_results.pivot_rows(ballot) if r.index == 9)
    slots = range(shown.start, shown.end)
    if any(ins[s] for ins in ballot.inserted for s in range(shown.start, shown.end + 1)):
        raise ValueError('the shown vote must contain only pivot positions')
    shipped = (p_work.run(run, 'trogs26') / 'rover' / transcript_name(base)).read_text(encoding='utf8')
    if ballot.text != shipped:
        raise ValueError('the overview vote differs from the saved system output')

    heading(
        7, 400, 'Align the readings, then vote',
        f'Test output: {base}, row {shown.index}. Chain {ballot.k + 1} is the pivot.',
    )
    tie_col, gap_col = 5, 7  # the slots shown below: a tie kept by the pivot, a position removed by a gap vote
    pivot = ballot.aligned[ballot.k]
    tie = Counter(a[shown.start + tie_col] for a in ballot.aligned)
    gap = Counter(a[shown.start + gap_col] for a in ballot.aligned)
    top = max(tie.values())
    pivot_symbol = pivot[shown.start + tie_col]
    if (sum(n == top for n in tie.values()) < 2 or tie[pivot_symbol] != top
            or ballot.symbols[shown.start + tie_col] != pivot_symbol):
        raise ValueError('the first shown position is not a tie kept by the pivot')
    if not pivot[shown.start + gap_col] or ballot.symbols[shown.start + gap_col] or any(
            n >= gap[''] for s, n in gap.items() if s):
        raise ValueError('the second shown position is not removed by a gap vote over a pivot letter')
    table_x, table_step = 104, 13.2
    for col, _ in enumerate(slots):
        text(
            table_x + (col + 0.5) * table_step, 440, str(col + 1), 7.5, align='center',
            colour=figstyle.ACCENT if col in (tie_col, gap_col) else 'black',
        )
    for chain in range(6):
        is_vote = chain == 5
        y = 450 + chain * 16 + (6 if is_vote else 0)
        text(3, y + 2, 'Vote' if is_vote else f'Chain {chain + 1}', 11, weight='bold' if is_vote else 'normal')
        for col, position in enumerate(slots):
            ch = ballot.symbols[position] if is_vote else ballot.aligned[chain][position]
            colour = missing if not ch else (
                substitution if not is_vote and ch != ballot.symbols[position] else 'black'
            )
            x = table_x + col * table_step
            cell(x, y, table_step, 16, highlighted=col in (tie_col, gap_col))
            text(
                x + table_step / 2, y + 8, ch or '·', 11.5,
                align='center', va='center', colour=colour, family=glyph_font,
            )
    def votes(counts: Counter) -> str:
        return ', '.join(f"{'gap' if not s else s} {n}" for s, n in sorted(counts.items(), key=lambda p: (-p[1], p[0])))
    text(3, 562, f"Position {tie_col + 1}: {votes(tie)}. The tie keeps pivot {pivot_symbol}.", 10)
    text(3, 579, f"Position {gap_col + 1}: {votes(gap)}. The gap wins; no letter is emitted.", 10)
    with plt.rc_context({'savefig.bbox': None, 'savefig.pad_inches': 0}):
        return [figstyle.save_vector(fig, 'overview_back')]


GENERATORS = {
    'teaser': fig_relief.teaser,
    'relief': fig_relief.relief,
    'registration': registration,
    'overview_front': overview_front,
    'overview_back': overview_back,
    'detector': fig_tokens.detector,
    'neighbourhood': neighbourhood,
    'augmentation': augmentation,
    'tta': tta,
    'confusion': fig_tokens.confusion,
    'tokens': fig_tokens.tokens,
    'rows': rows,
    'lm_prior': lm_prior,
    'lattice': lattice,
    'rover': rover,
    'sweep': sweep,
    'squeezes': squeezes,
    'example': fig_decoder.example,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('figures', nargs='*', metavar='FIGURE')
    args = parser.parse_args()
    unknown = sorted(set(args.figures) - set(GENERATORS))
    if unknown:
        parser.error(f'unknown figures {unknown}')
    configure_hooks()
    work = Work.from_env()
    cfg = load_cfg()
    figstyle.setup()
    for name in args.figures or GENERATORS:
        for path in GENERATORS[name](work, cfg):
            print(path, flush=True)


if __name__ == '__main__':
    main()
