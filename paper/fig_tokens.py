"""Figures on the tokens: the detector's detections, the page network's reading of them, and its confusions.

All three are out of fold over the 224 public squeezes: each squeeze is taken with the detector record and the
page-network posteriors of the P chain that never saw it (`figstyle.record`, `figstyle.page_posteriors`; the two
networks combined as the decoder reads them). The data are read once per process and kept in memory.

detector -> figs/detector.{pdf,tex}
    (a) Calibration. Line: the isotonic map of the fused score to `p_letter`, one curve per chain (each fitted on
        the other four folds, `trogs26.detect.calibrate`). Dots: in eighteen bins of equal width in the fused score,
        the share of detections that win the one-to-one match (IoU >= `detector.calibration.match_iou`,
        `calibrate.page_labels`) to an annotated Rotation1 letter, against their mean fused score.
    (b) Distribution of `p_letter` over the detections that win that match and those that do not (counts, log scale).
    (c) Coverage. Left axis: the share of annotated Rotation1 letters (`detect.annotations.read_letters`) that some
        detection with `p_letter` at least the threshold overlaps at rotated IoU >= 0.3 (the claim IoU) and at
        >= 0.5; not one-to-one. Right axis: detections left per annotated letter at that threshold.

tokens -> figs/tokens.{pdf,tex}
    Targets are the claim rule's (`pages.with_targets`): a token is a claimed letter or none.
    (a) ROC curves for "the token claims a letter": the keep probability k of the combined networks and the
        detector's `p_letter`. Dots: the threshold 0.5 on each. Inset: low false-positive rates.
    (b) Reliability of k: the share of tokens that claim a letter in ten equal bins of k (dots, at the bin's mean k),
        and under it the number of tokens per bin.
    (c) Letter reading: among tokens that claim a letter, the share whose claimed letter is among the n most
        probable of the 27 letters (log-probabilities renormalised over the letters; the beam takes the top
        `char_topk`), for all claiming tokens and by keep probability (the decoder's thresholds 0.5 and `k_hi`).

confusion -> figs/confusion.{pdf,tex}
    Claiming tokens: the claimed letter against the most probable of the 27 letters, each row normalised by the
    claimed letter's count (log colour scale from 10^-3, below which cells share the lightest shade; white: never).
    Letters in order of how often they are claimed; on the right, how often (log scale). Outlined: the five most
    frequent confusions. The letters are drawn as the Greek capitals they transliterate (`trogs26.lm.open.PROXY`),
    the three signs as the dataset writes them.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.patches import Rectangle
from sklearn.metrics import roc_auc_score, roc_curve

import figstyle
from trogs26.detect.annotations import read_letters
from trogs26.detect.calibrate import page_labels
from trogs26.geometry import rotated_iou_matrix
from trogs26.lm.open import GREEK, PROXY
from trogs26.pagenet import pages
from trogs26.pagenet.decode import KEEPS

OBSERVED = '#E08214'     # a measured share, against a fitted or predicted one
DETECTOR = '#636363'     # the detector, against the page network (figstyle.P_COLOUR)
WORDS = ('One', 'Two', 'Three', 'Four', 'Five')
GREEK_OF = dict(zip(PROXY, GREEK))  # dataset letter -> the Greek capital it stands for

_CACHE: dict = {}


def load(work, cfg) -> dict:
    """The tokens of the 224 public squeezes, out of fold (module docstring): one array per quantity over all tokens
    in squeeze order (`score`, `p`, `matched`, `target`, `k`, `lp`, `fold`), and per annotated Rotation1 letter
    `cover['claim']` and `cover['match']`, the highest `p_letter` among the detections that overlap it at the claim
    IoU and at the match IoU (-1: none does); `cover_iou` holds the two IoUs."""
    key = (str(work.root), cfg['data']['folds'])
    if key in _CACHE:
        return _CACHE[key]
    alphabet, recipe = cfg['alphabet'], cfg['pagenet']
    match_iou = cfg['detector']['calibration']['match_iou']
    cover_iou = {'claim': recipe['label_iou'], 'match': match_iou}
    names = ('score', 'p', 'matched', 'target', 'k', 'lp', 'fold')
    out = {name: [] for name in names} | {'cover': {name: [] for name in cover_iou}}
    for fold, bases in sorted(figstyle.folds(cfg).items()):
        posteriors = figstyle.page_posteriors(work, cfg, fold)
        for base in bases:
            det = figstyle.record(work, cfg, base)
            boxes = det['boxes']
            quads = np.array([b['quad'] for b in boxes], np.float64).reshape(-1, 4, 2)
            p = np.array([b['p_letter'] for b in boxes], np.float64)
            letters = read_letters(work.annotation(base, 1))[0]
            ious = rotated_iou_matrix(quads, letters)
            out['score'].append(np.array([b['score'] for b in boxes], np.float64))
            out['p'].append(p)
            out['matched'].append(page_labels(quads, letters.astype(np.float32), match_iou) > 0)
            for name, iou in cover_iou.items():
                out['cover'][name].append(np.where(ious >= iou, p[:, None], -1.0).max(0, initial=-1.0))
            page = pages.with_targets(pages.build(det, base), det, work, alphabet, recipe)
            ids, log_probs = posteriors[base]
            if not np.array_equal(ids, page.det_ids):
                raise ValueError(f'{base}: the page network and the detector record cover different detections')
            out['target'].append(page.target)
            out['lp'].append(log_probs)
            out['k'].append(figstyle.keep_of(log_probs))
            out['fold'].append(np.full(len(page), fold))
    out = {name: np.concatenate(out[name]) for name in names} | {
        'cover': {name: np.concatenate(v) for name, v in out['cover'].items()}, 'cover_iou': cover_iou}
    _CACHE[key] = out
    return out


def pct(x: float, digits: int = 1) -> str:
    return f'{100 * x:.{digits}f}\\%'


def tex_letter(letter: str) -> str:
    """A letter or sign as LaTeX text."""
    return '\\&' if letter == '&' else letter


def detector(work, cfg) -> list:
    """figs/detector.{pdf,tex}: calibration of `p_letter`, its distribution and the letters the detections cover."""
    d = load(work, cfg)
    score, p, matched = d['score'], d['p'], d['matched']
    n_det, n_letters = len(p), len(d['cover']['claim'])
    folds = np.unique(d['fold'])
    fig, axes = plt.subplots(1, 3, figsize=(figstyle.TEXTWIDTH, 1.9))
    fig.subplots_adjust(left=0.068, right=0.935, bottom=0.21, top=0.9, wspace=0.62)
    a, b, c = axes

    # (a) the map and the observed match rate
    for i, fold in enumerate(folds):
        mine = d['fold'] == fold
        order = np.argsort(score[mine], kind='stable')
        s, q = score[mine][order], p[mine][order]
        edge = np.r_[True, q[1:] != q[:-1]] | np.r_[q[1:] != q[:-1], True]  # the map is linear between plateaus
        a.plot(s[edge], q[edge], color=figstyle.ACCENT, lw=0.7, alpha=0.8, label='map to $p_{\\mathrm{letter}}$'
               if i == 0 else None)
    bins = 18
    edges = np.linspace(0, score.max(), bins + 1)
    which = np.clip(np.searchsorted(edges, score, side='right') - 1, 0, bins - 1)
    a.plot([score[which == i].mean() for i in range(bins)], [matched[which == i].mean() for i in range(bins)],
           'o', ms=3, mfc='white', mec=OBSERVED, mew=1.0, label='share matched', zorder=3)
    a.set_xlim(0, 0.92)
    a.set_ylim(0, 1.02)
    a.set_xlabel('fused detector score')
    a.set_ylabel('probability of a match')
    a.legend(loc='upper left', handlelength=1.2, borderaxespad=0.1)

    # (b) p_letter of the detections that win a match and those that do not
    edges = np.linspace(0, 1, 21)
    width = np.diff(edges)[0]
    centres = (edges[:-1] + edges[1:]) / 2
    for shift, mask, colour, label in ((-0.25, matched, figstyle.LETTER, 'matched'),
                                       (0.25, ~matched, figstyle.NONE, 'unmatched')):
        counts, _ = np.histogram(p[mask], edges)
        b.bar(centres + shift * width, counts, width=width / 2, color=colour, lw=0, label=f'{label} ({mask.sum():,})')
    b.set_yscale('log')
    b.set_xlim(0, 1)
    b.set_ylim(1e2, 3e5)
    b.set_xlabel('$p_{\\mathrm{letter}}$')
    b.set_ylabel('detections')
    b.legend(loc='upper center', handlelength=1.0, borderaxespad=0.1, ncols=1)

    # (c) letters covered, and what covering them costs
    thresholds = np.linspace(0, 1, 201)
    styles = {'claim': ('-', figstyle.ACCENT), 'match': ('--', figstyle.LETTER)}
    for name, (line, colour) in styles.items():
        cover = d['cover'][name]
        c.plot(thresholds, [(cover >= t).mean() for t in thresholds], line, color=colour,
               label=f'IoU $\\geq$ {d["cover_iou"][name]:g}')
    c.set_xlim(0, 1)
    c.set_ylim(0, 1.02)
    c.set_xlabel('$p_{\\mathrm{letter}}$ threshold')
    c.set_ylabel('annotated letters covered')
    right = c.twinx()
    right.spines['right'].set_visible(True)
    right.plot(thresholds, [(p >= t).sum() / n_letters for t in thresholds], color=DETECTOR, lw=1.0)
    right.set_ylim(0, 1.02 * n_det / n_letters)
    right.set_ylabel('detections per letter', color=DETECTOR)
    right.tick_params(axis='y', colors=DETECTOR)
    right.spines['right'].set_color(DETECTOR)
    c.legend(loc='lower left', handlelength=1.6, borderaxespad=0.1)
    for ax, letter in zip(axes, 'abc'):
        figstyle.panel_label(ax, letter, x=-0.14 if letter == 'a' else -0.2)

    half = p >= KEEPS
    cover_half = (d['cover']['claim'] >= KEEPS).mean()
    macros = {
        'DetectorDetections': figstyle.num(n_det),
        'DetectorLetters': figstyle.num(n_letters),
        'DetectorMatched': figstyle.num(int(matched.sum())),
        'DetectorUnmatched': figstyle.num(int((~matched).sum())),
        'DetectorCoverageClaim': pct((d['cover']['claim'] >= 0).mean()),
        'DetectorCoverageMatch': pct((d['cover']['match'] >= 0).mean()),
        'DetectorPerLetter': f'{n_det / n_letters:.2f}',
        'DetectorLow': figstyle.num(int((~half).sum())),
        'DetectorLowShare': pct((~half).mean()),
        'DetectorHalfCoverage': pct(cover_half),
        'DetectorHalfPerLetter': f'{half.sum() / n_letters:.2f}',
    }
    return [figstyle.save_vector(fig, 'detector'), figstyle.write_macros('detector', macros)]


def reading_ranks(lp: np.ndarray, target: np.ndarray, letters: int) -> np.ndarray:
    """Rank (0 = most probable) of the claimed letter among the `letters` letters of every token that claims one.
    Ranking the letters' log-probabilities is the same as ranking them renormalised over the letters."""
    claim = target < letters
    order = np.argsort(-lp[claim, :letters], axis=1, kind='stable')
    return (order == target[claim][:, None]).argmax(1)


def tokens(work, cfg) -> list:
    """figs/tokens.{pdf,tex}: how well the page network separates letters from none and reads the letters."""
    d = load(work, cfg)
    letters = len(cfg['alphabet'])
    k, p, lp = d['k'], d['p'], d['lp']
    claim = d['target'] < letters
    top, k_hi = cfg['decode']['char_topk'], cfg['decode']['k_hi']
    fig = plt.figure(figsize=(figstyle.TEXTWIDTH, 1.9))
    grid = fig.add_gridspec(1, 3, left=0.075, right=0.985, bottom=0.21, top=0.9, wspace=0.42)
    roc = fig.add_subplot(grid[0])
    inner = grid[1].subgridspec(2, 1, height_ratios=(3, 1.25), hspace=0.1)
    rel = fig.add_subplot(inner[0])
    hist = fig.add_subplot(inner[1], sharex=rel)
    read = fig.add_subplot(grid[2])

    # (a) ROC for "the token claims a letter"
    zoom = roc.inset_axes([0.5, 0.37, 0.47, 0.4])
    aucs = {}
    for key, name, score, colour in (('keep', '$k$', k, figstyle.P_COLOUR),
                                     ('detector', '$p_{\\mathrm{letter}}$', p, DETECTOR)):
        fpr, tpr, _ = roc_curve(claim, score)
        aucs[key] = roc_auc_score(claim, score)
        kept = score >= KEEPS
        corner = (~claim & kept).sum() / (~claim).sum(), (claim & kept).sum() / claim.sum()
        for ax in (roc, zoom):
            ax.plot(fpr, tpr, color=colour, label=f'{name}: AUC {aucs[key]:.3f}')
            ax.plot(*corner, 'o', ms=3.5, color=colour, mec='white', mew=0.5, zorder=3)
    roc.set_xlim(0, 1)
    roc.set_ylim(0, 1.02)
    roc.set_xlabel('false-positive rate')
    roc.set_ylabel('true-positive rate')
    roc.legend(*roc.get_legend_handles_labels(), loc='lower right', handlelength=1.0, borderaxespad=0.1)
    box = (0.0, 0.25, 0.75, 1.0)  # x0, x1, y0, y1 of the zoom
    zoom.set_xlim(*box[:2])
    zoom.set_ylim(*box[2:])
    zoom.set_xticks([0, 0.1, 0.2])
    zoom.set_yticks([0.8, 0.9, 1.0])
    zoom.tick_params(labelsize=6.5, length=2, pad=1.5)
    zoom.patch.set_alpha(0.9)
    roc.add_patch(Rectangle((box[0], box[2]), box[1] - box[0], box[3] - box[2], fill=False, ec=figstyle.MUTED,
                            lw=0.5, ls='--'))

    # (b) reliability of k
    edges = np.linspace(0, 1, 11)
    which = np.minimum(np.digitize(k, edges) - 1, len(edges) - 2)
    counts = np.bincount(which, minlength=len(edges) - 1)
    mean_k = np.array([k[which == i].mean() for i in range(len(counts))])
    rate = np.array([claim[which == i].mean() for i in range(len(counts))])
    rel.plot([0, 1], [0, 1], color=figstyle.MUTED, lw=0.6, ls='--')
    rel.plot(mean_k, rate, color=figstyle.P_COLOUR, lw=0.8)
    rel.scatter(mean_k, rate, s=14, c=mean_k, cmap=figstyle.KEEP_CMAP, vmin=0, vmax=1, edgecolors='k', linewidths=0.4,
                zorder=3)
    rel.set_ylim(0, 1.02)
    rel.set_ylabel('share matched\nto a letter')
    plt.setp(rel.get_xticklabels(), visible=False)
    hist.bar((edges[:-1] + edges[1:]) / 2, counts, width=np.diff(edges) * 0.9,
             color=plt.get_cmap(figstyle.KEEP_CMAP)((edges[:-1] + edges[1:]) / 2), lw=0)
    hist.set_yscale('log')
    hist.set_ylim(3e2, 2e5)
    hist.set_yticks([1e3, 1e4, 1e5])
    hist.set_xlim(0, 1)
    hist.set_xlabel('keep probability $k$')
    hist.set_ylabel('tokens')

    # (c) the claimed letter among the n most probable letters
    ranks = reading_ranks(lp, d['target'], letters)
    kc = k[claim]
    groups = (('all', np.ones(len(kc), bool), 'k', 'o', 1.3),
              (f'$k \\geq {k_hi:g}$', kc >= k_hi, plt.get_cmap(figstyle.KEEP_CMAP)(0.95), 'o', 0.8),
              (f'${KEEPS:g} \\leq k < {k_hi:g}$', (kc >= KEEPS) & (kc < k_hi), plt.get_cmap(figstyle.KEEP_CMAP)(0.25),
               's', 0.8),
              (f'$k < {KEEPS:g}$', kc < KEEPS, plt.get_cmap(figstyle.KEEP_CMAP)(0.02), '^', 0.8))
    ns = np.arange(1, top + 1)
    for label, mask, colour, marker, lw in groups:
        read.plot(ns, [(ranks[mask] < n).mean() for n in ns], marker=marker, ms=3, lw=lw, color=colour,
                  label=f'{label} ({mask.sum():,})')
    read.set_xlim(0.7, top + 0.3)
    read.set_ylim(0.5, 1.0)
    read.set_xticks(ns)
    read.set_xlabel('letter among the $n$ most probable')
    read.set_ylabel('share of matched letters')
    read.legend(loc='lower right', handlelength=1.4, borderaxespad=0.1, labelspacing=0.3)
    for ax, letter, x in ((roc, 'a', -0.2), (rel, 'b', -0.2), (read, 'c', -0.2)):
        figstyle.panel_label(ax, letter, x=x)

    kept = k >= KEEPS
    macros = {
        'TokensTokens': figstyle.num(len(k)),
        'TokensClaims': figstyle.num(int(claim.sum())),
        'TokensNone': figstyle.num(int((~claim).sum())),
        'TokensKept': figstyle.num(int(kept.sum())),
        'TokensAucKeep': f'{aucs["keep"]:.3f}',
        'TokensAucDetector': f'{aucs["detector"]:.3f}',
        'TokensLost': figstyle.num(int((claim & ~kept).sum())),
        'TokensLostShare': pct((claim & ~kept).sum() / claim.sum()),
        'TokensSpurious': figstyle.num(int((~claim & kept).sum())),
        'TokensSpuriousShare': pct((~claim & kept).sum() / kept.sum()),
        'TokensTopOne': pct((ranks < 1).mean()),
        'TokensTopEight': pct((ranks < top).mean()),
    }
    return [figstyle.save_vector(fig, 'tokens'), figstyle.write_macros('tokens', macros)]


def confusion(work, cfg) -> list:
    """figs/confusion.{pdf,tex}: the claimed letter against the most probable letter, among claiming tokens."""
    d = load(work, cfg)
    alphabet = cfg['alphabet']
    size = len(alphabet)
    claim = d['target'] < size
    claimed, read = d['target'][claim], d['lp'][claim, :size].argmax(1)
    counts = np.zeros((size, size), np.int64)
    np.add.at(counts, (claimed, read), 1)
    order = np.argsort(-counts.sum(1), kind='stable')
    counts = counts[np.ix_(order, order)]
    labels = [GREEK_OF.get(alphabet[i], alphabet[i]) for i in order]
    rows = counts.sum(1, keepdims=True)
    share = np.divide(counts, rows, out=np.zeros(counts.shape), where=rows > 0)

    side = 3.4  # inches: the figure is square; the matrix, its count strip and its colour bar are placed by hand
    fig = plt.figure(figsize=(side, side))
    matrix, strip, gap, left, top = 2.45, 0.5, 0.05, 0.42, 0.04

    def place(x, y, w, h):
        return [x / side, y / side, w / side, h / side]

    ax = fig.add_axes(place(left, side - top - matrix, matrix, matrix))
    counts_ax = fig.add_axes(place(left + matrix + gap, side - top - matrix, strip, matrix), sharey=ax)
    cax = fig.add_axes(place(left, 0.44, matrix, 0.07))
    low = 1e-3
    cmap = LinearSegmentedColormap.from_list('blues', plt.get_cmap('Blues')(np.linspace(0.12, 1, 256)))
    cells = np.arange(size + 1) - 0.5
    image = ax.pcolormesh(cells, cells, np.ma.masked_equal(share, 0), cmap=cmap, norm=LogNorm(low, 1),
                          edgecolors='face', linewidth=0.3)
    ax.set_xlim(cells[0], cells[-1])
    ax.set_ylim(cells[-1], cells[0])
    ax.set_aspect('equal')
    off = counts - np.diag(np.diag(counts))
    worst = np.argsort(-off.ravel(), kind='stable')[:len(WORDS)]
    for flat in worst:
        i, j = divmod(int(flat), size)
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=figstyle.NONE, lw=0.9))
    ticks = range(size)
    ax.set_xticks(ticks, labels, fontsize=6.5)
    ax.set_yticks(ticks, labels, fontsize=6.5)
    ax.tick_params(length=0, pad=1.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel('most probable letter')
    ax.set_ylabel('annotated letter')
    bar = fig.colorbar(image, cax=cax, orientation='horizontal', extend='min')
    bar.set_label('share of the annotated letter\'s tokens')
    bar.ax.tick_params(labelsize=7)
    counts_ax.barh(np.arange(size), rows[:, 0], height=0.8, color=figstyle.LETTER, lw=0)
    counts_ax.set_xscale('log')
    counts_ax.set_xlim(10, 2 * rows.max())
    counts_ax.set_xticks([1e2, 1e4])
    counts_ax.set_xlabel('letters')
    counts_ax.tick_params(axis='y', left=False, labelleft=False)
    counts_ax.spines['left'].set_visible(False)

    macros = {
        'ConfusionTokens': figstyle.num(int(claim.sum())),
        'ConfusionErrors': figstyle.num(int(off.sum())),
        'ConfusionErrorShare': pct(off.sum() / counts.sum()),
    }
    for word, flat in zip(WORDS, worst):
        i, j = divmod(int(flat), size)
        macros |= {f'ConfusionTop{word}From': tex_letter(labels[i]), f'ConfusionTop{word}To': tex_letter(labels[j]),
                   f'ConfusionTop{word}Count': figstyle.num(int(counts[i, j])),
                   f'ConfusionTop{word}Share': pct(counts[i, j] / rows[i, 0])}
    return [figstyle.save_vector(fig, 'confusion'), figstyle.write_macros('confusion', macros)]
