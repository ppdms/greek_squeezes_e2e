"""Figures of the P decoder: how tokens become rows, how a row is searched, and one squeeze end to end (drawn by
paper/make_figures.py; nothing here is used by the system).

rows     figs/rows.jpg, figs/rows.tex: how the decoder forms the rows of a page. Two regions of the showcase squeeze
         (`figstyle.SHOWCASE`, read out of fold by the P chain that never saw it) in Rotation1, with every token the
         way `PageNetDecoder.page_rows` treats it, computed by the same functions (`pages.text_angle`,
         `rows.group_rows`, `rows.attach`) from the two page networks' posteriors (`figstyle.page_posteriors`): kept
         tokens (keep probability k = 1 - P(none) >= 0.5) solid in their row's colour; weak tokens (0.05 <= k < 0.5)
         that join a row dashed in that row's colour; weak tokens that join none, and tokens below 0.05, grey. A thin
         line runs through each row's tokens in reading order, and a tinted band marks where a weak token joins the
         row: within the row's span widened by `decode.extras.span_margin` letter heights and within half a letter
         height of the row's median line. The macros count the page's tokens by how the decoder treats them.
lattice  figs/lattice.pdf, figs/lattice.tex: the joint search on one real row. The row is chosen by a rule
         (`choose_row`) among the rows of the showcase squeeze, decoded out of fold exactly as the system decodes
         it (`decoder.decode_rows`, with the page cache pass): of the rows of 10 to 25 positions whose decode
         without the language model (lambda = 0, the (w, beta) that is best for lambda = 0 in the out-of-fold sweep)
         differs from the decode at the operating point (`decode.variants.P.combo`), those closer to the reference
         row at the operating point, the one that gains the most, ties to the fewest errors at the operating point. Per
         position: the token's relit crop (as the page network sees it), its three most probable letters with
         their probabilities (the 27-letter posterior renormalised, `decode.search.top_letters`), its keep
         probability k; below, the letters each decode emits from each position (a position that emits nothing is
         marked), and the reference row aligned to the operating point's letters.
example  figs/example.jpg, figs/example.tex: one squeeze followed through a P chain: the squeeze of 150 to 450
         Core characters whose out-of-fold error rate at the operating point is nearest the median (`typical_squeeze`,
         from the out-of-fold sweep), Rotation1 with every token the decoder uses outlined by its keep probability,
         and its Core reference rows against the decoded rows with the edit operations marked (the LaTeX tabular
         `\\ExampleRows`).
"""
from __future__ import annotations

import json
import math
from typing import NamedTuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon
from rapidfuzz.distance import Levenshtein

import figstyle
from trogs26 import official, score
from trogs26.decode.cache import CacheLM
from trogs26.decode.decoder import chain_decoder, decode_page, decode_rows
from trogs26.decode import rows as row_grouping
from trogs26.decode.search import Combo, candidate_pool, select, top_letters
from trogs26.dino.crops import relief_page_crops
from trogs26.layout import Work
from trogs26.pagenet import pages
from trogs26.pagenet.decode import KEEPS

ROW_COLOURS = ('#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9')  # Okabe-Ito, row after row
GREY = '#3A3A3A'
SUB, DEL, INS = '#C0392B', '#8E44AD', '#D35400'  # the colours report.tex gives subcol, delcol, inscol


# --------------------------------------------------------------------------------------------- the rows of a page
class PageRows(NamedTuple):
    """What `PageNetDecoder.page_rows` makes of a page, with the token indices it drops."""
    keep: np.ndarray          # (n,) keep probability 1 - P(none) of every token
    angle: float              # page angle, radians
    rows: list[np.ndarray]    # token indices of each row in reading order: its kept tokens and the weak ones joined
    row_of: np.ndarray        # (n,) row of the tokens in a row, -1 for the others
    span: np.ndarray          # (rows, 2) along-range a weak token may join, px: the kept tokens' widened by margin
    level: np.ndarray         # (rows,) median across-position of the row's kept tokens, px
    centre: np.ndarray        # (n, 2) token centres, px


def page_rows(cfg: dict, det: dict, log_probs: np.ndarray) -> PageRows:
    """The rows of a page as `PageNetDecoder.page_rows` forms them, with indices into the detector record.

    Tokens with k >= 0.5 give the page angle (`pages.text_angle`) and the rows (`rows.group_rows`); a token with
    `drop_below` <= k < 0.5 joins the row it sits on (`rows.attach`) or is dropped; a row is read in order of its
    tokens' centres along the page angle."""
    drop_below = cfg['decode']['variants']['P']['drop_below']
    margin = cfg['decode']['extras']['span_margin']
    keep = figstyle.keep_of(log_probs)
    candidates = np.flatnonzero(keep >= drop_below)
    letter_height = float(det['letter_height_px'])
    angle = pages.text_angle(det['boxes'], letter_height, keep, KEEPS)
    quads = np.array([b['quad'] for b in det['boxes']], np.float64).reshape(-1, 4, 2)
    centre = quads.mean(1)
    along = centre @ np.array([math.cos(angle), math.sin(angle)])
    across = centre @ np.array([-math.sin(angle), math.cos(angle)])
    strong = np.flatnonzero(keep[candidates] >= KEEPS)
    weak = np.flatnonzero(keep[candidates] < KEEPS)
    members = [strong[row] for row in row_grouping.group_rows(quads[candidates][strong], angle)]
    rows_u = [along[candidates][row] for row in members]
    rows_v = [across[candidates][row] for row in members]
    if len(weak):
        joins = row_grouping.attach(rows_u, rows_v, along[candidates][weak], across[candidates][weak], letter_height,
                                    margin)
        members = [np.concatenate([row, weak[joins == r]]) for r, row in enumerate(members)]
    rows = [candidates[row[np.argsort(along[candidates][row], kind='stable')]] for row in members]
    row_of = np.full(len(keep), -1)
    for r, row in enumerate(rows):
        row_of[row] = r
    span = np.array([[u.min() - margin * letter_height, u.max() + margin * letter_height] for u in rows_u])
    level = np.array([np.median(v) for v in rows_v])
    return PageRows(keep, angle, rows, row_of, span, level, centre)


def check_rows(decoder, base: str, layout: PageRows) -> None:
    """The decoder's own rows of `base` carry the keep probabilities of `layout`'s rows in the same order."""
    own = decoder.page_rows(base)
    if len(own) != len(layout.rows) or any(not np.array_equal(k, layout.keep[row])
                                           for (_, k), row in zip(own, layout.rows)):
        raise AssertionError(f'{base}: the rows drawn are not the rows the decoder forms')


# -------------------------------------------------------------------------------------------------- figure: rows
def _draw_tokens(ax, det: dict, layout: PageRows, window: tuple[float, float, float, float], lh: float,
                 drop_below: float) -> None:
    """The tokens of `layout` and the rows' reading lines over the axes' region `window` (x0, y0, x1, y1, px); the
    tokens below `drop_below` are drawn solid, the weak ones that join no row dashed."""
    x0, y0, x1, y1 = window
    c, s = math.cos(layout.angle), math.sin(layout.angle)
    for r in range(len(layout.rows)):
        u0, u1 = layout.span[r]
        v = layout.level[r]
        corners = np.array([[u * c - w * s, u * s + w * c]
                            for u, w in ((u0, v - lh / 2), (u1, v - lh / 2), (u1, v + lh / 2), (u0, v + lh / 2))])
        ax.add_patch(Polygon(corners, closed=True, fc=ROW_COLOURS[r % len(ROW_COLOURS)], ec='none', alpha=0.13,
                             zorder=2))
    quads = np.array([b['quad'] for b in det['boxes']], np.float64).reshape(-1, 4, 2)
    inside = ((quads[:, :, 0].max(1) >= x0) & (quads[:, :, 0].min(1) <= x1)
              & (quads[:, :, 1].max(1) >= y0) & (quads[:, :, 1].min(1) <= y1))
    for i in np.flatnonzero(inside):
        r, k = layout.row_of[i], layout.keep[i]
        if r >= 0:
            colour = ROW_COLOURS[r % len(ROW_COLOURS)]
            ax.add_patch(Polygon(quads[i], closed=True, fill=False, ec=colour, lw=0.9 if k >= KEEPS else 0.8,
                                 ls='-' if k >= KEEPS else (0, (2.2, 1.2)), zorder=4))
        else:
            ax.add_patch(Polygon(quads[i], closed=True, fill=False, ec=GREY, alpha=0.8, lw=0.55,
                                 ls='-' if k < drop_below else (0, (2.2, 1.2)), zorder=3))
    for r, row in enumerate(layout.rows):
        colour = ROW_COLOURS[r % len(ROW_COLOURS)]
        xy = layout.centre[row]
        ax.plot(xy[:, 0], xy[:, 1], '-', color='white', lw=2.0, solid_capstyle='round', zorder=5, alpha=0.8)
        ax.plot(xy[:, 0], xy[:, 1], '-o', color=colour, lw=0.8, ms=1.6, mew=0, solid_capstyle='round', zorder=6)


def _angle_guide(ax, origin: tuple[float, float], angle: float, lh: float) -> None:
    """The page angle as a dotted line from `origin` (page px) and the letter height as a bar across the rows, both
    labelled, in a blank part of a panel."""
    halo = [patheffects.withStroke(linewidth=2.2, foreground='white')]
    c, s = math.cos(angle), math.sin(angle)
    x, y = origin
    ax.plot([x, x + 4.4 * lh * c], [y, y + 4.4 * lh * s], color='black', lw=0.9, ls=(0, (1, 1.4)), zorder=8,
            path_effects=halo)
    ax.text(x + 2.2 * lh * c, y + 2.2 * lh * s - 0.12 * lh, f'page angle ${math.degrees(angle):.2f}^\\circ$',
            ha='center', va='bottom', fontsize=7, zorder=8, path_effects=halo)
    bx, by = x + 0.2 * lh, y + 0.8 * lh
    ax.plot([bx, bx - lh * s], [by, by + lh * c], color='black', lw=0.9, zorder=8, path_effects=halo)
    ax.text(bx + 0.3 * lh, by + 0.5 * lh * c, '1 letter height', ha='left', va='center', fontsize=7, zorder=8,
            path_effects=halo)


def rows(work: Work, cfg: dict) -> list:
    base = figstyle.SHOWCASE
    det = figstyle.record(work, cfg, base)
    fold = figstyle.fold_of(cfg, base)
    det_ids, log_probs = figstyle.page_posteriors(work, cfg, fold)[base]
    if not np.array_equal(det_ids, [int(b['id']) for b in det['boxes']]):
        raise ValueError(f'{base}: the page networks read other detections than the record holds')
    layout = page_rows(cfg, det, log_probs)
    check_rows(chain_decoder(work, figstyle.oof_chain(work, cfg, fold), cfg, 'P'), base, layout)
    gray = figstyle.views(work, base)[0]
    lh = float(det['letter_height_px'])
    keep = layout.keep
    drop_below = cfg['decode']['variants']['P']['drop_below']
    weak = (keep >= drop_below) & (keep < KEEPS)
    joined = weak & (layout.row_of >= 0)
    if int(((keep >= KEEPS) & (layout.row_of >= 0)).sum()) != int((keep >= KEEPS).sum()):
        raise AssertionError(f'{base}: a kept token is in no row')

    width, height, gap, top = figstyle.TEXTWIDTH, 2.5, 0.08, 0.36
    panel_w = (width - gap) / 2
    panel_h = height - top
    fig = plt.figure(figsize=(width, height))
    windows = [(2100.0, 4450.0, 12.5), (900.0, 1400.0, 12.5)]
    shape = (panel_w / panel_h)
    for p, (x0, y0, size) in enumerate(windows):
        ax = fig.add_axes([p * (panel_w + gap) / width, 0.0, panel_w / width, panel_h / height])
        w = size * lh
        x1, y1 = x0 + w, y0 + w / shape
        patch = gray[int(y0):int(y1), int(x0):int(x1)].astype(np.float32)
        lo, hi = np.percentile(patch, (1, 99))
        ax.imshow(patch, cmap='gray', vmin=lo - 0.15 * (hi - lo), vmax=hi + 0.1 * (hi - lo),
                  extent=(int(x0), int(x1), int(y1), int(y0)), interpolation='antialiased', zorder=1)
        _draw_tokens(ax, det, layout, (x0, y0, x1, y1), lh, drop_below)
        if p == 0:
            _angle_guide(ax, (x0 + 0.60 * (x1 - x0), y0 + 0.50 * (y1 - y0)), layout.angle, lh)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)
        ax.set_axis_off()
        ax.text(0.012, 0.985, f'({"ab"[p]})', transform=ax.transAxes, ha='left', va='top', fontweight='bold',
                bbox={'fc': 'white', 'ec': 'none', 'pad': 1.5, 'alpha': 0.85}, zorder=10)
    handles = [Line2D([], [], color=GREY, lw=0.9, label=f'kept, $k\\geq {KEEPS:g}$'),
               Line2D([], [], color=GREY, lw=0.8, ls=(0, (2.2, 1.2)), label='weak, joins a row'),
               Line2D([], [], color=GREY, lw=0.55, alpha=0.8, ls=(0, (2.2, 1.2)), label='weak, joins none'),
               Line2D([], [], color=GREY, lw=0.55, alpha=0.8, label=f'$k<{drop_below:g}$, dropped'),
               Line2D([], [], color=GREY, lw=0.8, marker='o', ms=2, mew=0, label='reading order'),
               Patch(fc=GREY, alpha=0.18, label='where a weak token joins')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=3, handlelength=2.2,
               columnspacing=2.0, handletextpad=0.5, borderaxespad=0.0, labelspacing=0.25)
    paths = [figstyle.save_raster(fig, 'rows')]
    angle = math.degrees(layout.angle)
    n_rows = len(layout.rows)
    kept = int((keep >= KEEPS).sum())
    macros = {
        'RowsBase': base.replace('_', '\\_'), 'RowsTokens': figstyle.num(len(keep)),
        'RowsRows': str(n_rows), 'RowsKept': figstyle.num(kept),
        'RowsWeakJoined': str(int(joined.sum())), 'RowsWeakDropped': str(int((weak & ~joined).sum())),
        'RowsDropped': figstyle.num(int((keep < drop_below).sum())),
        'RowsAngle': f'${angle:.2f}$',
    }
    paths.append(figstyle.write_macros('rows', macros))
    return paths


# ------------------------------------------------------------------------------------- the search on one row
POSITIONS = (10, 25)  # a row shown in `lattice` has this many positions at least and at most


class RowDecode(NamedTuple):
    """One row of the showcase page, read two ways, with the reference row it is compared to."""
    base: str
    index: int                  # the row's index in the page's rows
    n_rows: int
    candidates: int             # rows of the page that meet the rule
    tokens: np.ndarray          # (n,) token indices (detector record) of the row's positions, reading order
    emit: np.ndarray            # (n, 27) letter log-probabilities of the positions
    keep: np.ndarray            # (n,) keep probabilities
    texts: dict[str, str]       # 'none': decode without the language model; 'op': at the operating point
    positions: dict[str, list[int]]  # for each text, the position every letter comes from
    combos: dict[str, Combo]
    reference: str              # the reference row nearest the operating point's text
    errors: dict[str, int]      # edit distance of each text to the reference row
    terms: dict[str, dict[str, float]]  # each text's terms of the score at the operating point


def best_without_lm(work: Work, cfg: dict) -> Combo:
    """The (w, beta) of lambda = 0 with the fewest out-of-fold Core errors (both views, all squeezes) in the P
    chains' sweep; the first of equals in the sweep's order."""
    sweep = json.loads((work.run(cfg['name'], 'oof') / 'sweep_P.json').read_text())
    totals: dict[str, int] = {}
    for scores in sweep['scores'].values():
        for key, (e1, _, e2, _) in scores.items():
            totals[key] = totals.get(key, 0) + e1 + e2
    combos = [Combo(*(float(v) for v in key.split('|'))) for key in totals]
    return min((c for c in combos if c.lam == 0.0), key=lambda c: totals[c.key])


def emitting_positions(text: str, emit: np.ndarray, keep: np.ndarray, search) -> list[int]:
    """The positions of a row that emit the letters of `text`, one each in order: the path with the largest sum of
    letter log-probabilities and emit and skip log-probabilities of the keep probability (the weighting under which
    `decode.search.candidate_pool` keeps the best path of a text). A position may be skipped only if k < k_hi, and
    emits only one of its top `char_topk` letters."""
    letters = [dict(top_letters(emit[j], search.alphabet, search.topk)) for j in range(len(emit))]
    k = np.clip(np.asarray(keep, np.float64), 1e-4, 1 - 1e-4)
    skippable = k < search.k_hi
    log_keep = np.where(skippable, np.log(k), 0.0)
    log_skip = np.where(skippable, np.log1p(-k), 0.0)
    n, m = len(emit), len(text)
    best = np.full((n + 1, m + 1), -np.inf)
    best[0, 0] = 0.0
    for j in range(n):
        for i in range(m + 1):
            if best[j, i] == -np.inf:
                continue
            if skippable[j]:
                best[j + 1, i] = max(best[j + 1, i], best[j, i] + log_skip[j])
            if i < m and text[i] in letters[j]:
                best[j + 1, i + 1] = max(best[j + 1, i + 1], best[j, i] + letters[j][text[i]] + log_keep[j])
    if best[n, m] == -np.inf:
        raise ValueError(f'{text!r} cannot be read off the row')
    emitted, i = [], m
    for j in range(n - 1, -1, -1):
        if i and text[i - 1] in letters[j] and np.isclose(
                best[j + 1, i], best[j, i - 1] + letters[j][text[i - 1]] + log_keep[j]):
            emitted.append(j)
            i -= 1
    if i:
        raise AssertionError(f'{text!r}: path lost at {i} letters')
    return emitted[::-1]


def choose_row(work: Work, cfg: dict, base: str) -> RowDecode:
    """The row `lattice` shows, among the rows of `base` decoded out of fold by the system (`decode_rows`, with the
    page cache pass) at lambda = 0 (the (w, beta) best for it, `best_without_lm`) and at the operating point: of the
    rows of `POSITIONS` positions whose two texts differ and whose text at the operating point is closer to its
    reference row (the reference row nearest the operating point's text, edit distance), the one the operating
    point improves most, ties to the fewest errors there, then the earliest row."""
    fold = figstyle.fold_of(cfg, base)
    decoder = chain_decoder(work, figstyle.oof_chain(work, cfg, fold), cfg, 'P')
    det = figstyle.record(work, cfg, base)
    layout = page_rows(cfg, det, decoder.posteriors[base][1])
    check_rows(decoder, base, layout)
    combos = {'none': best_without_lm(work, cfg), 'op': Combo(**cfg['decode']['variants']['P']['combo'])}
    inputs = decoder.page_rows(base)
    texts = decode_rows(inputs, list(combos.values()), decoder.lm, decoder.search, decoder.cache)
    reference = score.core_text(work, base, 1).split('\n')
    found = []
    for r, ((emit, _), (none, op)) in enumerate(zip(inputs, texts)):
        near = min(reference, key=lambda row: Levenshtein.distance(op, row))
        errors = {'none': Levenshtein.distance(none, near), 'op': Levenshtein.distance(op, near)}
        if POSITIONS[0] <= len(emit) <= POSITIONS[1] and none != op and errors['op'] < errors['none']:
            found.append((errors['op'] - errors['none'], errors['op'], r, near, errors))
    if not found:
        raise ValueError(f'{base}: no row meets the rule')
    _, _, r, near, errors = min(found)
    emit, keep = inputs[r]
    row_texts = dict(zip(combos, texts[r]))
    # the row's pool of the second pass, rebuilt as `decode_rows` builds it, for the terms of the score
    cache = decoder.cache
    first = [select(candidate_pool(e, k, decoder.lm, decoder.search), [cache.ref])[0] for e, k in inputs]
    others = [text for j, text in enumerate(first) if j != r and text]
    pool = candidate_pool(emit, keep, CacheLM(decoder.lm, others, cache.gamma, cache.order, decoder.search.alphabet),
                          decoder.search)
    if select(pool, list(combos.values())) != texts[r]:
        raise AssertionError(f'{base} row {r}: the rebuilt pool does not give the decoded texts')
    op = combos['op']
    terms = {}
    for name, text in row_texts.items():
        i = pool.texts.index(text)
        parts = {'letters': float(pool.visual[i]), 'keep': op.w * float(pool.keep[i]),
                 'lm': op.lam * float(pool.lm[i]), 'length': op.beta * float(pool.length[i])}
        terms[name] = {**parts, 'total': sum(parts.values())}
    return RowDecode(base, r, len(inputs), len(found), layout.rows[r], emit, keep, row_texts,
                     {name: emitting_positions(text, emit, keep, decoder.search) for name, text in row_texts.items()},
                     combos, near, errors, terms)


def edit_tags(hyp: str, ref: str) -> tuple[list[str], list[str], dict[int, int]]:
    """The edit operations aligning `hyp` to `ref` (`Levenshtein.opcodes`): a tag for each letter of `hyp` ('equal',
    'sub' or 'ins': a letter the reference lacks), a tag for each letter of `ref` ('equal', 'sub' or 'del': a letter
    `hyp` lacks), and for each letter of `ref` that pairs with one of `hyp` its index there."""
    hyp_tags, ref_tags, pairs = ['equal'] * len(hyp), ['equal'] * len(ref), {}
    for tag, i1, i2, j1, j2 in Levenshtein.opcodes(hyp, ref):
        common = i2 - i1 if tag == 'equal' else min(i2 - i1, j2 - j1)
        for d in range(common):
            pairs[j1 + d] = i1 + d
            if tag == 'replace':
                hyp_tags[i1 + d] = ref_tags[j1 + d] = 'sub'
        for i in range(i1 + common, i2):
            hyp_tags[i] = 'ins'
        for j in range(j1 + common, j2):
            ref_tags[j] = 'del'
    return hyp_tags, ref_tags, pairs


def reference_columns(ref: str, pairs: dict[int, int], hyp_columns: list[int]) -> list[float]:
    """The column of each letter of `ref`: that of the letter of the aligned text it pairs with. The letters of
    `ref` between two paired ones that pair with none take the columns between them that the aligned text leaves
    empty, or if there are too few spread evenly between them."""
    out: list[float] = [0.0] * len(ref)
    for j, i in pairs.items():
        out[j] = float(hyp_columns[i])
    taken = {int(out[j]) for j in pairs}
    for j in range(len(ref)):
        if j in pairs:
            continue
        a = max((s for s in pairs if s < j), default=None)
        b = min((s for s in pairs if s > j), default=None)
        lo = out[a] if a is not None else -1.0
        hi = out[b] if b is not None else max([*hyp_columns, lo]) + 1.0
        run = [s for s in range(len(ref)) if (a is None or s > a) and (b is None or s < b) and s not in pairs]
        free = [c for c in range(int(lo) + 1, int(hi)) if c not in taken]
        index = run.index(j)
        out[j] = float(free[index]) if len(free) >= len(run) else lo + (index + 1) * (hi - lo) / (len(run) + 1)
    return out


# ----------------------------------------------------------------------------------------------- figure: lattice
MINUS = '\u2212'
PROBABILITY = LinearSegmentedColormap.from_list('probability', ['#FFFFFF', figstyle.ACCENT])
TAG_COLOUR = {'equal': 'black', 'sub': SUB, 'ins': INS, 'del': DEL}


def _signed(value: float, digits: int = 1) -> str:
    return f'{value:.{digits}f}'.replace('-', MINUS)


def _luminance(rgba) -> float:
    r, g, b = rgba[:3]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def lattice(work: Work, cfg: dict) -> list:
    base = figstyle.SHOWCASE
    row = choose_row(work, cfg, base)
    n = len(row.keep)
    alphabet = cfg['alphabet']
    det = figstyle.record(work, cfg, base)
    crop = cfg['dino']['crop']
    _, relit = relief_page_crops(det, *figstyle.views(work, base), crop['size'], crop['scale'], ['relit'])
    thumbs = relit['relit'][row.tokens].transpose(0, 2, 3, 1)

    width = figstyle.TEXTWIDTH
    left, right = 1.16, 0.36
    cw = (width - left - right) / n
    side = cw - 0.03
    cell_h, cell_gap = 0.18, 0.025
    y_crop = 0.0
    y_cells = y_crop + side + 0.04
    y_keep = y_cells + 3 * cell_h + 2 * cell_gap + 0.035
    y_head = y_keep + 0.20 + 0.20
    y_text = y_head + 0.20
    text_h = 0.30
    y_terms = y_text + 2 * text_h + 0.18
    term_h = 0.23
    height = y_terms + 2 * term_h + 0.02
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_axis_off()

    def column(j: float) -> float:
        return left + (j + 0.5) * cw

    letters = {name: [None] * n for name in row.texts}
    for name, text in row.texts.items():
        for ch, j in zip(text, row.positions[name]):
            letters[name][j] = ch
    keep_cmap = plt.get_cmap(figstyle.KEEP_CMAP)
    for j in range(n):
        x = column(j)
        k = float(row.keep[j])
        ax.imshow(cv2.resize(thumbs[j], (96, 96), interpolation=cv2.INTER_AREA),
                  extent=(x - side / 2, x + side / 2, y_crop + side, y_crop), interpolation='nearest', zorder=2)
        ax.add_patch(plt.Rectangle((x - side / 2, y_crop), side, side, fill=False, ec=keep_cmap(k), lw=1.1, zorder=3))
        order = np.argsort(-row.emit[j], kind='stable')[:3]
        for c, idx in enumerate(order):
            p = float(np.exp(row.emit[j][idx]))
            y = y_cells + c * (cell_h + cell_gap)
            chosen = letters['op'][j] == alphabet[idx]
            ax.add_patch(plt.Rectangle((x - cw / 2 + 0.012, y), cw - 0.024, cell_h, fc=PROBABILITY(p),
                                       ec='black' if chosen else '#BBBBBB', lw=1.0 if chosen else 0.4, zorder=3))
            colour = 'white' if p > 0.55 else 'black'
            ax.text(x - cw / 2 + 0.03, y + cell_h / 2, alphabet[idx], ha='left', va='center_baseline', color=colour,
                    family='DejaVu Sans', fontsize=9, zorder=4)
            percent = f'{100 * p:.0f}' if p >= 0.005 else '<1'
            ax.text(x + cw / 2 - 0.025, y + cell_h / 2, percent, ha='right', va='center_baseline', color=colour,
                    fontsize=8, zorder=4)
        face = keep_cmap(k)
        ax.add_patch(plt.Rectangle((x - cw / 2 + 0.012, y_keep), cw - 0.024, 0.20, fc=face, ec='#BBBBBB', lw=0.4,
                                   zorder=3))
        ax.text(x, y_keep + 0.10, f'{k:.2f}', ha='center', va='center_baseline', fontsize=8, zorder=4,
                color='white' if _luminance(face) < 0.45 else 'black')

    names = ('op', 'ref')
    centres = {name: y_text + (i + 0.5) * text_h for i, name in enumerate(names)}
    tags = {name: edit_tags(row.texts[name], row.reference) for name in ('op',)}
    for name in ('op',):
        for ch, j, tag in zip(row.texts[name], row.positions[name], tags[name][0]):
            ax.text(column(j), centres[name], ch, ha='center', va='center_baseline', family='DejaVu Sans', fontsize=11,
                    color=TAG_COLOUR[tag], zorder=4)
        for j in sorted(set(range(n)) - set(row.positions[name])):
            ax.text(column(j), centres[name], '\u2013', ha='center', va='center_baseline', fontsize=11,
                    color='#555555', zorder=4)
    columns = reference_columns(row.reference, tags['op'][2], row.positions['op'])
    for ch, x, tag in zip(row.reference, columns, tags['op'][1]):
        ax.text(column(x), centres['ref'], ch, ha='center', va='center_baseline', family='DejaVu Sans', fontsize=11,
                color=TAG_COLOUR[tag], zorder=4)
    ax.plot([left, left + n * cw], [y_text - 0.05] * 2, color='#BBBBBB', lw=0.4)
    ax.plot([left, left + n * cw], [y_text + 2 * text_h + 0.0] * 2, color='#BBBBBB', lw=0.4, ls=(0, (1, 1.5)))

    label = dict(ha='right', va='center_baseline', fontsize=8.5)
    xl = left - 0.07
    ax.text(xl, y_crop + side / 2, 'relit crop', **label)
    ax.text(xl, y_cells + (3 * cell_h + 2 * cell_gap) / 2, 'top three\nletters (%)', ha='right',
            va='center', fontsize=8.5, linespacing=1.2)
    ax.text(xl, y_keep + 0.10, 'keep $k$', **label)
    ax.text(xl, centres['op'], 'Final reading', **label)
    ax.text(xl, centres['ref'], 'reference', **label)
    for name in ('op',):
        ax.text(width - right / 2 + 0.03, centres[name], str(row.errors[name]), ha='center', va='center_baseline',
                fontsize=9)
    ax.text(width - right / 2 + 0.03, y_head, 'errors', ha='center', va='center_baseline', fontsize=8)

    ax.text(0, y_crop, '(a)', fontweight='bold', va='top', ha='left')
    ax.text(0, y_head, '(b)', fontweight='bold', va='center_baseline', ha='left')
    ax.text(0, y_terms + term_h / 2, '(c)', fontweight='bold', va='center_baseline', ha='left')

    heads = ('Letters', 'Keep / skip', 'Text model', 'Length', 'Total')
    x0, step = left + 0.05, (width - left - right - 0.1) / len(heads)
    for h, head in enumerate(heads):
        ax.text(x0 + (h + 0.5) * step, y_terms + term_h / 2, head, ha='center', va='center', fontsize=8.5)
    best = max(row.terms, key=lambda name: row.terms[name]['total'])
    for i, name in enumerate(('op',)):
        y = y_terms + (i + 1.5) * term_h
        ax.text(xl, y, 'Final reading', **label)
        for h, key in enumerate(('letters', 'keep', 'lm', 'length', 'total')):
            ax.text(x0 + (h + 0.5) * step, y, _signed(row.terms[name][key]), ha='center', va='center_baseline',
                    fontsize=8.5, fontweight='bold' if key == 'total' and name == best else 'normal')
    ax.plot([left, width - right], [y_terms + term_h] * 2, color='#BBBBBB', lw=0.4)
    op = row.combos['op']
    ax.text(xl, y_terms + term_h / 2, f'at $({op.lam:g},{op.w:g},{op.beta:g})$', ha='right', va='center_baseline',
            fontsize=8)

    _key(fig, ax, left + n * cw, y_head,
         [('extra', INS), ('wrong', SUB), ('missing', DEL), ('\u2013 omitted', '#555555')])
    paths = [figstyle.save_vector(fig, 'lattice')]
    macros = {
        'LatticeBase': base.replace('_', '\\_'), 'LatticeRow': str(row.index + 1), 'LatticeRows': str(row.n_rows),
        'LatticeCandidates': str(row.candidates), 'LatticePositions': str(n),
        'LatticeReferenceLetters': str(len(row.reference)),
        'LatticeNoLmW': f'{row.combos["none"].w:g}', 'LatticeNoLmBeta': f'{row.combos["none"].beta:g}',
        'LatticeNoLmErrors': str(row.errors['none']), 'LatticeOpErrors': str(row.errors['op']),
        'LatticeNoLmSkipped': str(n - len(row.texts['none'])), 'LatticeOpSkipped': str(n - len(row.texts['op'])),
    }
    for name, prefix in (('none', 'LatticeNoLm'), ('op', 'LatticeOp')):
        for key, term in (('letters', 'Letters'), ('keep', 'Keep'), ('lm', 'Lm'), ('length', 'Length'),
                          ('total', 'Score')):
            macros[prefix + term] = f'${row.terms[name][key]:.1f}$'
    paths.append(figstyle.write_macros('lattice', macros))
    return paths


def _key(fig, ax, x_right: float, y: float, parts: list[tuple[str, str]], gap: float = 0.1) -> None:
    """Words in their colours on one line ending at `x_right`, centred on `y` (axes data in inches)."""
    renderer = fig.canvas.get_renderer()
    texts = [ax.text(0, y, word, color=colour, ha='left', va='center_baseline', fontsize=8.5) for word, colour in parts]
    widths = [t.get_window_extent(renderer).width / fig.dpi for t in texts]
    x = x_right - sum(widths) - gap * (len(parts) - 1)
    for text, w in zip(texts, widths):
        text.set_x(x)
        x += w + gap


# ---------------------------------------------------------------------------------------------- figure: example
def typical_squeeze(work: Work, cfg: dict) -> str:
    """The squeeze of 150-450 Core characters (Rotation1) whose error rate at the P decoder's operating point
    (`decode.variants.P.combo`) in the run's out-of-fold sweep is nearest the median of those squeezes' (ties: the
    name)."""
    sweep = json.loads((work.run(cfg['name'], 'oof') / 'sweep_P.json').read_text())
    key = Combo(**cfg['decode']['variants']['P']['combo']).key
    rates = {base: (v[key][0] + v[key][2]) / (v[key][1] + v[key][3]) for base, v in sweep['scores'].items()
             if 150 <= v[key][1] <= 450}
    median = float(np.median(list(rates.values())))
    return min(rates, key=lambda b: (abs(rates[b] - median), b))


def annotated_rows(work: Work, base: str, view: int) -> list[str]:
    """The rows of the transcript section of annotation view `view` of `base` as written, `*` (an illegible letter
    without a box) included and spaces removed. The organisers' reader drops the `*`, so no scorer mode sees them."""
    phase, rows = 0, []
    for line in work.annotation(base, view).read_text().splitlines():
        if line.startswith('#'):
            phase += 1
        elif phase == 3:
            rows.append(line.replace(' ', '').replace('\t', ''))
    return rows


def free_reference(work: Work, base: str, view: int) -> str:
    """The Core reference of annotation view `view` with its `*` kept: the confident letters of each row (as
    `score.core_text`), rows joined by newlines, a row without a letter dropped, every `*` where it stands. The
    letters, `*` removed, are checked to be the scorer's own reference."""
    ref = score.reference(work, base, view)
    rows = [official.v2.convert(row).upper() for row in annotated_rows(work, base, view)]
    rows = [row for row in rows if row.replace('*', '')]
    scorer = official.v2.convert(ref.text).upper().replace(' ', '').replace('\t', '')
    if '\n'.join(row.replace('*', '') for row in rows) != scorer:
        raise ValueError(f'{base} view {view}: the transcript rows are not the scorer\'s reference rows')
    flags = iter(ref.confident)
    kept = [''.join(ch for ch in row if ch == '*' or next(flags)) for row in rows]
    return '\n'.join(row for row in kept if row.replace('*', ''))


def free_distance(decoded: str, reference: str) -> int:
    """Edit distance of `decoded` to `reference` in which every `*` of the reference is free: it matches any letter
    (not a newline) at cost 0, or nothing at cost 0. Every other insertion, deletion and substitution costs 1;
    newlines are characters, as in the scorer."""
    previous = [0]
    for ch in reference:
        previous.append(previous[-1] + (0 if ch == '*' else 1))
    for c in decoded:
        current = [previous[0] + 1]
        for j, r in enumerate(reference):
            match = 0 if r == c or (r == '*' and c != '\n') else 1
            current.append(min(previous[j + 1] + 1, current[j] + (0 if r == '*' else 1), previous[j] + match))
        previous = current
    return previous[-1]


def marked_rows(reference: str, decoded: str) -> tuple[list[tuple[int | None, str, str]], dict[str, int]]:
    """LaTeX rows of the reference and the decoded text with the edit operations of their alignment marked:
    substituted letters in both, deleted ones in the reference, inserted ones in the decoded text. A row break of
    either text starts a new line of both, so aligned letters stay side by side; each line carries the number of
    its reference row, or None for a line of the decoded text's own (where it breaks a row the reference does not;
    a line that holds no reference letter yet passes its number on to the next)."""
    lines: list[list] = [[1, '', '']]
    counts = {'substitutions': 0, 'deletions': 0, 'insertions': 0}
    style = {'replace': 'subcol', 'delete': 'delcol', 'insert': 'inscol'}

    def mark(ch: str, colour: str | None) -> str:
        ch = {'&': '\\&'}.get(ch, ch)
        return f'\\textcolor{{{colour}}}{{\\underline{{{ch}}}}}' if colour else ch

    row = 1
    for tag, i1, i2, j1, j2 in Levenshtein.opcodes(reference, decoded):
        if tag == 'equal':
            for ch in reference[i1:i2]:
                if ch == '\n':
                    row += 1
                    lines.append([row, '', ''])
                else:
                    lines[-1][1] += mark(ch, None)
                    lines[-1][2] += mark(ch, None)
            continue
        for ch in reference[i1:i2]:
            if ch == '\n':
                row += 1
                lines.append([row, '', ''])
            else:
                lines[-1][1] += mark(ch, style[tag])
        for ch in decoded[j1:j2]:
            if ch == '\n':
                number = None if lines[-1][1] else lines[-1][0]
                if not lines[-1][1]:
                    lines[-1][0] = None
                lines.append([number, '', ''])
            else:
                lines[-1][2] += mark(ch, style[tag])
        if tag == 'replace':
            counts['substitutions'] += max(i2 - i1, j2 - j1) - abs((i2 - i1) - (j2 - j1))
            counts['deletions' if i2 - i1 > j2 - j1 else 'insertions'] += abs((i2 - i1) - (j2 - j1))
        elif tag == 'delete':
            counts['deletions'] += i2 - i1
        elif tag == 'insert':
            counts['insertions'] += j2 - j1
    return [tuple(line) for line in lines], counts


def example(work: Work, cfg: dict) -> list:
    base = typical_squeeze(work, cfg)
    settings = cfg['decode']['variants']['P']
    fold = figstyle.fold_of(cfg, base)
    decoder = chain_decoder(work, figstyle.oof_chain(work, cfg, fold), cfg, 'P')
    det = figstyle.record(work, cfg, base)
    det_ids, log_probs = figstyle.page_posteriors(work, cfg, fold)[base]
    if not np.array_equal(det_ids, [int(b['id']) for b in det['boxes']]):
        raise ValueError(f'{base}: the page networks read other detections than the record holds')
    keep = figstyle.keep_of(log_probs)
    drop_below = settings['drop_below']

    gray = figstyle.views(work, base)[0]
    width = 1.46  # inches; the saved JPEG is 1.5 in wide with figstyle's padding
    pixels = int(round(width * 300))
    small = cv2.resize(gray, (pixels, int(round(pixels * gray.shape[0] / gray.shape[1]))), interpolation=cv2.INTER_AREA)
    bar = 0.34
    image_h = width * gray.shape[0] / gray.shape[1]
    fig = plt.figure(figsize=(width, image_h + bar))
    ax = fig.add_axes([0, bar / (image_h + bar), 1, image_h / (image_h + bar)])
    ax.imshow(small, cmap='gray', vmin=0, vmax=255 * 1.25, extent=(0, gray.shape[1], gray.shape[0], 0),
              interpolation='nearest')
    cmap = plt.get_cmap(figstyle.KEEP_CMAP)
    quads = np.array([b['quad'] for b in det['boxes']], np.float64).reshape(-1, 4, 2)
    for i in np.flatnonzero(keep >= drop_below):
        strong = keep[i] >= KEEPS
        ax.add_patch(Polygon(quads[i], closed=True, fill=False, ec=cmap(keep[i]), lw=0.5,
                             ls='-' if strong else (0, (3, 1.5))))
    ax.set_xlim(0, gray.shape[1])
    ax.set_ylim(gray.shape[0], 0)
    ax.set_axis_off()
    cax = fig.add_axes([0.1, 0.12 / (image_h + bar), 0.8, 0.07 / (image_h + bar)])
    colourbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1)), cax=cax,
                             orientation='horizontal')
    colourbar.set_label('keep probability $k$', fontsize=7, labelpad=1)
    colourbar.set_ticks([drop_below, KEEPS, 1], labels=[f'{drop_below:g}', f'{KEEPS:g}', '1'])
    colourbar.outline.set_linewidth(0.4)
    cax.tick_params(labelsize=6.5, length=2, width=0.4, pad=1)
    paths = [figstyle.save_raster(fig, 'example')]

    combo = Combo(**settings['combo'])
    decoded = decode_page(decoder, base, [combo])[combo]
    errors, chars = score.view_errors(work, base, 1, decoded)
    sweep = json.loads((work.run(cfg['name'], 'oof') / 'sweep_P.json').read_text())['scores'][base][combo.key]
    if (errors, chars) != (sweep[0], sweep[1]):
        raise AssertionError(f'{base}: the decode ({errors}/{chars}) is not the sweep\'s ({sweep[0]}/{sweep[1]})')
    v2 = official.v2
    reference = score.core_text(work, base, 1)
    decoded = '\n'.join(v2.convert(decoded).upper().replace(' ', '').replace('\t', '').split('\n'))
    with_stars = free_reference(work, base, 1)
    if free_distance(decoded, with_stars.replace('*', '')) != errors:
        raise AssertionError(f'{base}: the edit distance does not reproduce the Core errors without the `*`')
    free_errors = free_distance(decoded, with_stars)
    table, counts = marked_rows(reference, decoded)
    cells = [f'{number or ""} & \\textsf{{{r}}} & \\textsf{{{d}}}' for number, r, d in table]
    columns = '@{}r@{\\quad}l@{\\qquad}l@{}'
    header = 'Row & Reference & System'
    lines = ['%', f'\\begin{{tabular}}{{{columns}}}\\toprule', header + ' \\\\\\midrule']
    lines += [f'{row} \\\\' for row in cells]
    lines += ['\\bottomrule\\end{tabular}']
    macros = {
        'ExampleBase': base.replace('_', '\\_'), 'ExampleErrors': str(errors), 'ExampleChars': str(chars),
        'ExampleFreeErrors': str(free_errors),
        'ExampleIllegible': str(sum(row.count('*') for row in annotated_rows(work, base, 1))),
        'ExampleTokens': figstyle.num(len(keep)), 'ExampleKept': figstyle.num(int((keep >= KEEPS).sum())),
        'ExampleWeak': figstyle.num(int(((keep >= drop_below) & (keep < KEEPS)).sum())),
        'ExampleDropped': figstyle.num(int((keep < drop_below).sum())),
        **{f'Example{name.capitalize()}': str(n) for name, n in counts.items()},
        'ExampleRows': '\n'.join(lines),
    }
    paths.append(figstyle.write_macros('example', macros))
    return paths
