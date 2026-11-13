"""Figures of a squeeze and its relief: `teaser` and `relief`.

Both follow the showcase squeeze `figstyle.SHOWCASE` (a public squeeze, held out by P chain 4). The relief is the
one the page network reads: `trogs26.relief.fuse` of the two photographs with the registration and the letter
positions of the detector record that chain reads (the boxes with `p_letter` >= `trogs26.dino.crops.RELIEF_LETTERS`),
as `trogs26.dino.crops.relief_page_crops` computes it. No annotation is read.

teaser   (figs/teaser.jpg, figs/teaser.tex) The whole page, cropped to the paper (the span the paper of both views
         covers, less half a letter height on every side), in four panels. (a) Rotation1 and (b) Rotation2 as
         photographed; the crop of (b) is that of (a) carried through the inverse registration. (c) The relit
         relief, `crops.relief_page(f, 'relit')`, the page the page network's crops are cut from: the gradient
         recovered from both views, shaded from three azimuths, one per RGB channel. (d) Every detection of the
         record as its oriented box on a lightened Rotation1, coloured by its keep probability k = 1 - P(none) from
         the page network (`figstyle.page_posteriors`, the two networks combined). The macros count the detections
         by k and give the letter height and the registration grade.

relief   (figs/relief.jpg, figs/relief.tex) How the relief is recovered, on a window of the page's first text rows
         at the working scale `f.scale`. The window is 10.5 letter heights wide and as high as the panel (7.3); it
         is the one holding the most kept letters (k >= 0.5) among those on a half-letter grid whose top edge lies
         in the top fifth of the page and that lie on the paper of both views (the middle one of those holding as
         many). (a) Rotation1; (b) Rotation2 resampled onto it through the registration
         (`relief.second_view_on_first`); (c) the shading of the view lit along x and (d) of the view lit along y,
         each relative to the paper around it (s = I / I_paper - 1; `Fusion.s2` is view 2 after the tile-wise
         residual alignment), grey = 0; (e) the mixed derivatives of the two shadings (`relief.mixed_derivatives`
         of `Fusion.s1`, `Fusion.s2`): d/dy of the x-lit one in magenta and d/dx of the y-lit one in green, each in
         its own robust units (`relief.to_uint8`, 8 robust sigmas to the ends), view 2's with the sign that makes the
         two correlate positively, as `fuse` applies it. Both estimate h_xy, so where they agree the composite is
         grey, white or black, and where they differ it is tinted. (f) The relit relief of the same window. The
         macros give the global agreement of the mixed derivatives (`Fusion.agreement`) and, for the registration
         alone, before the residual alignment; the median residual shift; view 2's mean share of the gradient
         (`Fusion.share`); which view is lit along x; whether the overall sign was flipped; the working scale;
         and the window's size, kept letters, share and agreement.
"""
from __future__ import annotations

import cv2
import figstyle
import matplotlib
import numpy as np
from figstyle import TEXTWIDTH

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402

from trogs26 import relief as relief_lib  # noqa: E402
from trogs26.dino import crops  # noqa: E402
from trogs26.layout import Work  # noqa: E402
from trogs26.registration import Registration  # noqa: E402

DPI = 300
GAP = 0.05             # inches between panels
LABEL_ROOM = 0.15      # inches above each row of panels, for the panel labels
BOTTOM_ROOM = 0.06     # inches below the teaser, for the colourbar's lowest tick label
PAPER_COLUMNS = 0.9    # the teaser's crop keeps the columns and rows that the paper of both views covers at least
PAPER_ROWS = 0.9       # this much, less EDGE_LH on every side: the shadings are artefacts where the paper ends
EDGE_LH = 0.5          # (and the scan has a dark line along its edge)
WINDOW_WIDTH_LH = 10.5  # the relief figure's window, letter heights
WINDOW_TOP = 0.2       # its top edge lies in this top fraction of the page
WINDOW_STEP_LH = 0.5   # and is searched on this grid
SHADING_SPREAD = 4.0   # robust sigmas of a shading that reach the ends of the grey scale (`relief.to_uint8`)
MIXED_SPREAD = 8.0     # the same for a mixed derivative: its strokes are sharper, 4 would saturate them
TAG_COLOURS = ('#B0008A', '#1B7F1B')  # the x-lit view's mixed derivative (magenta), the y-lit view's (green)

_PAGES: dict[tuple[str, str], dict] = {}


def _page(work: Work, cfg: dict, base: str) -> dict:
    """The squeeze `base`'s detector record, grey photographs, registration and relief fusion, computed once."""
    key = (str(work.root), base)
    if key not in _PAGES:
        record = figstyle.record(work, cfg, base)
        gray1, gray2 = figstyle.views(work, base)
        registration = Registration.from_dict(record['registration'])
        letters = [b['quad'] for b in record['boxes'] if b['p_letter'] >= crops.RELIEF_LETTERS]
        fusion = relief_lib.fuse(gray1, gray2, registration, record['letter_height_px'], letters)
        _PAGES[key] = {'record': record, 'gray1': gray1, 'gray2': gray2, 'registration': registration,
                       'fusion': fusion, 'letters': len(letters)}
    return _PAGES[key]


def _keep(work: Work, cfg: dict, base: str, record: dict) -> np.ndarray:
    """Keep probability of each of the record's boxes, from its chain's two page networks."""
    ids, log_probs = figstyle.page_posteriors(work, cfg, figstyle.fold_of(cfg, base))[base]
    by_id = dict(zip(ids.tolist(), figstyle.keep_of(log_probs)))
    return np.array([by_id[int(b['id'])] for b in record['boxes']])


def _to_working(points: np.ndarray, scale: float) -> np.ndarray:
    """Original-pixel coordinates (pixel centres at integers) in the working frame, as `relief.fuse` carries them."""
    return (np.asarray(points, np.float64) + .5) * scale - .5


def _paper_box(f: relief_lib.Fusion) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1), working pixels, half-open: the span of columns and rows the paper of both views covers,
    less `EDGE_LH` letter heights on every side."""
    columns = np.nonzero(f.both.mean(0) >= PAPER_COLUMNS)[0]
    rows = np.nonzero(f.both.mean(1) >= PAPER_ROWS)[0]
    edge = round(EDGE_LH * f.letter_px)
    return int(columns[0]) + edge, int(rows[0]) + edge, int(columns[-1]) + 1 - edge, int(rows[-1]) + 1 - edge


def _fit(image: np.ndarray, width: float, height: float) -> np.ndarray:
    """`image` reduced (area average) to `width` by `height` inches at `DPI`."""
    size = (max(1, round(width * DPI)), max(1, round(height * DPI)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def _frame(fig, left: float, bottom: float, width: float, height: float):
    """An axes at the given inch position and size, ticks and spines off."""
    ax = fig.add_axes([left / fig.get_figwidth(), bottom / fig.get_figheight(), width / fig.get_figwidth(),
                       height / fig.get_figheight()])
    ax.set_axis_off()
    return ax


def _show(ax, image: np.ndarray, width: float, height: float, extent=None, **kwargs) -> None:
    """Draw `image` reduced to the axes' pixels; the image keeps the coordinates of `extent` if given."""
    ax.imshow(_fit(image, width, height), extent=extent, interpolation='nearest', aspect='auto', **kwargs)


def _save(fig, name: str):
    """figs/<name>.jpg at exactly the figure's size (not trimmed to its content)."""
    with plt.rc_context({'savefig.bbox': 'standard'}):
        return figstyle.save_raster(fig, name, DPI)


def _tag(ax, parts: list[tuple[str, str]], x: float = 0.1) -> None:
    """What `ax` shows, above it beside its panel label: `parts` are (text, colour), set side by side from `x`."""
    renderer = ax.figure.canvas.get_renderer()
    for text, colour in parts:
        artist = ax.text(x, 1.0, text, transform=ax.transAxes, ha='left', va='bottom', color=colour)
        x = artist.get_window_extent(renderer).transformed(ax.transAxes.inverted()).x1 + 0.015


def _r2_box(page: dict, box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """Rotation2's crop (original pixels, half-open) of the same paper as view 1's `box` (working pixels): `box`
    carried through the inverse registration, with view 1's aspect ratio so that the panels match."""
    scale = page['fusion'].scale
    x0, y0, x1, y1 = (v / scale for v in box)
    corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    moved = page['registration'].inverse().apply(corners)
    width = np.mean([np.linalg.norm(moved[1] - moved[0]), np.linalg.norm(moved[2] - moved[3])])
    height = np.mean([np.linalg.norm(moved[3] - moved[0]), np.linalg.norm(moved[2] - moved[1])])
    zoom = (width / (x1 - x0) + height / (y1 - y0)) / 2
    cx, cy = moved.mean(0)
    h2, w2 = page['gray2'].shape
    half_w, half_h = zoom * (x1 - x0) / 2, zoom * (y1 - y0) / 2
    return (max(0, round(cx - half_w)), max(0, round(cy - half_h)), min(w2, round(cx + half_w)),
            min(h2, round(cy + half_h)))


def _lightened(gray: np.ndarray) -> np.ndarray:
    """Grey photograph with its contrast halved around white-ish paper: an underlay that coloured outlines read on."""
    return (0.45 * gray.astype(np.float32) + 0.55 * 255).astype(np.uint8)


def teaser(work: Work, cfg: dict) -> list:
    base = figstyle.SHOWCASE
    page = _page(work, cfg, base)
    record, f = page['record'], page['fusion']
    keep = _keep(work, cfg, base, record)
    x0, y0, x1, y1 = _paper_box(f)
    ox0, oy0, ox1, oy1 = (round(v / f.scale) for v in (x0, y0, x1, y1))   # the same crop, original pixels
    r2 = _r2_box(page, (x0, y0, x1, y1))
    crop1 = page['gray1'][oy0:oy1, ox0:ox1]
    crop2 = page['gray2'][r2[1]:r2[3], r2[0]:r2[2]]
    relit = crops.relief_page(f, 'relit')[y0:y1, x0:x1]

    n_panels, colourbar, cb_pad, cb_room = 4, 0.07, 0.06, 0.40
    aspect = (x1 - x0) / (y1 - y0)
    width = (TEXTWIDTH - (n_panels - 1) * GAP - cb_pad - colourbar - cb_room) / n_panels
    height = width / aspect
    fig = plt.figure(figsize=(TEXTWIDTH, height + LABEL_ROOM + BOTTOM_ROOM))
    axes = [_frame(fig, i * (width + GAP), BOTTOM_ROOM, width, height) for i in range(n_panels)]
    extent = (-.5, ox1 - ox0 - .5, oy1 - oy0 - .5, -.5)
    _show(axes[0], crop1, width, height, extent, cmap='gray', vmin=0, vmax=255)
    _show(axes[1], crop2, width, height, None, cmap='gray', vmin=0, vmax=255)
    _show(axes[2], relit, width, height)
    _show(axes[3], _lightened(crop1), width, height, extent, cmap='gray', vmin=0, vmax=255)
    quads = np.array([b['quad'] for b in record['boxes']], np.float64) - [ox0, oy0]
    order = np.argsort(keep)                      # surest boxes on top
    cmap = plt.get_cmap(figstyle.KEEP_CMAP)
    axes[3].add_collection(PolyCollection(quads[order], facecolors='none', edgecolors=cmap(keep[order]),
                                          linewidths=0.45))
    axes[3].set_xlim(extent[0], extent[1])
    axes[3].set_ylim(extent[2], extent[3])
    for i, ax in enumerate(axes):
        figstyle.panel_label(ax, 'abcd'[i], x=0.11, y=1.0)

    cax = _frame(fig, 4 * width + 3 * GAP + cb_pad, BOTTOM_ROOM, colourbar, height)
    cax.set_axis_on()
    gradient = np.linspace(1, 0, 256)[:, None]
    cax.imshow(gradient, cmap=figstyle.KEEP_CMAP, aspect='auto', extent=(0, 1, 0, 1), vmin=0, vmax=1)
    cax.set_xticks([])
    cax.yaxis.tick_right()
    cax.yaxis.set_label_position('right')
    ticks = [0, 0.05, 0.5, 1]
    cax.set_yticks(ticks, ['0', '0.05', '0.5', '1'])
    cax.set_ylabel('keep probability $k$', labelpad=2)
    for tick in ticks[1:3]:
        cax.axhline(tick, color='k', lw=0.5)
    for side in cax.spines.values():
        side.set_visible(True)

    path = _save(fig, 'teaser')

    base_k = len(keep)
    kept, weak = int((keep >= .5).sum()), int(((keep >= .05) & (keep < .5)).sum())
    registration = record['registration']
    macros = {
        'TeaserSqueeze': base.replace('_', '\\_'),
        'TeaserFold': str(figstyle.fold_of(cfg, base)),
        'TeaserTokens': figstyle.num(base_k),
        'TeaserKept': figstyle.num(kept),
        'TeaserWeak': figstyle.num(weak),
        'TeaserDropped': figstyle.num(base_k - kept - weak),
        'TeaserReliefLetters': figstyle.num(page['letters']),
        'TeaserLetterHeight': f'{record["letter_height_px"]:.0f}',
        'TeaserRegistration': registration['quality'].replace('_', ' '),
        'TeaserInliers': f'{100 * registration["inlier_fraction"]:.1f}',
        'TeaserXView': f'Rotation{f.x_view}',
        'TeaserYView': f'Rotation{3 - f.x_view}',
    }
    return [path, figstyle.write_macros('teaser', macros)]


def _window(f: relief_lib.Fusion, centres: np.ndarray, width: float, height: float) -> tuple[int, int]:
    """Top-left corner (x, y), working pixels, of the `width` by `height` window that holds the most of the
    working-frame `centres` among windows on a `WINDOW_STEP_LH` grid whose top edge lies in the top `WINDOW_TOP` of
    the page and that lie on the paper of both views. Of several with the most, the middle one in reading order
    (windows a fraction of a letter apart hold the same letters; the middle one cuts the rows least)."""
    rows, columns = f.both.shape
    step = WINDOW_STEP_LH * f.letter_px
    found = []
    for y in np.arange(0, WINDOW_TOP * rows, step):
        for x in np.arange(0, columns - width, step):
            top, left = int(y), int(x)
            if y + height > rows or f.both[top:top + int(height), left:left + int(width)].mean() < 0.99:
                continue
            count = int(((centres[:, 0] >= x) & (centres[:, 0] < x + width) & (centres[:, 1] >= y)
                         & (centres[:, 1] < y + height)).sum())
            found.append((count, top, left))
    best = max(count for count, _, _ in found)
    ties = sorted((top, left) for count, top, left in found if count == best)
    top, left = ties[len(ties) // 2]
    return left, top


def _correlation(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    x, y = a[mask].astype(np.float64), b[mask].astype(np.float64)
    x, y = x - x.mean(), y - y.mean()
    return float((x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum()))


def relief(work: Work, cfg: dict) -> list:
    base = figstyle.SHOWCASE
    page = _page(work, cfg, base)
    record, f = page['record'], page['fusion']
    keep = _keep(work, cfg, base, record)
    if not f.x_view:
        raise ValueError(f'{base}: both views are lit along the same axis, so there is no x-lit view to show')

    rows, columns, gap = 2, 3, GAP
    width = (TEXTWIDTH - (columns - 1) * gap) / columns
    height = 1.45
    lh = f.letter_px
    win_w = WINDOW_WIDTH_LH * lh
    win_h = win_w * height / width
    kept_quads = np.array([b['quad'] for b, k in zip(record['boxes'], keep) if k >= .5], np.float64)
    centres = _to_working(kept_quads.mean(1), f.scale)
    left, top = _window(f, centres, win_w, win_h)
    box = (slice(top, top + int(win_h)), slice(left, left + int(win_w)))

    # the system's own quantities, whole page, then the window
    one = cv2.resize(page['gray1'], None, fx=f.scale, fy=f.scale, interpolation=cv2.INTER_AREA) \
        if f.scale < 1 else page['gray1']
    two = relief_lib.second_view_on_first(page['gray2'], page['registration'], one.shape[:2], f.scale)
    x_lit, y_lit = (f.s1, f.s2) if f.x_view == 1 else (f.s2, f.s1)
    sigma = relief_lib.MIXED_LH * lh
    m1, m2 = relief_lib.mixed_derivatives(f.s1, f.s2, f.x_view, sigma)
    signed = _correlation(m1, m2, f.both)
    sign = 1.0 if signed >= 0 else -1.0
    m_x, m_y = (m1, sign * m2) if f.x_view == 1 else (sign * m2, m1)   # d/dy of the x-lit, d/dx of the y-lit view
    assert abs(abs(signed) - f.agreement) < 1e-3
    local = sign * _correlation(m1[box], m2[box], f.both[box])
    # the registration alone: view 2 flattened as `fuse` does, before the tile-wise residual alignment
    unit2 = two.astype(np.float32) / 255
    paper2 = relief_lib.paper_mask(unit2, lh)
    s2_registered = relief_lib.flatten(unit2, paper2, relief_lib.FLATTEN_LH * lh)
    registered = abs(_correlation(*relief_lib.mixed_derivatives(f.s1, s2_registered, f.x_view, sigma),
                                  f.mask & paper2))
    relit = crops.relief_page(f, 'relit')

    def grey(s: np.ndarray, spread: float) -> np.ndarray:
        return relief_lib.to_uint8(s, f.both, True, spread)[box]

    ex, ey = grey(m_x, MIXED_SPREAD), grey(m_y, MIXED_SPREAD)
    composite = np.stack([ex, ey, ex], axis=-1)      # magenta: the x-lit view's derivative, green: the y-lit view's
    images = [one[box], two[box], grey(x_lit, SHADING_SPREAD), grey(y_lit, SHADING_SPREAD), composite, relit[box]]
    lit = {f.x_view: 'x', 3 - f.x_view: 'y'}
    tags = [[(f'R1, lit along ${lit[1]}$', 'k')],
            [(f'R2 aligned to R1, lit along ${lit[2]}$', 'k')],
            [('x-lit shading $s_x\\approx h_x$', 'k')],
            [('y-lit shading $s_y\\approx h_y$', 'k')],
            [('$\\partial_y s_x$', TAG_COLOURS[0]), ('$\\partial_x s_y$', TAG_COLOURS[1]), ('agree: grey', 'k')],
            [('relit relief', 'k')]]

    fig = plt.figure(figsize=(TEXTWIDTH, rows * (height + LABEL_ROOM)))
    for i, image in enumerate(images):
        r, c = divmod(i, columns)
        ax = _frame(fig, c * (width + gap), (rows - 1 - r) * (height + LABEL_ROOM), width, height)
        gray = image.ndim == 2
        _show(ax, image, width, height, None, **({'cmap': 'gray', 'vmin': 0, 'vmax': 255} if gray else {}))
        figstyle.panel_label(ax, 'abcdef'[i], x=0.08, y=1.0)
        _tag(ax, tags[i])
    path = _save(fig, 'relief')

    macros = {
        'ReliefAgreement': f'{f.agreement:.2f}',
        'ReliefAgreementRegistered': f'{registered:.2f}',
        'ReliefWindowAgreement': f'{local:.2f}',
        'ReliefRelativeSign': 'positive' if sign > 0 else 'negative',
        'ReliefShiftPx': f'{f.shift_px:.1f}',
        'ReliefShiftLh': f'{f.shift_px / lh:.3f}',
        'ReliefShare': f'{100 * f.share[f.both].mean():.0f}',
        'ReliefWindowShare': f'{100 * f.share[box].mean():.0f}',
        'ReliefXView': f'Rotation{f.x_view}',
        'ReliefYView': f'Rotation{3 - f.x_view}',
        'ReliefFlipped': 'flipped' if f.flipped else 'not flipped',
        'ReliefScale': f'{f.scale:.2f}',
        'ReliefLetterPx': f'{lh:.0f}',
        'ReliefWindowWidth': f'{WINDOW_WIDTH_LH:.1f}',
        'ReliefWindowHeight': f'{win_h / lh:.1f}',
        'ReliefWindowKept': figstyle.num(int(((centres[:, 0] >= left) & (centres[:, 0] < left + int(win_w))
                                              & (centres[:, 1] >= top) & (centres[:, 1] < top + int(win_h))).sum())),
    }
    return [path, figstyle.write_macros('relief', macros)]
