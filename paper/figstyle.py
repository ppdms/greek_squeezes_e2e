"""The look the paper's figures share, and the reads several of them need (paper/make_figures.py draws them).

Sizes are in inches: report.tex's text block is TEXTWIDTH wide. Figure text is set in the paper's own fonts (Latin
Modern, located with kpsewhich). Figures of images are written as JPEG (`save_raster`) to keep the PDF small, plots
as PDF (`save_vector`). Numbers a figure's caption or the text quotes are written beside it as LaTeX macros
(`write_macros`), so the paper cannot disagree with the data.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

from trogs26 import data  # noqa: E402
from trogs26.layout import Chain, Work  # noqa: E402
from trogs26.pagenet.infer import posteriors  # noqa: E402

FIGS = Path(__file__).resolve().parent / 'figs'
TEXTWIDTH = 6.37  # inches: A4 with 0.95 in margins
SHOWCASE = 'IG_II2_1973a'  # the public squeeze the illustrations follow

ACCENT = '#1F4E79'   # the paper's accent colour
LETTER = '#2166AC'   # a token that is (or is read as) a letter
NONE = '#B2182B'     # a token that is (or is read as) none
MUTED = '#8C8C8C'
P_COLOUR = '#2166AC'  # the page-network chains
KEEP_CMAP = 'RdYlBu'  # keep probability: 0 red (none) .. 1 blue (letter)


def _font(name: str) -> str:
    """Register the TeX font file `name` with matplotlib; its family name."""
    path = subprocess.run(['kpsewhich', name], capture_output=True, text=True, check=True).stdout.strip()
    font_manager.fontManager.addfont(path)
    return font_manager.FontProperties(fname=path).get_name()


def setup() -> None:
    """Fonts, sizes and line widths of every figure."""
    serif = _font('lmroman10-regular.otf')
    for face in ('lmroman10-bold.otf', 'lmroman10-italic.otf'):
        _font(face)
    mono = _font('lmmono10-regular.otf')
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': [serif], 'font.monospace': [mono],
        'font.size': 9, 'axes.titlesize': 9, 'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'legend.fontsize': 8, 'mathtext.fontset': 'cm',
        'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.minor.width': 0.4,
        'ytick.minor.width': 0.4, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'lines.linewidth': 1.0,
        'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
        'figure.dpi': 100, 'savefig.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
        'image.interpolation': 'antialiased',
    })


def panel_label(ax, letter: str, x: float = -0.02, y: float = 1.02) -> None:
    """'(a)' at the top left of `ax`, outside the axes by default."""
    ax.text(x, y, f'({letter})', transform=ax.transAxes, ha='right', va='bottom', fontweight='bold')


def save_vector(fig, name: str) -> Path:
    """figs/<name>.pdf, without a creation date, so the same data gives the same bytes."""
    FIGS.mkdir(exist_ok=True)
    path = FIGS / f'{name}.pdf'
    fig.savefig(path, metadata={'CreationDate': None})
    plt.close(fig)
    return path


def save_raster(fig, name: str, dpi: int = 300, quality: int = 90) -> Path:
    """figs/<name>.jpg."""
    FIGS.mkdir(exist_ok=True)
    path = FIGS / f'{name}.jpg'
    fig.savefig(path, dpi=dpi, pil_kwargs={'quality': quality, 'optimize': True})
    plt.close(fig)
    return path


def num(n: int) -> str:
    """A LaTeX integer with a grouped thousands separator, 12{,}345."""
    return f'{n:,}'.replace(',', '{,}')


def write_macros(name: str, macros: dict[str, str], extra: list[str] = ()) -> Path:
    """figs/<name>.tex: one \\newcommand per macro (names of letters only), then the `extra` lines verbatim."""
    FIGS.mkdir(exist_ok=True)
    path = FIGS / f'{name}.tex'
    lines = [f'% Written by paper/make_figures.py {name}; do not edit.',
             *(f'\\newcommand{{\\{key}}}{{{value}}}' for key, value in macros.items()), *extra]
    path.write_text('\n'.join(lines) + '\n')
    return path


def folds(cfg: dict) -> dict[int, tuple[str, ...]]:
    """The P chains' folds: fold -> its squeezes."""
    return data.folds(cfg['data']['folds'])


def oof_chain(work: Work, cfg: dict, fold: int) -> Chain:
    """P chain `fold` over the squeezes it holds out."""
    return work.chain(fold, 'oof', folds(cfg)[fold])


def fold_of(cfg: dict, base: str) -> int:
    return data.fold_of(folds(cfg))[base]


def record(work: Work, cfg: dict, base: str) -> dict:
    """The detector record the P chain holding `base` out reads."""
    return json.loads(oof_chain(work, cfg, fold_of(cfg, base)).det(base).read_text())


def page_posteriors(work: Work, cfg: dict, fold: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """base -> (detection ids, (n, 28) log-probabilities) of P chain `fold`'s two networks combined, as the decoder
    reads them (`trogs26.pagenet.infer.posteriors`)."""
    return posteriors(oof_chain(work, cfg, fold), cfg['decode']['variants']['P']['pagenet'], cfg['alphabet'])


def keep_of(log_probs: np.ndarray) -> np.ndarray:
    """Keep probability 1 - P(none) of every token."""
    return 1.0 - np.exp(log_probs[:, -1].astype(np.float64))


def views(work: Work, base: str) -> tuple[np.ndarray, np.ndarray]:
    """The two photographs of `base`, uint8 grey."""
    return tuple(cv2.imread(str(work.image(base, v)), cv2.IMREAD_GRAYSCALE) for v in (1, 2))
