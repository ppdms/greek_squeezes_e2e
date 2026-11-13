"""Preparing crops for the DINOv3 encoder of the page network: normalisation, augmentation and reading-time views.

The input normalisation is the pretrained model's `preprocessor_config.json` (see `pretrained.py`). `prepare` turns
uint8 crops into normalised floats, with the augmentation of training (`relief_style`, brightness, affine, `erase`)
or the fixed rotation and zoom of one reading view.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from ..relief import RELIT_AZIMUTHS


def normalisation(directory: Path, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-channel mean and std, shaped for (n, 3, h, w) batches, from the `preprocessor_config.json` in `directory`."""
    config = json.loads((directory / 'preprocessor_config.json').read_text())
    return (torch.tensor(config['image_mean'], device=device).view(1, 3, 1, 1),
            torch.tensor(config['image_std'], device=device).view(1, 3, 1, 1))


def rotate_zoom(x: torch.Tensor, angles: torch.Tensor, scales: torch.Tensor,
                shifts: torch.Tensor | None = None) -> torch.Tensor:
    """(n, 3, h, w) crops each rotated by its angle (radians) and zoomed by its scale about the centre, then moved
    by its shift ((n, 2), fractions of the crop's side along x and y), in one affine resampling with border
    replication."""
    affine = torch.zeros(len(x), 2, 3, device=x.device)
    affine[:, 0, 0] = torch.cos(angles) / scales
    affine[:, 0, 1] = -torch.sin(angles) / scales
    affine[:, 1, 0] = torch.sin(angles) / scales
    affine[:, 1, 1] = torch.cos(angles) / scales
    if shifts is not None:
        affine[:, :, 2] = 2 * shifts
    grid = torch.nn.functional.affine_grid(affine, x.shape, align_corners=False)
    return torch.nn.functional.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=False)


def _blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Gaussian blur of width `sigma` pixels (separable, reflecting borders)."""
    radius = max(1, math.ceil(3 * sigma))
    taps = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    kernel = torch.exp(-taps ** 2 / (2 * sigma ** 2))
    kernel = (kernel / kernel.sum()).view(1, 1, -1)
    n, c, h, w = x.shape
    x = torch.nn.functional.pad(x.reshape(n * c, 1, h, w), (radius, radius, radius, radius), mode='reflect')
    x = torch.nn.functional.conv2d(x, kernel.unsqueeze(2))
    return torch.nn.functional.conv2d(x, kernel.unsqueeze(3)).view(n, c, h, w)


def relit_mix(turn: torch.Tensor, gains: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
    """(3, 3) map of the relit channels' shadings about mid-grey to those of the same relief with its lights turned
    by `turn` radians, its gradient components p and q scaled by `gains` (2,) and everything by `depth`."""
    azimuths = torch.tensor([math.radians(a) for a in RELIT_AZIMUTHS], device=gains.device, dtype=gains.dtype)
    render = -torch.stack([torch.cos(azimuths), torch.sin(azimuths)], 1)   # channels from (p, q)
    recover = render.T * (2 / 3)                                           # (p, q) from channels
    rotation = torch.stack([torch.stack([torch.cos(turn), torch.sin(turn)]),
                            torch.stack([-torch.sin(turn), torch.cos(turn)])])
    # render @ recover projects on the channels a gradient can give; the 1/3 keeps what they share besides
    return depth * (render @ rotation @ torch.diag(gains) @ recover + 1 / 3)


def relief_style(x: torch.Tensor, style: dict) -> torch.Tensor:
    """Relit crops (n, 3, h, w in [0, 1]) as another squeeze might show the same relief; one draw for the batch.

    A relit channel is the relief's shading under one of three canonical lights, a fixed combination of the height
    gradient (p, q) about mid-grey (`relief.relit`). So the gradient is recovered from the channels, changed, and
    lit again (exact up to the channels' separate scalings to uint8): the lights turn by U(-rig, rig) degrees; p and
    q are each scaled by exp(U(-tilt, tilt)), as under lights of unequal elevation; with probability `fade` one of
    them is scaled by U(0, 1) more, as where the second view is faded out; everything is scaled by U(depth), a
    shallower or deeper relief. Then a Gaussian blur of U(0, blur) pixels (applied above 0.2) and Gaussian noise
    of standard deviation U(0, noise).
    """
    device = x.device
    turn = math.radians(style['rig']) * (2 * torch.rand((), device=device) - 1)
    gains = torch.exp(style['tilt'] * (2 * torch.rand(2, device=device) - 1))
    if torch.rand((), device=device) < style['fade']:
        gains[torch.randint(2, (), device=device)] *= torch.rand((), device=device)
    low, high = style['depth']
    depth = low + (high - low) * torch.rand((), device=device)
    x = torch.einsum('kj,njhw->nkhw', relit_mix(turn, gains, depth), x - 0.5)
    sigma = style['blur'] * float(torch.rand((), device=device))
    if sigma > 0.2:
        x = _blur(x, sigma)
    x = x + style['noise'] * torch.rand((), device=device) * torch.randn_like(x)
    return (x + 0.5).clamp_(0, 1)


def erase(x: torch.Tensor, probability: float, area: tuple[float, float]) -> torch.Tensor:
    """Crops with, each with `probability`, one rectangle of U(area) of the crop (sides in a ratio up to two)
    set to mid-grey: flat paper where a stroke may have been."""
    n, _, h, w = x.shape
    device = x.device
    low, high = area
    share = low + (high - low) * torch.rand(n, device=device)
    ratio = torch.exp(math.log(2) * (2 * torch.rand(n, device=device) - 1))
    tall, wide = (share * ratio).sqrt().clamp(max=1), (share / ratio).sqrt().clamp(max=1)
    top, left = torch.rand(n, device=device) * (1 - tall), torch.rand(n, device=device) * (1 - wide)
    yy = ((torch.arange(h, device=device) + 0.5) / h).view(1, h, 1)
    xx = ((torch.arange(w, device=device) + 0.5) / w).view(1, 1, w)
    inside = ((yy >= top.view(n, 1, 1)) & (yy < (top + tall).view(n, 1, 1))
              & (xx >= left.view(n, 1, 1)) & (xx < (left + wide).view(n, 1, 1)))
    inside &= (torch.rand(n, device=device) < probability).view(n, 1, 1)
    return torch.where(inside.unsqueeze(1), torch.full_like(x, 0.5), x)


def prepare(batch: torch.Tensor, device: torch.device, normalise: tuple[torch.Tensor, torch.Tensor],
            augment: dict | None = None, view: tuple[float, float] = (0.0, 1.0)) -> torch.Tensor:
    """uint8 (n, 3, 224, 224) crops -> normalised float on `device`.

    With `augment`: first, with `relief` (relit crops only), the batch as another squeeze might show it
    (`relief_style`); per-crop brightness gain 1 + gain * U(-.5, .5) and bias bias * U(-.5, .5), then a random
    rotation within +-rotation_degrees, zoom within `scale` and, with `shift`, a move by U(-shift, shift) of the
    crop's side along each axis; last, with `erase` ({'probability', 'area'}), `erase`. `view` (degrees, scale)
    rotates and zooms every crop alike (a fixed view at reading time; (0, 1) leaves the crops as they are). Crops
    are never flipped.
    """
    x = batch.to(device, non_blocking=True).float().div_(255)
    if augment:
        n = len(x)
        if augment.get('relief'):
            x = relief_style(x, augment['relief'])
        gain = 1 + augment['gain'] * (torch.rand(n, 1, 1, 1, device=device) - 0.5)
        bias = augment['bias'] * (torch.rand(n, 1, 1, 1, device=device) - 0.5)
        x.mul_(gain).add_(bias).clamp_(0, 1)
        angles = (torch.rand(n, device=device) * 2 - 1) * math.radians(augment['rotation_degrees'])
        low, high = augment['scale']
        scales = low + torch.rand(n, device=device) * (high - low)
        shifts = (augment['shift'] * (2 * torch.rand(n, 2, device=device) - 1) if augment.get('shift') else None)
        x = rotate_zoom(x, angles, scales, shifts)
        if augment.get('erase'):
            x = erase(x, augment['erase']['probability'], augment['erase']['area'])
    degrees, scale = view
    if (degrees, scale) != (0.0, 1.0):
        n = len(x)
        x = rotate_zoom(x, torch.full((n,), math.radians(degrees), device=device),
                        torch.full((n,), float(scale), device=device))
    mean, std = normalise
    return x.sub_(mean).div_(std)
