"""Training-time augmentation of the detector, applied to a batch of tiles on the device the batch lives on.

Ultralytics' own transforms are off. Every training batch gets, per tile: a random rotation (+-60 degrees), scale
(0.6-1.6) and shift (+-10%) with white borders, in 15% of the tiles also a smooth elastic displacement (9x9 control
grid, up to 6 px); the oriented boxes follow (four corners warped, clamped to the tile, refitted by the minimum-area
rectangle; a box smaller than one pixel is dropped). Then, per channel gamma 0.7-1.4 and contrast 0.75-1.25, a swap
of the red and green channel (p 0.5), a mono dropout (p 0.2: one of the two view channels copied to all three), a
Gaussian blur (p 0.2, sigma 0.4-1.5) and a JPEG round trip (p 0.25, quality 35-90, no chroma subsampling) computed
with a block DCT on the device. No flips.

The order of the random draws is part of the recipe: with the same seed and batches the same tiles are produced.
Run on CUDA tensors for training; the code is device-agnostic (CPU for smoke runs).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

VERSION = 'det_gpu_augfull_v1'
POLICY = {
    'geometry': {
        'rotation_degrees': [-60.0, 60.0],
        'scale': [0.6, 1.6],
        'translation_fraction': [-0.1, 0.1],
        'elastic_probability': 0.15,
        'elastic_control_grid': [9, 9],
        'elastic_max_displacement_px': 6.0,
        'interpolation': 'bilinear',
        'border': 'white',
        'obb_policy': 'warp four corners, clamp to tile, refit minimum-area rectangle',
    },
    'photometry': {
        'gamma_per_channel': [0.7, 1.4],
        'contrast_per_channel': [0.75, 1.25],
        'rgb_swap_probability': 0.5,
        'mono_dropout_probability': 0.2,
        'mono_channels': ['R', 'G'],
        'mono_policy': 'duplicate one uniformly selected view channel to all channels',
        'gaussian_blur_probability': 0.2,
        'gaussian_blur_kernel': 5,
        'gaussian_blur_sigma': [0.4, 1.5],
        'jpeg_probability': 0.25,
        'jpeg_quality': [35, 90],
        'jpeg_chroma_subsampling': 'none',
    },
    'flips': False,
}


def _uniform(shape, low, high, *, device, dtype):
    return torch.rand(shape, device=device, dtype=dtype) * (high - low) + low


def _obb_corners(boxes, width, height):
    cx, cy = boxes[:, 0] * width, boxes[:, 1] * height
    box_w, box_h = boxes[:, 2] * width, boxes[:, 3] * height
    angle = boxes[:, 4]
    cos, sin = angle.cos(), angle.sin()
    v1 = torch.stack((box_w * cos / 2, box_w * sin / 2), dim=1)
    v2 = torch.stack((-box_h * sin / 2, box_h * cos / 2), dim=1)
    center = torch.stack((cx, cy), dim=1)
    return torch.stack((center + v1 + v2, center + v1 - v2,
                        center - v1 - v2, center - v1 + v2), dim=1)


def _fit_obb(points, width, height):
    """Minimum-area rectangle of each quadrilateral, as normalised (cx, cy, w, h, angle) with w >= h."""
    edges = torch.roll(points, shifts=-1, dims=1) - points
    angles = torch.atan2(edges[:, :, 1], edges[:, :, 0])
    cos, sin = angles.cos().unsqueeze(2), angles.sin().unsqueeze(2)
    x, y = points[:, None, :, 0], points[:, None, :, 1]
    x_rot = x * cos + y * sin
    y_rot = -x * sin + y * cos
    x_min, x_max = x_rot.amin(dim=2), x_rot.amax(dim=2)
    y_min, y_max = y_rot.amin(dim=2), y_rot.amax(dim=2)
    widths, heights = x_max - x_min, y_max - y_min
    best = (widths * heights).argmin(dim=1, keepdim=True)

    def gather(value):
        return value.gather(1, best).squeeze(1)

    box_w, box_h = gather(widths), gather(heights)
    angle = gather(angles)
    cx_rot, cy_rot = (gather(x_min) + box_w / 2), (gather(y_min) + box_h / 2)
    cos, sin = angle.cos(), angle.sin()
    cx = cx_rot * cos - cy_rot * sin
    cy = cx_rot * sin + cy_rot * cos
    swap = box_w < box_h
    final_w = torch.where(swap, box_h, box_w)
    final_h = torch.where(swap, box_w, box_h)
    angle = angle + swap.to(angle.dtype) * (torch.pi / 2)
    angle = torch.remainder(angle + torch.pi / 4, torch.pi) - torch.pi / 4
    return torch.stack((cx / width, cy / height, final_w / width, final_h / height, angle), dim=1)


def _apply_geometry(images, boxes, batch_idx, angle, scale, tx, ty, control):
    batch, _, height, width = images.shape
    device, dtype = images.device, images.dtype
    cos, sin = angle.cos(), angle.sin()
    inverse_scale = scale.reciprocal()
    tx_norm, ty_norm = tx * (2 / width), ty * (2 / height)
    theta = torch.empty((batch, 2, 3), device=device, dtype=dtype)
    theta[:, 0, 0] = cos * inverse_scale
    theta[:, 0, 1] = sin * inverse_scale
    theta[:, 0, 2] = -(cos * inverse_scale * tx_norm + sin * inverse_scale * ty_norm)
    theta[:, 1, 0] = -sin * inverse_scale
    theta[:, 1, 1] = cos * inverse_scale
    theta[:, 1, 2] = -(-sin * inverse_scale * tx_norm + cos * inverse_scale * ty_norm)
    # The inverse image map samples at A^-1(y - d(y)). OBB corners therefore
    # solve the matching forward equation y = A(x) + d(y), not d(A(x)).
    grid = F.affine_grid(theta, images.shape, align_corners=False)
    flow = F.interpolate(control, size=(height, width), mode='bilinear', align_corners=False)
    flow_x, flow_y = flow[:, 0] * (2 / width), flow[:, 1] * (2 / height)
    source_dx = (cos * inverse_scale)[:, None, None] * flow_x + \
                (sin * inverse_scale)[:, None, None] * flow_y
    source_dy = (-sin * inverse_scale)[:, None, None] * flow_x + \
                (cos * inverse_scale)[:, None, None] * flow_y
    grid = grid - torch.stack((source_dx, source_dy), dim=-1)
    warped = F.grid_sample(images - 1, grid, mode='bilinear', padding_mode='zeros', align_corners=False) + 1
    warped = warped.clamp_(0, 1)

    if boxes.numel() == 0:
        return warped, boxes, torch.zeros((0,), device=device, dtype=torch.bool)

    indices = batch_idx.long()
    points = _obb_corners(boxes, width, height)
    center_x, center_y = (width - 1) / 2, (height - 1) / 2
    points_centered = points - points.new_tensor([center_x, center_y])
    box_cos, box_sin = cos[indices, None], sin[indices, None]
    box_scale = scale[indices, None]  # one scale per box, shared by its four corners
    x = box_scale * (box_cos * points_centered[:, :, 0] -
                     box_sin * points_centered[:, :, 1]) + center_x + tx[indices, None]
    y = box_scale * (box_sin * points_centered[:, :, 0] +
                     box_cos * points_centered[:, :, 1]) + center_y + ty[indices, None]
    affine_points = torch.stack((x, y), dim=-1)
    points = affine_points
    box_control = control.index_select(0, indices)
    for _ in range(8):
        point_grid = torch.stack((2 * (points[:, :, 0] + .5) / width - 1,
                                  2 * (points[:, :, 1] + .5) / height - 1), dim=-1)
        displacement = F.grid_sample(box_control, point_grid.unsqueeze(1), mode='bilinear',
                                     padding_mode='border', align_corners=False)
        displacement = displacement.squeeze(2).transpose(1, 2)
        points = affine_points + displacement
    points[..., 0].clamp_(0, width)
    points[..., 1].clamp_(0, height)
    fitted = _fit_obb(points, width, height)
    valid = torch.isfinite(fitted).all(dim=1) & (fitted[:, 2] * width >= 1) & (fitted[:, 3] * height >= 1)
    return warped, fitted, valid


def _gaussian_blur(images):
    batch, channels, height, width = images.shape
    sigma = _uniform((batch, 1, 1), .4, 1.5, device=images.device, dtype=images.dtype)
    axis = torch.arange(-2, 3, device=images.device, dtype=images.dtype)
    kernel = torch.exp(-(axis[None, :, None] ** 2 + axis[None, None, :] ** 2) / (2 * sigma.square()))
    kernel = kernel / kernel.sum(dim=(1, 2), keepdim=True)
    weights = kernel.repeat_interleave(channels, dim=0).unsqueeze(1)
    blurred = F.conv2d(images.reshape(1, batch * channels, height, width), weights, padding=2,
                       groups=batch * channels)
    blurred = blurred.reshape_as(images)
    selected = torch.rand((batch, 1, 1, 1), device=images.device) < .2
    return torch.where(selected, blurred, images)


_LUMA = [[16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55],
         [14, 13, 16, 24, 40, 57, 69, 56], [14, 17, 22, 29, 51, 87, 80, 62],
         [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92],
         [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]]
_CHROMA = [[17, 18, 24, 47, 99, 99, 99, 99], [18, 21, 26, 66, 99, 99, 99, 99],
           [24, 26, 56, 99, 99, 99, 99, 99], [47, 66, 99, 99, 99, 99, 99, 99],
           [99, 99, 99, 99, 99, 99, 99, 99], [99, 99, 99, 99, 99, 99, 99, 99],
           [99, 99, 99, 99, 99, 99, 99, 99], [99, 99, 99, 99, 99, 99, 99, 99]]


def _jpeg_roundtrip(images):
    """JPEG compression and decompression (4:4:4) of every tile at its own random quality."""
    batch, channels, height, width = images.shape
    pad_h, pad_w = (-height) % 8, (-width) % 8
    source = F.pad(images, (0, pad_w, 0, pad_h), mode='replicate') if pad_h or pad_w else images
    padded_h, padded_w = source.shape[-2:]
    red, green, blue = source.unbind(dim=1)
    y = .299 * red + .587 * green + .114 * blue
    cb = -.168736 * red - .331264 * green + .5 * blue
    cr = .5 * red - .418688 * green - .081312 * blue
    ycbcr = torch.stack((y, cb, cr), dim=1) * 255
    ycbcr[:, 0] -= 128

    blocks = ycbcr.reshape(batch, channels, padded_h // 8, 8, padded_w // 8, 8).permute(0, 1, 2, 4, 3, 5)
    pos = torch.arange(8, device=images.device, dtype=images.dtype)
    freq = pos[:, None]
    dct = torch.cos(torch.pi * (2 * pos[None, :] + 1) * freq / 16)
    dct = dct * (2 / 8) ** .5
    dct[0] = (1 / 8) ** .5
    coeff = torch.matmul(torch.matmul(dct, blocks), dct.T)

    luma = torch.tensor(_LUMA, device=images.device, dtype=images.dtype)
    chroma = torch.tensor(_CHROMA, device=images.device, dtype=images.dtype)
    tables = torch.stack((luma, chroma, chroma), dim=0)[None, :, None, None]
    quality = _uniform((batch,), 35, 90, device=images.device, dtype=images.dtype)
    factor = torch.where(quality < 50, 5000 / quality, 200 - 2 * quality)
    quant = torch.floor((tables * factor[:, None, None, None, None, None] + 50) / 100).clamp_(1, 255)
    coeff = torch.round(coeff / quant) * quant
    restored = torch.matmul(torch.matmul(dct.T, coeff), dct)
    restored = restored.permute(0, 1, 2, 4, 3, 5).reshape(batch, channels, padded_h, padded_w)
    restored[:, 0] += 128
    y, cb, cr = (restored[:, index] / 255 for index in range(3))
    red = y + 1.402 * cr
    green = y - .344136 * cb - .714136 * cr
    blue = y + 1.772 * cb
    return torch.stack((red, green, blue), dim=1)[:, :, :height, :width].clamp_(0, 1)


def augment_batch(batch: dict) -> dict:
    """Augment `batch` in place and return it: `img` (B, 3, H, W) float RGB in [0, 1], `bboxes` (n, 5) normalised
    cx, cy, w, h, angle, `cls` (n, 1) and `batch_idx` (n,), the latter three filtered to the boxes that survive."""
    images = batch['img']
    if images.ndim != 4 or images.shape[1] != 3 or not images.is_floating_point():
        raise ValueError('augmentation expects normalized float RGB batches')
    batch_size, _, height, width = images.shape
    device, dtype = images.device, images.dtype
    degrees = _uniform((batch_size,), -60., 60., device=device, dtype=dtype)
    angle = degrees * (torch.pi / 180)
    scale = _uniform((batch_size,), .6, 1.6, device=device, dtype=dtype)
    tx = _uniform((batch_size,), -.1 * width, .1 * width, device=device, dtype=dtype)
    ty = _uniform((batch_size,), -.1 * height, .1 * height, device=device, dtype=dtype)
    control = _uniform((batch_size, 2, 9, 9), -6., 6., device=device, dtype=dtype)
    elastic = torch.rand((batch_size, 1, 1, 1), device=device) < .15
    control = control * elastic.to(dtype)
    boxes, batch_idx = batch['bboxes'], batch['batch_idx']
    images, fitted, valid = _apply_geometry(images, boxes, batch_idx, angle, scale, tx, ty, control)
    batch['img'] = images
    if boxes.numel():
        batch['bboxes'] = fitted[valid]
        batch['cls'] = batch['cls'][valid]
        batch['batch_idx'] = batch_idx[valid]

    gamma = _uniform((batch_size, 3, 1, 1), .7, 1.4, device=device, dtype=dtype)
    contrast = _uniform((batch_size, 3, 1, 1), .75, 1.25, device=device, dtype=dtype)
    images = ((images.clamp(0, 1).pow(gamma) - .5) * contrast + .5).clamp_(0, 1)
    swap = torch.rand((batch_size, 1, 1, 1), device=device) < .5
    images = torch.where(swap, images[:, [1, 0, 2]], images)
    mono = torch.rand((batch_size,), device=device) < .2
    channel = torch.randint(0, 2, (batch_size,), device=device)
    mono_image = images[torch.arange(batch_size, device=device), channel].unsqueeze(1).expand_as(images)
    images = torch.where(mono[:, None, None, None], mono_image, images)
    images = _gaussian_blur(images)
    jpeg = _jpeg_roundtrip(images)
    jpeg_selected = torch.rand((batch_size, 1, 1, 1), device=device) < .25
    batch['img'] = torch.where(jpeg_selected, jpeg, images).contiguous()
    return batch
