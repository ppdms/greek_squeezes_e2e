import cv2
import numpy as np
import pytest
import torch

from trogs26 import relief
from trogs26.dino.model import relit_mix
from trogs26.registration import Registration

LETTER = 40.0


def strokes(shape=(400, 520), seed=0, letter=LETTER, count=90):
    """A height field of grooves: random straight strokes about a tenth of a letter wide, depth 1."""
    rng = np.random.default_rng(seed)
    canvas = np.zeros(shape, np.float32)
    for _ in range(count):
        x0, y0 = rng.uniform(40, shape[1] - 40), rng.uniform(40, shape[0] - 40)
        angle = rng.uniform(0, np.pi)
        dx, dy = np.cos(angle) * letter / 2, np.sin(angle) * letter / 2
        cv2.line(canvas, (int(x0 - dx), int(y0 - dy)), (int(x0 + dx), int(y0 + dy)), 1.0, thickness=4)
    return -cv2.GaussianBlur(canvas, (0, 0), 1.5)


def shade(height, along_x, sign, gain=1.5):
    """uint8 view lit along x or y from one side: 0.5 (1 + gain * sign * directional derivative)."""
    d = cv2.Sobel(height, cv2.CV_32F, 1 if along_x else 0, 0 if along_x else 1, ksize=3, scale=1 / 8)
    return np.clip(255 * 0.5 * (1 + gain * sign * d), 0, 255).astype(np.uint8)


def identity():
    return Registration('similarity', {'matrix': [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]}, None, 'pass', 'test', {})


def correlation(a, b, mask):
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def test_an_identity_registration_leaves_the_second_view_in_place():
    view = np.random.default_rng(2).integers(0, 256, (130, 170), dtype=np.uint8)
    assert np.array_equal(relief.second_view_on_first(view, identity(), view.shape, 1.0), view)


@pytest.mark.parametrize('x_view, signs', [(1, (1, 1)), (2, (1, -1)), (1, (-1, 1))])
def test_two_views_lit_along_x_and_y_give_back_the_height(x_view, signs):
    h = strokes()
    one = shade(h, x_view == 1, signs[0])
    two = shade(h, x_view == 2, signs[1])
    f = relief.fuse(one, two, identity(), LETTER)
    assert f.x_view == x_view
    inner = f.mask.copy()
    inner[:60], inner[-60:], inner[:, :60], inner[:, -60:] = False, False, False, False
    k = relief.curvature(f)
    truth = cv2.Laplacian(cv2.GaussianBlur(h, (0, 0), relief.CURVATURE_LH * LETTER), cv2.CV_32F)
    assert abs(correlation(k, truth, inner)) > 0.8
    assert abs(correlation(relief.height(f), h - cv2.GaussianBlur(h, (0, 0), LETTER), inner)) > 0.8


def test_a_residual_shift_of_the_second_view_is_measured_and_removed():
    h = strokes(seed=1)
    one = shade(h, True, 1)
    moved = cv2.warpAffine(shade(h, False, 1), np.float32([[1, 0, 3], [0, 1, -2]]), one.shape[::-1],
                           borderMode=cv2.BORDER_REFLECT)        # view 2's content sits 3 px right, 2 px up
    f = relief.fuse(one, moved, identity(), LETTER)
    centre = f.shift[:, 200, 260]
    assert centre == pytest.approx([3, -2], abs=0.5)
    aligned = relief.fuse(one, shade(h, False, 1), identity(), LETTER)
    assert f.agreement == pytest.approx(aligned.agreement, abs=0.05)


def test_a_page_smaller_than_the_alignment_tiles_is_still_aligned():
    h = strokes((200, 520), seed=3, count=40)                   # 5 letters high: below the 8-letter coarse tiles
    one = shade(h, True, 1)
    moved = cv2.warpAffine(shade(h, False, 1), np.float32([[1, 0, 3], [0, 1, -2]]), one.shape[::-1],
                           borderMode=cv2.BORDER_REFLECT)
    f = relief.fuse(one, moved, identity(), LETTER)
    assert f.shift[:, 100, 260] == pytest.approx([3, -2], abs=0.5)


@pytest.mark.parametrize('sign', [1, -1])
def test_views_further_apart_than_a_stroke_align_whatever_their_relative_sign(sign):
    h = strokes(seed=7)
    one = shade(h, True, 1)
    moved = cv2.warpAffine(shade(h, False, sign), np.float32([[1, 0, 0], [0, 1, 13]]), one.shape[::-1],
                           borderMode=cv2.BORDER_REFLECT)        # a third of a letter down: strokes do not overlap
    f = relief.fuse(one, moved, identity(), LETTER)
    assert f.shift[:, 200, 260] == pytest.approx([0, 13], abs=1.0)
    assert f.agreement > 0.5


def test_a_stitched_part_placed_letters_apart_is_found_by_the_coarse_pass():
    letter = 20.0
    h = strokes((640, 640), seed=4, letter=letter, count=500)
    one = shade(h, True, 1)
    two = shade(h, False, 1)
    top = cv2.warpAffine(two, np.float32([[1, 0, 30], [0, 1, 0]]), two.shape[::-1], borderMode=cv2.BORDER_REFLECT)
    two[:320] = top[:320]                                       # the upper part of view 2 sits 1.5 letters right
    f = relief.fuse(one, two, identity(), letter)
    assert f.shift[:, 120, 320] == pytest.approx([30, 0], abs=1.0)
    assert f.shift[:, 520, 320] == pytest.approx([0, 0], abs=1.0)


def test_view_two_fades_out_where_it_shows_something_else():
    h = strokes(seed=5)
    other = strokes(seed=6)
    one = shade(h, True, 1)
    two = shade(h, False, 1)
    two[:, 260:] = shade(other, False, 1)[:, 260:]              # right half of view 2: another page
    f = relief.fuse(one, two, identity(), LETTER)
    assert f.share[100:300, 40:180].mean() > 0.9
    assert f.share[100:300, 360:480].mean() < 0.1
    assert np.abs(f.q[100:300, 360:480]).mean() < 0.1 * np.abs(f.q[100:300, 40:180]).mean()   # q is view 2's part


def test_turning_the_lights_by_a_third_moves_each_relit_channel_to_the_next():
    f64 = {'dtype': torch.float64}
    still, one, whole = torch.tensor(0.0, **f64), torch.tensor(1.0, **f64), torch.ones(2, **f64)
    assert torch.allclose(relit_mix(still, whole, one), torch.eye(3, **f64), atol=1e-12)
    # a relief with gradient (p, q) = (0.3, -0.2), lit from 90, 210 and 330 degrees
    azimuths = torch.tensor(np.radians(relief.RELIT_AZIMUTHS), **f64)
    shading = -(0.3 * torch.cos(azimuths) - 0.2 * torch.sin(azimuths))
    turned = relit_mix(torch.tensor(np.radians(120.0), **f64), whole, one) @ shading
    assert torch.allclose(turned, shading.roll(-1), atol=1e-12)
    # without q, what is left is the shading of (p, 0)
    faded = relit_mix(still, torch.tensor([1.0, 0.0], **f64), one) @ shading
    assert torch.allclose(faded, -0.3 * torch.cos(azimuths), atol=1e-12)
