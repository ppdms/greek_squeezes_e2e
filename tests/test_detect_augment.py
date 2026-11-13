"""Training augmentation: oriented boxes follow the image through the geometric transform; draws are reproducible."""
from __future__ import annotations

import cv2
import numpy as np
import pytest
import torch

from trogs26.detect import augment

SIDE = 256


def page_with_letters():
    """White page with dark axis-aligned letters (cx, cy, w, h) and the matching normalised OBB targets."""
    letters = [(70, 80, 40, 24), (180, 70, 30, 50), (120, 190, 56, 20)]
    image = np.ones((SIDE, SIDE), np.float32)
    for cx, cy, w, h in letters:
        image[cy - h // 2:cy + h // 2, cx - w // 2:cx + w // 2] = 0.
    boxes = torch.tensor([[cx / SIDE, cy / SIDE, w / SIDE, h / SIDE, 0.] for cx, cy, w, h in letters])
    return torch.from_numpy(image)[None, None].repeat(1, 3, 1, 1), boxes


def dark_rectangle_near(mask: np.ndarray, centre) -> tuple:
    """Minimum-area rectangle of the dark connected component containing (or nearest to) `centre`."""
    count, labels = cv2.connectedComponents(mask.astype(np.uint8))
    best = None
    for index in range(1, count):
        ys, xs = np.nonzero(labels == index)
        distance = np.hypot(xs.mean() - centre[0], ys.mean() - centre[1])
        if best is None or distance < best[0]:
            best = (distance, np.column_stack([xs, ys]).astype(np.float32))
    (cx, cy), (w, h), _ = cv2.minAreaRect(best[1])
    return cx, cy, max(w, h), min(w, h)


@pytest.mark.parametrize('degrees,scale,shift', [(0., 1., (0., 0.)), (25., 1.2, (10., -6.)), (-50., .8, (-12., 9.))])
def test_boxes_follow_the_letters_through_rotation_scale_and_shift(degrees, scale, shift):
    images, boxes = page_with_letters()
    warped, fitted, valid = augment._apply_geometry(
        images, boxes, torch.zeros(len(boxes)), torch.tensor([np.deg2rad(degrees)], dtype=torch.float32),
        torch.tensor([scale]), torch.tensor([shift[0]]), torch.tensor([shift[1]]), torch.zeros(1, 2, 9, 9))
    assert valid.all()
    mask = warped[0, 0].numpy() < .5
    for box in fitted.numpy():
        cx, cy, w, h = box[:4] * SIDE
        found = dark_rectangle_near(mask, (cx, cy))
        assert found[:2] == pytest.approx((cx, cy), abs=1.5)
        assert found[2:] == pytest.approx((max(w, h), min(w, h)), abs=2.0)


def test_a_letter_pushed_out_of_the_tile_loses_its_box():
    images, boxes = page_with_letters()
    valid = augment._apply_geometry(
        images, boxes, torch.zeros(len(boxes)), torch.zeros(1), torch.ones(1), torch.tensor([-300.]), torch.zeros(1),
        torch.zeros(1, 2, 9, 9))[2]
    assert not valid.any()


def test_an_augmented_batch_is_well_formed_and_reproducible_from_the_seed():
    images, boxes = page_with_letters()

    def batch():
        return {'img': images.repeat(4, 1, 1, 1), 'bboxes': boxes.repeat(4, 1),
                'cls': torch.zeros(12, 1), 'batch_idx': torch.arange(4).repeat_interleave(3).float()}

    torch.manual_seed(5)
    first = augment.augment_batch(batch())
    torch.manual_seed(5)
    again = augment.augment_batch(batch())
    torch.manual_seed(6)
    other = augment.augment_batch(batch())
    assert torch.equal(first['img'], again['img']) and torch.equal(first['bboxes'], again['bboxes'])
    assert not torch.equal(first['img'], other['img'])
    assert first['img'].shape == (4, 3, SIDE, SIDE) and 0. <= first['img'].min() and first['img'].max() <= 1.
    assert len(first['bboxes']) == len(first['cls']) == len(first['batch_idx']) <= 12
    assert torch.isfinite(first['bboxes']).all() and (first['bboxes'][:, 2:4] > 0).all()


def test_the_jpeg_round_trip_leaves_a_flat_page_flat():
    flat = torch.full((2, 3, 32, 40), .6)
    assert (augment._jpeg_roundtrip(flat) - flat).abs().max() < 2 / 255
