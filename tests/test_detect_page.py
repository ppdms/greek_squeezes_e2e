"""Page geometry of the detector: scale frames, tiling and ownership, box canonicalisation, training tiles/targets."""
from __future__ import annotations

import json
import math

import cv2
import numpy as np
import pytest

from trogs26.detect import annotations, page, tiles


def letter(cx, cy, width, height, angle):
    """A rectangle's corners in the official order NW, SW, SE, NE."""
    return page.obb_to_quad([cx, cy, width, height, angle])


def test_windows_cover_every_pixel_and_every_centre_is_owned_by_a_window_that_covers_it():
    for size in [(100, 80), (1280, 1280), (1281, 700), (3000, 2500)]:
        windows = page.tile_windows(size, 1280, .25)
        covered = np.zeros((size[1], size[0]), bool)
        for x0, y0, x1, y1 in windows:
            assert 0 <= x0 < x1 <= size[0] and 0 <= y0 < y1 <= size[1] and x1 - x0 <= 1280 and y1 - y0 <= 1280
            covered[y0:y1, x0:x1] = True
        assert covered.all()
        rng = np.random.default_rng(0)
        for point in rng.uniform([0, 0], size, size=(200, 2)):
            x0, y0, x1, y1 = windows[page.tile_owner(point, windows)]
            assert x0 <= point[0] <= x1 and y0 <= point[1] <= y1


def test_a_detection_is_owned_by_the_window_with_the_nearest_centre():
    windows = [(0, 0, 100, 100), (50, 0, 150, 100)]
    assert page.tile_owner((60, 50), windows) == 0 and page.tile_owner((90, 50), windows) == 1
    assert page.tile_owner((200, 50), windows) == -1


def test_scale_frame_inverts_and_scales_letters_to_the_target_height():
    frame = page.ScaleFrame.from_height((4000, 3000), 80., 40.)
    assert frame.size == (2000, 1500)
    points = np.array([[0., 0.], [123.4, 567.8], [3999., 2999.]])
    assert frame.inverse(frame.apply(points)) == pytest.approx(points)
    assert page.ScaleFrame.from_dict(frame.to_dict()) == frame
    with pytest.raises(ValueError):
        page.ScaleFrame.from_height((10, 10), 0., 40.)


@pytest.mark.parametrize('angle', [-0.9, -0.3, 0., 0.25, 0.7])
def test_canonical_quads_are_idempotent_official_order_rectangles_with_the_letter_height(angle):
    quad = letter(500., 300., 40., 90., angle)
    canonical = page.canonical_quad(quad[::-1], angle)  # any corner order in, official order out
    assert canonical == pytest.approx(quad, abs=1e-3)
    assert page.canonical_quad(canonical, angle) == pytest.approx(canonical, abs=1e-3)
    edges = np.roll(canonical, -1, axis=0) - canonical
    assert edges[0, 0] * edges[1, 1] - edges[0, 1] * edges[1, 0] < 0
    assert page.median_height([quad], angle) == pytest.approx(90., abs=1e-3)


def test_reading_angle_is_the_common_tilt_of_the_letters_whatever_their_aspect():
    quads = [letter(100. * i, 50., 30., 60., .3) for i in range(1, 6)] + [letter(80., 300., 60., 30., .3)]
    assert page.estimate_reading_angle(quads) == pytest.approx(.3, abs=1e-3)
    assert page.estimate_reading_angle([]) == 0.


def test_mono_pages_are_the_normalised_view_in_three_channels_and_packed_pages_end_in_the_minimum():
    gray = np.random.default_rng(0).integers(0, 255, (120, 160), dtype=np.uint8)
    frame = page.ScaleFrame((160, 120), (80, 60))
    mono = page.make_page(gray, None, None, frame, packed=False)
    assert mono.shape == (60, 80, 3) and (mono[..., 0] == mono[..., 1]).all() and (mono[..., 1] == mono[..., 2]).all()
    with pytest.raises(ValueError):
        page.make_page(gray, None, None, frame, packed=True)


def test_merged_is_a_property_of_either_file_name():
    assert page.is_merged('a_Rotation1_300dpi.png', 'a_Merged_Rotation2_300dpi.png')
    assert not page.is_merged('/data/IG_II2_1_Rotation1.png', 'IG_II2_1_Rotation2.png')


def test_annotation_boxes_smaller_than_nine_pixels_are_dropped_like_the_organisers_parser_does(tmp_path):
    big, small = '0 0 0 10 10 10 10 0', '0 0 0 2 2 2 2 0'
    path = tmp_path / 'x_Rotation1_300dpi_letters.txt'
    path.write_text(f'# Baseline angle\n0.25\n# Boxes: x1 y1 x2 y2 x3 y3 x4 y4\n{big}\n{small}\n# Transcript\nAB\n')
    quads, angle = annotations.read_letters(path)
    assert quads.shape == (1, 4, 2) and angle == .25


def test_label_lines_clip_letters_at_the_tile_edge_and_drop_slivers():
    side = 100
    inside = letter(50., 50., 20., 40., 0.)
    edge = letter(98., 50., 20., 40., 0.)  # half of it is cut off by the tile
    sliver = letter(109.9, 50., 20., 40., 0.)  # 0.1 px of it remains: 4 square pixels
    lines = tiles.label_lines([inside, edge, sliver], 0., (0, 0, side, side), side)
    assert len(lines) == 2
    clipped = np.array(lines[1].split()[1:], float).reshape(4, 2) * side
    assert clipped[:, 0].max() == pytest.approx(side) and cv2.contourArea(clipped.astype(np.float32)) == pytest.approx(
        (side - 88) * 40, rel=1e-3)


@pytest.mark.parametrize('angle', [-0.7, -0.2, 0., 0.4, 1.0])
def test_obb_targets_are_the_labelled_rectangle(angle):
    side = 320
    quad = letter(160., 150., 36., 80., angle)
    line = '0 ' + ' '.join(f'{v:.8f}' for v in (quad / side).ravel())
    classes, boxes = tiles.obb_targets(line + '\n', side)
    assert classes.tolist() == [[0.]] and boxes.shape == (1, 5)
    cx, cy, width, height, theta = boxes[0] * [side, side, side, side, 1]
    assert width >= height and -math.pi / 4 <= theta < 3 * math.pi / 4
    rebuilt = page.obb_to_quad([cx, cy, width, height, theta])
    # the same rectangle: every rebuilt corner is a corner of the label (corner order is free)
    for corner in rebuilt:
        assert np.abs(quad - corner).sum(1).min() < 1e-2


def test_held_out_squeezes_contribute_only_their_pack_tiles_and_train_tiles_never_include_them(tmp_path):
    folds = {0: ('a',), 1: ('b',), 2: ('c',)}
    (tmp_path / 'squeezes').mkdir()
    for base in 'abc':
        record = {'packed': True, 'tiles': {view: [f'{base}__{view}__t000', f'{base}__{view}__t001']
                                            for view in tiles.VIEWS}}
        (tmp_path / 'squeezes' / f'{base}.json').write_text(json.dumps(record))
    train, held = tiles.split_tiles(tmp_path, folds, 1)
    assert held == ['b__pack__t000', 'b__pack__t001']
    assert len(train) == 12 and not any(name.startswith('b__') for name in train) and train == sorted(train)
