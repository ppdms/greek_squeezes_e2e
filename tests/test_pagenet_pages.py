"""The page network's targets and attention neighbourhoods (trogs26/pagenet/pages.py)."""
from __future__ import annotations

import numpy as np

from trogs26.pagenet import pages


def test_claim_by_p_gives_a_letter_to_the_surer_of_two_detections_on_it():
    ious = np.array([[0.8], [0.5]])
    p = np.array([0.6, 0.9])
    assert pages.claims(ious, p, 0.3, 'p') == {1: 0}
    assert pages.claims(ious, p, 0.3, 'iou') == {0: 0}


def test_claim_by_p_claims_as_many_letters_as_possible_before_preferring_the_surer_detection():
    # detection 1 is surer but overlaps letter 0 only; detection 0 overlaps both letters, so it takes letter 1
    ious = np.array([[0.6, 0.4], [0.7, 0.0]])
    assert pages.claims(ious, np.array([0.5, 0.99]), 0.3, 'p') == {0: 1, 1: 0}


def test_claims_need_the_overlap_threshold():
    ious = np.array([[0.29, 0.0], [0.0, 0.3]])
    assert pages.claims(ious, np.array([0.9, 0.1]), 0.3, 'p') == {1: 1}


def test_neighbours_count_distance_across_rows_v_weight_times():
    # token 1 is one letter height along the row, token 2 0.7 across it
    u, v = np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 0.7])
    assert pages.neighbours(u, v, 2, 2.0)[0][0].tolist() == [0, 1]
    assert pages.neighbours(u, v, 2, 1.0)[0][0].tolist() == [0, 2]


def test_every_token_comes_first_among_its_neighbours_even_at_a_shared_position():
    idx, mask = pages.neighbours(np.zeros(3), np.zeros(3), 3, 2.0)
    assert idx[:, 0].tolist() == [0, 1, 2]
    assert [sorted(row) for row in idx.tolist()] == [[0, 1, 2]] * 3
    assert mask.all()


def test_a_page_with_fewer_tokens_than_neighbours_pads_with_the_token_itself_masked():
    idx, mask = pages.neighbours(np.array([0.0, 1.0]), np.zeros(2), 4, 2.0)
    assert idx.tolist() == [[0, 1, 0, 0], [1, 0, 1, 1]]
    assert mask.tolist() == [[True, True, False, False]] * 2
