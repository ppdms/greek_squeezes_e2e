import math

import numpy as np

from trogs26.decode import rows
from trogs26.decode.decoder import Cache, decode_rows
from trogs26.decode.search import GAP, Combo, Search, candidate_pool, grid_combos, select, top_letters


class NoLM:
    """A language model with no opinion: only emissions and keep probabilities decide."""

    def cond_logprob(self, letter, context):
        return 0.0

    def text_logprob(self, text):
        return 0.0


class NeverAA:
    """A language model that all but forbids an A right after an A."""

    def cond_logprob(self, letter, context):
        return -20.0 if context.endswith('A') and letter == 'A' else -0.7

    def text_logprob(self, text):
        return sum(self.cond_logprob(letter, text[:i]) for i, letter in enumerate(text))


class HalfLM:
    """A language model that gives every letter probability 1/2."""

    def cond_logprob(self, letter, context):
        return math.log(0.5)

    def text_logprob(self, text):
        return math.log(0.5) * len(text)


SEARCH = Search('AB', beam=8, topk=2, k_hi=0.95, passes=(Combo(0.0, 1.0, 0.0), Combo(1.0, 2.0, 2.5)))


def test_grid_is_lam_major():
    assert grid_combos({'lam': [1, 2], 'w': [3], 'beta': [4, 5]}) == [Combo(1, 3, 4), Combo(1, 3, 5), Combo(2, 3, 4),
                                                                      Combo(2, 3, 5)]
    assert grid_combos({'lam': [1], 'w': [3], 'beta': [4], 'nn': [0.5, 0.7]}) == [Combo(1, 3, 4, 0.5),
                                                                                 Combo(1, 3, 4, 0.7)]
    assert Combo(1.25, 2.0, 4.0).key == '1.25|2.00|4.00'
    assert Combo(1.25, 2.0, 4.0, 0.7).key == '1.25|2.00|4.00|0.70'


def test_equal_log_probs_keep_alphabet_order():
    assert top_letters(np.array([-1.0, -1.0, -0.5]), 'ABC', 2) == [('C', -0.5), ('A', -1.0)]


def test_length_bonus_decides_whether_an_uncertain_box_is_read():
    emit = np.log(np.array([[0.9, 0.1], [0.4, 0.6]]))
    pool = candidate_pool(emit, np.array([0.99, 0.5]), NoLM(), SEARCH)
    assert 'AB' in pool.texts and 'A' in pool.texts
    # emitting the second box costs log 0.6 + w log 0.5 and gains beta; skipping costs w log 0.5
    assert select(pool, [Combo(0.0, 1.0, 0.0), Combo(0.0, 1.0, 2.0)]) == ['A', 'AB']


def test_confident_boxes_are_never_skipped():
    emit = np.log(np.array([[0.9, 0.1], [0.4, 0.6]]))
    pool = candidate_pool(emit, np.array([0.99, 0.99]), NoLM(), SEARCH)
    assert all(len(text) == 2 for text in pool.texts)


def test_the_language_model_does_not_read_across_a_gap():
    combo = [Combo(1.0, 1.0, 0.0)]
    emit, keep = np.log(np.array([[0.9, 0.1], [0.8, 0.2]])), np.array([0.99, 0.99])
    # side by side the second A is turned into a B; a gap apart both stay
    assert select(candidate_pool(emit, keep, NeverAA(), SEARCH), combo) == ['AB']
    assert select(candidate_pool(emit, keep, NeverAA(), SEARCH, along=[0.0, 2.0]), combo) == ['A' + GAP + 'A']
    # a doubtful letter between them, once dropped, leaves a gap as well
    emit, keep = np.log(np.array([[0.9, 0.1], [0.5, 0.5], [0.8, 0.2]])), np.array([0.99, 0.3, 0.99])
    assert select(candidate_pool(emit, keep, NeverAA(), SEARCH), combo) == ['AB']
    assert select(candidate_pool(emit, keep, NeverAA(), SEARCH, along=[0.0, 0.6, 1.2]), combo) == ['A' + GAP + 'A']


def test_first_occurrence_rows_drops_repeats_and_empty_rows():
    assert rows.first_occurrence_rows([[0, 1], [1, 2], [0], []]) == [[0, 1], [2]]


def test_a_point_joins_the_row_it_sits_on_within_the_span_margin():
    # one row of two letters at u = 10 and 50, v = 100; letters are 10 high, so the span reaches u = -15 .. 75
    row_u, row_v = [np.array([10.0, 50.0])], [np.array([100.0, 100.0])]
    u, v = np.array([30.0, 30.0, 200.0]), np.array([102.0, 140.0, 100.0])
    # inside the row; too far above it; too far right
    assert rows.attach(row_u, row_v, u, v, letter_height=10.0, span_margin=2.5).tolist() == [0, -1, -1]


def test_the_cache_lets_a_squeeze_vote_on_its_own_ambiguous_row():
    search = Search('AB', beam=8, topk=2, k_hi=0.95, passes=(Combo(0.0, 1.0, 0.0), Combo(1.0, 1.0, 0.0)))
    ambiguous = (np.log([[0.45, 0.55]]), np.array([1.0]))  # the image slightly prefers B
    clear = (np.log([[0.99, 0.01]] * 3), np.ones(3))  # the squeeze's other row reads AAA
    combos = [Combo(1.0, 1.0, 0.0)]
    without = decode_rows([ambiguous, clear], combos, HalfLM(), search, None)
    with_cache = decode_rows([ambiguous, clear], combos, HalfLM(), search, Cache(0.9, 2, combos[0]))
    assert without == [['B'], ['AAA']]
    assert with_cache == [['A'], ['AAA']]
