import json
import math
import pickle

import pytest

from trogs26.decode.cache import CacheLM
from trogs26.lm import artifact
from trogs26.lm.ppm import PPM, Backward, Counts
from trogs26.lm.twosided import TwoSided


def test_counts_are_per_row_and_ignore_foreign_letters():
    counts = Counts.from_lines(['ABA', 'a-b', ''], order=2, alphabet='AB')
    assert counts.counts[0] == {}
    assert counts.counts[1][''] == {'A': 3, 'B': 2}
    assert counts.counts[2] == {'A': {'B': 2}, 'B': {'A': 1}}


def test_ppm_escapes_to_shorter_contexts_and_excludes_seen_letters():
    model = PPM(Counts.from_lines(['AB'], order=2, alphabet='ABC'), max_order=2)
    # After 'A' the bigram context has seen only B: P(B) = 1/2, the rest escapes with 1/2.
    # The empty context then offers A (B is excluded): P(A) = 1/2 * 1/2; the unseen C gets what is left.
    assert math.exp(model.cond_logprob('B', 'A')) == pytest.approx(0.5)
    assert math.exp(model.cond_logprob('A', 'A')) == pytest.approx(0.25)
    assert math.exp(model.cond_logprob('C', 'A')) == pytest.approx(0.25)
    # An unseen context falls straight to the empty context: A and B get 1/4 each, C the escape mass 1/2.
    assert math.exp(model.cond_logprob('C', 'C')) == pytest.approx(0.5)
    assert sum(math.exp(model.cond_logprob(x, 'A')) for x in 'ABC') == pytest.approx(1.0)


def test_ppm_reads_no_context_beyond_max_order():
    counts = Counts.from_lines(['AB'], order=2, alphabet='ABC')
    assert math.exp(PPM(counts, max_order=1).cond_logprob('B', 'A')) == pytest.approx(0.25)


def test_text_logprob_adds_the_letters_in_context():
    model = PPM(Counts.from_lines(['AB'], order=2, alphabet='ABC'), max_order=2)
    assert model.text_logprob('AB') == pytest.approx(math.log(0.25) + math.log(0.5))
    assert model.text_logprob('') == 0.0


def test_cache_mixes_probabilities_not_log_probabilities():
    class Base:
        def cond_logprob(self, letter, context):
            return math.log(0.25)

    cache = CacheLM(Base(), ['AB'], gamma=0.2, order=2, alphabet='ABC')
    # cache P(B | A) = 1/2
    assert math.exp(cache.cond_logprob('B', 'A')) == pytest.approx(0.8 * 0.25 + 0.2 * 0.5)
    with pytest.raises(ValueError):
        CacheLM(Base(), ['AB'], gamma=0.0, order=2, alphabet='ABC')


def test_reading_right_to_left_equals_the_model_of_the_reversed_corpus():
    lines = ['ABACAB', 'CABBA', 'BCA', 'AAC']
    backward = Backward(Counts.from_lines(lines, order=4, alphabet='ABCD'), max_order=4)
    mirrored = PPM(Counts.from_lines([line[::-1] for line in lines], order=4, alphabet='ABCD'), max_order=4)
    for context in ('', 'A', 'BA', 'CAB', 'ABCA', 'DD', 'BBAC'):
        for letter in 'ABCD':
            assert backward.cond_logprob(letter, context) == pytest.approx(mirrored.cond_logprob(letter, context))


def test_two_sided_scores_a_first_letter_from_what_follows_it():
    counts = Counts.from_lines(['AB', 'AB', 'CB', 'AC'], order=2, alphabet='ABC')
    forward, backward = PPM(counts, 2), Backward(counts, 2)
    model = TwoSided(forward, backward)
    # 'AB': A from the B after it, B from the A before it and from nothing after it (which cancels)
    assert model.text_logprob('AB') == pytest.approx(backward.cond_logprob('A', 'B') + forward.cond_logprob('B', 'A'))
    assert model.cond_logprob('B', 'A') == forward.cond_logprob('B', 'A')


def test_artifact_round_trip(tmp_path):
    counts = Counts.from_lines(['ABBA', 'BA'], order=3, alphabet='AB')
    artifact.save(counts, tmp_path, {'created_by': 'test'})
    loaded = artifact.load_counts(tmp_path)
    assert (loaded.order, loaded.alphabet, loaded.counts) == (3, 'AB', counts.counts)
    assert artifact.load(tmp_path, max_order=2).order == 2


def test_other_classes_in_a_model_file_are_refused(tmp_path):
    (tmp_path / 'lm.json').write_text(json.dumps({'model': 'bad.pkl'}))
    (tmp_path / 'bad.pkl').write_bytes(pickle.dumps(json.dumps))
    with pytest.raises(pickle.UnpicklingError):
        artifact.load_counts(tmp_path)
