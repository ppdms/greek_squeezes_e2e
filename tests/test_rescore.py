import json
import math

import numpy as np
import pytest
import torch

from trogs26.decode.rescore import neural_mix
from trogs26.decode.search import Pool
from trogs26.layout import Work
from trogs26.lm import neural
from trogs26.lm.twosided import TwoSided


class Prefixed:
    """A model whose letters depend on the letter read before them."""

    def __init__(self, after_a: float):
        self.after_a = after_a

    def cond_logprob(self, letter: str, context: str) -> float:
        p = 0.8 if letter == 'A' else 0.2
        return math.log(self.after_a if context.endswith('A') else p)


def save_model(directory, alphabet, seed):
    torch.manual_seed(seed)
    arch = neural.Arch(16, 1, 2, 2, 0.0, 4)
    directory.mkdir(parents=True)
    torch.save(neural.CharLM(arch, len(alphabet)).state_dict(), directory / 'model.pt')
    (directory / 'config.json').write_text(json.dumps({'architecture': arch.__dict__, 'symbols': len(alphabet),
                                                       'alphabet': alphabet}))


def mixed_change(model_dir, lm, runs, a):
    """The change of the lm sum of a text with runs `runs`, letter by letter."""
    model, _ = neural.load_model(model_dir, torch.device('cpu'))
    lut, change = neural.alphabet_lut('AB'), 0.0
    for run in runs:
        nn = neural.score(model, [neural.encode(run, lut)])
        for i, letter in enumerate(run):
            p = math.exp(lm.cond_logprob(letter, run[:i]))
            change += math.log((1 - a) * p + a * math.exp(nn[i])) - math.log(p)
    return change


@pytest.mark.parametrize('two_sided', [False, True])
def test_neural_mix_replaces_the_forward_letters_and_keeps_the_rest(tmp_path, two_sided):
    models = tmp_path / 'models'
    save_model(models / 'fwd' / 'f0', 'AB', 0)
    work = Work(tmp_path)
    cfg = {'alphabet': 'AB', 'decode': {'variants': {'P': {'neural': {'model': 'fwd'}}}}}
    mix = neural_mix(work, work.chain(0, 'oof', ['x']), cfg, 'P')
    forward = Prefixed(0.5)
    lm = TwoSided(forward, Prefixed(0.3)) if two_sided else forward
    texts = ['ABAAB|BA', 'BA', 'ABAAB', '|']  # a run longer than the window, runs shared between texts, none
    sums = np.array([-5.0, -2.0, -4.0, 0.0])
    pool = Pool(texts, np.zeros(4), np.ones(4), sums, np.zeros(4))
    for a, out in zip([0.3, 0.6], mix.rescore(pool, lm, [0.3, 0.6])):
        for text, before, after in zip(texts, sums, out.lm):
            runs = [part for part in text.split('|') if part]
            assert after == pytest.approx(before + mixed_change(models / 'fwd' / 'f0', forward, runs, a), abs=1e-9)
        assert out.texts == texts and (out.keep == pool.keep).all()


def test_pages_outside_the_folds_read_the_model_of_every_fold_and_a_variant_without_one_has_none(tmp_path):
    save_model(tmp_path / 'models' / 'fwd' / 'all', 'AB', 0)
    work = Work(tmp_path)
    cfg = {'alphabet': 'AB', 'decode': {'variants': {'P': {'neural': {'model': 'fwd'}}, 'Q': {}}}}
    assert neural_mix(work, work.chain(0, 'new', ['x']), cfg, 'P') is not None
    assert neural_mix(work, work.chain(0, 'new', ['x']), cfg, 'Q') is None


def test_a_neural_weight_outside_zero_one_or_a_model_of_another_alphabet_is_refused(tmp_path):
    save_model(tmp_path / 'models' / 'fwd' / 'f0', 'AB', 0)
    work = Work(tmp_path)
    cfg = {'alphabet': 'ABC', 'decode': {'variants': {'P': {'neural': {'model': 'fwd'}}}}}
    with pytest.raises(ValueError):
        neural_mix(work, work.chain(0, 'oof', ['x']), cfg, 'P')
    cfg['alphabet'] = 'AB'
    mix = neural_mix(work, work.chain(0, 'oof', ['x']), cfg, 'P')
    pool = Pool(['AB'], np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1))
    with pytest.raises(ValueError):
        mix.rescore(pool, Prefixed(0.5), [0.3, 1.0])
