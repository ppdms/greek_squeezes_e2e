"""Reading a page with a page network, and the decoder's input from a chain's networks."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from trogs26.pagenet import infer, pages
from trogs26.pagenet import model as net

ALPHABET = 'AB'


class _Backbone(nn.Module):
    """Stands in for the encoder: a class token and four patch tokens per crop, from the crop's mean."""
    config = type('Config', (), {'hidden_size': 8, 'num_register_tokens': 0})()

    def forward(self, pixel_values):
        tokens = pixel_values.mean((1, 2, 3))[:, None, None] * torch.arange(1.0, 6.0)[None, :, None]
        return type('Output', (), {'last_hidden_state': tokens.expand(len(pixel_values), 5, 8)})()


def _record() -> dict:
    boxes = []
    for row in range(3):
        for column in range(8):
            cx, cy = 20 + 12 * column, 30 + 40 * row + 0.3 * column
            boxes.append({'id': len(boxes), 'quad': [[cx - 5, cy - 10], [cx - 5, cy + 10], [cx + 5, cy + 10],
                                                     [cx + 5, cy - 10]],
                          'score': 0.9, 'p_letter': 0.9 - 0.01 * len(boxes),
                          'view_support': {'pack': 0.9, 'mono1': None, 'mono2': None}})
    return {'boxes': boxes, 'letter_height_px': 20.0, 'registration': {'inlier_fraction': 0.99}}


def _network_and_page():
    page = pages.build(_record(), 'x')
    torch.manual_seed(0)
    arch = {'node_dim': len(pages.NODE), 'pair_dim': len(pages.PAIR), 'classes': 5, 'd': 12, 'layers': 1,
            'heads': 2, 'dropout': 0.0}
    checkpoint = {'recipe': {'neighbours': 6, 'v_weight': 2.0},
                  'node_stats': {'mean': [0.0] * len(pages.NODE), 'std': [1.0] * len(pages.NODE)}}
    crops = np.random.default_rng(0).integers(0, 256, (len(page), 3, 8, 8), dtype=np.uint8)
    return net.build(arch, _Backbone()).eval(), page, crops, checkpoint


def test_a_page_is_read_into_one_distribution_per_token_and_equal_views_average_to_one():
    model, page, crops, checkpoint = _network_and_page()
    normalise = torch.zeros(1, 3, 1, 1), torch.ones(1, 3, 1, 1)
    once = infer.read_page(model, page, crops, checkpoint, torch.device('cpu'), normalise, 7, [(0.0, 1.0)])
    twice = infer.read_page(model, page, crops, checkpoint, torch.device('cpu'), normalise, 7,
                            [(0.0, 1.0), (0.0, 1.0)])
    assert once.shape == (len(page), 5) and once.dtype == np.float32
    assert np.allclose(np.exp(once).sum(1), 1, atol=1e-5)
    assert np.allclose(once, twice, atol=1e-5)


def write(path, det_ids, log_probs, alphabet=ALPHABET) -> None:
    np.savez(path, bases=np.array(['x', 'x', 'y']), det_ids=np.array(det_ids, np.int64),
             log_probs=np.asarray(log_probs, np.float32), alphabet=np.array([*alphabet, 'none']))


def test_two_networks_combine_to_the_normalised_product_of_their_distributions(tmp_path):
    rng = np.random.default_rng(0)
    a, b = rng.dirichlet(np.ones(3), 3), rng.dirichlet(np.ones(3), 3)
    write(tmp_path / 'pagenet_n1.npz', [4, 2, 7], np.log(a))
    write(tmp_path / 'pagenet_n2.npz', [4, 2, 7], np.log(b))
    out = infer.posteriors(SimpleNamespace(dir=tmp_path), ['n1', 'n2'], ALPHABET)
    product = np.sqrt(a * b)
    expected = np.log(product / product.sum(1, keepdims=True))
    assert list(out) == ['x', 'y']
    assert out['x'][0].tolist() == [4, 2] and out['y'][0].tolist() == [7]
    assert np.allclose(np.concatenate([out['x'][1], out['y'][1]]), expected, atol=1e-6)


def test_networks_that_read_different_detections_are_refused(tmp_path):
    write(tmp_path / 'pagenet_n1.npz', [4, 2, 7], np.full((3, 3), np.log(1 / 3)))
    write(tmp_path / 'pagenet_n2.npz', [4, 2, 8], np.full((3, 3), np.log(1 / 3)))
    with pytest.raises(ValueError):
        infer.posteriors(SimpleNamespace(dir=tmp_path), ['n1', 'n2'], ALPHABET)


def test_a_network_of_another_alphabet_is_refused(tmp_path):
    write(tmp_path / 'pagenet_n1.npz', [4, 2, 7], np.full((3, 3), np.log(1 / 3)), alphabet='BA')
    with pytest.raises(AssertionError):
        infer.posteriors(SimpleNamespace(dir=tmp_path), ['n1'], ALPHABET)
