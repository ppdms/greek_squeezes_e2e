"""The page network (`trogs26.pagenet.model`): the mask vector and the architecture a checkpoint stores."""
from __future__ import annotations

import torch
from torch import nn

from trogs26.pagenet import model as net


class _Backbone(nn.Module):
    """Stands in for the encoder, which `context` does not use."""
    config = type('Config', (), {'hidden_size': 8, 'num_register_tokens': 0})()


ARCH = {'node_dim': 3, 'pair_dim': 2, 'classes': 5, 'd': 12, 'layers': 2, 'heads': 2, 'dropout': 0.0}


def _network() -> net.PageNet:
    torch.manual_seed(0)
    return net.build(ARCH, _Backbone())


def _tokens(n: int = 10, k: int = 4) -> tuple[torch.Tensor, ...]:
    g = torch.Generator().manual_seed(1)
    return (torch.randn(n, 12, generator=g), torch.randn(n, 3, generator=g),
            torch.randint(0, n, (n, k), generator=g), torch.ones(n, k, dtype=torch.bool),
            torch.randn(n, k, 2, generator=g))


def test_a_checkpoints_architecture_may_hold_the_keys_of_options_the_network_does_not_have():
    torch.manual_seed(0)
    stored = {**ARCH, 'readings_dim': 0, 'photo': False, 'recycle': False}
    assert list(net.build(stored, _Backbone()).state_dict()) == list(_network().state_dict())


def test_a_masked_token_is_classified_from_its_neighbours_not_from_its_encoding():
    model = _network().eval()
    visual, node, idx, mask, pair = _tokens()
    masked = torch.zeros(len(visual), dtype=torch.bool)
    masked[0] = True
    other = visual.clone()
    other[0] += torch.linspace(-2.0, 2.0, other.shape[1])  # not a constant: layer norm would remove that
    with torch.no_grad():
        same = model.context(visual, node, idx, mask, pair, masked)
        changed = model.context(other, node, idx, mask, pair, masked)
        unmasked = model.context(other, node, idx, mask, pair)
        original = model.context(visual, node, idx, mask, pair)
    assert torch.equal(same, changed)
    assert not torch.allclose(unmasked[0], original[0])
