"""The page network (see the package docstring).

A trained network lives in `models/pagenet_{version}/f{k}/`: `pagenet.pt` (state dict, architecture, feature
statistics, alphabet, recipe) with the backbone's `config.json`, the input normalisation `preprocessor_config.json`
and Meta's `LICENSE.md`, which governs the pretrained DINOv3 weights the encoder derives from.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, DINOv3ViTModel


class NeighbourAttention(nn.Module):
    """Multi-head attention of each token over its k neighbours, biased per head by their relative geometry."""

    def __init__(self, d: int, heads: int, pair_dim: int, dropout: float):
        super().__init__()
        self.heads, self.dh = heads, d // heads
        self.qkv = nn.Linear(d, 3 * d)
        self.bias = nn.Sequential(nn.Linear(pair_dim, 64), nn.GELU(), nn.Linear(64, heads))
        self.out = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, idx: torch.Tensor, mask: torch.Tensor, pair: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        q, k, v = self.qkv(x).view(n, 3, self.heads, self.dh).unbind(1)
        scores = torch.einsum('nhd,nkhd->nhk', q, k[idx]) / math.sqrt(self.dh) + self.bias(pair).permute(0, 2, 1)
        scores = scores.float().masked_fill(~mask[:, None, :], float('-inf'))
        attention = self.drop(torch.softmax(scores, -1)).to(v.dtype)
        return self.out(torch.einsum('nhk,nkhd->nhd', attention, v[idx]).reshape(n, -1))


class Block(nn.Module):
    def __init__(self, d: int, heads: int, pair_dim: int, dropout: float):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attention = NeighbourAttention(d, heads, pair_dim, dropout)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, idx, mask, pair):
        x = x + self.drop(self.attention(self.norm1(x), idx, mask, pair))
        return x + self.drop(self.ff(self.norm2(x)))


class PageNet(nn.Module):
    """Crop encoder, geometry embedding and neighbour transformer with one head of 27 letters + none per token.

    `reader` is a head on the crop encoding alone; it supervises the encoder during training (as a reader would be
    trained) and is not used for reading pages.
    """

    def __init__(self, backbone: nn.Module, node_dim: int, pair_dim: int, classes: int, d: int, layers: int,
                 heads: int, dropout: float):
        super().__init__()
        self.backbone = backbone
        hidden = backbone.config.hidden_size
        self.registers = int(getattr(backbone.config, 'num_register_tokens', 0))
        self.visual = nn.Sequential(nn.Linear(2 * hidden, d), nn.LayerNorm(d))
        self.reader = nn.Linear(d, classes)
        self.geometry = nn.Sequential(nn.Linear(node_dim, d), nn.GELU(), nn.Linear(d, d))
        self.mask_token = nn.Parameter(torch.zeros(d))
        self.blocks = nn.ModuleList(Block(d, heads, pair_dim, dropout) for _ in range(layers))
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, classes)

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        """(n, d) encodings of normalised (n, 3, 224, 224) crops: the class token and the mean patch token of each."""
        h = self.backbone(pixel_values=pixels).last_hidden_state
        return self.visual(torch.cat([h[:, 0], h[:, 1 + self.registers:].mean(1)], -1))

    def context(self, visual: torch.Tensor, node: torch.Tensor, idx: torch.Tensor, mask: torch.Tensor,
                pair: torch.Tensor, masked: torch.Tensor | None = None) -> torch.Tensor:
        """Class logits (n, classes) of n tokens with encodings `visual` (n, d), standardised node features (n, f),
        neighbour indices and mask (n, k) and pair features (n, k, p). Tokens flagged `masked` lose their encoding to
        a learned mask vector (training: their class must come from their neighbours)."""
        x = visual
        if masked is not None:
            x = torch.where(masked[:, None], self.mask_token.to(x.dtype), x)
        h = x + self.geometry(node)
        for block in self.blocks:
            h = block(h, idx, mask, pair)
        return self.head(self.norm(h))


def build(arch: dict, backbone: nn.Module) -> PageNet:
    """The network of architecture `arch` (a checkpoint's `arch`) around `backbone`."""
    return PageNet(backbone, arch['node_dim'], arch['pair_dim'], arch['classes'], arch['d'], arch['layers'],
                   arch['heads'], arch['dropout'])


def from_pretrained(directory: Path, arch: dict, drop_path: float) -> PageNet:
    """A network whose encoder carries the pretrained DINOv3 weights in `directory`, with stochastic depth
    `drop_path` in each encoder layer while training; everything else is fresh."""
    backbone = DINOv3ViTModel.from_pretrained(str(directory), local_files_only=True, attn_implementation='sdpa',
                                              drop_path_rate=drop_path)
    return build(arch, backbone)


def load(directory: Path, device: torch.device) -> tuple[PageNet, dict]:
    """The trained network in `directory` in eval mode on `device`, and its checkpoint (without the weights)."""
    checkpoint = torch.load(directory / 'pagenet.pt', map_location='cpu', weights_only=True)
    config = AutoConfig.from_pretrained(str(directory), local_files_only=True)
    model = build(checkpoint['arch'], AutoModel.from_config(config, attn_implementation='sdpa'))
    model.load_state_dict(checkpoint.pop('model'))
    return model.to(device).eval(), checkpoint
