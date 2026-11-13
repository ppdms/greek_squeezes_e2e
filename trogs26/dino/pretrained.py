"""The pretrained DINOv3 ViT-S/16 weights the page networks are fine-tuned from.

Source: Hugging Face `facebook/dinov3-vits16-pretrain-lvd1689m` (gated: the account fetching it must have accepted
Meta's DINOv3 licence on the model page and expose its token as HF_TOKEN). The weights are pinned by commit and by
the SHA-256 of `model.safetensors`, so a run can never silently train from different weights. Meta's `LICENSE.md` is
downloaded with them and copied beside every model trained from them (`pagenet.train.train_fold`).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

FILES = ('config.json', 'preprocessor_config.json', 'LICENSE.md', 'model.safetensors')


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def check(directory: Path, expected_sha256: str) -> None:
    """Raise unless `directory` holds every file of the pretrained model and its weights match `expected_sha256`."""
    missing = [name for name in FILES if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f'{directory}: missing {missing}')
    found = sha256(directory / 'model.safetensors')
    if found != expected_sha256:
        raise ValueError(f'{directory}/model.safetensors has SHA-256 {found}, expected {expected_sha256}')


def fetch(directory: Path, repo: str, revision: str, expected_sha256: str) -> None:
    """Download revision `revision` of `repo` into `directory` and verify it."""
    from huggingface_hub import snapshot_download

    snapshot_download(repo, revision=revision, local_dir=directory, allow_patterns=list(FILES))
    check(directory, expected_sha256)
