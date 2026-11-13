"""What outputs record about their origin: the configuration digest and the code version (`stamp`), and for runs
also the counted fallbacks and seconds (`summary`)."""
from __future__ import annotations

import os
import subprocess

from . import config, fallbacks


def git_state() -> dict:
    """Commit of this checkout and whether the working tree differs from it. Summaries are written by the
    drivers, which run in the checkout, or on a Compute Engine job VM, which runs `git archive` of the commit its
    startup script exports as TROGS26_COMMIT (no .git there; the tree is the commit's)."""
    if not (config.REPO / '.git').exists():
        if not os.environ.get('TROGS26_COMMIT'):
            raise RuntimeError(f'{config.REPO} is not a git checkout and TROGS26_COMMIT is not set')
        return {'commit': os.environ['TROGS26_COMMIT'], 'dirty': False}
    git = ['git', '-C', str(config.REPO)]
    commit = subprocess.run([*git, 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
    dirty = bool(subprocess.run([*git, 'status', '--porcelain'], capture_output=True, text=True,
                                check=True).stdout.strip())
    return {'commit': commit, 'dirty': dirty}


def stamp(cfg: config.Config) -> dict:
    """Configuration name, file (with base and override for an experiment) and digest of the merged
    configuration, git commit and whether the working tree was dirty."""
    return {'config': cfg['name'], 'config_source': cfg.source, 'config_digest': config.digest(cfg), **git_state()}


def summary(cfg: config.Config, seconds: float, counts: dict[str, int] | None = None, **fields) -> dict:
    """Run summary: `fields` plus the stamp, fallback counts (default: this process's) and wall-clock seconds."""
    return {**fields, **stamp(cfg),
            'fallbacks': fallbacks.counts() if counts is None else counts, 'seconds': round(seconds, 1)}
