"""Scoring with the organisers' v2 scorer.

Two entry points over the same scorer:
- `view_errors`: Core errors of one transcript against one annotation view, cached. Decoding sweeps call it
  thousands of times, so it reads each annotation once instead of copying files per call.
- `official_scores`: the scorer's own `run_evaluations` on one squeeze at a time, for reported scores. The
  scorer finds a squeeze's two annotation files by file-name prefix, which fails when one squeeze's name is the
  prefix of another's; a fresh directory per squeeze avoids that.

A squeeze whose annotations have no confidence section and whose letter counts disagree between views or with the
transcript cannot be scored by the scorer's own routine. Its letters are all counted as confident instead, the
flag `all_confident` marks it, and `fallbacks` counts it.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

from . import fallbacks, official
from .layout import Work

_LETTERS_SUFFIX = '_letters.txt'
_COUNT_FIELDS = ('err', 'char', 'all_err', 'all_char', 'opt_err', 'opt_char')


def transcript_name(base: str) -> str:
    """File name the scorer looks for a squeeze's transcript under; it names the squeeze by cutting `base` at
    `_Rotation` or `_Merged`."""
    return official.v2.clip_squeeze(base) + '_transcript.txt'


class Reference(NamedTuple):
    """One annotation view as the scorer compares against: all letters, which of them are confident, and whether
    that vector is the all-confident substitute."""
    text: str
    confident: list[bool]
    all_confident: bool


@lru_cache(maxsize=None)
def reference(work: Work, base: str, view: int) -> Reference:
    """Reference of `base`'s annotation view `view`: confident = confident in either view, as the scorer does."""
    v2 = official.v2
    text = v2.getRowTranscript(str(work.annotation(base, view)), all_conf=True)
    views = [v2.readBoxFile(str(work.annotation(base, v)), all_conf=True, get_conf=True) for v in (1, 2)]
    try:
        assert len(views[0]) == len(views[1])
        confident = [a or b for a, b in zip(*views)]
        v2.evaluate('', text, confident)  # raises where the scorer would
    except AssertionError:
        fallbacks.note('scorer_all_confident', 1, f'{base} view {view}')
        return Reference(text, [True] * (len(text) - text.count('\n')), True)
    return Reference(text, confident, False)


@lru_cache(maxsize=None)
def view_errors(work: Work, base: str, view: int, text: str) -> tuple[int, int]:
    """Core (errors, characters) of transcript `text` against annotation view `view` of `base`."""
    ref = reference(work, base, view)
    result = official.v2.evaluate(text, ref.text, ref.confident)
    return int(result[3]), int(result[6])


def squeeze_errors(work: Work, base: str, text: str) -> list[int]:
    """[errors, characters] of both views of `base`: [err1, chars1, err2, chars2]."""
    return [*view_errors(work, base, 1, text), *view_errors(work, base, 2, text)]


def core_text(work: Work, base: str, view: int) -> str:
    """The text Core errors of annotation view `view` are counted against, normalised as the scorer does: its
    confident letters by row, rows left empty dropped."""
    ref = reference(work, base, view)
    rows = official.v2.convert(ref.text).upper().replace(' ', '').replace('\t', '').split('\n')
    flags = iter(ref.confident)
    return '\n'.join(row for row in (''.join(c for c in r if next(flags)) for r in rows) if row)


# ----------------------------------------------------------------------------- official, one squeeze at a time
def group_letters_files(gt_dir: Path) -> dict[str, list[Path]]:
    """Squeeze -> its two letters files. The squeeze is the file name cut at `_Rotation` or `_Merged`
    (`IAS_27119_Merged_Rotation2` pairs with `IAS_27119_Rotation1`; `_Copy1` and `_Copy2` are distinct squeezes)."""
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(gt_dir.iterdir()):
        if path.is_file() and path.name.endswith(_LETTERS_SUFFIX):
            if '_Rotation' not in path.name:
                raise ValueError(f'not a rotation letters file: {path.name}')
            groups[official.v2.clip_squeeze(path.name)].append(path)
    bad = {key: [p.name for p in files] for key, files in groups.items() if len(files) != 2}
    if bad:
        raise ValueError(f'squeezes without exactly two rotation files: {bad}')
    return dict(groups)


def _transcript_names(letters_name: str) -> list[str]:
    """Prediction file names the scorer tries for a letters file, in its order."""
    return [letters_name[:-len(_LETTERS_SUFFIX)] + '_transcript.txt', transcript_name(letters_name)]


def _has_confidence_section(path: Path) -> bool:
    text = path.read_text(encoding='utf-8', errors='replace')
    return sum(1 for line in text.splitlines() if line.startswith('#')) >= 5


def _row(name: str, counts: tuple, missing: bool, all_confident: bool) -> dict:
    err, all_err, opt_err, char, all_char, opt_char = counts
    return {'file': name, 'err': int(err), 'char': int(char), 'all_err': int(all_err), 'all_char': int(all_char),
            'opt_err': int(opt_err), 'opt_char': int(opt_char), 'missing_prediction': missing,
            'all_confident': all_confident}


def _score_squeeze(files: list[Path], pred_dir: Path) -> list[dict]:
    """The scorer's `run_evaluations` on the two files of one squeeze, staged in fresh directories."""
    v2 = official.v2
    missing = {p.name: not any((pred_dir / n).is_file() for n in _transcript_names(p.name)) for p in files}
    try:
        with tempfile.TemporaryDirectory(prefix='trogs26_score_') as tmp:
            gt_tmp, pred_tmp = Path(tmp) / 'gt', Path(tmp) / 'pred'
            gt_tmp.mkdir()
            pred_tmp.mkdir()
            for path in files:
                shutil.copy2(path, gt_tmp / path.name)
                for name in _transcript_names(path.name):
                    if (pred_dir / name).is_file() and not (pred_tmp / name).exists():
                        shutil.copy2(pred_dir / name, pred_tmp / name)
            *_, n_err, n_all_err, n_opt_err, n_char, n_all_char, n_opt_char, names = \
                v2.run_evaluations(str(pred_tmp), str(gt_tmp))
        return [_row(name, (n_err[i], n_all_err[i], n_opt_err[i], n_char[i], n_all_char[i], n_opt_char[i]),
                     missing[name], False) for i, name in enumerate(names)]
    except AssertionError:
        if any(_has_confidence_section(p) for p in files):
            raise
    rows = []
    for path in files:
        pred = ''
        for name in _transcript_names(path.name):
            if (pred_dir / name).is_file():
                pred = (pred_dir / name).read_text(encoding='utf8')
                break
        text = v2.getRowTranscript(str(path), all_conf=True)
        r = v2.evaluate(pred, text, [True] * (len(text) - text.count('\n')))
        rows.append(_row(path.name, (r[3], r[4], r[5], r[6], r[7], r[8]), missing[path.name], True))
    fallbacks.note('scorer_all_confident', len(files), files[0].name)
    return rows


def official_scores(pred_dir: Path, gt_dir: Path, squeezes: list[str]) -> dict:
    """Official Core (`cer`), All (`all_cer`) and Opt (`opt_cer`) scores of the transcripts in `pred_dir` on
    `squeezes` (squeeze names as `group_letters_files` defines them), each squeeze scored on its own. `per_squeeze`
    sums the two views of a squeeze; `all_confident` marks squeezes scored with the substitute confidence."""
    groups = group_letters_files(gt_dir)
    unknown = [key for key in squeezes if key not in groups]
    if unknown:
        raise KeyError(f'squeezes not in {gt_dir}: {unknown[:5]}')
    per_file = [{**row, 'squeeze': key} for key in squeezes for row in _score_squeeze(groups[key], pred_dir)]
    total: Counter = Counter()
    per_squeeze: dict[str, Counter] = defaultdict(Counter)
    flagged = set()
    for row in per_file:
        for field in _COUNT_FIELDS:
            total[field] += row[field]
            per_squeeze[row['squeeze']][field] += row[field]
        if row['all_confident']:
            flagged.add(row['squeeze'])
    return {
        'errors': {'cer': total['err'], 'all': total['all_err'], 'opt': total['opt_err']},
        'chars': {'cer': total['char'], 'all': total['all_char'], 'opt': total['opt_char']},
        'cer': total['err'] / max(total['char'], 1),
        'all_cer': total['all_err'] / max(total['all_char'], 1),
        'opt_cer': total['opt_err'] / max(total['opt_char'], 1),
        'missing': sorted({row['squeeze'] for row in per_file if row['missing_prediction']}),
        'all_confident': sorted(flagged),
        'per_squeeze': [{'squeeze': key, **{f: int(c[f]) for f in _COUNT_FIELDS},
                         'cer': c['err'] / max(c['char'], 1), 'all_cer': c['all_err'] / max(c['all_char'], 1),
                         'all_confident': key in flagged} for key, c in sorted(per_squeeze.items())],
    }


def score_directory(pred_dir: Path, gt_dir: Path) -> dict:
    """`official_scores` of every squeeze that has a transcript in `pred_dir` (`<squeeze>_transcript.txt`)."""
    squeezes = sorted(p.name[:-len('_transcript.txt')] for p in pred_dir.glob('*_transcript.txt'))
    return official_scores(pred_dir, gt_dir, squeezes)


def main() -> None:
    parser = argparse.ArgumentParser(description='Official scores of a directory of transcripts.')
    parser.add_argument('pred_dir', type=Path)
    parser.add_argument('gt_dir', type=Path, help='directory of the squeezes\' *_letters.txt annotation files')
    args = parser.parse_args()
    result = score_directory(args.pred_dir, args.gt_dir)
    print(json.dumps({k: v for k, v in result.items() if k != 'per_squeeze'}, indent=1))


if __name__ == '__main__':
    main()
