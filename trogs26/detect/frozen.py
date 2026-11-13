"""The detector records of the work tree's chains: copies them, unchanged, from the detector work tree.

`SOURCE` is the work tree the detectors ran in (`detect.train`, `detect.infer`, `detect.calibrate` with
`--folds-file data/folds_train181.json`): its chain j holds out fold j of `data/folds_train181.json`
(`chains/f{j}/oof/det`) and has read every image-only squeeze (`chains/f{j}/new/det`). No detector saw the page of a
record it copies:

  --group oof   squeeze s of fold k (the configuration's folds) goes to chains/f{k}/oof/det: a TRAIN squeeze's
                out-of-fold record (from the detector chain that held it out), a validation or internal_test
                squeeze's record from detector chain k
  --group new   every squeeze of `--splits` goes to chains/f{k}/new/det from detector chain k, for every chain k

`SOURCE`'s `detector/calibration.json`, the maps that calibrated the records (`detect.calibrate`), is copied to
`detector/calibration.json` beside them. `detector/frozen_{group}.json` lists each record's source and sha256 and the
calibration's.

    TROGS26_WORK=<work> python -m trogs26.detect.frozen SOURCE --group oof|new [--splits trogs26] \
        [--config configs/pagenet_only_style_neural.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .. import config, data
from ..layout import Work
from . import records


def sources(source: Path, fold_map: dict[int, tuple[str, ...]], group: str,
            splits: list[str]) -> dict[tuple[int, str], Path]:
    """(chain k, squeeze) -> the detector record that chain k gets."""
    if group == 'new':
        bases = sorted(b for split in splits for b in data.splits()[split])
        return {(k, base): Work(source).chain(k, 'new', ()).det(base) for k in fold_map for base in bases}
    train = set(data.splits()['train'])
    out = {}
    for k, bases in fold_map.items():
        for base in bases:
            if base in train:
                hits = sorted(source.glob(f'chains/f*/oof/det/{base}.json'))
                if len(hits) != 1:
                    raise FileNotFoundError(f'{base}: {len(hits)} out-of-fold records under {source}')
                out[k, base] = hits[0]
            else:
                out[k, base] = Work(source).chain(k, 'new', ()).det(base)
    return out


def place(work: Work, source: Path, fold_map: dict[int, tuple[str, ...]], group: str, splits: list[str]) -> dict:
    """Copy every record of `sources` and `source`'s calibration into `work` and write the manifest; returns its
    summary."""
    copied = {}
    for (k, base), path in sorted(sources(source, fold_map, group, splits).items()):
        raw = path.read_bytes()
        record = json.loads(raw)
        records.validate(record)
        if record['base'] != base or any(box['p_letter'] is None for box in record['boxes']):
            raise ValueError(f'{path} is not a calibrated record of {base}')
        target = work.chain(k, group, ()).det(base)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        copied[f'f{k}/{base}'] = {'source': str(path.relative_to(source)), 'sha256': hashlib.sha256(raw).hexdigest(),
                                  'detections': len(record['boxes'])}
    maps = (source / 'detector' / 'calibration.json').read_bytes()
    work.detector.mkdir(parents=True, exist_ok=True)
    (work.detector / 'calibration.json').write_bytes(maps)
    manifest = {'source': str(source), 'group': group,
                'calibration': {'source': 'detector/calibration.json', 'sha256': hashlib.sha256(maps).hexdigest()},
                'records': copied}
    path = work.detector / f'frozen_{group}.json'
    path.write_text(json.dumps(manifest, indent=1, sort_keys=True))
    chains = sorted({key.split('/')[0] for key in copied})
    return {'manifest': str(path), 'records': len(copied),
            'detections': {c: sum(v['detections'] for key, v in copied.items() if key.startswith(c + '/'))
                           for c in chains}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('source', type=Path, help='the detector work tree')
    parser.add_argument('--group', choices=('oof', 'new'), required=True)
    parser.add_argument('--splits', default='trogs26', help="image-only splits of --group new (comma-separated)")
    parser.add_argument('--config', default='configs/pagenet_only_style_neural.json')
    args = parser.parse_args()
    cfg = config.load(args.config)
    print(json.dumps(place(Work.from_env(), args.source.resolve(), data.folds(cfg['data']['folds']), args.group,
                           args.splits.split(','))))


if __name__ == '__main__':
    main()
