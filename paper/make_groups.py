"""Public squeezes of one inscription, for the out-of-fold figure on the squeezes that have no related squeeze in
another fold (paper/make_results.py).

    TROGS26_WORK=<work> .venv/bin/python paper/make_groups.py

Two squeezes are related when they share an identity. A squeeze named after IG II² has the numbers of its name
(`IG_II2_2061b+2093b`: 2061 and 2093; fragments, copies and parts of one number share it); any other squeeze has
the canonical identity of `<work>/phi_raw/exclusions.json`, the PHI exclusion manifest: for `Agora_I_*` the IG II²
number of the Agora inventory concordance, for `IAS*` the Krateros record of its scan, else the first IG reference
of the PHI texts its transcript matched (public squeezes: the annotation; TROGS-26: the system's unscored readings).
A text match can be a formulaic parallel rather than the inscription itself, so it may relate squeezes that are not
one inscription. Groups are the connected components over the public squeezes; data/related_public224.json lists
those of two or more and the identities a TROGS-26 squeeze shares with a public one.
"""
from __future__ import annotations

import hashlib
import json
import re

from trogs26 import data
from trogs26.config import REPO
from trogs26.layout import Work

OUT = REPO / 'data' / 'related_public224.json'
FOLDS = 'data/folds_public224.json'
_IG2 = re.compile(r'IG II\^2 (\d+)[a-z]?')


def identities(base: str, manifest: dict[str, dict]) -> list[str]:
    """`base`'s identities: the IG II² numbers of its name, else its canonical identity in the manifest."""
    if base.startswith('IG_II2_'):
        head = base[len('IG_II2_'):].split('_')[0]
        numbers = sorted({int(m.group()) for part in head.split('+') if (m := re.match(r'\d+', part))})
        if not numbers:
            raise ValueError(f'no IG II² number in {base}')
        return [f'IG II^2 {n}' for n in numbers]
    canonical = manifest[base]['canonical_identity']
    m = _IG2.fullmatch(canonical)
    return [f'IG II^2 {int(m.group(1))}' if m else canonical]


def components(bases: list[str], keys: dict[str, list[str]]) -> list[list[str]]:
    """Squeezes connected by a shared identity, each group sorted, groups by their first squeeze."""
    parent = {base: base for base in bases}

    def root(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    first: dict[str, str] = {}
    for base in bases:
        for key in keys[base]:
            if key in first:
                parent[root(base)] = root(first[key])
            else:
                first[key] = base
    groups: dict[str, list[str]] = {}
    for base in bases:
        groups.setdefault(root(base), []).append(base)
    return sorted(sorted(g) for g in groups.values())


def main() -> None:
    path = Work.from_env().root / 'phi_raw' / 'exclusions.json'
    manifest = {row['base']: row for row in json.loads(path.read_text())['targets']}
    public = data.all_bases(data.folds(FOLDS))
    test = data.splits()['trogs26']
    keys = {base: identities(base, manifest) for base in [*public, *test]}
    groups = [g for g in components(public, keys) if len(g) > 1]
    public_keys = {key: base for base in public for key in keys[base]}
    shared = [{'trogs26': base, 'identity': key, 'public224': sorted(b for b in public if key in keys[b])}
              for base in test for key in keys[base] if key in public_keys]
    OUT.write_text(json.dumps({
        'kind': 'related_public224',
        'note': __doc__.split('\n\n')[2].replace('\n', ' '),
        'identification': {'file': 'phi_raw/exclusions.json',
                           'sha256': hashlib.sha256(path.read_bytes()).hexdigest()},
        'groups': [{'identities': sorted({k for b in g for k in keys[b]}), 'squeezes': g} for g in groups],
        'trogs26_shared': shared,
    }, indent=1, ensure_ascii=False) + '\n')
    print(f'{len(groups)} groups of {sum(map(len, groups))} public squeezes; '
          f'{len(shared)} TROGS-26 identities shared with public squeezes: {shared}')


if __name__ == '__main__':
    main()
