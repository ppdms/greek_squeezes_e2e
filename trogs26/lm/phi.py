"""The PHI Greek Inscriptions, staged for the calibration of decontamination (`decontaminate.calibration`): the raw
crawl's visible text in corpus lines and the publication references each document cites.

The crawl (`<root>/raw/phi`: `chunks/chunk-*.jsonl.gz`, one JSON record per document, and `completion_manifest.json`)
is read only once every chunk's size and SHA-256 match the manifest and the documents are exactly those it counts. A
document's text is its `lineation` (else `raw_text`) in Leiden markup, read by `visible_lines`: only letters seen on
the stone enter; restorations [...], editorial additions <...>, resolved abbreviations (...) and erasures [[...]] never
do; a lacuna, a line end, a dash and a vacat split the text; markup that does not balance quarantines the document.
Letters are mapped as `open.normalise` maps them. A document's references (`references`) are the IG and Agora
citations of its edition reference and metadata; the calibration finds a public squeeze's editions by the IG numbers
of its name (`name_references`).

PHI's text is not openly licensed: it is read for that measurement only, and no corpus or model holds it.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

from .open import _classify

_OPEN = {'[': ']', '<': '>', '⟨': '⟩', '(': ')', '⟦': '⟧', '{': '}'}
_DROP = frozenset({'<', '⟨', '(', '⟦'})  # what these enclose is not on the stone
_INTERPUNCTS = ':··∙⋅•∶⁚'
_VAC = re.compile(r'\bvac(?:at)?\.?(?!\w)', re.IGNORECASE)


def _sha256(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def visible_lines(text: str | list[str]) -> list[str]:
    """Corpus lines of the letters of a Leiden text that are visible on the stone (module docstring). Letters in
    curly braces (written on the stone in error) and underdotted letters stay; raises ValueError on markup that does
    not balance."""
    if not isinstance(text, str):
        text = '\n'.join(text)
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = text.replace('[[', '⟦').replace(']]', '⟧').replace('〚', '⟦').replace('〛', '⟧')
    text = unicodedata.normalize('NFD', _VAC.sub('\n', text))
    stack: list[str] = []
    lines: list[str] = []
    run: list[str] = []

    def split() -> None:
        if run:
            lines.append(''.join(run))
            run.clear()

    for ch in text:
        if unicodedata.category(ch).startswith('M'):
            continue
        if ch == '\n':
            split()
        elif ch in _OPEN:
            stack.append(ch)
            if ch == '[' and not any(mark in _DROP for mark in stack):
                split()
        elif stack and ch == _OPEN[stack[-1]]:
            stack.pop()
        elif ch in ']}>⟩⟧' or (ch == ')' and '(' in stack):
            raise ValueError('unbalanced Leiden delimiter')
        elif any(mark in _DROP for mark in stack):
            continue
        elif ch in '—–-.*…':
            split()
        elif ch.isspace() or ch.isdigit() or ch in ',;!?\'"/|=⁷':
            continue
        else:
            token = ':' if ch in _INTERPUNCTS else _classify(ch, True)
            if token and token != '\n':
                if '[' not in stack:
                    run.append(token)
            elif not (token == '' and unicodedata.name(ch, '').startswith('GREEK')):
                split()  # anything but a letter, an interpunct or a dropped Greek letter (digamma, ...)
    if stack:
        raise ValueError('unclosed Leiden delimiter')
    split()
    return lines


def references(value: str) -> set[str]:
    """The IG and Agora citations in `value`, e.g. 'IG II^2 1234', 'IG II^3,1 567', 'IG IX,1^2 2:123' (part 1 of the
    second edition of IG IX, fascicle 2), 'Agora XVI 12'; every number of a join ('IG II² 333+334') is one citation,
    letters after a number (fragments a, b, ...) are dropped."""
    value = re.sub(r'<[^>]*>', '', value)
    value = unicodedata.normalize('NFKC', value).upper().replace('_', ' ')
    value = value.replace('^', '').replace('I.G.', 'IG').replace('I. G.', 'IG')
    value = re.sub(r'(\bIG\s+[IVX]+)\s*\(([23])\)', r'\1\2 ', value)
    value = re.sub(r'\bIG\s+II\s*3\s+([1-9])(?:\s*[,.:]\s*|\s+)(\d+)', r'IG II3,\1 \2', value)
    found: set[str] = set()
    number_list = r'(\d+[A-Z]*(?:\s*\+\s*\d+[A-Z]*)*)'
    ig = (r'\bIG\s+([IVX]+)\s*([23])?'  # volume and its edition: IG II², IG IV²,1
          r'(?:\s*[,;]\s*(\d)([23])?(?:\s+([1-9])(?=\s*:|\s+\d))?)?'  # part, its edition, fascicle: IG IX,1² 2:123
          r'(?:\s*:\s*|\s+)' + number_list)
    for match in re.finditer(ig, value):
        volume, edition, part, part_edition, fascicle, numbers = match.groups()
        volume += (f'^{edition}' if edition else '') + (f',{part}' if part else '')
        volume += (f'^{part_edition}' if part_edition else '') + (f' {fascicle}:' if fascicle else ' ')
        found.update(f'IG {volume}{int(number)}' for number in re.findall(r'\d+', numbers))
    for match in re.finditer(r'\bAGORA\s+([IVX]+)\s*[,.:]?\s+' + number_list, value):
        found.update(f'Agora {match[1]} {int(number)}' for number in re.findall(r'\d+', match[2]))
    return found


def name_references(base: str) -> set[str]:
    """The IG citations a squeeze's name makes ('IG_II2_2061b+2093b' -> IG II^2 2061 and 2093). An Agora or IAS
    inventory number names a stone, not a publication, so it cites nothing."""
    return {ref for ref in references(base) if ref.startswith('IG ')}


def source_chunks(raw: Path) -> tuple[list[Path], dict]:
    """The chunk files of the crawl in `raw` (`<root>/raw/phi`), each checked against `completion_manifest.json`."""
    manifest = json.loads((raw / 'completion_manifest.json').read_text())
    if manifest.get('format') != 'phi_completion_manifest_v1' or manifest.get('missing_ids'):
        raise ValueError(f'{raw}: no complete PHI crawl')
    entries = [entry for entry in manifest['files'] if entry['path'].startswith('chunks/')]
    if len(entries) != manifest['chunks']:
        raise ValueError(f'{raw}: the manifest lists {len(entries)} chunks, not {manifest["chunks"]}')
    paths = []
    for entry in entries:
        name = entry['path']
        if not re.fullmatch(r'chunks/chunk-[0-9a-f]+\.jsonl\.gz', name):
            raise ValueError(f'{raw}: unexpected member {name}')
        path = raw / name
        if path.stat().st_size != entry['bytes'] or _sha256(path) != entry['sha256']:
            raise ValueError(f'{path}: size or SHA-256 differs from the manifest')
        paths.append(path)
    return paths, manifest


def documents(paths: list[Path]):
    for path in paths:
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            for line in stream:
                yield json.loads(line)['document']


def extract(root: Path) -> dict:
    """Write <root>/staging/phi.txt, phi.docs.tsv (as `open.extract` writes them; documents `phi:<id>`) and
    phi.refs.tsv (document, its references joined by '|'), and return the counters: documents, quarantined (markup
    that does not balance), empty (no visible letter), lines, letters."""
    raw, staging = root / 'raw' / 'phi', root / 'staging'
    paths, manifest = source_chunks(raw)
    stats: Counter = Counter()
    seen: set[str] = set()
    with open(staging / 'phi.txt', 'w', encoding='utf-8') as text, \
            open(staging / 'phi.docs.tsv', 'w', encoding='utf-8') as index, \
            open(staging / 'phi.refs.tsv', 'w', encoding='utf-8') as cited:
        for document in documents(paths):
            identity = str(document['id'])
            if identity in seen:
                raise ValueError(f'PHI document {identity} twice in the crawl')
            seen.add(identity)
            stats['source_documents'] += 1
            try:
                lines = visible_lines(document['lineation'] if 'lineation' in document else document['raw_text'])
            except ValueError:
                stats['quarantined'] += 1
                continue
            if not lines:
                stats['empty'] += 1
                continue
            letters = sum(len(line) for line in lines)
            text.writelines(line + '\n' for line in lines)
            index.write(f'phi:{identity}\tphi\t{len(lines)}\t{letters}\n')
            refs = references(f'{document.get("edition_reference", "")} {document.get("metadata", "")}')
            cited.write(f'phi:{identity}\t{"|".join(sorted(refs))}\n')
            stats.update(documents=1, lines=len(lines), letters=letters)
    if stats['source_documents'] != manifest['parsed_documents']:
        raise ValueError(f'{raw}: {stats["source_documents"]} documents, the manifest counts '
                         f'{manifest["parsed_documents"]}')
    return {'phi': {**stats, 'manifest_sha256': _sha256(raw / 'completion_manifest.json')}}


def read_refs(path: Path) -> dict[str, set[str]]:
    """document -> its references, from a refs file `extract` writes."""
    out = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        doc, refs = line.split('\t')
        out[doc] = set(refs.split('|')) if refs else set()
    return out
