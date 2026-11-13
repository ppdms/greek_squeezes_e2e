"""External Greek text for the language models: download, normalise, decontaminate, count.

Three corpus families: literary Greek (Perseus canonical-greekLit, Open Greek and Latin First1KGreek), documentary
papyri (DDbDP) and the EpiDoc corpora of inscriptions (I.Sicily, IAph2007, IOSPE, IGCyr2/GVCyr2, IIP, the Cretan
Institutional Inscriptions). `SOURCES` records each source's URL and licence (three of the nine are NonCommercial). A
corpus is a line file of the proxy alphabet, `<root>/<family>.txt`, with `<family>.docs.tsv` (one row per document:
id, source, lines, letters) beside it. The PHI Greek Inscriptions (a crawl of inscriptions.packhum.org, not openly
licensed) are staged (`PHI`) only for the calibration of `decontaminate --calibrate`: no corpus or model holds their
text.

    python -m trogs26.lm.open download
    python -m trogs26.lm.open extract [--family literary ... phi] [--jobs 3]    raw TEI/EpiDoc/PHI -> <root>/staging
    TROGS26_WORK=<work> python -m trogs26.lm.open inputs --test-annotations DIR
                                                          <root>/public224.json, <root>/references.json
    python -m trogs26.lm.open decontaminate [--calibrate]                 staging -> <root>/<family>.txt
    python -m trogs26.lm.open count --name clean --corpora literary papyri epigraphic --order 6 \
        --sources ... [--check 1500]
    python -m trogs26.lm.open bits --models clean:6 --weights 0.3 0.45 0.6 --out bits.json

Extraction keeps every Greek edition of every document and drops what is no edition of a text in Greek (other
languages, transliterations, lemmatised copies, editions imported from another database: `_SKIP_SUBTYPES`).
`decontaminate` (`trogs26.lm.decontaminate`) is the one step that removes text because of the squeezes: it checks
the staged corpora against the transcripts of all 274 squeezes, the 224 public and the 50 test squeezes.

`<root>` is `--root` or `<TROGS26_WORK>/../open_text`, `--lm-dir` (default `<TROGS26_WORK>/lm`) holds the language
models, raw downloads are under `<root>/raw`, `manifest.json` records sources, licences, commits and per-corpus
sizes. Only `inputs` reads annotations (the public ones from TROGS26_WORK, the test ones from DIR); the other steps
run anywhere (the numpy-only VM). Normalisation is `normalise`; `Counts` for ~10^8 letters are built by
`count_codes` (numpy), equal to `Counts.from_lines`.
"""
from __future__ import annotations

import argparse
import datetime
import glob
import hashlib
import json
import math
import os
import re
import subprocess
import time
import unicodedata
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np

from .. import config, data
from ..layout import Work
from . import artifact
from .ppm import Counts

GREEK = 'ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ'
PROXY = 'ABGDEZHQIKLMNXOPRSTYFCUW'
LETTERS = PROXY  # the proxy letters; ':' (interpunct) is the only other character a corpus line holds
_SPLIT = '\x1e'  # in a raw text stream: a gap, a line break, a block boundary; contexts never cross it
_XML_LANG = '{http://www.w3.org/XML/1998/namespace}lang'
_INTERPUNCT = '·\u0387∶⁝⁚⸱⸫⋮•∙⋅⁞⸪⸬⸭:'
_INTERPUNCT_LABELS = frozenset({'interpunct', 'dipunct', 'tripunct', 'punct', 'stop', 'dot'})  # <g> types and refs
_UNWRITTEN = frozenset({'omitted', 'subaudible', 'abbreviation', 'explanation', 'parallel', 'ellipsis', 'expansion'})
_DROPPED_LETTERS = ('DIGAMMA', 'KOPPA', 'SAMPI', 'STIGMA')
_LETTER_CATEGORIES = ('Lu', 'Ll', 'Lt', 'Lo')
# edition subtypes that are no edition of the inscription itself: transliterations, lemmatised and segmented copies,
# editions imported from another database
_SKIP_SUBTYPES = frozenset({'simple-lemmatized', 'transliteration', 'transliteration-primary',
                            'transcription_segmented', 'PHI', 'phi'})
_VERSE_GROUP = 20  # consecutive verse lines of a literary text that make one corpus line


# ---------------------------------------------------------------------------------------------------------------
# Normalisation

_LETTER = {**dict(zip(GREEK, PROXY)), **dict(zip(GREEK.lower(), PROXY)),
           'ς': 'S', 'ϲ': 'S', 'Ϲ': 'S', 'ϐ': 'B', 'ϑ': 'Q', 'ϴ': 'Q', 'ϒ': 'Y', 'ϕ': 'F', 'ϖ': 'P', 'ϰ': 'K',
           'ϱ': 'R', 'ϵ': 'E', 'µ': 'M', '\u0345': 'I', '\u037a': 'I', _SPLIT: '\n'}


def _classify(ch: str, punct: bool) -> str:
    """The corpus text of one character (after NFD): a proxy letter, ':' (interpunct, if `punct`), '\\n' (marks
    missing text: a gap, or a Greek symbol that is no letter) or '' (dropped: marks, digits, Latin, punctuation)."""
    if ch in _LETTER:
        return _LETTER[ch]
    category = unicodedata.category(ch)
    if category[0] == 'M':
        return ''
    if ch in _INTERPUNCT:
        return ':' if punct else ''
    name = unicodedata.name(ch, '')
    if name.startswith(('GREEK', 'ANCIENT GREEK')):
        if category in _LETTER_CATEGORIES:
            return '' if any(word in name for word in _DROPPED_LETTERS) else '\n'
        return '\n' if category in ('Nl', 'No', 'So', 'Sm') else ''
    if category in _LETTER_CATEGORIES and not name.startswith('LATIN'):
        return '\n'  # a letter of another script (Hebrew, Cyrillic, ...): a different language, not Greek text
    return ''


class _Table(dict):
    """str.translate table that classifies each code point on first use."""

    def __init__(self, punct: bool):
        super().__init__()
        self.punct = punct

    def __missing__(self, code: int) -> str:
        value = self[code] = _classify(chr(code), self.punct)
        return value


_TABLES = {True: _Table(True), False: _Table(False)}


def normalise(raw: str, punct: bool) -> list[str]:
    """Lines of the proxy alphabet in `raw` (text with `_SPLIT` marks): NFD, only the 24 letters, the iota
    subscript as an iota after its vowel, every sigma S, interpuncts as ':' if `punct`; a split at every mark of
    missing text. Empty lines are dropped."""
    text = unicodedata.normalize('NFD', raw).translate(_TABLES[punct])
    return [line for line in text.split('\n') if line]


# ---------------------------------------------------------------------------------------------------------------
# TEI / EpiDoc

class Rules(NamedTuple):
    """What of a TEI text is the transcription."""
    punct: bool
    blocks: frozenset  # elements that start and end a corpus line
    drop: frozenset  # elements whose text is not transcribed (their tail is)
    drop_split: frozenset  # likewise, and they separate the text around them
    gaps: frozenset  # elements that mark missing text
    prefer: tuple  # of the alternatives in <choice>, the first of these elements is kept
    verse: bool  # group consecutive <l> into one line


EPIDOC = Rules(
    True, frozenset({'div', 'ab', 'p', 'lg', 'l', 'sp', 'list', 'item', 'table', 'row', 'cell'}),
    frozenset({'note', 'ex', 'del', 'rdg', 'figure', 'figDesc', 'graphic', 'bibl', 'listBibl', 'certainty'}),
    frozenset({'head', 'label', 'speaker', 'stage'}),
    frozenset({'gap', 'space', 'lacunaStart', 'lacunaEnd'}), ('sic', 'orig', 'abbr'), False)
LITERARY = Rules(
    False, frozenset({'div', 'ab', 'p', 'lg', 'sp', 'list', 'item', 'table', 'row', 'cell', 'castList'}),
    frozenset({'note', 'del', 'surplus', 'rdg', 'figure', 'bibl', 'listBibl', 'certainty'}),
    frozenset({'head', 'label', 'speaker', 'stage', 'trailer', 'argument', 'titlePart', 'byline', 'epigraph'}),
    frozenset({'gap', 'lacunaStart', 'lacunaEnd'}), ('corr', 'reg', 'expan'), True)


def _local(tag) -> str:
    return tag.rsplit('}', 1)[-1] if isinstance(tag, str) else ''


def is_greek(lang: str) -> bool:
    """Whether an `xml:lang` value names Greek text in Greek script ('grc', not its transliteration)."""
    return lang == 'grc' or (lang.startswith('grc-') and 'Latn' not in lang)


class _Walker:
    """Collects the transcription of a TEI element into a raw stream (text and `_SPLIT` marks)."""

    def __init__(self, rules: Rules):
        self.rules, self.out, self.verse = rules, [], 0

    def split(self) -> None:
        self.out.append(_SPLIT)
        self.verse = 0

    def children(self, el: ET.Element) -> None:
        for child in el:
            self.walk(child)
            if child.tail:
                self.out.append(child.tail)

    def glyph(self, el: ET.Element) -> None:
        """<g>: an interpunct (in an inscription or papyrus) is ':', an alphabetic glyph is its letters, any other
        symbol (cross, numeral, ornament) separates the text."""
        label = set(re.split(r'[^a-z]+', f"{el.get('type', '')} {el.get('ref', '')}".lower()))
        text = ''.join(el.itertext()).strip()
        if self.rules.punct and (label & _INTERPUNCT_LABELS or (text and set(text) <= set(_INTERPUNCT))):
            self.out.append(':')
        elif 'alphabetic' in label:
            self.out.append(text)
        else:
            self.split()

    def walk(self, el: ET.Element) -> None:
        rules, tag = self.rules, _local(el.tag)
        lang = el.get(_XML_LANG) or el.get('lang')
        if lang and not is_greek(lang):
            self.split()
        elif tag in rules.gaps or tag == 'lb':
            self.split()
        elif tag == 'g':
            self.glyph(el)
        elif tag in rules.drop_split:
            self.split()
        elif tag in rules.drop or (tag == 'supplied' and el.get('reason') in _UNWRITTEN):
            pass
        elif tag == 'choice':
            kids = list(el)
            chosen = next((k for k in kids if _local(k.tag) in rules.prefer), kids[0] if kids else None)
            if chosen is not None:
                self.walk(chosen)
            self.out.extend(k.tail for k in kids if k.tail)
        elif tag == 'app':
            for child in el:
                if _local(child.tag) == 'lem':
                    self.walk(child)
        else:
            block = tag in rules.blocks
            if block:
                self.split()
            elif tag == 'l' and rules.verse:
                if self.verse and self.verse % _VERSE_GROUP == 0:
                    self.split()
                self.verse += 1
            if el.text:
                self.out.append(el.text)
            self.children(el)
            if block:
                self.split()


def transcription(el: ET.Element, rules: Rules) -> list[str]:
    """Corpus lines of the TEI element `el` (its transcription under `rules`)."""
    walker = _Walker(rules)
    walker.walk(el)
    return normalise(''.join(walker.out), rules.punct)


def _editions(root: ET.Element, stats: Counter) -> list[ET.Element]:
    """The Greek edition divs of an EpiDoc file: no transliteration, lemmatised or imported copy (`_SKIP_SUBTYPES`);
    where both a transcription and a diplomatic edition exist, the transcription."""
    found = []
    for div in root.iter():
        if _local(div.tag) != 'div' or div.get('type') != 'edition':
            continue
        lang, subtype = div.get(_XML_LANG) or div.get('lang'), div.get('subtype')
        if subtype in _SKIP_SUBTYPES:
            stats[f'skipped_edition:{subtype}'] += 1
        elif lang and not is_greek(lang):
            stats['skipped_lang'] += 1
        else:
            found.append(div)
    if any(d.get('subtype') == 'transcription' for d in found):
        found = [d for d in found if d.get('subtype') != 'diplomatic']
    return found


def epidoc_documents(stem: str, root: ET.Element, stats: Counter) -> list[tuple[str, list[str]]]:
    """[(document id, lines)] of one EpiDoc file: its Greek editions, one document per file."""
    lines = [line for div in _editions(root, stats) for line in transcription(div, EPIDOC)]
    return [(stem, lines)] if lines else []


def literary_documents(stem: str, root: ET.Element, stats: Counter) -> list[tuple[str, list[str]]]:
    """[(document id, lines)] of one literary TEI file: one document per top-level text part (book, speech, ...),
    or the whole edition if it has fewer than two."""
    body = next((e for e in root.iter() if _local(e.tag) == 'body'), None)
    if body is None:
        stats['no_body'] += 1
        return []
    edition = next((e for e in body.iter() if _local(e.tag) == 'div' and e.get('type') == 'edition'), body)
    parts = [e for e in edition if _local(e.tag) == 'div']
    if len(parts) < 2:
        lines = transcription(edition, LITERARY)
        return [(stem, lines)] if lines else []
    docs = [(f'{stem}#{i + 1}', transcription(part, LITERARY)) for i, part in enumerate(parts)]
    return [(doc, lines) for doc, lines in docs if lines]


# ---------------------------------------------------------------------------------------------------------------
# Sources

@dataclass(frozen=True)
class Source:
    name: str
    family: str  # literary | papyri | epigraphic
    glob: str  # the text files, relative to <root>/raw
    repo: str  # directory of <root>/raw holding the clone (for the commit) or the download
    url: str
    licence: str
    licence_basis: str  # where the licence was read
    commercial: bool  # False if the licence has a NonCommercial clause

    @property
    def parse(self):
        return literary_documents if self.family == 'literary' else epidoc_documents


SOURCES = (
    Source('perseus_greeklit', 'literary', 'canonical-greekLit/data/*/*/*-grc*.xml', 'canonical-greekLit',
           'https://github.com/PerseusDL/canonical-greekLit', 'CC BY-SA 4.0',
           'repository license.md and README ("unless otherwise indicated, all contents")', True),
    Source('first1k_greek', 'literary', 'First1KGreek/data/*/*/*-grc*.xml', 'First1KGreek',
           'https://github.com/OpenGreekAndLatin/First1KGreek', 'CC BY-SA 4.0',
           'repository license.md and the <licence> element of each file', True),
    Source('ddbdp', 'papyri', 'idp.data/DDbDP/*/*.xml', 'idp.data', 'https://github.com/papyri/idp.data',
           'CC BY 3.0', 'repository README (LICENSE) and the <availability> of each file', True),
    Source('isicily', 'epigraphic', 'ISicily/inscriptions/*.xml', 'ISicily', 'https://github.com/ISicily/ISicily',
           'CC BY 4.0', 'repository licence.txt and the <licence> of each file', True),
    Source('iaph2007', 'epigraphic', 'IAph2007/*/x/*.xml', 'IAph2007', 'https://doi.org/10.17863/CAM.117714',
           'CC BY 2.5 (each file; the Cambridge repository record says CC BY)',
           'publicationStmt of each file; repository.cam.ac.uk record 97b051f9', True),
    Source('iospe', 'epigraphic', 'iospe/kiln/webapps/ROOT/content/xml/tei/inscriptions/*.xml', 'iospe',
           'https://github.com/kingsdigitallab/iospe', 'CC BY 2.0 UK (catalogue record; no data licence in the repo)',
           'Open Digital Epigraphy Hub, open-epihub.cnr.it/resources/90 (the repository LICENSE, MIT, is for code)',
           True),
    Source('igcyr2', 'epigraphic', 'IGCyr2/IGCyr2-GVCyr2-inscriptions/*.xml', 'IGCyr2',
           'https://doi.org/10.6092/unibo/amsacta/7796', 'CC BY-NC-SA 4.0',
           'amsacta.unibo.it/id/eprint/7796 and the <availability> of each file', False),
    Source('iip', 'epigraphic', 'iip-texts/epidoc-files/*.xml', 'iip-texts',
           'https://github.com/Brown-University-Library/iip-texts', 'CC BY-NC 4.0',
           'inscriptionsisraelpalestine.org/copyright', False),
    Source('cretan', 'epigraphic', 'CretanInscriptions/webapps/ROOT/content/xml/epidoc/*.xml', 'CretanInscriptions',
           'https://github.com/IreneVagionakis/CretanInscriptions', 'CC BY-NC-SA 4.0',
           'repository Data-LICENSE', False),
)
FAMILIES = ('literary', 'papyri', 'epigraphic')
PHI = 'phi'  # the staged PHI crawl, read by the calibration of decontamination only
# clone -> (URL, commit); idp.data is checked out sparsely (DDbDP only)
_REPOS = {'canonical-greekLit': ('https://github.com/PerseusDL/canonical-greekLit.git',
                                 '282e6f5a3ec12e6ca67e44cd4fa3487526e3b421'),
          'First1KGreek': ('https://github.com/OpenGreekAndLatin/First1KGreek.git',
                           'ed91552c52530c8f5a3ddf2dd767e899372a804c'),
          'idp.data': ('https://github.com/papyri/idp.data.git', '75141dd5c0bd1434eb659ab8dcb8795add29a81f'),
          'ISicily': ('https://github.com/ISicily/ISicily.git', 'fffc6718a80a1e6773dab33a1230c4f34297cc93'),
          'iip-texts': ('https://github.com/Brown-University-Library/iip-texts.git',
                        '0b7dc8358ccdfd0c9391f049da4839fbd91c26e5'),
          'iospe': ('https://github.com/kingsdigitallab/iospe.git', '571aec42aa0b85ed8b0433594e57a130df87c640'),
          'CretanInscriptions': ('https://github.com/IreneVagionakis/CretanInscriptions.git',
                                 '70fc617724034433870910fe3637331d9cbf2014')}
# archive -> (URL, SHA-256)
_ZIPS = {'IGCyr2': ('https://amsacta.unibo.it/7796/1/IGCyr2-GVCyr2.zip',
                    '68e14eacdbd45f2d6e090cc5bdcbd283a347eba2469a798f49c1c8ddc046df13'),
         'IAph2007': ('https://www.repository.cam.ac.uk/bitstreams/6941385a-729f-483b-a7ab-ae1363a21779/download',
                      '87de3599fc9b610d3c28f0e1aebbabf5915f9da5c30f1419ebd8d3bafe36f8f5')}


def default_root() -> Path:
    return Path(os.environ['TROGS26_WORK']).parent / 'open_text'


def download(root: Path) -> None:
    """The clones at the commits of `_REPOS` (depth 1; DDbDP only of idp.data) and the archives of `_ZIPS` (checked
    against their SHA-256) into <root>/raw; present ones are kept. The PHI crawl, for the calibration, is not
    downloaded: its chunks and completion manifest are placed in <root>/raw/phi."""
    raw = root / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    for name, (url, commit) in _REPOS.items():
        clone = raw / name
        if (clone / '.git').exists():
            continue
        git = ['git', '-C', str(clone)]
        subprocess.run(['git', 'init', '-q', str(clone)], check=True)
        subprocess.run([*git, 'remote', 'add', 'origin', url], check=True)
        sparse = name == 'idp.data'
        if sparse:
            subprocess.run([*git, 'sparse-checkout', 'set', '--cone', 'DDbDP'], check=True)
        subprocess.run([*git, 'fetch', '-q', '--depth', '1', *(['--filter=blob:none'] if sparse else []), 'origin',
                        commit], check=True)
        subprocess.run([*git, 'checkout', '-q', 'FETCH_HEAD'], check=True)
    for name, (url, digest) in _ZIPS.items():
        target = raw / name
        if not target.exists():
            archive = raw / f'{name}.zip'
            urllib.request.urlretrieve(url, archive)
            if hashlib.sha256(archive.read_bytes()).hexdigest() != digest:
                raise ValueError(f'{archive}: SHA-256 differs from the pinned archive')
            with zipfile.ZipFile(archive) as z:
                z.extractall(target)
    for inner in glob.glob(str(raw / 'IAph2007' / '*' / '*.zip')):
        with zipfile.ZipFile(inner) as z:
            z.extractall(Path(inner).parent / 'x')


def _revision(raw: Path, source: Source) -> dict:
    """Where a source came from: the clone's commit and date, the archive's SHA-256, or the crawl manifest's."""
    repo = raw / source.repo
    if (repo / '.git').exists():
        out = subprocess.run(['git', '-C', str(repo), 'log', '-1', '--format=%H %cI'], capture_output=True,
                             text=True, check=True).stdout.split()
        return {'commit': out[0], 'commit_date': out[1]}
    archive = raw / f'{source.repo}.zip'
    if archive.exists():
        return {'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
                'downloaded': datetime.date.fromtimestamp(archive.stat().st_mtime).isoformat()}
    crawl = repo / 'completion_manifest.json'
    if crawl.exists():
        return {'manifest_sha256': hashlib.sha256(crawl.read_bytes()).hexdigest(),
                'crawled': json.loads(crawl.read_text())['created_utc']}
    return {}


# ---------------------------------------------------------------------------------------------------------------
# Extraction

def _parse_file(task: tuple[str, str, str]) -> tuple[list[tuple[str, list[str]]], Counter]:
    """(source name, family, path) -> the file's documents and its counters; a file that does not parse is
    counted, not fatal."""
    name, family, path = task
    stats: Counter = Counter()
    source = next(s for s in SOURCES if s.name == name)
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        stats['parse_error'] += 1
        return [], stats
    stem = Path(path).stem
    docs = source.parse(f'{name}:{stem}' if family != 'literary' else stem, root, stats)
    stats['files'] += 1
    stats['documents'] += len(docs)
    return docs, stats


def _files(raw: Path, source: Source) -> list[Path]:
    return sorted(Path(p) for p in glob.glob(str(raw / source.glob)) if not p.endswith('__cts__.xml'))


def _literary_files(raw: Path) -> dict[str, list[Path]]:
    """One file per work (CTS work URN tlgNNNN.tlgNNN), canonical-greekLit before First1KGreek, the largest first."""
    chosen: dict[str, tuple[str, Path]] = {}
    for source in (s for s in SOURCES if s.family == 'literary'):
        by_work: dict[str, Path] = {}
        for path in _files(raw, source):
            work = '.'.join(path.name.split('.')[:2])
            if work not in by_work or path.stat().st_size > by_work[work].stat().st_size:
                by_work[work] = path
        for work, path in by_work.items():
            chosen.setdefault(work, (source.name, path))
    out: dict[str, list[Path]] = {}
    for source_name, path in chosen.values():
        out.setdefault(source_name, []).append(path)
    return out


def extract(root: Path, families: list[str], jobs: int) -> None:
    """Write <root>/staging/<family>.txt, .docs.tsv and .stats.json for each family (before decontamination); `PHI`
    is read by `phi.extract`."""
    raw, staging = root / 'raw', root / 'staging'
    staging.mkdir(parents=True, exist_ok=True)
    for family in families:
        started = time.perf_counter()
        if family == PHI:
            from .phi import extract as extract_phi
            stats = extract_phi(root)
            (staging / f'{PHI}.stats.json').write_text(json.dumps(stats, indent=1, sort_keys=True))
            print(family, f'{time.perf_counter() - started:.0f} s', json.dumps(stats, sort_keys=True), flush=True)
            continue
        sources = [s for s in SOURCES if s.family == family]
        literary = _literary_files(raw) if family == 'literary' else {}
        tasks = [(s.name, family, str(p)) for s in sources
                 for p in (literary.get(s.name, []) if family == 'literary' else _files(raw, s))]
        per_source = {s.name: Counter() for s in sources}
        sizes = {s.name: Counter() for s in sources}
        with ProcessPoolExecutor(jobs) as pool, open(staging / f'{family}.txt', 'w', encoding='utf-8') as text, \
                open(staging / f'{family}.docs.tsv', 'w', encoding='utf-8') as index:
            for (name, _, _), (docs, stats) in zip(tasks, pool.map(_parse_file, tasks, chunksize=16)):
                per_source[name].update(stats)
                for doc, lines in docs:
                    text.writelines(line + '\n' for line in lines)
                    letters = sum(len(line) for line in lines)
                    index.write(f'{doc}\t{name}\t{len(lines)}\t{letters}\n')
                    sizes[name].update(lines=len(lines), letters=letters)
        stats = {name: {**per_source[name], **sizes[name]} for name in per_source}
        (staging / f'{family}.stats.json').write_text(json.dumps(stats, indent=1, sort_keys=True))
        print(family, f'{time.perf_counter() - started:.0f} s', json.dumps(stats, sort_keys=True), flush=True)


# ---------------------------------------------------------------------------------------------------------------
# Corpus files

class Docs(NamedTuple):
    ids: list[str]
    sources: list[str]
    lines: np.ndarray  # lines per document
    letters: np.ndarray


def read_docs(path: Path) -> Docs:
    rows = [line.rstrip('\n').split('\t') for line in path.read_text(encoding='utf-8').splitlines()]
    return Docs([r[0] for r in rows], [r[1] for r in rows], np.array([int(r[2]) for r in rows]),
                np.array([int(r[3]) for r in rows]))


# ---------------------------------------------------------------------------------------------------------------
# Inputs from the annotations

def view_letters(path: Path) -> str:
    """Every annotated letter of one view's letters file as the scorer reads it (`getRowTranscript` with all
    confidence levels, Latin proxies), rows joined without separators, the 24 letters only."""
    from .. import official
    text = official.v2.convert(official.v2.getRowTranscript(str(path), all_conf=True)).upper()
    return ''.join(ch for ch in text if ch in LETTERS)


def write_inputs(root: Path, work: Work, cfg: dict, test_annotations: Path) -> None:
    """<root>/public224.json: the folds and the transcript rows of the 224 public squeezes (for the bits grid and the
    neural model); <root>/references.json: cohort -> squeeze -> the letters of each view (`view_letters`), of the 224
    public squeezes (annotations of TROGS26_WORK) and of the 50 test squeezes (the letters files in
    `test_annotations`), for `decontaminate`. The steps after this one need no annotation."""
    from .build import transcript_rows
    from ..score import group_letters_files
    folds = data.folds(cfg['data']['folds'])
    public = data.all_bases(folds)
    rows = {base: transcript_rows(work, base) for base in public}
    (root / 'public224.json').write_text(json.dumps(
        {'folds': {str(k): list(v) for k, v in folds.items()}, 'rows': rows}))
    test = group_letters_files(test_annotations)
    if sorted(test) != sorted(data.splits()['trogs26']):
        raise ValueError(f'{test_annotations} does not hold exactly the TROGS-26 squeezes')
    references = {'public224': {base: [view_letters(work.annotation(base, v)) for v in (1, 2)] for base in public},
                  'trogs26': {base: [view_letters(path) for path in paths] for base, paths in sorted(test.items())}}
    (root / 'references.json').write_text(json.dumps(references))


def read_inputs(root: Path) -> tuple[dict[int, list[str]], dict[str, list[str]]]:
    raw = json.loads((root / 'public224.json').read_text())
    return {int(k): v for k, v in raw['folds'].items()}, raw['rows']


def write_manifest(root: Path, families: list[str]) -> None:
    """Update <root>/manifest.json with the sources of `families`, their corpus sizes and what decontamination
    removed from each."""
    path = root / 'manifest.json'
    manifest = json.loads(path.read_text()) if path.exists() else {'sources': {}, 'corpora': {}}
    raw = root / 'raw'
    report = json.loads((root / 'decontamination.json').read_text())
    for source in SOURCES:
        if source.family in families:
            manifest['sources'][source.name] = {**asdict(source), **_revision(raw, source)}
    for family in families:
        docs = read_docs(root / f'{family}.docs.tsv')
        manifest['corpora'][family] = {
            'file': f'{family}.txt', 'documents': len(docs.ids), 'lines': int(docs.lines.sum()),
            'letters': int(docs.letters.sum()),
            'letters_by_source': {s: int(docs.letters[[i for i, x in enumerate(docs.sources) if x == s]].sum())
                                  for s in sorted(set(docs.sources))},
            'decontamination': report['families'][family],
            'staging': json.loads((root / 'staging' / f'{family}.stats.json').read_text())}
    manifest['updated'] = datetime.datetime.now().astimezone().isoformat(timespec='seconds')
    manifest['scope'] = ('every Greek edition of every document (no non-Greek edition, transliteration, lemmatised '
                         'copy or edition imported from another database), decontaminated against the 224 public and '
                         'the 50 test squeezes (trogs26.lm.decontaminate: rule in decontamination.json)')
    manifest['normalisation'] = (f'NFD; {GREEK} -> {PROXY} (final and lunate sigma S, iota subscript I after its '
                                 'vowel); interpuncts ":" in inscriptions and papyri only; all else dropped; gaps, '
                                 'line breaks, block boundaries and non-letter Greek symbols split')
    path.write_text(json.dumps(manifest, indent=1))


# ---------------------------------------------------------------------------------------------------------------
# Counting

def _level(keys: np.ndarray, totals: np.ndarray, m: int, base: int, alphabet: str) -> dict[str, Counter]:
    """counts[m] of `Counts`: context -> Counter(letter -> count) from the sorted m-gram codes and their counts."""
    chars = np.array(list(alphabet))
    if len(keys) == 0:
        return {}
    last = chars[keys % base].tolist()
    totals = totals.tolist()
    if m == 1:
        return {'': Counter(dict(zip(last, totals)))}
    context = keys // base
    starts = np.flatnonzero(np.r_[True, context[1:] != context[:-1]])
    ends = starts[1:].tolist() + [len(keys)]
    letters = np.frombuffer(alphabet.encode('ascii'), dtype=np.uint8)
    digits = np.stack([letters[(context[starts] // base ** (m - 2 - j)) % base] for j in range(m - 1)], axis=1)
    names = np.ascontiguousarray(digits).view(f'S{m - 1}').reshape(-1).astype(f'U{m - 1}').tolist()
    level: dict[str, Counter] = {}
    for name, a, b in zip(names, starts.tolist(), ends):
        counter: Counter = Counter()
        dict.update(counter, zip(last[a:b], totals[a:b]))
        level[name] = counter
    return level


def _unique_counts(parts: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    keys = np.concatenate([k for k, _ in parts])
    totals = np.concatenate([c for _, c in parts])
    order = np.argsort(keys, kind='stable')
    keys, totals = keys[order], totals[order]
    first = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
    return keys[first], np.add.reduceat(totals, first)


def count_codes(codes: np.ndarray, order: int, alphabet: str) -> Counts:
    """`Counts.from_lines` over letter indices: `codes` holds an index into `alphabet` per letter and the separator
    `len(alphabet)` between lines; n-grams never cross a separator."""
    base = len(alphabet) + 1
    separator = len(alphabet)
    levels: list[dict[str, Counter]] = [{}]
    block = 1 << 24
    for m in range(1, order + 1):
        parts: list[tuple[np.ndarray, np.ndarray]] = []
        size = 0
        for start in range(0, max(len(codes) - m + 1, 0), block):
            stop = min(start + block, len(codes) - m + 1)
            gram = np.zeros(stop - start, dtype=np.int64)
            clean = np.ones(stop - start, dtype=bool)
            for j in range(m):
                piece = codes[start + j:stop + j]
                gram = gram * base + piece
                clean &= piece != separator
            parts.append(np.unique(gram[clean], return_counts=True))
            size += len(parts[-1][0])
            if size > 2 * block:
                parts = [_unique_counts(parts)]
                size = len(parts[0][0])
        keys, totals = _unique_counts(parts) if parts else (np.zeros(0, np.int64), np.zeros(0, np.int64))
        levels.append(_level(keys, totals, m, base, alphabet))
    return Counts(order, alphabet, levels)


def count_lines(lines, order: int, alphabet: str) -> Counts:
    """`Counts.from_lines(lines, order, alphabet)` by numpy (lines in `alphabet` letters already)."""
    index = {ch: i for i, ch in enumerate(alphabet)}
    parts = []
    for line in lines:
        parts.append(np.array([index[ch] for ch in line if ch in index], dtype=np.int64))
        parts.append(np.array([len(alphabet)], dtype=np.int64))
    return count_codes(np.concatenate(parts) if parts else np.zeros(0, np.int64), order, alphabet)


def selected_bytes(raw: np.ndarray, line_ok: np.ndarray, stride: int) -> np.ndarray:
    """The bytes of the lines of `raw` (a line file) where `line_ok`, keeping every `stride`-th of those lines."""
    ends = raw == 10
    line = np.cumsum(ends, dtype=np.int32) - ends  # line index of each byte
    keep = line_ok[line]
    if stride > 1:
        rank = np.cumsum(line_ok, dtype=np.int64) - 1  # position of a line among the lines that are ok
        keep &= rank[line] % stride == 0
    return raw[keep]


def count_corpus(pieces: list[np.ndarray], order: int, alphabet: str) -> tuple[Counts, dict]:
    """Counts of the line files' bytes `pieces`; every character outside `alphabet` is dropped, as `keep_chars` does."""
    lut = np.full(256, 255, dtype=np.uint8)
    for i, ch in enumerate(alphabet):
        lut[ord(ch)] = i
    lut[10] = len(alphabet)
    codes = []
    for raw in pieces:
        mapped = lut[raw]
        codes.append(mapped[mapped != 255])
    codes = np.concatenate(codes)
    lines = int((codes == len(alphabet)).sum())
    return count_codes(codes, order, alphabet), {'lines': lines, 'letters': int(len(codes) - lines)}


def verify_counts(corpus: Path, order: int, alphabet: str, sample: int) -> None:
    """Raise unless `Counts.from_lines`, `count_lines` and `count_corpus` agree exactly on `sample` lines of
    `corpus` (evenly spaced)."""
    lines = corpus.read_text(encoding='utf-8').splitlines()
    lines = lines[::max(len(lines) // sample, 1)][:sample]
    expected = Counts.from_lines(lines, order, alphabet)
    raw = np.frombuffer(''.join(line + '\n' for line in lines).encode('ascii'), dtype=np.uint8)
    through_lines = count_lines(lines, order, alphabet)
    through_bytes, size = count_corpus([raw], order, alphabet)
    if not (expected.counts == through_lines.counts == through_bytes.counts):
        raise AssertionError(f'fast counts differ from Counts.from_lines on {corpus.name}')
    print(f'fast counts == Counts.from_lines: {corpus.name}, order {order}, {len(lines)} lines, {size["letters"]} '
          f'letters, {sum(len(c) for c in expected.counts)} contexts', flush=True)


def build_model(root: Path, lm_dir: Path, alphabet: str, name: str, corpora: list[str], order: int,
                cap: int | None, sources: list[str] | None, check: int) -> None:
    """Count the corpora (lines of `sources` only, if given; every n-th line so a corpus has about `cap` letters, if
    given) into the language model artifact `lm_dir/open_<name>` of the given `order`."""
    pieces, used = [], {}
    for family in corpora:
        docs = read_docs(root / f'{family}.docs.tsv')
        ok = np.array([sources is None or s in sources for s in docs.sources])
        letters = int(docs.letters[ok].sum())
        stride = max(round(letters / cap), 1) if cap else 1
        if check:
            verify_counts(root / f'{family}.txt', order, alphabet, check)
        raw = np.fromfile(root / f'{family}.txt', dtype=np.uint8)
        pieces.append(selected_bytes(raw, np.repeat(ok, docs.lines), stride))
        used[family] = {'letters_available': letters, 'stride': stride}
    started = time.perf_counter()
    counts, size = count_corpus(pieces, order, alphabet)
    wanted = {s.name: {'licence': s.licence, 'url': s.url} for s in SOURCES if s.family in corpora
              and (sources is None or s.name in sources)}
    artifact.save(counts, lm_dir / f'open_{name}', {
        'created_by': 'trogs26.lm.open', 'created': datetime.date.today().isoformat(), 'corpora': used,
        'sources': wanted, **size, 'cap': cap, 'seconds': round(time.perf_counter() - started, 1)})
    path = root / 'manifest.json'
    manifest = json.loads(path.read_text()) if path.exists() else {'sources': {}, 'corpora': {}}
    manifest.setdefault('models', {})[f'open_{name}'] = {'order': order, 'corpora': used, 'cap': cap, **size,
                                                           'sources': sorted(wanted)}
    path.write_text(json.dumps(manifest, indent=1))
    print(f'open_{name}', size, {m: len(c) for m, c in enumerate(counts.counts)}, flush=True)


# ---------------------------------------------------------------------------------------------------------------
# Out-of-fold bits

def letter_logprobs(model, rows: list[str]) -> np.ndarray:
    """log P(letter | row so far) of every letter of `rows` (contexts restart at each row, as in decoding)."""
    return np.array([model.cond_logprob(ch, row[:i]) for row in rows for i, ch in enumerate(row)])


def bits_grid(root: Path, lm_dir: Path, cfg: dict, models: list[str], weights: list[float], out: Path) -> None:
    """Bits per letter of every fold's public transcript rows under (1 - a) P_chain + a P_open, `models` given as
    `name:max_order` (artifact `lm_dir/open_<name>`), P_chain the PPM that excludes the fold; a = 0 is the plain
    chain model. Checked against `Mixture` on a sample."""
    from .mixture import Mixture
    from .ppm import keep_chars
    folds, rows = read_inputs(root)
    alphabet, reach = cfg['alphabet'], cfg['lm']['decode_max_order']
    fold_rows = {k: [r for r in (keep_chars(row, alphabet) for base in folds[k] for row in rows[base]) if r]
                 for k in sorted(folds)}
    chain = {k: artifact.load(lm_dir / cfg['lm']['oof'].format(fold=k), reach) for k in fold_rows}
    base_lp = {k: letter_logprobs(chain[k], fold_rows[k]) for k in fold_rows}
    letters = {k: len(base_lp[k]) for k in fold_rows}
    result: dict = {'letters': {str(k): v for k, v in letters.items()}, 'bits': {'chain': {}}}

    def pooled(per_fold: dict[int, float]) -> dict[str, float]:
        total = sum(per_fold[k] * letters[k] for k in per_fold) / sum(letters.values())
        return {**{f'f{k}': round(v, 5) for k, v in per_fold.items()}, 'all': round(total, 5)}

    result['bits']['chain'] = pooled({k: -base_lp[k].sum() / math.log(2) / letters[k] for k in fold_rows})
    for spec in models:
        name, order = spec.split(':')
        open_lm = artifact.load(lm_dir / f'open_{name}', int(order))
        open_lp = {k: letter_logprobs(open_lm, fold_rows[k]) for k in fold_rows}
        table = {'open alone': pooled({k: -open_lp[k].sum() / math.log(2) / letters[k] for k in fold_rows})}
        for a in weights:
            mixed = {k: np.logaddexp(math.log1p(-a) + base_lp[k], math.log(a) + open_lp[k]) for k in fold_rows}
            table[f'{a:g}'] = pooled({k: -mixed[k].sum() / math.log(2) / letters[k] for k in fold_rows})
        check = Mixture(chain[0], open_lm, weights[-1])
        sample = fold_rows[0][:50]
        by_class = sum(check.text_logprob(row) for row in sample)
        direct = sum(np.logaddexp(math.log1p(-weights[-1]) + letter_logprobs(chain[0], [row]),
                                  math.log(weights[-1]) + letter_logprobs(open_lm, [row])).sum() for row in sample)
        table['mixture_class_check'] = abs(by_class - float(direct))
        result['bits'][spec] = table
        del open_lm, check
        print(spec, {a: row['all'] for a, row in table.items() if isinstance(row, dict)}, flush=True)
    out.write_text(json.dumps(result, indent=1))
    print(markdown(result))


def markdown(result: dict) -> str:
    """The bits grid as a table: a row per model and weight, a column per fold and overall."""
    folds = sorted(result['letters'], key=int)
    lines = ['| model | a | ' + ' | '.join(f'f{k}' for k in folds) + ' | all |',
             '|---|---|' + '---|' * (len(folds) + 1)]
    for model, table in result['bits'].items():
        for a, row in ([('', table)] if model == 'chain' else table.items()):
            if isinstance(row, dict):
                lines.append(f'| {model} | {a} | ' + ' | '.join(f'{row[f"f{k}"]:.4f}' for k in folds)
                             + f' | {row["all"]:.4f} |')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--root', type=Path, help='corpus directory (default <TROGS26_WORK>/../open_text)')
    parser.add_argument('--lm-dir', type=Path, help='language model directory (default <TROGS26_WORK>/lm)')
    parser.add_argument('--config', default='configs/pagenet_only_style_neural.json')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('download')
    p = sub.add_parser('inputs')
    p.add_argument('--test-annotations', type=Path, required=True, help='directory of the TROGS-26 letters files')
    p = sub.add_parser('decontaminate')  # always every family: rarity is counted over all of them
    p.add_argument('--calibrate', action='store_true', help='measure the calibration on the staged PHI crawl')
    p = sub.add_parser('extract')
    p.add_argument('--family', nargs='+', choices=(*FAMILIES, PHI), default=list(FAMILIES))
    p.add_argument('--jobs', type=int, default=3)
    p = sub.add_parser('manifest')
    p.add_argument('--family', nargs='+', choices=FAMILIES, default=list(FAMILIES))
    p = sub.add_parser('count')
    p.add_argument('--name', required=True, help='artifact open_<name>')
    p.add_argument('--corpora', nargs='+', choices=FAMILIES, required=True)
    p.add_argument('--order', type=int, default=6)
    p.add_argument('--cap', type=int, help='about this many letters per corpus (every n-th line)')
    p.add_argument('--sources', nargs='+', help='only these sources of the corpora')
    p.add_argument('--check', type=int, default=0, help='lines per corpus to compare with Counts.from_lines')
    p = sub.add_parser('bits')
    p.add_argument('--models', nargs='+', required=True, help='name:max_order of artifacts open_<name>')
    p.add_argument('--weights', nargs='+', type=float, default=[0.05, 0.1, 0.2, 0.3, 0.5])
    p.add_argument('--out', type=Path, default=Path('bits.json'))
    args = parser.parse_args()
    root = args.root or default_root()
    lm_dir = args.lm_dir or Path(os.environ.get('TROGS26_WORK', '.')) / 'lm'
    cfg = config.load(args.config)
    if args.command == 'download':
        download(root)
    elif args.command == 'inputs':
        write_inputs(root, Work.from_env(), cfg, args.test_annotations)
    elif args.command == 'extract':
        extract(root, args.family, args.jobs)
    elif args.command == 'decontaminate':
        from .decontaminate import decontaminate
        decontaminate(root, list(FAMILIES), args.calibrate)
        write_manifest(root, list(FAMILIES))
    elif args.command == 'manifest':
        write_manifest(root, args.family)
    elif args.command == 'count':
        build_model(root, lm_dir, cfg['alphabet'], args.name, args.corpora, args.order, args.cap, args.sources,
                    args.check)
    else:
        bits_grid(root, lm_dir, cfg, args.models, args.weights, args.out)


if __name__ == '__main__':
    main()
