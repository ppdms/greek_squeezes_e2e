"""Decontamination of the external corpora against the transcripts of every squeeze: the 224 public and the 50 test.

    python -m trogs26.lm.open decontaminate [--calibrate]    <root>/staging -> <root>/<family>.txt,
                                                              decontamination.json

References. Every annotated letter of both views of each squeeze (`<root>/references.json`, which `open inputs`
writes from the public and the test annotations), rows joined without separators; a squeeze's 15-grams are those of
either view. A document's text is its lines joined without separators, so a 15-gram is found across a line break.

Rarity. The document frequency df(g) of a squeeze's 15-gram g is the number of staged documents, of every corpus
family, whose text holds g. A gram is rare if df(g) <= `RARE`. A rare gram is text of one inscription (its editions,
copies and quotations together are far fewer than `RARE` documents); formulae, titles and common names recur in more
documents, are not rare and stay.

Identity. A document of an inscription family (`INSCRIPTIONS`) is a squeeze's inscription (an edition, a fragment, a
copy of it) if it holds >= `DROP_AT` distinct rare 15-grams of that squeeze that are at least `SHARE` of the 15-grams
of the shorter of the two texts (the squeeze's or the document's). Such a document is dropped whole. A document that
shares only a rare stretch with a squeeze (another list naming the same person, another decree repeating a clause) is
not its inscription.

Spans. In every other document, of every family, every occurrence of a rare 15-gram of any squeeze is cut out: its
letters are removed and its line is split there, so no n-gram of the counts and no training window of the neural
model (neither crosses a line) holds a letter of it. A literary text quoting an epigram keeps the rest of its text.

Check. The written corpora are scanned again, line by line (what the models read): no rare 15-gram of any squeeze may
occur in any line.

Calibration (`calibrate`). `RARE` must lie above the number of documents that hold an inscription's own text (its
editions and copies) and below that of formulae; `SHARE` below the share of an edition. The open corpora hold no
edition of a squeeze; the PHI Greek Inscriptions (`open.PHI`, staged by `open extract --family phi` from its crawl)
hold editions of most public squeezes, under the IG numbers of their names (`phi.name_references`). PHI is read for
this measurement only: it is not written, counted or trained on, and the corpora do not depend on it. The report
gives, for the public squeezes only, the df over the corpus families and PHI together of each of their grams found in
a PHI document citing an IG number of the squeeze's name (own text sits at low df, formulae at high df), and the
share of each such document holding >= `DROP_AT` grams of the squeeze that are rare by that df. The df without PHI
is never larger, so a gram rare with PHI is rare without it and a share is never smaller: the corpora are cut at
least as much as the calibration assumes.

No other filter decides what text of the staged corpora is used (`open` drops only what is no Greek edition of a
text before staging).
"""
from __future__ import annotations

import datetime
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import NamedTuple

import numpy as np

from .open import LETTERS, PHI, read_docs
from .phi import name_references, read_refs

GRAM = 15
RARE = 10
DROP_AT = 5
SHARE = 0.04
INSCRIPTIONS = frozenset({'epigraphic'})  # families whose documents are inscriptions
_BASE = np.uint64(1099511628211)  # FNV prime, the multiplier of the rolling hash
_END = len(LETTERS) + 1  # code of a document's end
_INDEX = {ch: i + 1 for i, ch in enumerate(LETTERS)}


def _sha256(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _hashes(codes: np.ndarray) -> np.ndarray:
    """uint64 hash of every `GRAM`-window of the 1-d `codes` (window i starts at i)."""
    n = len(codes) - GRAM + 1
    h = np.zeros(max(n, 0), dtype=np.uint64)
    with np.errstate(over='ignore'):
        for j in range(GRAM):
            h = h * _BASE + codes[j:j + n].astype(np.uint64)
    return h


class Grams(NamedTuple):
    """The distinct 15-grams of the squeezes, in ascending hash order."""
    texts: list[str]
    keys: np.ndarray  # uint64 hash of each
    codes: np.ndarray  # (n, GRAM) uint8 letter codes
    holders: list[tuple[int, ...]]  # the squeezes (indices into the sorted names) whose views hold each


def gram_table(views: dict[str, list[str]]) -> Grams:
    """The 15-grams of `views` (squeeze -> the letters of each of its views)."""
    holders: dict[str, set[int]] = defaultdict(set)
    for k, name in enumerate(sorted(views)):
        for text in views[name]:
            for i in range(len(text) - GRAM + 1):
                holders[text[i:i + GRAM]].add(k)
    texts = list(holders)
    codes = np.array([[_INDEX[ch] for ch in gram] for gram in texts], dtype=np.uint8).reshape(-1, GRAM)
    keys = np.zeros(len(texts), dtype=np.uint64)
    with np.errstate(over='ignore'):
        for j in range(GRAM):
            keys = keys * _BASE + codes[:, j].astype(np.uint64)
    order = np.argsort(keys, kind='stable')
    return Grams([texts[i] for i in order], keys[order], codes[order],
                 [tuple(sorted(holders[texts[i]])) for i in order])


def letter_codes(corpus: Path, lines_per_doc: np.ndarray) -> np.ndarray:
    """The letters of a corpus file as codes 1..24, each document's text (its lines joined) followed by `_END`."""
    lut = np.zeros(256, dtype=np.uint8)
    for ch, code in _INDEX.items():
        lut[ord(ch)] = code
    lut[1] = _END
    raw = np.fromfile(corpus, dtype=np.uint8)
    newlines = np.flatnonzero(raw == 10)
    if len(newlines) != int(lines_per_doc.sum()):
        raise ValueError(f'{corpus}: {len(newlines)} lines, its docs file counts {int(lines_per_doc.sum())}')
    if (lines_per_doc == 0).any():
        raise ValueError(f'{corpus}: a document without lines')
    raw[newlines[np.cumsum(lines_per_doc) - 1]] = 1
    codes = lut[raw]
    return codes[codes != 0]


class Hits(NamedTuple):
    """Every occurrence of a gram of a `Grams` in a corpus."""
    doc: np.ndarray  # document index
    offset: np.ndarray  # letter offset of the occurrence in the document's text
    gram: np.ndarray  # index into the `Grams`


def scan(codes: np.ndarray, grams: Grams) -> Hits:
    """The occurrences of `grams` in `codes` (`letter_codes`), each checked letter by letter."""
    ends = np.flatnonzero(codes == _END)
    starts = np.r_[0, ends[:-1] + 1]
    last = len(grams.keys) - 1
    positions, found = [], []
    step = 1 << 25
    for start in range(0, max(len(codes) - GRAM + 1, 0), step):
        window = codes[start:start + step + GRAM - 1]
        h = _hashes(window)
        at = np.minimum(np.searchsorted(grams.keys, h), last)
        pos = np.flatnonzero(grams.keys[at] == h)
        same = (window[pos[:, None] + np.arange(GRAM)] == grams.codes[at[pos]]).all(axis=1)
        positions.append(start + pos[same])
        found.append(at[pos[same]])
    position = np.concatenate(positions) if positions else np.zeros(0, np.int64)
    doc = np.searchsorted(ends, position)
    return Hits(doc.astype(np.int64), (position - starts[doc]).astype(np.int64),
                np.concatenate(found).astype(np.int64) if found else np.zeros(0, np.int64))


def doc_frequency(hits: list[Hits], count: int) -> np.ndarray:
    """df of each of `count` grams: the documents holding it, over the corpora of `hits`."""
    df = np.zeros(count, dtype=np.int64)
    for h in hits:
        pairs = np.unique(h.doc * count + h.gram)
        df += np.bincount(pairs % count, minlength=count)
    return df


def shared_rare(h: Hits, rare: np.ndarray, grams: Grams) -> dict[int, Counter]:
    """document -> Counter(squeeze -> distinct rare grams of that squeeze the document holds)."""
    keep = rare[h.gram]
    out: dict[int, Counter] = defaultdict(Counter)
    for doc, gram in {(int(d), int(g)) for d, g in zip(h.doc[keep], h.gram[keep])}:
        for k in grams.holders[gram]:
            out[doc][k] += 1
    return out


def share(n: int, squeeze_grams: int, doc_letters: int) -> float:
    """`n` shared grams as a share of the 15-grams of the shorter text, the squeeze's or the document's."""
    return n / max(min(squeeze_grams, doc_letters - GRAM + 1), 1)


def inscription_of(per: Counter, squeeze_grams: np.ndarray, doc_letters: int) -> list[tuple[int, int]]:
    """The squeezes a document is an inscription of by its text (module docstring), as (squeeze, distinct rare
    grams), the most grams first (ties: the first squeeze); `per` is the document's `shared_rare` entry."""
    return sorted(((k, n) for k, n in per.items()
                   if n >= DROP_AT and share(n, int(squeeze_grams[k]), doc_letters) >= SHARE),
                  key=lambda item: (-item[1], item[0]))


def cut(lines: list[str], spans: list[tuple[int, int]]) -> list[str]:
    """`lines` (one document, without line ends) less the letters at the offsets of `spans` ((start, stop) letter
    offsets of its text), each line split where a letter is removed; pieces without a letter are dropped."""
    removed = np.zeros(sum(sum(ch in _INDEX for ch in line) for line in lines), dtype=bool)
    for a, b in spans:
        removed[a:b] = True
    out, i = [], 0
    for line in lines:
        piece: list[str] = []
        for ch in line:
            if ch in _INDEX:
                i += 1
                if removed[i - 1]:
                    out.append(''.join(piece))
                    piece = []
                    continue
            piece.append(ch)
        out.append(''.join(piece))
    return [piece for piece in out if any(ch in _INDEX for ch in piece)]


def write_family(staging: Path, out: Path, family: str, dropped: set[int], spans: dict[int, list[tuple[int, int]]]
                 ) -> dict:
    """Write <out>/<family>.txt and .docs.tsv: the staged documents less `dropped`, those of `spans` cut."""
    docs = read_docs(staging / f'{family}.docs.tsv')
    size = Counter()
    with open(staging / f'{family}.txt', encoding='utf-8') as src, \
            open(out / f'{family}.txt', 'w', encoding='utf-8') as text, \
            open(out / f'{family}.docs.tsv', 'w', encoding='utf-8') as index:
        for d, (doc, source) in enumerate(zip(docs.ids, docs.sources)):
            lines = [next(src).rstrip('\n') for _ in range(int(docs.lines[d]))]
            if d in dropped:
                continue
            if d in spans:
                lines = cut(lines, spans[d])
                if not lines:
                    size['emptied_documents'] += 1
                    continue
            letters = sum(len(line) for line in lines)
            text.writelines(line + '\n' for line in lines)
            index.write(f'{doc}\t{source}\t{len(lines)}\t{letters}\n')
            size.update(documents=1, lines=len(lines), letters=letters)
    return dict(size)


def decontaminate(root: Path, families: list[str], calibrate: bool = False) -> dict:
    """Decontaminate the staged `families` of `root` (module docstring), write the corpora and
    <root>/decontamination.json (with the calibration if `calibrate`), and return the report."""
    started = time.perf_counter()
    staging = root / 'staging'
    cohorts = json.loads((root / 'references.json').read_text())
    cohort_of = {name: cohort for cohort, views in cohorts.items() for name in views}
    views = {name: texts for per in cohorts.values() for name, texts in per.items()}
    names = sorted(views)
    grams = gram_table(views)
    docs = {f: read_docs(staging / f'{f}.docs.tsv') for f in families}
    hits = {f: scan(letter_codes(staging / f'{f}.txt', docs[f].lines), grams) for f in families}
    df = doc_frequency(list(hits.values()), len(grams.texts))
    rare = df <= RARE
    shared = {f: shared_rare(hits[f], rare, grams) for f in families}
    squeeze_grams = np.bincount([k for holders in grams.holders for k in holders], minlength=len(names))

    # identity: documents of an inscription family that are a squeeze's inscription by their text
    identified: dict[str, dict[int, dict]] = {f: {} for f in families}
    for f in families:
        if f in INSCRIPTIONS:
            for d, per in shared[f].items():
                found = inscription_of(per, squeeze_grams, int(docs[f].letters[d]))
                if found:
                    k, n = found[0]
                    identified[f][d] = {'squeeze': names[k], 'rare_grams': n,
                                        'share': round(share(n, int(squeeze_grams[k]), int(docs[f].letters[d])), 4)}

    # spans: every occurrence of a rare gram in a document that stays
    spans: dict[str, dict[int, list[tuple[int, int]]]] = {}
    report_families = {}
    staged = {}
    for f in families:
        h = hits[f]
        keep = rare[h.gram] & ~np.isin(h.doc, np.array(sorted(identified[f]), dtype=np.int64))
        per_doc: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for d, offset in zip(h.doc[keep].tolist(), h.offset[keep].tolist()):
            per_doc[d].append((offset, offset + GRAM))
        spans[f] = dict(per_doc)
        staged[f] = {'documents': len(docs[f].ids), 'letters': int(docs[f].letters.sum()),
                     'sha256': _sha256(staging / f'{f}.txt')}
        size = write_family(staging, root, f, set(identified[f]), spans[f])
        dropped = sorted(identified[f])
        report_families[f] = {
            'staged': staged[f], 'written': size,
            'dropped_documents': len(dropped),
            'dropped_letters': int(docs[f].letters[dropped].sum()) if dropped else 0,
            'cut_documents': len(spans[f]),
            'cut_occurrences': int(keep.sum()),
            'cut_letters': int(sum(len(merge(s)) for s in spans[f].values())),
        }
        print(f, json.dumps(report_families[f]), flush=True)

    # the check, line by line, and what each squeeze still shares with the written corpora (grams that are not rare)
    after = {}
    for f in families:
        lines = read_docs(root / f'{f}.docs.tsv').lines
        h = scan(letter_codes(root / f'{f}.txt', np.ones(int(lines.sum()), dtype=np.int64)), grams)
        after[f] = h._replace(doc=np.searchsorted(np.cumsum(lines) - 1, h.doc))  # line -> document
    remaining_rare = int(sum(rare[h.gram].sum() for h in after.values()))
    if remaining_rare:
        raise AssertionError(f'{remaining_rare} occurrences of rare grams remain in lines of the written corpora')
    report_squeezes = squeeze_rows(names, cohort_of, views, grams, df, rare, shared, identified, spans, hits,
                                   after, docs)
    report = {
        'created_by': 'trogs26.lm.decontaminate', 'created': datetime.date.today().isoformat(),
        'seconds': round(time.perf_counter() - started, 1),
        'rule': {'gram': GRAM, 'rare': RARE, 'drop_at': DROP_AT, 'share': SHARE,
                 'inscription_families': sorted(INSCRIPTIONS & set(families))},
        'references': {'sha256': _sha256(root / 'references.json'),
                       'squeezes': {c: len(v) for c, v in cohorts.items()},
                       'grams': len(grams.texts), 'grams_found': int((df > 0).sum()),
                       'grams_rare_found': int(((df > 0) & rare).sum()),
                       'grams_by_df': df_histogram(df)},
        'families': report_families,
        'check': {'rare_occurrences_remaining': remaining_rare},
        'calibration': calibration(staging, names, cohort_of, grams, hits, squeeze_grams) if calibrate else None,
        'cross_cohort': cross_cohort(names, cohort_of, grams),
        'squeezes': report_squeezes,
    }
    (root / 'decontamination.json').write_text(json.dumps(report, indent=1))
    return report


def merge(spans: list[tuple[int, int]]) -> list[int]:
    """The distinct letter offsets the spans cover."""
    return sorted({i for a, b in spans for i in range(a, b)})


def calibration(staging: Path, names: list[str], cohort_of: dict[str, str], grams: Grams, hits: dict[str, Hits],
                squeeze_grams: np.ndarray) -> dict:
    """The calibration on the staged PHI crawl (module docstring), with df over it and the corpus families
    (`hits`): for its documents citing an IG number of a public squeeze's name, df -> the distinct (squeeze, gram)
    pairs of that df whose gram such a document holds, and the shares (`share`) of the rare grams of the documents
    holding >= DROP_AT of them, smallest first."""
    phi_docs = read_docs(staging / f'{PHI}.docs.tsv')
    h = scan(letter_codes(staging / f'{PHI}.txt', phi_docs.lines), grams)
    df = doc_frequency([*hits.values(), h], len(grams.texts))
    shared = shared_rare(h, df <= RARE, grams)
    refs = read_refs(staging / f'{PHI}.refs.tsv')
    by_ref: dict[str, list[int]] = defaultdict(list)
    for k, name in enumerate(names):
        if cohort_of[name] == 'public224':
            for ref in name_references(name):
                by_ref[ref].append(k)
    named: dict[int, set[int]] = defaultdict(set)  # PHI document -> the public squeezes its references name
    for d, doc in enumerate(phi_docs.ids):
        for ref in refs[doc]:
            named[d].update(by_ref.get(ref, ()))
    named = {d: ks for d, ks in named.items() if ks}
    pairs = {(k, g) for d, g in zip(h.doc.tolist(), h.gram.tolist()) if d in named
             for k in grams.holders[g] if k in named[d]}
    values = Counter(int(df[g]) for _, g in pairs)
    shares = sorted(share(n, int(squeeze_grams[k]), int(phi_docs.letters[d])) for d, ks in named.items() for k in ks
                    if (n := shared.get(d, Counter())[k]) >= DROP_AT)
    return {'corpus': PHI, 'staged': {'documents': len(phi_docs.ids), 'letters': int(phi_docs.letters.sum()),
                                      'sha256': _sha256(staging / f'{PHI}.txt')},
            'squeezes': len({k for k, _ in pairs}), 'documents': len(named), 'pairs': len(pairs),
            'df': {str(v): n for v, n in sorted(values.items())},
            'edition_shares': [round(s, 4) for s in shares]}


def df_histogram(df: np.ndarray) -> dict[str, int]:
    """Grams found by df band."""
    bands = [(1, 1), (2, 2), (3, 5), (6, 10), (11, 20), (21, 50), (51, 100), (101, 1000), (1001, None)]
    out = {}
    for lo, hi in bands:
        mask = df >= lo if hi is None else (df >= lo) & (df <= hi)
        out[f'{lo}+' if hi is None else (f'{lo}' if lo == hi else f'{lo}-{hi}')] = int(mask.sum())
    return out


def cross_cohort(names: list[str], cohort_of: dict[str, str], grams: Grams) -> dict:
    """For each squeeze, the most distinct 15-grams it shares with one squeeze of another cohort."""
    pairs: Counter = Counter()
    for holders in grams.holders:
        for a in holders:
            for b in holders:
                if a < b and cohort_of[names[a]] != cohort_of[names[b]]:
                    pairs[(a, b)] += 1
    top = pairs.most_common(10)
    return {'max_shared_grams': top[0][1] if top else 0,
            'top_pairs': [{'squeezes': [names[a], names[b]], 'shared_grams': n} for (a, b), n in top]}


def squeeze_rows(names, cohort_of, views, grams, df, rare, shared, identified, spans, hits, after, docs) -> list:
    """Per squeeze: its grams, those found and rare, the documents dropped as its inscription, the occurrences cut,
    and what the written corpora still share with it (formulae)."""
    holder_grams: dict[int, list[int]] = defaultdict(list)
    for g, holders in enumerate(grams.holders):
        for k in holders:
            holder_grams[k].append(g)
    cut_by: Counter = Counter()
    for f, h in hits.items():
        keep = rare[h.gram] & ~np.isin(h.doc, np.array(sorted(identified[f]), dtype=np.int64))
        for g in h.gram[keep].tolist():
            for k in grams.holders[g]:
                cut_by[k] += 1
    left: dict[int, Counter] = defaultdict(Counter)  # squeeze -> document -> distinct grams still shared
    left_grams: dict[int, set[int]] = defaultdict(set)  # squeeze -> grams still found
    for f, h in after.items():
        for d, g in {(int(d), int(g)) for d, g in zip(h.doc, h.gram)}:
            for k in grams.holders[g]:
                left[k][(f, d)] += 1
                left_grams[k].add(g)
    rows = []
    for k, name in enumerate(names):
        own = np.array(holder_grams[k], dtype=np.int64)
        dropped = [{'family': f, 'document': docs[f].ids[d], **{x: y for x, y in info.items() if x != 'squeeze'}}
                   for f in identified for d, info in sorted(identified[f].items()) if info['squeeze'] == name]
        top = max(left[k].values(), default=0)
        rows.append({'squeeze': name, 'cohort': cohort_of[name], 'letters': [len(v) for v in views[name]],
                     'grams': len(own), 'grams_found': int((df[own] > 0).sum()) if len(own) else 0,
                     'grams_rare_found': int(((df[own] > 0) & rare[own]).sum()) if len(own) else 0,
                     'documents_dropped': dropped, 'occurrences_cut': cut_by[k],
                     'grams_left': len(left_grams[k]), 'max_document_grams_left': top})
    return rows
