import math
import random
import xml.etree.ElementTree as ET
from collections import Counter

import numpy as np
import pytest

from trogs26.lm import mixture, open as open_text
from trogs26.lm.ppm import PPM, Counts

ALPHABET = 'ABCDEFGHIKLMNOPQRSTUWXYZ&:)'
TEI = '<TEI xmlns="http://www.tei-c.org/ns/1.0"><text><body>{}</body></text></TEI>'


def test_normalise_letters_sigma_iota_and_interpunct():
    assert open_text.normalise('ᾳ ᾷ ΑΙ Σοφοκλῆς ϲ', False) == ['AIAIAISOFOKLHSS']
    assert open_text.normalise('ὁ·δῆμος:', True) == ['O:DHMOS:']
    assert open_text.normalise('ὁ·δῆμος:', False) == ['ODHMOS']
    assert open_text.normalise('µ ΨΩ ϑ', False) == ['MUWQ']


def test_normalise_splits_at_missing_text_but_not_at_dropped_characters():
    # a gap mark and a Greek symbol that is no letter split; digamma, digits, Latin and punctuation do not
    assert open_text.normalise('ΑΒ\x1eΓΔ', False) == ['AB', 'GD']
    assert open_text.normalise('Α𐅵Β', False) == ['A', 'B']
    assert open_text.normalise('ΑϜΒ 12 abc, Γ', False) == ['ABG']
    assert open_text.normalise('Α\u05d0Β', False) == ['A', 'B']  # Hebrew letter: another language


def _lines(body: str, rules=open_text.EPIDOC) -> list[str]:
    return open_text.transcription(ET.fromstring(TEI.format(body)), rules)


def test_epidoc_keeps_what_is_written():
    body = ('<div type="edition"><ab><lb n="1"/>αβ<supplied reason="lost">γ</supplied>'
            '<supplied reason="omitted">ζ</supplied><unclear>δ</unclear><expan><abbr>ε</abbr><ex>ηθ</ex></expan>'
            '<lb n="2"/><choice><sic>ι</sic><corr>κ</corr></choice><choice><reg>λ</reg><orig>μ</orig></choice>'
            '<del>ν</del><gap reason="lost"/>ξ<app><lem>ο</lem><rdg>π</rdg></app><note>ρ</note>σ'
            '<g type="punct">:</g>τ<g type="cross"/>υ<foreign xml:lang="la">φ</foreign>χ</ab></div>')
    assert _lines(body) == ['ABGDE', 'IM', 'XOS:T', 'Y', 'C']


def test_edition_selection_skips_other_languages_and_copies_that_are_no_edition_of_the_inscription():
    stats = Counter()
    copies = ''.join(f'<div type="edition" subtype="{subtype}"><ab>δ</ab></div>'
                     for subtype in sorted(open_text._SKIP_SUBTYPES))
    root = ET.fromstring(TEI.format(
        '<div type="edition" xml:lang="grc"><ab>αβ</ab></div><div type="edition" xml:lang="la"><ab>γ</ab></div>'
        + copies + '<div type="edition" subtype="transcription"><ab>ε</ab></div>'
        '<div type="edition" subtype="diplomatic"><ab>ζ</ab></div>'))
    assert [open_text.transcription(d, open_text.EPIDOC) for d in open_text._editions(root, stats)] == [['AB'], ['E']]
    assert sum(n for kind, n in stats.items() if kind.startswith('skipped_edition:')) == len(open_text._SKIP_SUBTYPES)
    assert stats['skipped_lang'] == 1


def test_literary_text_is_one_line_per_paragraph_without_notes_or_speakers():
    body = ('<div type="edition"><div type="textpart"><head>Τίτλος</head><p>αβ<note>γ</note>δ</p><p>ε</p></div>'
            '<div type="textpart"><sp><speaker>Ζ</speaker><l>η</l><l>θ</l></sp></div></div>')
    root = ET.fromstring(TEI.format(body))
    docs = open_text.literary_documents('w', root, Counter())
    assert docs == [('w#1', ['ABD', 'E']), ('w#2', ['HQ'])]


def test_fast_counts_equal_counts_from_lines():
    rng = random.Random(3)
    lines = [''.join(rng.choice('ABKNO:S') for _ in range(rng.randint(0, 14))) for _ in range(300)]
    lines += ['A', '', 'AB']
    for order in (1, 2, 5):
        expected = Counts.from_lines(lines, order, ALPHABET)
        assert open_text.count_lines(lines, order, ALPHABET).counts == expected.counts
        raw = np.frombuffer(''.join(line + '\n' for line in lines).encode(), dtype=np.uint8)
        counts, size = open_text.count_corpus([raw], order, ALPHABET)
        assert counts.counts == expected.counts and size['letters'] == sum(map(len, lines))


def test_selected_bytes_keeps_every_nth_line_of_the_wanted_lines():
    raw = np.frombuffer(b'A\nB\nC\nD\nE\n', dtype=np.uint8)
    wanted = np.array([True, False, True, True, True])
    assert open_text.selected_bytes(raw, wanted, 1).tobytes() == b'A\nC\nD\nE\n'
    assert open_text.selected_bytes(raw, wanted, 2).tobytes() == b'A\nD\n'


def test_mixture_is_the_probability_mixture_and_validates_its_weight():
    first = PPM(Counts.from_lines(['ABAB'], 2, 'ABC'), 2)
    second = PPM(Counts.from_lines(['BCBC'], 3, 'ABC'), 3)
    mix = mixture.Mixture(first, second, 0.25)
    for letter in 'ABC':
        expected = 0.75 * math.exp(first.cond_logprob(letter, 'B')) + 0.25 * math.exp(second.cond_logprob(letter, 'B'))
        assert math.exp(mix.cond_logprob(letter, 'B')) == pytest.approx(expected)
    assert sum(math.exp(mix.cond_logprob(x, 'AB')) for x in 'ABC') == pytest.approx(1.0)
    assert mix.text_logprob('AB') == pytest.approx(mix.cond_logprob('A', '') + mix.cond_logprob('B', 'A'))
    assert mix.text_logprob('') == 0.0
    for bad in (0.0, 1.0):
        with pytest.raises(ValueError):
            mixture.Mixture(first, second, bad)
    with pytest.raises(ValueError):
        mixture.Mixture(first, PPM(Counts.from_lines(['AB'], 2, 'AB'), 2), 0.5)


def test_chain_lm_is_a_mixture_only_with_lm_open(tmp_path):
    from trogs26.decode.decoder import chain_lm
    from trogs26.layout import Chain, Work
    from trogs26.lm import artifact

    work, chain = Work(tmp_path), Chain(tmp_path, 1, 'oof', ())
    artifact.save(Counts.from_lines(['ABAB'], 3, 'ABC'), work.lm('chain_f1'), {})
    artifact.save(Counts.from_lines(['BCBCBC'], 4, 'ABC'), work.lm('open_x'), {})
    cfg = {'lm': {'oof': 'chain_f{fold}', 'final': 'chain_final', 'decode_max_order': 2}}
    assert isinstance(chain_lm(work, chain, cfg), PPM)
    cfg['lm']['open'] = {'name': 'open_x', 'weight': 0.3, 'max_order': 3}
    lm = chain_lm(work, chain, cfg)
    assert isinstance(lm, mixture.Mixture) and lm.weight == 0.3 and (lm.base.order, lm.other.order) == (2, 3)
