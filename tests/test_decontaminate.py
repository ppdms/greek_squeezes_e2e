import json
import random

from trogs26.lm import decontaminate as dc
from trogs26.lm.open import LETTERS, read_docs
from trogs26.lm.phi import name_references


def _text(rng: random.Random, n: int) -> str:
    return ''.join(rng.choice(LETTERS) for _ in range(n))


def _stage(root, family: str, docs: list[tuple[str, list[str]]]) -> None:
    staging = root / 'staging'
    staging.mkdir(exist_ok=True)
    with open(staging / f'{family}.txt', 'w') as text, open(staging / f'{family}.docs.tsv', 'w') as index:
        for doc, lines in docs:
            text.writelines(line + '\n' for line in lines)
            index.write(f'{doc}\t{family}\t{len(lines)}\t{sum(map(len, lines))}\n')


def _written(root, family: str) -> dict[str, list[str]]:
    docs = read_docs(root / f'{family}.docs.tsv')
    lines = (root / f'{family}.txt').read_text().splitlines()
    out, i = {}, 0
    for doc, n in zip(docs.ids, docs.lines.tolist()):
        out[doc], i = lines[i:i + n], i + n
    return out


def test_editions_are_dropped_shared_stretches_cut_and_formulae_kept(tmp_path):
    rng = random.Random(0)
    formula = _text(rng, 20)
    public, test = _text(rng, 200), _text(rng, 120)
    stone = formula + public  # the public squeeze's inscription opens with a formula
    (tmp_path / 'references.json').write_text(json.dumps(
        {'public224': {'IG_II2_100': [stone, stone[:-3]]}, 'trogs26': {'IAS_1': [test, test]}}))
    # the formula is in RARE + 1 documents: not rare (each ends with it: nothing after it can extend it to a gram of
    # the stone)
    others = [(f'e{i}', [_text(rng, 40), _text(rng, 30) + formula]) for i in range(dc.RARE)]
    # an edition with its lines broken mid-word: its grams span line ends
    # another inscription sharing a rare stretch of DROP_AT grams with the stone, far below SHARE of either text
    pre, post = _text(rng, 300), _text(rng, 100)
    stretch = public[60:60 + dc.DROP_AT + dc.GRAM - 1]
    _stage(tmp_path, 'epigraphic', [('ed', [stone[:50], stone[50:]]), ('list', [pre + stretch + post])] + others)
    head, tail = _text(rng, 40), _text(rng, 40)
    _stage(tmp_path, 'literary', [('quote', [head + test[10:40] + tail])])
    # PHI, for the calibration only: an edition its reference names, and RARE documents holding the stretch the other
    # inscription shares (rare without PHI, so cut from it above; not rare with PHI)
    _stage(tmp_path, 'phi', [('phi:0', [stone])] + [(f'phi:{i}', [stretch]) for i in range(1, dc.RARE + 1)])
    ref = next(iter(name_references('IG_II2_100')))
    (tmp_path / 'staging' / 'phi.refs.tsv').write_text(
        f'phi:0\t{ref}\n' + ''.join(f'phi:{i}\t\n' for i in range(1, dc.RARE + 1)))

    report = dc.decontaminate(tmp_path, ['literary', 'epigraphic'], calibrate=True)
    written = {f: (tmp_path / f'{f}.txt').read_text() for f in ('literary', 'epigraphic')}

    # the edition dropped by its text, the other inscription cut where it shares the stretch
    assert _written(tmp_path, 'epigraphic') == {**dict(others), 'list': [pre, post]}
    assert report['families']['epigraphic']['dropped_documents'] == 1
    assert _written(tmp_path, 'literary') == {'quote': [head, tail]}  # the quotation cut, its line split there
    rows = {row['squeeze']: row for row in report['squeezes']}
    assert rows['IAS_1']['occurrences_cut'] == 30 - dc.GRAM + 1
    assert rows['IG_II2_100']['max_document_grams_left'] == len(formula) - dc.GRAM + 1
    # the calibration measures the named edition, with df over PHI too; PHI is never written and changes nothing
    calibration = report['calibration']
    assert (calibration['documents'], len(calibration['edition_shares'])) == (1, 1)
    assert calibration['df'][str(dc.RARE + 3)] == dc.DROP_AT  # the stretch: ed, list, the edition and RARE more
    assert not (tmp_path / 'phi.txt').exists()
    assert dc.decontaminate(tmp_path, ['literary', 'epigraphic'])['calibration'] is None
    assert {f: (tmp_path / f'{f}.txt').read_text() for f in written} == written
