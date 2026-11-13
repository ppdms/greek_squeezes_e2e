from trogs26.layout import Work
from trogs26.rover import medoid, member_texts, rover


def test_plurality_letter_wins():
    assert rover(['ABC', 'ABC', 'ABD'], ins_frac=0.5, gap_weight=1.0) == 'ABC'


def test_fewer_than_three_members_give_the_medoid():
    assert rover(['ABC', 'XYZ'], ins_frac=0.5, gap_weight=1.0) == 'ABC'


def test_medoid_ties_go_to_the_earliest_member():
    assert medoid(['AB', 'AC']) == 'AB'
    assert medoid(['AB', 'AC', 'AD']) == 'AB'


def test_gap_votes_count_gap_weight_times():
    texts = ['AB', 'AB', 'AB', 'A', 'A']
    assert rover(texts, ins_frac=0.5, gap_weight=1.0) == 'AB'  # 3 letter votes against 2 gaps
    assert rover(texts, ins_frac=0.5, gap_weight=2.0) == 'A'  # 3 against 2 * 2


def test_empty_rows_are_dropped():
    assert rover(['A\n\nB', 'A\n\nB', 'A\n\nB'], ins_frac=0.5, gap_weight=1.0) == 'A\nB'


def test_members_are_read_variant_major_then_chain_by_chain(tmp_path):
    work = Work(tmp_path)
    for variant in ('P', 'Q'):
        for fold in (0, 1):
            directory = work.run('run', 'trogs26') / f'f{fold}' / variant
            directory.mkdir(parents=True)
            (directory / 'S_1_transcript.txt').write_text(f'{variant}{fold}', encoding='utf8')
    texts = member_texts(work, 'run', 'trogs26', ['P', 'Q'], [0, 1], ['S_1'])
    assert [member['S_1'] for member in texts] == ['P0', 'P1', 'Q0', 'Q1']
