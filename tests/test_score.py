from trogs26 import score
from trogs26.layout import Work

ANNOTATION = """# baseline angle
0.0
# boxes
{boxes}
# transcript
{text}
# lines
# confidence
{confidence}
"""


def write_squeeze(work: Work, base: str, text: str, confidence: str) -> None:
    boxes = '\n'.join(f'{20 * i} 0 {20 * i + 10} 0 {20 * i + 10} 10 {20 * i} 10' for i in range(len(text)))
    work.annotations.mkdir(parents=True, exist_ok=True)
    for view in (1, 2):
        work.annotation(base, view).write_text(
            ANNOTATION.format(boxes=boxes, text=text, confidence=' '.join(confidence)))


def test_official_scores_match_per_view_scores_when_one_squeeze_name_prefixes_another(tmp_path):
    work = Work(tmp_path)
    write_squeeze(work, 'S_1', 'ABC', '110')  # Core letters: AB
    write_squeeze(work, 'S_10', 'ABC', '111')
    predictions = tmp_path / 'pred'
    predictions.mkdir()
    (predictions / 'S_1_transcript.txt').write_text('AC')
    (predictions / 'S_10_transcript.txt').write_text('ABC')
    assert score.squeeze_errors(work, 'S_1', 'AC') == [1, 2, 1, 2]
    assert score.squeeze_errors(work, 'S_10', 'ABC') == [0, 3, 0, 3]
    result = score.official_scores(predictions, work.annotations, ['S_1', 'S_10'])
    assert result['errors']['cer'] == 2 and result['chars']['cer'] == 10
    assert [row['err'] for row in result['per_squeeze']] == [2, 0]
    assert result['missing'] == [] and result['all_confident'] == []


def test_missing_transcript_is_reported(tmp_path):
    work = Work(tmp_path)
    write_squeeze(work, 'S_1', 'ABC', '111')
    (tmp_path / 'pred').mkdir()
    result = score.official_scores(tmp_path / 'pred', work.annotations, ['S_1'])
    assert result['missing'] == ['S_1'] and result['errors']['cer'] == 6
