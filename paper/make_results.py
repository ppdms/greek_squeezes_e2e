"""Numbers of the page-network system for paper/report.tex, read from the run outputs.

    TROGS26_WORK=<work> .venv/bin/python paper/make_results.py

Reads the out-of-fold sweeps runs/<run>/oof/sweep_P.json of `FINAL_RUNS`, one per weight of the external language
model (Core errors and characters of every public squeeze at every point of the grid `decode.variants.P.grid`), as one
grid whose points are '<external weight>|<combo key>'. The operating point (`lm.open.weight` and
`decode.variants.P.combo`) is checked to be its minimum; the nested result reads each fold at the point with the
fewest errors on the other four. With the groups of related public squeezes data/related_public224.json
(paper/make_groups.py), the frozen TROGS-26 score paper/data/trogs26_score.json and the external text's counts and
decontamination report (`corpus_macros`) it writes paper/results.tex (macros) and paper/results_table.tex (the main
results table).
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import NamedTuple

from trogs26 import config, data
from trogs26.config import REPO
from trogs26.decode.search import Combo, grid_combos
from trogs26.layout import Work

PAPER = Path(__file__).resolve().parent
RELATED = REPO / 'data' / 'related_public224.json'
VARIANT = 'P'
NEAR = 0.01
FINAL_RUNS = {0.3: 'clean_open30', 0.45: 'clean_open45', 0.6: 'clean_open60'}  # external weight -> sweep run
FINAL_CONFIG = 'configs/pagenet_only_style_neural.json'


def pct(errors: int, chars: int) -> str:
    return f'{100 * errors / chars:.2f}\\%'


def num(n: int) -> str:
    return f'{n:,}'.replace(',', '{,}')


def millions(n: int) -> str:
    return f'{n / 1e6:.1f}'


def totals(sweep: dict) -> dict[str, list[int]]:
    """combo key -> [errors, characters] over every squeeze and both views."""
    out: dict[str, list[int]] = {}
    for base, per_combo in sweep['scores'].items():
        for key, (e1, c1, e2, c2) in per_combo.items():
            acc = out.setdefault(key, [0, 0])
            acc[0] += e1 + e2
            acc[1] += c1 + c2
    return out


def load_sweep(work: Work, run_name: str) -> dict:
    return json.loads((work.run(run_name, 'oof') / f'sweep_{VARIANT}.json').read_text())


def point(weight: float, key: str) -> str:
    """A point of the joint grid: external weight and combo key."""
    return f'{weight:.2f}|{key}'


def final_run(cfg: dict) -> str:
    """The sweep run of the configured external weight, the run that holds the operating point."""
    return FINAL_RUNS[cfg['lm']['open']['weight']]


def joint_sweep(work: Work, bases: list[str], grid: list[str]) -> dict:
    """The sweeps of `FINAL_RUNS` as one: scores[base][point(weight, combo key)]; each must hold every squeeze of
    `bases` at exactly the combo keys `grid`."""
    scores: dict[str, dict] = {}
    for weight, run in FINAL_RUNS.items():
        sweep = load_sweep(work, run)
        if sorted(sweep['scores']) != sorted(bases):
            raise ValueError(f'sweep {run} does not cover the folds file')
        if any(sorted(per_combo) != grid for per_combo in sweep['scores'].values()):
            raise ValueError(f'sweep {run} does not hold exactly the points of the grid')
        for base, per_combo in sweep['scores'].items():
            scores.setdefault(base, {}).update({point(weight, key): v for key, v in per_combo.items()})
    return {'scores': scores}


class Nested(NamedTuple):
    """The nested result over some squeezes: each fold read at the point with the fewest errors on the other folds'
    squeezes of them (ties: the smaller key)."""
    errors: int
    chars: int
    chosen: dict[int, str]  # fold -> its point
    per_fold: dict[int, list[int]]  # fold -> [errors, characters]


def nested(sweep: dict, folds: dict[int, tuple[str, ...]], bases: set[str]) -> Nested:
    """The nested result over `bases` (`Nested`), both views."""
    fold_of = data.fold_of(folds)
    chosen, per_fold = {}, {}
    for k in sorted(folds):
        table = totals({'scores': {b: s for b, s in sweep['scores'].items() if b in bases and fold_of[b] != k}})
        key = chosen[k] = min(table, key=lambda c: (table[c][0], c))
        per_fold[k] = [0, 0]
        for b in sorted(bases):
            if fold_of[b] == k:
                e1, c1, e2, c2 = sweep['scores'][b][key]
                per_fold[k][0] += e1 + e2
                per_fold[k][1] += c1 + c2
    return Nested(sum(e for e, _ in per_fold.values()), sum(c for _, c in per_fold.values()), chosen, per_fold)


def split_groups(folds: dict[int, tuple[str, ...]]) -> list[list[str]]:
    """The groups of related public squeezes that lie in more than one fold."""
    fold_of = data.fold_of(folds)
    groups = [g['squeezes'] for g in json.loads(RELATED.read_text())['groups']]
    return [g for g in groups if len({fold_of[b] for b in g}) > 1]


def corpus_macros(work: Work, cfg: dict) -> dict[str, str]:
    """Numbers of the external text: its counts (`lm.open.name`) and the decontamination report
    open_text/decontamination.json (trogs26.lm.decontaminate), with its calibration on PHI."""
    system = json.loads((work.root / 'lm' / cfg['lm']['open']['name'] / 'lm.json').read_text())
    report = json.loads((work.root / 'open_text' / 'decontamination.json').read_text())
    families, rule, calibration = report['families'], report['rule'], report['calibration']
    df = {int(k): v for k, v in calibration['df'].items()}
    own = sum(v for k, v in df.items() if k <= rule['rare'])
    found = Counter(row['cohort'] for row in report['squeezes'] if row['documents_dropped'])
    return {
        'ExtLetters': millions(system['letters']),
        'DeconGram': str(rule['gram']),
        'DeconRare': str(rule['rare']),
        'DeconDropAt': str(rule['drop_at']),
        'DeconShare': f'{100 * rule["share"]:g}\\%',
        'DeconDropped': num(sum(f['dropped_documents'] for f in families.values())),
        'DeconPublicFound': str(found['public224']),
        'DeconTestFound': str(found['trogs26']),
        'DeconCutDocuments': num(sum(f['cut_documents'] for f in families.values())),
        'DeconCutLetters': num(sum(f['cut_letters'] for f in families.values())),
        'DeconCalibPhiDocuments': num(calibration['staged']['documents']),
        'DeconCalibDocuments': num(calibration['documents']),
        'DeconCalibSqueezes': str(calibration['squeezes']),
        'DeconCalibShare': f'{100 * own / sum(df.values()):.1f}\\%',
        'DeconCalibEditions': str(len(calibration['edition_shares'])),
        'DeconCalibShareMin': f'{100 * calibration["edition_shares"][0]:.1f}\\%',
    }


def main() -> None:
    cfg = config.load(FINAL_CONFIG)
    work = Work.from_env()
    folds = data.folds(cfg['data']['folds'])
    settings = cfg['decode']['variants'][VARIANT]
    combo = Combo(**settings['combo'])
    open_weight = cfg['lm']['open']['weight']
    grid = sorted(c.key for c in grid_combos(cfg['decode']['grids'][settings['grid']]))
    sweep = joint_sweep(work, data.all_bases(folds), grid)
    operating = point(open_weight, combo.key)
    grid_totals = totals(sweep)
    if operating not in grid_totals:
        raise ValueError(f'the operating point {operating} is not a point of the swept grid')
    errors, chars = grid_totals[operating]
    least = min(e for e, _ in grid_totals.values())
    if errors != least:
        raise ValueError(f'operating point {operating} makes {errors} errors, minimum is {least}')
    near = sum(1 for e, _ in grid_totals.values() if e <= least * (1 + NEAR))
    public = nested(sweep, folds, set(sweep['scores']))
    no_merged_bases = {b for b in sweep['scores'] if 'Merged' not in b}
    no_merged = nested(sweep, folds, no_merged_bases)
    split = split_groups(folds)
    related = {b for g in split for b in g}
    unrelated = set(sweep['scores']) - related
    rel = nested(sweep, folds, related)
    unrel = nested(sweep, folds, unrelated)

    official = json.loads((PAPER / 'data' / 'trogs26_score.json').read_text())
    trogs = official['errors']['cer'], official['chars']['cer']

    macros = {
        'DevOofErrors': num(errors),
        'DevOofChars': num(chars),
        'DevOofCer': pct(errors, chars),
        'DevNoMergedErrors': num(no_merged.errors),
        'DevNoMergedChars': num(no_merged.chars),
        'DevNoMergedCer': pct(no_merged.errors, no_merged.chars),
        'DevNestedErrors': num(public.errors),
        'DevNestedChars': num(public.chars),
        'DevNestedCer': pct(public.errors, public.chars),
        'DevNestedAgree': ('none', 'one', 'two', 'three', 'four', 'all')[
            sum(key == operating for key in public.chosen.values())],
        'DevCombo': f'$(\\lambda, w, \\beta, \\nu) = ({combo.lam:g}, {combo.w:g}, {combo.beta:g}, {combo.nn:g})$',
        'DevScoreWeights': f'$(\\lambda, w, \\beta) = ({combo.lam:g}, {combo.w:g}, {combo.beta:g})$',
        'DevOpenWeight': f'{open_weight:g}',
        'DevNeuralWeight': f'{combo.nn:g}',
        'DevOpenWeights': ', '.join(f'{w:g}' for w in FINAL_RUNS),
        'DevGridPoints': str(len(grid_totals)),
        'DevNearBest': str(near),
        'DevSplitGroups': str(len(split)),
        'DevSplitSqueezes': str(len(related)),
        'DevSplitErrors': num(rel.errors),
        'DevSplitChars': num(rel.chars),
        'DevSplitCer': pct(rel.errors, rel.chars),
        'DevUnsplitSqueezes': str(len(unrelated)),
        'DevUnsplitErrors': num(unrel.errors),
        'DevUnsplitChars': num(unrel.chars),
        'DevUnsplitCer': pct(unrel.errors, unrel.chars),
        'DevTrogsErrors': num(trogs[0]),
        'DevTrogsChars': num(trogs[1]),
        'DevTrogsCer': pct(*trogs),
        'DevTrogsAllErrors': num(official['errors']['all']),
        'DevTrogsAllChars': num(official['chars']['all']),
        'DevTrogsAllCer': pct(official['errors']['all'], official['chars']['all']),
        'DevTrogsOptErrors': num(official['errors']['opt']),
        'DevTrogsOptChars': num(official['chars']['opt']),
        'DevTrogsOptCer': pct(official['errors']['opt'], official['chars']['opt']),
        **corpus_macros(work, cfg),
    }

    (PAPER / 'results.tex').write_text(
        '% Written by paper/make_results.py from the run outputs; do not edit.\n'
        + ''.join(f'\\newcommand{{\\{k}}}{{{v}}}\n' for k, v in macros.items())
    )

    fold_rates = ', '.join(pct(*public.per_fold[k]) for k in sorted(folds))
    lines = [
        '% Written by paper/make_results.py from the run outputs; do not edit.',
        '\\begin{table}[htbp]\\centering',
        '\\begin{tabular}{@{}lrrr@{}}\\toprule',
        'Evaluation set & Errors & Characters & Core \\cer{} \\\\\\midrule',
        f'Public (224 squeezes) & {num(public.errors)} & {num(public.chars)} '
        f'& \\textbf{{{pct(public.errors, public.chars)}}} \\\\',
        f'\\quad Without \\code{{Merged}} ({len(no_merged_bases)}) & {num(no_merged.errors)} & {num(no_merged.chars)} '
        f'& {pct(no_merged.errors, no_merged.chars)} \\\\',
        f'\\quad No related squeeze in another fold ({len(unrelated)}) & {num(unrel.errors)} & {num(unrel.chars)} '
        f'& {pct(unrel.errors, unrel.chars)} \\\\',
        f'TROGS-26 test (50 squeezes) & {num(trogs[0])} & {num(trogs[1])} '
        f'& \\textbf{{{pct(*trogs)}}} \\\\\\bottomrule',
        '\\end{tabular}',
        '\\caption{\\textbf{Character error rates against both annotated views.} '
        'Public squeezes use one held-out chain each, and each fold is read at the weights chosen on the other '
        'four; test squeezes use readers trained on all public data and a vote over five chains. '
        f'Public fold rates (0--4): {fold_rates}.}}',
        '\\label{tab:results}',
        '\\end{table}\n',
    ]
    (PAPER / 'results_table.tex').write_text('\n'.join(lines))
    print(json.dumps(macros, indent=1))


if __name__ == '__main__':
    main()
