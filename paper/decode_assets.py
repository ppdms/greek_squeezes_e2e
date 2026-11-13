"""The decode and letter-model parts of the saved figure assets of one public squeeze, under the final system.

    TROGS26_WORK=<work tree> FIGURE_ASSET_DIR=<the squeeze's saved assets> \\
        python paper/decode_assets.py --base IG_II2_493_Copy1 --fold 4

Chain `--fold` reads the squeeze out of fold with configs/pagenet_only_style_neural_oof.json (the configured external
weight, operating point and neural model). Rewrites in FIGURE_ASSET_DIR:

  decode.json     rows (tokens, keep probabilities, best letters, the path of the chosen text, the best candidates),
                  the transcript with its errors and both reference views
  lm.npz, lm.json every language-model component's distribution over the next letter at every position of the chosen
                  texts

The page assets beside them (tokens.npz, pages, relief, augmentation) are kept: the rows must be tokens.npz's rows and
the page network's log-probabilities its own, else nothing is written.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from trogs26 import config, data, score  # noqa: E402
from trogs26.decode.decoder import chain_decoder, row_pools  # noqa: E402
from trogs26.decode.search import _KEEP_CLIP, GAP, Combo, plain, select, top_letters  # noqa: E402
from trogs26.decode.trace import row_path  # noqa: E402
from trogs26.layout import Work  # noqa: E402

CONFIG = ROOT / 'configs' / 'pagenet_only_style_neural_oof.json'
VARIANT = 'P'
COMPONENTS = ('chain', 'open', 'base', 'cache', 'neural', 'final', 'backward', 'prior', 'two_sided')
CANDIDATES = 8  # best candidates of each row kept


def saved_rows(decoder, base: str, tokens: np.lib.npyio.NpzFile) -> tuple[list[np.ndarray], list[tuple]]:
    """The squeeze's rows (token indices) and decoder inputs, in tokens.npz's row order; raises if the decoder reads
    other rows or other log-probabilities than the saved assets."""
    rows, *_ = decoder.page_tokens(base)
    inputs = decoder.page_rows(base)
    order = np.argsort([float(tokens['v'][r].mean()) for r in rows])
    rows, inputs = [rows[i] for i in order], [inputs[i] for i in order]
    for r, row in enumerate(rows):
        if not (np.all(tokens['row'][row] == r) and np.array_equal(tokens['col'][row], np.arange(len(row)))):
            raise ValueError(f'{base}: row {r} differs from the saved tokens')
    if not np.allclose(decoder.posteriors[base][1], tokens['lp'], atol=1e-5):
        raise ValueError(f'{base}: the page network log-probabilities differ from the saved tokens')
    return rows, inputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--base', required=True)
    parser.add_argument('--fold', type=int, required=True)
    args = parser.parse_args()
    out = Path(os.environ['FIGURE_ASSET_DIR'])
    work, cfg, base = Work.from_env(), config.load(str(CONFIG)), args.base
    bases = data.folds(cfg['data']['folds'])[args.fold]
    if base not in bases:
        raise SystemExit(f'{base} is not in fold {args.fold}')
    alphabet = cfg['alphabet']
    op = Combo(**cfg['decode']['variants'][VARIANT]['combo'])
    decoder = chain_decoder(work, work.chain(args.fold, 'oof', bases), cfg, VARIANT)
    rows, inputs = saved_rows(decoder, base, np.load(out / 'tokens.npz'))
    neural = decoder.neural
    device = next(neural.model.parameters()).device

    def letters(logprob) -> np.ndarray:
        return np.exp(np.array([logprob(a) for a in alphabet], np.float64))

    rows_out, raw_texts, positions, probs = [], [], [], []
    for r, ((pool, lm), inp, tokens) in enumerate(zip(row_pools(inputs, decoder.lm, decoder.search, decoder.cache),
                                                      inputs, rows)):
        scored = neural.rescore(pool, lm, [op.nn])[0]
        total = scored.visual + op.w * scored.keep + op.lam * scored.lm + op.beta * scored.length
        raw = select(scored, [op])[0]
        text = plain(raw)
        raw_texts.append(raw)
        emit, keep = inp[0], np.clip(np.asarray(inp[1], np.float64), _KEEP_CLIP, 1 - _KEEP_CLIP)
        rows_out.append({
            'tokens': [int(t) for t in tokens], 'keep': [float(k) for k in inp[1]],
            'place': [float(p) for p in inp[2]] if len(inp) > 2 else None,
            'top': [[[a, float(np.exp(lp))] for a, lp in top_letters(e, alphabet, 4)] for e in emit],
            'path': [int(j) for j in row_path(emit, keep, text, op.w, decoder.search)] if text else [],
            'raw': raw, 'text': text,
            'candidates': [{'text': scored.texts[i], 'visual': float(scored.visual[i]),
                            'keep': float(scored.keep[i]), 'lm': float(scored.lm[i]),
                            'length': float(scored.length[i]), 'total': float(total[i])}
                           for i in np.argsort(-total, kind='stable')[:CANDIDATES]],
            'pool_size': len(scored.texts)})
        forward, mixture = lm.forward, decoder.lm.forward
        for i, ch in enumerate(raw):
            if ch == GAP:
                continue
            before = raw[:i]
            before = before[before.rfind(GAP) + 1:]
            after = raw[i + 1:].split(GAP)[0]
            encoded = [int(c) for c in neural.neural.encode(before, neural.lut)] if before else []
            codes = [neural.model.bos] + encoded[-(neural.model.context - 1):]
            with torch.no_grad():
                logits = neural.model(torch.tensor([codes], device=device))[0, -1].float()
            p_nn = torch.softmax(logits, -1).cpu().numpy()[:len(alphabet)].astype(np.float64)
            p_cache = letters(lambda a: forward.cond_logprob(a, before))
            p_final = (1 - op.nn) * p_cache + op.nn * p_nn
            p_back = letters(lambda a: lm.backward.cond_logprob(a, after[::-1]))
            p_prior = letters(lambda a: forward.cond_logprob(a, ''))
            p_two = p_final * p_back / p_prior
            probs.append(np.stack([letters(lambda a: mixture.base.cond_logprob(a, before)),
                                   letters(lambda a: mixture.other.cond_logprob(a, before)),
                                   letters(lambda a: mixture.cond_logprob(a, before)),
                                   p_cache, p_nn, p_final, p_back, p_prior, p_two / p_two.sum()]))
            positions.append({'row': r, 'index': i, 'letter': ch, 'before': before, 'after': after})
        print(f'row {r + 1}/{len(rows)}: {len(tokens)} tokens, pool {len(scored.texts)}', flush=True)
    transcript = '\n'.join(t for t in (plain(raw) for raw in raw_texts) if t)
    errors = score.squeeze_errors(work, base, transcript)
    np.savez_compressed(out / 'lm.npz', probs=np.array(probs, np.float32), components=np.array(COMPONENTS),
                        alphabet=np.array(list(alphabet)))
    (out / 'lm.json').write_text(json.dumps({
        'positions': positions, 'components': COMPONENTS, 'neural_weight': op.nn,
        'cache_gamma': cfg['decode']['cache']['gamma'], 'open_weight': cfg['lm']['open']['weight']}))
    (out / 'decode.json').write_text(json.dumps({
        'combo': list(op), 'alphabet': alphabet, 'rows': rows_out, 'transcript': transcript, 'errors': errors,
        'reference': {str(v): score.core_text(work, base, v) for v in (1, 2)}}))
    print(f'transcript errors [err1, chars1, err2, chars2] = {errors}')


if __name__ == '__main__':
    main()
