"""Reading a chain's pages with its page network: every token's log-probabilities over 27 letters and none.

A page is read whole: every crop is encoded, then every token attends to its neighbours on the full page. With several
`read_views` (degrees, scale) the page is read once per view, every crop rotated and zoomed alike, and the views'
log-probabilities are averaged and renormalised. The result is one file per chain and version,
`chains/f{k}/{oof|new}/pagenet_{version}.npz`: `bases` and `det_ids` (one row per detection, pages in chain order,
detections in record order), `log_probs` (float32), `alphabet` (27 letters, then none) and `views`.
The network trained on every fold reads new pages only, a chain's into `pagenet_{version}_all.npz`; a variant decodes
those reads by naming `{version}_all` among its networks.
"""
from __future__ import annotations

import numpy as np
import torch

from .. import artifacts
from ..dino.model import normalisation, prepare
from ..layout import Chain, Work
from . import model as net
from . import pages
from .train import standardise


@torch.no_grad()
def read_page(model: net.PageNet, page: pages.Page, crops: np.ndarray, checkpoint: dict, device: torch.device,
              normalise, batch: int, views: list[tuple[float, float]]) -> np.ndarray:
    """(n, classes) float32 log-probabilities of the tokens of `page`, whose crops are `crops`, read in `views`."""
    cfg = checkpoint['recipe']
    autocast = torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda')
    idx, mask = pages.neighbours(page.u, page.v, cfg['neighbours'], cfg['v_weight'])
    pair = pages.pair_features(page.u, page.v, page.w, page.h, pages.logit(page.p), idx)
    node = torch.from_numpy(standardise(page.node, checkpoint['node_stats'])).to(device)
    idx, mask, pair = (torch.from_numpy(a).to(device) for a in (idx, mask, pair))
    log_probs = []
    for view in views:
        with autocast:
            visual = torch.cat([model.encode(prepare(torch.from_numpy(crops[i:i + batch]), device, normalise,
                                                     view=tuple(view)))
                                for i in range(0, len(crops), batch)])
            logits = model.context(visual, node, idx, mask, pair)
        log_probs.append(torch.log_softmax(logits.float(), -1))
    if len(log_probs) == 1:
        return log_probs[0].cpu().numpy()
    mean = torch.stack(log_probs).mean(0)
    return (mean - torch.logsumexp(mean, -1, keepdim=True)).cpu().numpy()


def read_chain(work: Work, chain: Chain, version: str, views: list[tuple[float, float]], device: torch.device,
               log=print, every_fold: bool = False) -> dict:
    """Read every page of `chain` in `views` with its network `models/pagenet_{version}/f{fold}` and write the
    chain's file; with `every_fold`, with the network trained on every fold (`models/pagenet_{version}/all`), which
    has seen every page of the folds, so `chain` must hold new pages."""
    if every_fold and chain.group != 'new':
        raise ValueError(f'the network trained on every fold has seen the {chain.group} pages')
    directory = work.models(f'pagenet_{version}', None if every_fold else chain.fold)
    model, checkpoint = net.load(directory, device)
    cfg = checkpoint['recipe']
    normalise = normalisation(directory, device)
    bases, det_ids, rows = [], [], []
    for base in chain.bases:
        det = artifacts.det_page(chain, base)
        page = pages.build(det, base)
        with np.load(chain.dino_crops(base, cfg['input'])) as z:
            if not np.array_equal(z['det_ids'], page.det_ids):
                raise ValueError(f'{base}: crop cache and detector record cover different detections')
            crops = z['crops']
        rows.append(read_page(model, page, crops, checkpoint, device, normalise, cfg['read_batch'], views))
        bases += [base] * len(page)
        det_ids.append(page.det_ids)
    path = chain.dir / f'pagenet_{version}{"_all" if every_fold else ""}.npz'
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(path) + '.partial.npz', bases=np.array(bases), det_ids=np.concatenate(det_ids),
             log_probs=np.concatenate(rows).astype(np.float32), alphabet=np.array(checkpoint['alphabet']),
             views=np.array(views, np.float64))
    (path.parent / (path.name + '.partial.npz')).replace(path)
    log(f'{path}: {len(chain.bases)} pages, {len(bases)} tokens')
    return {'path': str(path), 'pages': len(chain.bases), 'tokens': len(bases)}


def posteriors(chain: Chain, versions: list[str], alphabet: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """base -> (detection ids, (n, 28) log-probabilities) of the chain's files for `versions`: with several networks
    their log-probabilities averaged and renormalised (the normalised product of their distributions)."""
    members = []
    for version in versions:
        path = chain.dir / f'pagenet_{version}.npz'
        with np.load(path) as z:
            names = [str(a) for a in z['alphabet'].tolist()]
            if ''.join(names[:-1]) != alphabet or names[-1] != 'none':
                raise AssertionError(f'{path}: alphabet {names!r} is not {alphabet!r} + none')
            members.append((z['bases'], z['det_ids'], z['log_probs'].astype(np.float64)))
    bases, det_ids = members[0][0], members[0][1]
    if any(not np.array_equal(b, bases) or not np.array_equal(d, det_ids) for b, d, _ in members[1:]):
        raise ValueError(f'{chain.dir}: the networks {versions} read different detections')
    lp = np.mean([m[2] for m in members], axis=0)
    lp = (lp - np.logaddexp.reduce(lp, axis=1, keepdims=True)).astype(np.float32)
    out = {}
    for base in dict.fromkeys(bases.tolist()):
        rows = bases == base
        out[base] = (det_ids[rows], lp[rows])
    return out
