"""A neural character language model for the public transcripts: train on the open corpus, then score out of fold.

How well does a causal character transformer predict the public transcripts, against the PPM-C models the decoder
reads, the chain PPM of a fold alone and mixed with the external PPM of the configuration (`lm.open`: name, order)?
Yardstick, in bits per letter over the 62,246 letters of the 224 public transcripts, out of fold: chain 2.9666. The
decoder reads the models of `finetune` when a variant names one (`decode.rescore`).

    python -m trogs26.lm.neural train --root ROOT --out DIR [--sources ...] [--steps 6000] [--batch 512] [--lr 6e-4]
    python -m trogs26.lm.neural bits --root ROOT --model DIR --lm-dir LM_DIR --out OUT_DIR [--finetune-steps 0 50 ...]
    python -m trogs26.lm.neural finetune --root ROOT --model DIR --bits BITS --out MODELS

`gcp/jobs/neural_lm.sh` runs the first two. ROOT is the open corpus directory of `trogs26.lm.open` (`<family>.txt`,
`<family>.docs.tsv`, `public224.json`), LM_DIR holds the PPM artifacts.

Model. A pre-LayerNorm GPT over the 27 symbols of the alphabet plus BOS (id 27: "row start, the text before it is
unknown"): d_model 384, 8 layers, 6 heads, MLP ratio 4, dropout 0.1, learned positions, a window of 192 tokens, input
and output embeddings tied, causal `scaled_dot_product_attention`. Its distribution is over the 27 symbols (BOS is
never predicted). Every size is a command-line option. DIR holds `model.pt` (state dict), `config.json` (architecture,
alphabet, training arguments, sources, letters, final validation bits) and `log.json`.

`train`. The lines of the chosen sources (default: those of the external PPM `open_clean`) of the
families form one uint8 code array with a separator after each line; every 200th line is held out as validation text.
A training sequence is BOS
and up to `context` letters of ONE line (a context never crosses a line): the model reads BOS and the first
`context - 1` of them and predicts each next letter, so all `context` positions carry a loss. It starts at a uniformly
random letter of the corpus with probability 0.8 (so BOS is followed by the middle of a line, like a squeeze row,
which is a fragment) and at a line start with probability 0.2. Batches are cut from the array with numpy; what follows
a line's end is padding that the loss ignores. AdamW (betas 0.9, 0.95; no decay on norms), linear warmup, cosine decay
to 10% of the peak, gradient clipping at 1.0, bf16 autocast and TF32 matmuls on cuda. At the defaults a step holds
512 x 192 tokens, an estimated 13 GB of activations (not measured): it should fit an L4 (24 GB).

`bits`. Out-of-fold bits per letter on the public transcript rows, the context reset at every row, as in decoding. A
row is scored as BOS and its letters: letter i is predicted from BOS and letters 0..i-1. A row of at most `context`
letters is one pass. In a longer row the first `context` letters are scored in that pass and every later letter i in
a pass of its own over BOS and the `context - 1` letters before it, so each letter keeps the longest left context
the window allows (BOS then stands for "text before unknown", as in a training sequence that starts mid-line).
Scores are computed in float32. Per fold k (the folds of `public224.json`):
  * `base`: the trained model as it is.
  * `finetuned@S`: the base weights fine-tuned for S steps on the rows of the other four folds, each step one batch of
    those rows (BOS and the row) plus one batch of open-corpus sequences drawn as in training, the two mean losses
    averaged; constant learning rate, AdamW, no warmup. One run per fold up to the largest S, scored at each S on the
    way; S = 0 is the base model.
  * `finetuned@nested`: the S whose pooled bits on the other four folds' own out-of-fold scores are lowest.
  * mixtures, per letter in the probability domain, on a grid of weights in steps of 0.05: chain + open, chain +
    neural, chain + open + neural, for neural in {base, finetuned@nested}. `nested` rows use, for fold k, the weights
    that minimise the bits of the other four folds; `best` rows the one set of weights that minimises the bits of all
    five (descriptive only, it uses the scored folds). `chain+open@W` mixes the two at the configured weight W.
Everything is reported per fold, pooled over all letters, and separately for the first four letters of each row and
the later ones; OUT_DIR gets `letters_f{k}.npz` (per-letter natural-log probabilities of every model of fold k, in
the order of `open.bits_grid`, with `row_lengths`) and `bits.json`.

Fold exclusion. The base model is trained on open-corpus lines only, and the open corpus holds no rare letter 15-gram
of a public or test transcript (`trogs26.lm.decontaminate`): the base model has seen no squeeze's own text. The model
of fold k is the
base model fine-tuned on the rows of the folds other than k (`Lines.from_rows` over j != k) and scores only fold k's
rows, so no score of fold k's rows comes from a model that trained on them. The choices for fold k, the step count S
and the mixture weights, are made from the scores of the other four folds' rows only. Those are fold j's own
out-of-fold scores, and fold j's models (the PPM chain included) did train on fold k's rows: as with the mixture
weight of the existing yardstick, fold k's rows reach fold k's result only through the choice of S and of the
weights, which are picked on rows that are not fold k's.

`finetune`. The models the decoder mixes into its language model, fine-tuned as `bits` fine-tunes (the settings of
BITS, the `bits.json` of a `bits` run of the model DIR): MODELS/f<k> is the base model fine-tuned on the rows of the
folds other than k for fold k's nested step count, and reads fold k's pages out of fold; MODELS/all is the base model
fine-tuned on the rows of all five folds for the step count whose pooled bits over the five folds are lowest, and
reads every other page. Each directory holds `model.pt` and `config.json` as `train` writes them, the latter with the
fine-tuning settings added.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .. import config
from . import artifact
from .open import FAMILIES, SOURCES, letter_logprobs, read_docs, read_inputs, selected_bytes
from .ppm import keep_chars

# the sources of the external PPM `open_clean`
DEFAULT_SOURCES = ('perseus_greeklit', 'first1k_greek', 'ddbdp', 'isicily', 'iaph2007', 'iospe')
SEP = 255  # ends a line in a code array (never a token)
DROP = 254  # in the byte table: a byte that is neither a letter of the alphabet nor a line break
IGNORE = -1  # target of a position with nothing to predict
MID_LINE = 0.8  # share of training sequences that start at a random letter (the others at a line start)
VALIDATION_EVERY = 200  # every n-th line of the corpus is validation text
VALIDATION_LETTERS = 2_000_000
VALIDATION_BLOCK = 1024  # sequences drawn at a time for the validation sample
VALIDATION_SEED = 20_000  # the validation sample is the same for every run
LOG_EVERY = 100
VALIDATE_EVERY = 1000
LR_FLOOR = 0.1  # cosine decay ends at this share of the peak rate
CLIP = 1.0
YARDSTICK = {'chain': 2.9666}  # bits per letter, out of fold
GRID = 20  # mixture weights are multiples of 1 / GRID
EARLY = 4  # letters at the start of a row that are reported apart
LN2 = math.log(2)
SUBSETS = {'all': [0, 1], 'first4': [0], 'later': [1]}  # columns of a tally
TITLES = {'all': 'all letters', 'first4': f'first {EARLY} letters of each row', 'later': 'later letters'}
NEURAL = ('base', 'finetuned@nested')


# ---------------------------------------------------------------------------------------------------------------
# Model

@dataclass(frozen=True)
class Arch:
    d_model: int = 384
    layers: int = 8
    heads: int = 6
    mlp_ratio: int = 4
    dropout: float = 0.1
    context: int = 192


class Block(nn.Module):
    """Pre-LayerNorm transformer block: causal self-attention, then a GELU MLP, each on a residual branch."""

    def __init__(self, arch: Arch):
        super().__init__()
        d = arch.d_model
        self.heads, self.dropout = arch.heads, arch.dropout
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.fc1 = nn.Linear(d, arch.mlp_ratio * d, bias=False)
        self.fc2 = nn.Linear(arch.mlp_ratio * d, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(b, t, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        x = x + F.dropout(self.proj(y.transpose(1, 2).reshape(b, t, d)), self.dropout, self.training)
        return x + F.dropout(self.fc2(F.gelu(self.fc1(self.ln2(x)))), self.dropout, self.training)


class CharLM(nn.Module):
    """Causal character transformer over `symbols` letters and BOS (id `symbols`); `forward` maps token ids
    (batch, time) to logits (batch, time, symbols): position t predicts the letter that follows token t."""

    def __init__(self, arch: Arch, symbols: int):
        super().__init__()
        if arch.d_model % arch.heads:
            raise ValueError('d_model must be a multiple of heads')
        self.arch, self.symbols, self.bos, self.context = arch, symbols, symbols, arch.context
        self.tok = nn.Embedding(symbols + 1, arch.d_model)
        self.pos = nn.Embedding(arch.context, arch.d_model)
        self.drop = nn.Dropout(arch.dropout)
        self.blocks = nn.ModuleList(Block(arch) for _ in range(arch.layers))
        self.ln = nn.LayerNorm(arch.d_model)
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, std=0.02)
        for name, weight in self.named_parameters():
            if name.endswith(('proj.weight', 'fc2.weight')):
                nn.init.normal_(weight, std=0.02 / math.sqrt(2 * arch.layers))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.tok(ids) + self.pos(torch.arange(ids.size(1), device=ids.device)))
        for block in self.blocks:
            x = block(x)
        with torch.autocast(ids.device.type, enabled=False):  # the output layer in float32
            return F.linear(self.ln(x).float(), self.tok.weight[:self.symbols].float())


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def device_of(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


def load_model(directory: Path, device: torch.device) -> tuple[CharLM, dict]:
    """The model saved in `directory` and its `config.json`."""
    meta = json.loads((directory / 'config.json').read_text())
    model = CharLM(Arch(**meta['architecture']), meta['symbols'])
    model.load_state_dict(torch.load(directory / 'model.pt', map_location='cpu', weights_only=True))
    return model.to(device), meta


# ---------------------------------------------------------------------------------------------------------------
# Text

def alphabet_lut(alphabet: str) -> np.ndarray:
    """Table from a byte to its letter id in `alphabet`; a line break is `SEP`, any other byte `DROP`."""
    lut = np.full(256, DROP, dtype=np.uint8)
    for i, ch in enumerate(alphabet):
        lut[ord(ch)] = i
    lut[10] = SEP
    return lut


def encode(text: str, lut: np.ndarray) -> np.ndarray:
    """Letter ids of `text` (alphabet letters only, as `keep_chars` leaves it)."""
    return lut[np.frombuffer(text.encode('ascii'), dtype=np.uint8)]


class Lines:
    """Lines of letter ids in one uint8 array, each followed by `SEP`; `start` and `length` locate the non-empty ones.
    Sequences are cut from the array in numpy and never run past the `SEP` that ends their line."""

    def __init__(self, codes: np.ndarray, start: np.ndarray, length: np.ndarray):
        self.codes, self.start, self.length = codes, start, length
        self.total = np.cumsum(length)

    @classmethod
    def from_rows(cls, rows: list[np.ndarray]) -> Lines:
        """The rows (arrays of letter ids) as lines."""
        length = np.array([len(row) for row in rows], dtype=np.int64)
        start = np.cumsum(length + 1) - length - 1
        codes = np.full(int(length.sum()) + len(rows), SEP, dtype=np.uint8)
        for row, first in zip(rows, start.tolist()):
            codes[first:first + len(row)] = row
        return cls(codes, start, length)

    def select(self, keep: np.ndarray) -> Lines:
        """The lines where the boolean array `keep` is true (the same code array)."""
        return Lines(self.codes, self.start[keep], self.length[keep])

    def sample_letters(self, count: int, rng: np.random.Generator) -> np.ndarray:
        """Start offsets of `count` training sequences: a uniformly random letter with probability `MID_LINE`, else
        the start of a uniformly random line."""
        pick = rng.integers(0, self.total[-1], count)
        line = np.searchsorted(self.total, pick, side='right')
        offset = pick - (self.total[line] - self.length[line])
        fresh = rng.random(count) >= MID_LINE
        line = np.where(fresh, rng.integers(0, len(self.length), count), line)
        return self.start[line] + np.where(fresh, 0, offset)

    def sample_rows(self, count: int, rng: np.random.Generator, context: int) -> np.ndarray:
        """Start offsets of `count` sequences of uniformly random lines: the line start, or for a line longer than
        `context` a uniformly random start that leaves `context` letters."""
        line = rng.integers(0, len(self.length), count)
        spare = np.maximum(self.length[line] - context + 1, 1)
        return self.start[line] + (rng.random(count) * spare).astype(np.int64)

    def cut(self, start: np.ndarray, context: int, bos: int) -> tuple[np.ndarray, np.ndarray]:
        """Model inputs and targets (batch, context) of the sequences that begin at the offsets `start`: the inputs are
        BOS and the letters up to the last but one, the targets the letters, `IGNORE` after the line's end."""
        at = np.minimum(start[:, None] + np.arange(context), len(self.codes) - 1)
        chunk = self.codes[at]
        valid = np.logical_and.accumulate(chunk != SEP, axis=1)
        targets = np.where(valid, chunk.astype(np.int64), IGNORE)
        inputs = np.zeros_like(targets)
        inputs[:, 0] = bos
        inputs[:, 1:] = np.where(valid[:, :-1], targets[:, :-1], 0)
        return inputs, targets


def load_corpus(root: Path, sources: list[str], alphabet: str) -> Lines:
    """The lines of `sources` in the families of `root` (family order, file order), characters outside `alphabet`
    dropped as `keep_chars` does."""
    lut = alphabet_lut(alphabet)
    wanted = {s.family for s in SOURCES if s.name in sources}
    pieces = []
    for family in (f for f in FAMILIES if f in wanted):
        docs = read_docs(root / f'{family}.docs.tsv')
        ok = np.array([source in sources for source in docs.sources])
        if ok.any():
            raw = np.fromfile(root / f'{family}.txt', dtype=np.uint8)
            pieces.append(lut[selected_bytes(raw, np.repeat(ok, docs.lines), 1)])
    mapped = np.concatenate(pieces)
    codes = mapped[mapped != DROP]
    ends = np.flatnonzero(codes == SEP)
    length = np.diff(ends, prepend=-1) - 1
    keep = length > 0
    return Lines(codes, (ends - length)[keep], length[keep])


def split_lines(lines: Lines) -> tuple[Lines, Lines]:
    """(training lines, validation lines): every `VALIDATION_EVERY`-th line is validation text."""
    held = np.arange(len(lines.length)) % VALIDATION_EVERY == 0
    return lines.select(~held), lines.select(held)


def validation_set(lines: Lines, context: int, bos: int) -> tuple[np.ndarray, np.ndarray]:
    """A fixed sample of about `VALIDATION_LETTERS` letters of `lines`, sequences cut as in training."""
    rng = np.random.default_rng(VALIDATION_SEED)
    inputs, targets, letters = [], [], 0
    while letters < VALIDATION_LETTERS:
        x, y = lines.cut(lines.sample_letters(VALIDATION_BLOCK, rng), context, bos)
        inputs.append(x)
        targets.append(y)
        letters += int((y >= 0).sum())
    return np.concatenate(inputs), np.concatenate(targets)


# ---------------------------------------------------------------------------------------------------------------
# Optimisation

def autocast(device: torch.device):
    """bf16 autocast on cuda, nothing elsewhere."""
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda')


def letter_nll(logits: torch.Tensor, targets: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
    """Cross entropy in nats of `logits` (batch, time, symbols) against `targets`, `IGNORE` positions skipped."""
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=IGNORE,
                           reduction=reduction)


def step_loss(model: CharLM, inputs: np.ndarray, targets: np.ndarray) -> torch.Tensor:
    """Mean nats per letter of `model` on a numpy batch."""
    device = device_of(model)
    x, y = torch.from_numpy(inputs).to(device), torch.from_numpy(targets).to(device)
    with autocast(device):
        logits = model(x)
    return letter_nll(logits, y)


def make_optimizer(model: CharLM, lr: float, weight_decay: float) -> torch.optim.AdamW:
    """AdamW (betas 0.9, 0.95), weight decay on the matrices only."""
    params = list(model.parameters())
    groups = [{'params': [p for p in params if p.dim() >= 2], 'weight_decay': weight_decay},
              {'params': [p for p in params if p.dim() < 2], 'weight_decay': 0.0}]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), fused=device_of(model).type == 'cuda')


def update(model: CharLM, optimizer: torch.optim.AdamW, loss: torch.Tensor) -> None:
    """One optimizer step on `loss` with gradient clipping."""
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
    optimizer.step()


def lr_at(step: int, peak: float, warmup: int, steps: int) -> float:
    """Learning rate of the 0-based `step`: linear warmup, then cosine decay to `LR_FLOOR` of the peak."""
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(steps - warmup, 1)
    return peak * (LR_FLOOR + (1 - LR_FLOOR) * 0.5 * (1 + math.cos(math.pi * progress)))


@torch.no_grad()
def evaluate(model: CharLM, inputs: np.ndarray, targets: np.ndarray, batch: int = 512) -> float:
    """Bits per letter of `model` on prepared sequences."""
    device, was_training = device_of(model), model.training
    model.eval()
    nats, count = 0.0, 0
    for a in range(0, len(inputs), batch):
        x = torch.from_numpy(inputs[a:a + batch]).to(device)
        y = torch.from_numpy(targets[a:a + batch]).to(device)
        with autocast(device):
            logits = model(x)
        nats += float(letter_nll(logits, y, 'sum'))
        count += int((targets[a:a + batch] >= 0).sum())
    model.train(was_training)
    return nats / count / LN2


def setup(device: str, seed: int) -> torch.device:
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    return torch.device(device)


# ---------------------------------------------------------------------------------------------------------------
# train

def run_train(args: argparse.Namespace) -> None:
    """Pre-train on the open corpus and write DIR/model.pt, config.json and log.json."""
    device = setup(args.device, args.seed)
    alphabet = config.load(args.config)['alphabet']
    arch = Arch(args.d_model, args.layers, args.heads, args.mlp_ratio, args.dropout, args.context)
    started = time.perf_counter()
    corpus = load_corpus(args.root, args.sources, alphabet)
    fit, held = split_lines(corpus)
    print(f'corpus: {fit.total[-1]} training letters in {len(fit.length)} lines, {held.total[-1]} validation letters '
          f'in {len(held.length)} lines, loaded in {time.perf_counter() - started:.0f} s', flush=True)
    model = CharLM(arch, len(alphabet)).to(device)
    print(f'parameters {parameter_count(model):,}', flush=True)
    val_x, val_y = validation_set(held, arch.context, model.bos)
    print(f'validation sample: {int((val_y >= 0).sum())} letters in {len(val_x)} sequences', flush=True)
    optimizer = make_optimizer(model, args.lr, args.weight_decay)
    rng = np.random.default_rng(args.seed)
    history: list[dict] = []
    seen = interval_letters = 0  # letters with a loss: in all, and since the last log line
    interval_loss = torch.zeros((), device=device)
    validation = float('nan')
    mark = time.perf_counter()
    model.train()
    for step in range(1, args.steps + 1):
        for group in optimizer.param_groups:
            group['lr'] = lr_at(step - 1, args.lr, args.warmup, args.steps)
        x, y = fit.cut(fit.sample_letters(args.batch, rng), arch.context, model.bos)
        loss = step_loss(model, x, y)
        update(model, optimizer, loss)
        interval_loss += loss.detach()
        letters = int((y >= 0).sum())
        seen += letters
        interval_letters += letters
        if step % LOG_EVERY == 0:
            bits = float(interval_loss) / LOG_EVERY / LN2
            rate = interval_letters / (time.perf_counter() - mark)
            history.append({'step': step, 'train_bits': bits, 'letters_per_second': rate})
            print(f'step {step} train {bits:.4f} bits/letter {rate:,.0f} letters/s', flush=True)
            interval_loss.zero_()
            interval_letters = 0
            mark = time.perf_counter()
        if step % VALIDATE_EVERY == 0 or step == args.steps:
            validation = evaluate(model, val_x, val_y)
            history.append({'step': step, 'validation_bits': validation})
            print(f'step {step} validation {validation:.4f} bits/letter', flush=True)
            mark = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save({name: value.cpu() for name, value in model.state_dict().items()}, args.out / 'model.pt')
    (args.out / 'config.json').write_text(json.dumps({
        'architecture': asdict(arch), 'symbols': model.symbols, 'bos': model.bos, 'alphabet': alphabet,
        'parameters': parameter_count(model),
        'train': {'steps': args.steps, 'batch': args.batch, 'lr': args.lr, 'warmup': args.warmup,
                  'weight_decay': args.weight_decay, 'seed': args.seed, 'mid_line_share': MID_LINE,
                  'validation_every_line': VALIDATION_EVERY},
        'sources': args.sources,
        'letters': {'training_corpus': int(fit.total[-1]), 'validation_corpus': int(held.total[-1]),
                    'validation_sample': int((val_y >= 0).sum()), 'trained_on': seen},
        'validation_bits': validation}, indent=1))
    (args.out / 'log.json').write_text(json.dumps(history, indent=1))


# ---------------------------------------------------------------------------------------------------------------
# bits: scoring rows

def windows(rows: list[np.ndarray], context: int, bos: int) -> tuple[np.ndarray, np.ndarray]:
    """Model inputs that score every letter of `rows` once, and per input position the index (over the letters of all
    rows in order) of the letter it predicts, -1 where it scores none: a first window per row (BOS and the row's first
    `context - 1` letters, scoring its first `context` letters) and, for each later letter of a longer row, a window
    of BOS and the `context - 1` letters before it that scores that letter in its last position."""
    width = min(context, max(len(row) for row in rows))
    count = sum(1 + max(len(row) - context, 0) for row in rows)
    inputs = np.zeros((count, width), dtype=np.int64)
    where = np.full((count, width), -1, dtype=np.int64)
    w = offset = 0
    for row in rows:
        n = len(row)
        m = min(n, context)
        inputs[w, 0] = bos
        inputs[w, 1:m] = row[:m - 1]
        where[w, :m] = offset + np.arange(m)
        w += 1
        extra = n - context
        if extra > 0:
            inputs[w:w + extra, 0] = bos
            inputs[w:w + extra, 1:] = np.lib.stride_tricks.sliding_window_view(row, context - 1)[1:extra + 1]
            where[w:w + extra, -1] = offset + context + np.arange(extra)
            w += extra
        offset += n
    return inputs, where


@torch.no_grad()
def score(model: CharLM, rows: list[np.ndarray], batch: int = 512) -> np.ndarray:
    """log P(letter | BOS and the row so far) (natural log) of every letter of `rows`, in row order, the context
    reset at each row; long rows are scored window by window as `windows` lays out. Float32."""
    device, was_training = device_of(model), model.training
    model.eval()
    inputs, where = windows(rows, model.context, model.bos)
    flat = np.concatenate(rows).astype(np.int64)
    out = np.full(len(flat), np.nan)
    for a in range(0, len(inputs), batch):
        logp = F.log_softmax(model(torch.from_numpy(inputs[a:a + batch]).to(device)), dim=-1)
        slots = where[a:a + batch]
        scored = slots >= 0
        target = torch.from_numpy(flat[np.where(scored, slots, 0)]).to(device)
        chosen = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1).cpu().numpy()
        out[slots[scored]] = chosen[scored]
    model.train(was_training)
    if np.isnan(out).any():
        raise AssertionError('a letter of the rows was not scored')
    return out


def bits_per_letter(logp: np.ndarray) -> float:
    return float(-logp.sum() / LN2 / len(logp))


@dataclass(frozen=True)
class Finetune:
    steps: list[int]
    lr: float
    batch: int
    weight_decay: float
    seed: int


def finetune_fold(model: CharLM, base: dict, rows: Lines, open_lines: Lines | None, held_out: list[np.ndarray],
                  fine: Finetune, fold: int) -> dict[int, np.ndarray]:
    """Per-letter log-probabilities of `held_out` (fold `fold`'s rows) under the base weights `base` fine-tuned for
    each count of steps in `fine.steps` (and 0, the base model), on `rows` (the rows of the other folds: the caller
    leaves fold `fold` out) and on `open_lines`, scored along a single run."""
    model.load_state_dict(base)
    scores = {0: score(model, held_out)}
    print(f'fold {fold} base {bits_per_letter(scores[0]):.4f} bits/letter', flush=True)

    def scored(step: int) -> None:
        scores[step] = score(model, held_out)
        print(f'fold {fold} fine-tuned {step} steps {bits_per_letter(scores[step]):.4f} bits/letter', flush=True)

    fine_tune(model, rows, open_lines, fine, fold, scored)
    return scores


def fine_tune(model: CharLM, rows: Lines, open_lines: Lines, fine: Finetune, fold: int, after=None) -> None:
    """Train `model` in place for `max(fine.steps)` steps on `rows` and `open_lines`, the random streams seeded by
    `fine.seed` and `fold`; `after(step)` is called after each step of `fine.steps`."""
    torch.manual_seed(1000 * fine.seed + fold)
    rng = np.random.default_rng([fine.seed, fold])
    optimizer = make_optimizer(model, fine.lr, fine.weight_decay)
    model.train()
    for step in range(1, max(fine.steps) + 1):
        public = step_loss(model, *rows.cut(rows.sample_rows(fine.batch, rng, model.context), model.context,
                                            model.bos))
        corpus = step_loss(model, *open_lines.cut(open_lines.sample_letters(fine.batch, rng), model.context,
                                                  model.bos))
        update(model, optimizer, (public + corpus) / 2)
        if after is not None and step in fine.steps:
            after(step)


# ---------------------------------------------------------------------------------------------------------------
# bits: mixtures and tables

def public_rows(root: Path, cfg: dict) -> dict[int, list[str]]:
    """fold -> its transcript rows in the alphabet, empty rows dropped, in the order of `open.bits_grid`."""
    folds, rows = read_inputs(root)
    return {k: [r for r in (keep_chars(row, cfg['alphabet']) for base in folds[k] for row in rows[base]) if r]
            for k in sorted(folds)}


def ppm_scores(lm_dir: Path, cfg: dict, fold_rows: dict[int, list[str]]) -> dict[str, list[np.ndarray]]:
    """Per-letter log-probabilities (fold order) of the chain PPM of each fold and of the external PPM `lm.open`."""
    chain = [letter_logprobs(artifact.load(lm_dir / cfg['lm']['oof'].format(fold=k), cfg['lm']['decode_max_order']),
                             fold_rows[k]) for k in fold_rows]
    open_lm = artifact.load(lm_dir / cfg['lm']['open']['name'], cfg['lm']['open']['max_order'])
    return {'chain': chain, 'open': [letter_logprobs(open_lm, fold_rows[k]) for k in fold_rows]}


def grid(components: int) -> np.ndarray:
    """Every weight vector (a row) of `components` multiples of 1 / `GRID` that sum to 1."""
    return np.array([w for w in itertools.product(range(GRID + 1), repeat=components) if sum(w) == GRID]) / GRID


def tally(logp: list[np.ndarray], early: list[np.ndarray]) -> np.ndarray:
    """Bits per fold, split into [first `EARLY` letters of a row, later letters]: shape (folds, 2)."""
    bits = [-lp / LN2 for lp in logp]
    return np.array([[b[e].sum(), b[~e].sum()] for b, e in zip(bits, early)])


def mixture_tallies(parts: list[list[np.ndarray]], weights: np.ndarray, early: list[np.ndarray]) -> np.ndarray:
    """`tally` of the mixture of components `parts` (component -> fold -> log-probabilities) under every weight vector
    (row of `weights`), in the probability domain: shape (candidates, folds, 2)."""
    out = np.zeros((len(weights), len(early), 2))
    with np.errstate(divide='ignore'):
        log_weights = np.log(weights)
    for f, is_early in enumerate(early):
        stack = np.stack([component[f] for component in parts])
        for c, log_w in enumerate(log_weights):
            bits = -np.logaddexp.reduce(log_w[:, None] + stack, axis=0) / LN2
            out[c, f] = bits[is_early].sum(), bits[~is_early].sum()
    return out


def choose(totals: np.ndarray, fold: int) -> int:
    """The row of `totals` (candidate x fold bits) with the fewest bits on the folds other than index `fold`."""
    return int(np.argmin(totals.sum(axis=1) - totals[:, fold]))


def tables(models: dict[str, np.ndarray], counts: np.ndarray, folds: list[int]) -> dict[str, dict]:
    """Bits per letter of every model (tally) per fold and pooled over all letters, for each of `SUBSETS`."""
    out = {}
    for subset, cols in SUBSETS.items():
        n = counts[:, cols].sum(axis=1)
        out[subset] = {name: {**{f'f{k}': round(float(t[i, cols].sum() / n[i]), 5) for i, k in enumerate(folds)},
                              'all': round(float(t[:, cols].sum() / n.sum()), 5)} for name, t in models.items()}
    return out


def bits_markdown(result: dict, folds: list[int]) -> str:
    """The tables of `result['bits']` as markdown, one per subset of letters, a row per model."""
    lines = []
    for subset, title in TITLES.items():
        lines += [f'{title} ({result["letters"][subset]["all"]} letters), bits per letter', '',
                  '| model | ' + ' | '.join(f'f{k}' for k in folds) + ' | all |',
                  '|---|' + '---|' * (len(folds) + 1)]
        for name, row in result['bits'][subset].items():
            lines.append(f'| {name} | ' + ' | '.join(f'{row[f"f{k}"]:.4f}' for k in folds) + f' | {row["all"]:.4f} |')
        lines.append('')
    return '\n'.join(lines)


def run_bits(args: argparse.Namespace) -> None:
    """Out-of-fold bits per letter of the PPM models, the base and fine-tuned neural model and their mixtures."""
    device = setup(args.device, args.seed)
    cfg = config.load(args.config)
    alphabet = cfg['alphabet']
    model, meta = load_model(args.model, device)
    if meta['alphabet'] != alphabet:
        raise ValueError(f'the model reads {meta["alphabet"]!r}, the configuration {alphabet!r}')
    fold_rows = public_rows(args.root, cfg)
    folds = sorted(fold_rows)
    logp = ppm_scores(args.lm_dir, cfg, fold_rows)
    lut = alphabet_lut(alphabet)
    codes = {k: [encode(row, lut) for row in fold_rows[k]] for k in folds}
    steps = sorted(set(args.finetune_steps))
    fine = Finetune(steps, args.finetune_lr, args.finetune_batch, meta['train']['weight_decay'], args.seed)
    open_lines = split_lines(load_corpus(args.root, meta['sources'], alphabet))[0] if steps[-1] > 0 else None
    base = {name: value.clone() for name, value in model.state_dict().items()}
    scores = {}
    for k in folds:
        train = Lines.from_rows([row for j in folds if j != k for row in codes[j]])
        scores[k] = finetune_fold(model, base, train, open_lines, codes[k], fine, k)
    logp['base'] = [scores[k][0] for k in folds]
    for s in steps:
        if s > 0:
            logp[f'finetuned@{s}'] = [scores[k][s] for k in folds]
    early = [np.concatenate([np.arange(len(row)) for row in codes[k]]) < EARLY for k in folds]
    counts = np.array([[e.sum(), (~e).sum()] for e in early])
    # S for fold k: the fewest bits on the other folds' own out-of-fold scores
    by_steps = np.array([tally([scores[k][s] for k in folds], early).sum(axis=1) for s in steps])
    nested = {k: steps[choose(by_steps, i)] for i, k in enumerate(folds)}
    logp['finetuned@nested'] = [scores[k][nested[k]] for k in folds]
    print('nested step counts', nested, flush=True)

    models = {name: tally(lp, early) for name, lp in logp.items()}
    w = cfg['lm']['open']['weight']
    models[f'chain+open@{w:g}'] = mixture_tallies([logp['chain'], logp['open']], np.array([[1 - w, w]]), early)[0]
    mixtures = [('chain+open', ['chain', 'open'])] + [(f'chain+{n}', ['chain', n]) for n in NEURAL] \
        + [(f'chain+open+{n}', ['chain', 'open', n]) for n in NEURAL]
    chosen = {}
    for name, parts in mixtures:
        weights = grid(len(parts))
        found = mixture_tallies([logp[p] for p in parts], weights, early)
        totals = found.sum(axis=2)
        picks = [choose(totals, i) for i in range(len(folds))]
        best = int(np.argmin(totals.sum(axis=1)))
        models[f'{name} nested'] = np.array([found[c, i] for i, c in enumerate(picks)])
        models[f'{name} best'] = found[best]
        chosen[name] = {'components': parts, 'best': weights[best].tolist(),
                        'nested': {str(k): weights[c].tolist() for k, c in zip(folds, picks)}}
        print(f'{name} weights {parts}: best {weights[best].tolist()}, nested '
              + ', '.join(f'f{k} {weights[c].tolist()}' for k, c in zip(folds, picks)), flush=True)

    result = {'letters': {subset: {**{f'f{k}': int(counts[i, cols].sum()) for i, k in enumerate(folds)},
                                   'all': int(counts[:, cols].sum())} for subset, cols in SUBSETS.items()},
              'bits': tables(models, counts, folds), 'nested_steps': {str(k): s for k, s in nested.items()},
              'mixture_weights': chosen, 'yardstick': YARDSTICK,
              'rows': {str(k): len(codes[k]) for k in folds},
              'finetune': {'steps': steps, 'lr': args.finetune_lr, 'batch': args.finetune_batch,
                           'weight_decay': fine.weight_decay, 'seed': args.seed},
              'model': {'directory': str(args.model), 'parameters': meta['parameters'], 'sources': meta['sources'],
                        'validation_bits': meta['validation_bits']}}
    args.out.mkdir(parents=True, exist_ok=True)
    for i, k in enumerate(folds):
        np.savez(args.out / f'letters_f{k}.npz', row_lengths=np.array([len(row) for row in codes[k]]),
                 **{name: lp[i] for name, lp in logp.items()})
    (args.out / 'bits.json').write_text(json.dumps(result, indent=1))
    print(bits_markdown(result, folds))
    overall = result['bits']['all']
    print('reference check (bits per letter, all letters): '
          + ', '.join(f'{name} {overall[name]["all"]:.4f} (yardstick {value})' for name, value in YARDSTICK.items()),
          flush=True)


# ---------------------------------------------------------------------------------------------------------------
# finetune: the decoder's models

def run_finetune(args: argparse.Namespace) -> None:
    """Write MODELS/f<k> for every fold and MODELS/all (module docstring)."""
    cfg = config.load(args.config)
    device = setup(args.device, args.seed)
    model, meta = load_model(args.model, device)
    bits = json.loads(args.bits.read_text())
    ft = bits['finetune']
    pooled = bits['bits']['all']
    all_steps = min((s for s in ft['steps'] if s), key=lambda s: pooled[f'finetuned@{s}']['all'])
    if pooled[f'finetuned@{all_steps}']['all'] >= pooled['base']['all']:
        all_steps = 0
    lut = alphabet_lut(cfg['alphabet'])
    codes = {k: [encode(row, lut) for row in rows] for k, rows in public_rows(args.root, cfg).items()}
    open_lines = split_lines(load_corpus(args.root, meta['sources'], cfg['alphabet']))[0]
    base = {name: value.clone() for name, value in model.state_dict().items()}
    plans = [(f'f{k}', int(bits['nested_steps'][str(k)]), k, [k]) for k in sorted(codes)]
    plans.append(('all', all_steps, len(codes), []))
    for name, steps, stream, held in plans:
        model.load_state_dict(base)
        if steps:
            train = Lines.from_rows([row for j in codes if j not in held for row in codes[j]])
            fine = Finetune([steps], ft['lr'], ft['batch'], ft['weight_decay'], ft['seed'])
            fine_tune(model, train, open_lines, fine, stream)
        out = args.out / name
        out.mkdir(parents=True, exist_ok=True)
        torch.save({key: value.cpu() for key, value in model.state_dict().items()}, out / 'model.pt')
        (out / 'config.json').write_text(json.dumps({
            **meta, 'finetune': {**ft, 'steps': steps, 'stream': stream, 'held_out_folds': held,
                                 'base': str(args.model), 'bits': str(args.bits)}}, indent=1))
        note = f', {bits_per_letter(score(model, codes[held[0]])):.4f} bits/letter on its rows' if held else ''
        print(f'{name}: {steps} fine-tuning steps{note}', flush=True)


# ---------------------------------------------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--root', type=Path, required=True, help='open corpus directory')
    common.add_argument('--seed', type=int, default=0)
    common.add_argument('--device', default='cuda')
    common.add_argument('--config', default='configs/pagenet_only_style_neural.json')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('train', parents=[common])
    p.add_argument('--out', type=Path, required=True, help='model directory to write')
    p.add_argument('--sources', nargs='+', choices=[s.name for s in SOURCES], default=list(DEFAULT_SOURCES))
    p.add_argument('--steps', type=int, default=6000)
    p.add_argument('--batch', type=int, default=512)
    p.add_argument('--lr', type=float, default=6e-4)
    p.add_argument('--warmup', type=int, default=500)
    p.add_argument('--weight-decay', type=float, default=0.1)
    d = Arch()
    p.add_argument('--d-model', type=int, default=d.d_model)
    p.add_argument('--layers', type=int, default=d.layers)
    p.add_argument('--heads', type=int, default=d.heads)
    p.add_argument('--mlp-ratio', type=int, default=d.mlp_ratio)
    p.add_argument('--dropout', type=float, default=d.dropout)
    p.add_argument('--context', type=int, default=d.context)
    p = sub.add_parser('bits', parents=[common])
    p.add_argument('--model', type=Path, required=True, help='directory written by `train`')
    p.add_argument('--lm-dir', type=Path, required=True, help='PPM artifacts (chains and the external model)')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--finetune-steps', type=int, nargs='+', default=[0, 50, 100, 200, 400, 800])
    p.add_argument('--finetune-lr', type=float, default=1e-4)
    p.add_argument('--finetune-batch', type=int, default=64)
    p = sub.add_parser('finetune', parents=[common])
    p.add_argument('--model', type=Path, required=True, help='directory written by `train`')
    p.add_argument('--bits', type=Path, required=True, help='bits.json of a `bits` run of that model')
    p.add_argument('--out', type=Path, required=True, help='directory to write f0..f4 and all under')
    args = parser.parse_args()
    {'train': run_train, 'bits': run_bits, 'finetune': run_finetune}[args.command](args)


if __name__ == '__main__':
    main()
