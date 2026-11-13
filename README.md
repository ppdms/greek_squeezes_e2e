# Reading Greek Squeezes with Page Networks

Letter detection, page-network reading and decoding for Greek epigraphic squeezes, on the TROGS-26 data (Text
Recognition of Greek Squeezes): 224 public squeezes with annotated transcripts and 50 test squeezes. A letter detector
proposes boxes on both views of a squeeze; two page networks read every box in the context of its neighbours, from
crops of the relief recovered from both views; a beam decoder with character language models turns their posteriors
into rows of text; ROVER merges the transcripts of the five chains. `paper/report.pdf` describes the method and its
evaluation (also at https://data.ppdms.gr/report.pdf).

Nested out of fold on the 224 public squeezes: 12,612 errors in 131,473 Core characters (9.59% Core CER; the
operating point itself, chosen on all of them, makes 12,565, 9.56%; 7,935 in 96,628, 8.21%, without the `Merged`
squeezes). On the 50 TROGS-26 squeezes, scored with the organisers' scorer after the transcripts were frozen: 1,464
errors in 15,862 Core characters (9.23%; `paper/data/trogs26_score.json`).

## Layout

- `configs/` – `pagenet_only_style_neural.json`, the system; `pagenet_only_style_neural_oof.json`, the same system on
  the public folds, each fold read by the networks trained without it; `style_mild_5_epochs.json` and
  `style_mild_15_epochs.json`, the training recipes of the two page networks.
- `data/` – the public squeezes by split, the TROGS-26 image names and the folds: `folds_train181.json`, the 181
  TRAIN squeezes, for the detectors; `folds_public224.json`, all 224 public squeezes, for the page networks, the
  language models and every out-of-fold result.
- `trogs26/` – the package. `layout.py` defines the work tree every stage reads and writes (`TROGS26_WORK`).
  `detect/` holds the detectors; `registration.py` and `registration_images.py` map one view onto the other;
  `relief.py` and `dino/crops.py` cut the relit crops; `pagenet/` holds the page networks, `lm/` the language models
  and `decode/` the decoder; `rover.py` merges the chains and `score.py` scores with the organisers' scorer
  (`third_party/`, verbatim).
- `launchers/` – `gcp.py` (GPU jobs on Compute Engine, running the scripts in `gcp/jobs/`), `azure.py` (a pool of
  Azure VMs for the decoding) and `local.py` (a per-squeeze stage in parallel on this machine).
- `paper/` – the paper (`report.tex`); `make_results.py` and `make_figures.py` write its numbers and figures from the
  work tree, except the page-network diagram, drawn by hand in TikZ (`diagram/pagenet.tex`); `publish.sh` builds it
  and puts it at https://data.ppdms.gr/report.pdf.
- `tests/`

## Environment

```
uv venv --python 3.11.15 .venv
uv pip install --python .venv/bin/python -e '.[dev,paper]'
```

The `paper` extra serves `paper/make_figures.py`. The paper and its figures also need TeX Live with LuaLaTeX and the
Latin Modern fonts, which the figures locate with `kpsewhich`.

The GPU stages (detectors, page networks, the neural character model) run in the environment of
`gcp/requirements-gpu.txt` plus `gcp/requirements-gpu-nodeps.txt`, which `gcp/startup.sh` installs on each job's VM;
the decoding runs in the environment of `PINS` in `launchers/azure.py`, which it installs on the pool's VMs.

## Weights

Every model the system runs on a new squeeze is public, laid out as in the work tree, at
`hf://buckets/papadimas/greek_squeezes/trogs26/` (`MANIFEST.sha256` holds the sha256 of every file):

```
hf buckets sync hf://buckets/papadimas/greek_squeezes/trogs26 $TROGS26_WORK
```

- `models/pagenet_style_mild_5_epochs/` and `models/pagenet_style_mild_15_epochs/`: the page-network pair
  (`configs/style_mild_5_epochs.json`, `configs/style_mild_15_epochs.json`). `all/` is trained on all 224 public
  squeezes and reads new pages; `f{k}/` is trained on the public squeezes outside fold k of
  `data/folds_public224.json` and reads fold k.
- `models/detector/f{k}/`: the letter detector of chain k (YOLO26-L oriented boxes, epoch 59), trained on the TRAIN
  squeezes outside fold k of `data/folds_train181.json`, and `chains/f{k}/oof/det_raw/`: its raw out-of-fold records
  of the TRAIN squeezes, from which `python -m trogs26.detect.calibrate` fits again, with the TRAIN annotations, the
  calibration new pages' records go through.
- `lm/public224_order12[_exclude_f{k}]/`: PPM-C counts of the public transcripts (all 224, or without fold k), and
  `lm/open_clean/`: PPM-C counts of the openly licensed Greek text, decontaminated (step 5).
- `models/neural_lm_clean/`: the neural character model, `base/` trained on that text, `all/` fine-tuned on all 224
  public transcripts and `f{k}/` without fold k (step 6).

No model holds text of the PHI Greek inscriptions, whose terms do not allow redistribution. Step 5 reads a crawl of
them (its chunks and completion manifest as `trogs26/lm/phi.py` reads them; the crawler is not part of this
repository) for one measurement, the calibration of the decontamination thresholds; the corpora and models do not
depend on it.

The page networks derive from Meta's DINOv3 weights and carry the DINOv3 License (`LICENSE.md` beside each); the
detectors derive from Ultralytics' YOLO26 (AGPL-3.0).

## Stage by stage

Every command runs in the repository with `TROGS26_WORK` set to the work tree; `SYS` is
`configs/pagenet_only_style_neural.json`, `V` each of `style_mild_5_epochs` and `style_mild_15_epochs`, `K` each chain
0-4. The GPU stages run as Compute Engine jobs, `python launchers/gcp.py run NAME --inputs ... --outputs ... --cmd
"..."`, which run the committed tree on a fresh VM (`launchers/gcp.py --help`); the decoding runs on the Azure pool
(`launchers/azure.py`) or here.

1. **Detectors** (GPU, in a work tree of their own, `DETECTOR_WORK`): for every K, `python -m trogs26.detect.train
   --fold K --folds-file data/folds_train181.json`, then `python -m trogs26.detect.infer --fold K --folds-file
   data/folds_train181.json --group oof` and `... --group new --splits trogs26,validation,internal_test`; then
   `python -m trogs26.detect.calibrate --folds-file data/folds_train181.json`.
2. **Detector records** (CPU, seconds): `python -m trogs26.detect.frozen DETECTOR_WORK --group oof` and `...
   --group new --splits trogs26` copy the calibrated records into `chains/f{k}/{oof,new}/det/`, and
   `detector/calibration.json` beside them: every public squeeze goes to the chain of its fold of
   `data/folds_public224.json`, read by a detector that never saw it.
3. **Page networks** (GPU): `gcp/jobs/pagenet_fold.sh K configs/V.json` trains V without fold K
   (`models/pagenet_V/fK`); `gcp/jobs/pagenet_all.sh configs/V.json` trains it on all 224 (`models/pagenet_V/all`).
4. **Registration on the images** (GPU): `gcp/jobs/pagenet_images_read.sh oof configs/style_mild_5_epochs.json
   configs/style_mild_15_epochs.json` rewrites the out-of-fold records with the map chosen on the images, detects
   again the pages that map changes, cuts their relit crops and reads each fold with its pair
   (`chains/f{k}/oof/pagenet_V_images.npz`); with `new` it does the same for the TROGS-26 pages with the pair trained
   on all 224 (`chains/f{k}/new/pagenet_V_all_images.npz`). Both rewrite `det/` in place and keep a copy in
   `det_images/`.
5. **Language models** (CPU): `python -m trogs26.lm.build` (the public transcripts, without each fold and all 224);
   `python -m trogs26.lm.open inputs --test-annotations DIR` (where the public and the TROGS-26 annotations are: the
   squeezes' letters, `references.json`, the only use of the test annotations), then `gcp/jobs/open_text.sh`:
   `download` (the sources at pinned revisions), `extract` (and the PHI crawl in `open_text/raw/phi`),
   `decontaminate --calibrate` (`trogs26/lm/decontaminate.py`: every squeeze's inscription dropped, every rare
   15-gram of a public or test squeeze cut out of all other text; the calibration measured on PHI) and `count`:
   `lm/open_clean`.
6. **Neural character model** (GPU): `gcp/jobs/neural_lm.sh neural_lm_clean` trains it on the decontaminated text
   of `lm/open_clean`'s sources, measures its fine-tuning length and writes `f{k}` and `all`.
7. **Operating point** (CPU): for each external weight W of 30, 45 and 60, `python -m trogs26.decode.sweep --variant
   P --config configs/pagenet_only_style_neural_oof_openW.json --run clean_openW` decodes every public squeeze out of
   fold under every point of the grid (lam, w, beta and the neural weight nn); `paper/make_results.py` reads the
   nested result over the three, and their lowest-error point is `lm.open.weight` and `decode.variants.P.combo`.
8. **TROGS-26** (CPU): `python -m trogs26.decode.newpages --variant P --run pagenet_only_style_neural --splits
   trogs26`, then `python -m trogs26.rover --run pagenet_only_style_neural --split trogs26` (writes
   `runs/pagenet_only_style_neural/trogs26/rover/`).
9. **Scoring**: `python -m trogs26.score PRED_DIR GT_DIR` (the organisers' v2 Core errors).

## Detector

`trogs26/detect/` trains chain k's letter detector (YOLO26-L oriented boxes, Ultralytics 8.4.186) and writes the
detector records every later stage reads. The GPU environment is `gcp/requirements-gpu.txt` plus
`gcp/requirements-gpu-nodeps.txt` (Ultralytics, installed without its own OpenCV requirement); the tests of the
Ultralytics side skip where it is not installed.

`train` needs the annotations and images of every squeeze of the folds file and `pretrained/yolo26l-obb.pt`, and writes
`models/detector/f{K}/{weights.pt,receipt.json}`; `infer` needs `models/detector/f{K}` and both images of each page
(no annotation) and writes `chains/f{K}/{oof,new}/det_raw/{base}.json`; `calibrate` needs every fold's out-of-fold raw
records and their annotations and writes `detector/calibration.json` and `chains/f{k}/{oof,new}/det/{base}.json`.
`python -m trogs26.detect.<stage> --help` and the module docstrings give the details and the method.
