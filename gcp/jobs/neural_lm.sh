#!/bin/bash
# Neural character language model of the decoder (trogs26.lm.neural): `neural_lm.sh STAGE [TRAIN ARGS...]` trains a
# character transformer on the decontaminated external corpus lines of the sources of lm/open_clean (`train`;
# TRAIN ARGS are its options, e.g. --steps 12000), scores the 224 public transcripts out of fold with it, as it is and
# fine-tuned on the other four folds, alone and mixed with the chain and external PPM-C models (`bits`), and writes
# the decoder's fine-tuned models (`finetune`).
# Inputs: open_text (the corpus line files and docs.tsv of the three families, public224.json), lm/open_clean,
# lm/public224_order12_exclude_f0 ... lm/public224_order12_exclude_f4.
# Outputs (--outputs): models/STAGE (base: model.pt, config.json, log.json; f0 ... f4 and all) and neural_lm/STAGE
# (bits.json, letters_fK.npz). Run by launchers/gcp.py with TROGS26_WORK=/work.
set -euo pipefail
STAGE=${1:?usage: neural_lm.sh STAGE [TRAIN ARGS...]}
shift
python -m trogs26.lm.neural train --root /work/open_text --out "/work/models/$STAGE/base" "$@"
python -m trogs26.lm.neural bits --root /work/open_text --model "/work/models/$STAGE/base" --lm-dir /work/lm \
    --out "/work/neural_lm/$STAGE"
python -m trogs26.lm.neural finetune --root /work/open_text --model "/work/models/$STAGE/base" \
    --bits "/work/neural_lm/$STAGE/bits.json" --out "/work/models/$STAGE"
