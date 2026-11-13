#!/bin/bash
# Chain K's page network: `pagenet_fold.sh K CONFIG` trains it under CONFIG (its `pagenet` recipe) on the other
# folds of CONFIG's `data.folds`, then reads fold K out of fold. The crop caches of the recipe's input are cut first
# for every out-of-fold page from the public images (current caches are kept).
# Inputs: annotations, pretrained/dinov3-vits16-pretrain-lvd1689m, chains (every chain's calibrated out-of-fold
# detector records), the public images.
# Outputs (--outputs): models/pagenet_<version>/fK and chains/fK/oof/pagenet_<version>.npz. Run by launchers/gcp.py
# with TROGS26_WORK=/work.
set -euo pipefail
K=${1:?usage: pagenet_fold.sh K CONFIG}
CONFIG=${2:?usage: pagenet_fold.sh K CONFIG}
INPUT=$(python -c 'import sys; from trogs26 import config; print(config.load(sys.argv[1])["pagenet"]["input"])' "$CONFIG")
python -m trogs26.dino.stage crops --input "$INPUT" --group oof --jobs 4 --config "$CONFIG"
python -m trogs26.pagenet.stage train --fold "$K" --config "$CONFIG"
