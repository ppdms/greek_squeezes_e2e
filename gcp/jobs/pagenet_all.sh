#!/bin/bash
# The page network on every fold: `pagenet_all.sh CONFIG` trains CONFIG's page network (its `pagenet` recipe) on all
# the squeezes of CONFIG's `data.folds`, each through its own chain's out-of-fold detector records, then reads the
# TROGS-26 pages with it once per chain, from that chain's calibrated detector records. The crop caches of the
# recipe's input are cut first (current caches are kept).
# Inputs: annotations, pretrained/dinov3-vits16-pretrain-lvd1689m, chains/fK/oof/det and chains/fK/new/det of every
# chain, the public images and the TROGS-26 images (launchers/gcp.py run ... --trogs26-images).
# Outputs (--outputs): models/pagenet_<version>/all and chains/fK/new/pagenet_<version>_all.npz. Run by
# launchers/gcp.py with TROGS26_WORK=/work.
set -euo pipefail
CONFIG=${1:?usage: pagenet_all.sh CONFIG}
INPUT=$(python -c 'import sys; from trogs26 import config; print(config.load(sys.argv[1])["pagenet"]["input"])' "$CONFIG")
python -m trogs26.dino.stage crops --input "$INPUT" --group oof --jobs 4 --config "$CONFIG"
python -m trogs26.pagenet.stage train --all --config "$CONFIG"
python -m trogs26.dino.stage crops --input "$INPUT" --group new --splits trogs26 --jobs 4 --config "$CONFIG"
for K in 0 1 2 3 4; do
    python -m trogs26.pagenet.stage read --fold "$K" --all --group new --splits trogs26 --config "$CONFIG"
done
