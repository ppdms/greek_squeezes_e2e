#!/bin/bash
# Pages read through the images' registration: `pagenet_images_read.sh GROUP CONFIG...` rewrites every chain's
# detector records of GROUP (new: the TROGS-26 pages; oof: each chain's out-of-fold public squeezes) with the map chosen
# on the images (python -m trogs26.registration_images), detects again the pages that map changes (python -m
# trogs26.detect.redetect; both under configs/pagenet_only_style_neural.json), cuts their crops again and reads them
# once per chain with each CONFIG's network: on new pages the one trained on every fold (gcp/jobs/pagenet_all.sh), on a
# chain's out-of-fold pages the one trained without its fold (gcp/jobs/pagenet_fold.sh). Records of an earlier run
# (chains/fK/GROUP/det_images, if given) replace chains/fK/GROUP/det, and both stages leave alone what they have done
# already.
# Inputs: chains/fK/GROUP/det or det_images of every chain, models/detector/fK, detector/calibration.json,
# models/pagenet_<version>/all (new) or models/pagenet_<version>/fK (oof) of every CONFIG,
# pretrained/dinov3-vits16-pretrain-lvd1689m and the pages' images (new: launchers/gcp.py run ... --trogs26-images).
# Outputs (--outputs): chains/fK/new/pagenet_<version>_all_images.npz or chains/fK/oof/pagenet_<version>_images.npz,
# and chains/fK/GROUP/det_images (the rewritten records). Run by launchers/gcp.py with TROGS26_WORK=/work.
set -euo pipefail
GROUP=${1:?usage: pagenet_images_read.sh new|oof CONFIG...}
shift
case "$GROUP" in
    new) SPLITS=(--splits trogs26); ALL=(--all); SUFFIX=_all ;;
    oof) SPLITS=(); ALL=(); SUFFIX= ;;
    *) echo "unknown group $GROUP" >&2; exit 2 ;;
esac
for K in 0 1 2 3 4; do
    D="$TROGS26_WORK/chains/f$K/$GROUP"
    if [ -d "$D/det_images" ]; then
        rm -rf "$D/det"
        mv "$D/det_images" "$D/det"
    fi
done
SYSTEM=configs/pagenet_only_style_neural.json
python -m trogs26.registration_images --group "$GROUP" "${SPLITS[@]}" --jobs 8 --config "$SYSTEM"
python -m trogs26.detect.redetect --group "$GROUP" "${SPLITS[@]}" --config "$SYSTEM"
for CONFIG in "$@"; do
    read -r VERSION INPUT < <(python -c 'import sys; from trogs26 import config
net = config.load(sys.argv[1])["pagenet"]
print(net["version"], net["input"])' "$CONFIG")
    python -m trogs26.dino.stage crops --input "$INPUT" --group "$GROUP" "${SPLITS[@]}" --jobs 4 --config "$CONFIG"
    for K in 0 1 2 3 4; do
        python -m trogs26.pagenet.stage read --fold "$K" "${ALL[@]}" --group "$GROUP" "${SPLITS[@]}" --config "$CONFIG"
        mv "$TROGS26_WORK/chains/f$K/$GROUP/pagenet_${VERSION}${SUFFIX}.npz" \
           "$TROGS26_WORK/chains/f$K/$GROUP/pagenet_${VERSION}${SUFFIX}_images.npz"
    done
done
for K in 0 1 2 3 4; do
    cp -r "$TROGS26_WORK/chains/f$K/$GROUP/det" "$TROGS26_WORK/chains/f$K/$GROUP/det_images"
done
