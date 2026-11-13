#!/bin/bash
# The external corpora and their PPM-C model (trogs26.lm.open): the sources at their pinned revisions (download),
# staged in three families (extract; also the PHI crawl, for the calibration only), decontaminated against the 224
# public and the 50 test squeezes (decontaminate --calibrate, trogs26.lm.decontaminate), then counted at order 6:
# lm/open_clean (the openly licensed sources).
# Inputs: open_text/references.json and open_text/public224.json (`open inputs`, written where the annotations are),
# phi_raw/chunks, phi_raw/completion_manifest.json.
# Outputs (--outputs): open_text (corpora, docs, decontamination.json, manifest.json, staging; the raw sources are
# removed), lm/open_clean. Run by launchers/gcp.py with TROGS26_WORK=/work.
set -euo pipefail
ROOT=/work/open_text
OPEN="perseus_greeklit first1k_greek ddbdp isicily iaph2007 iospe"
mkdir -p $ROOT/raw
ln -sfn /work/phi_raw $ROOT/raw/phi
open() { python -m trogs26.lm.open --root $ROOT --lm-dir /work/lm "$@"; }
open download
open extract --family literary papyri epigraphic phi --jobs "$(nproc)"
open decontaminate --calibrate
open count --name clean --corpora literary papyri epigraphic --order 6 --sources $OPEN
rm -rf $ROOT/raw
