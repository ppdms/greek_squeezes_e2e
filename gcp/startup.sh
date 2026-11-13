#!/bin/bash
# Startup script of a trogs26 job VM (launchers/gcp.py). Runs once per VM: environment, code, images, inputs, the
# job's command, outputs, status; then the VM deletes itself. Everything goes to gs://$BUCKET/jobs/$JOB/; a failed
# command's outputs, whatever exists of them, go to gs://$BUCKET/jobs/$JOB/salvage/ (never to work/).
set -uo pipefail
[ -f /var/trogs26-started ] && exit 0
touch /var/trogs26-started

MD=http://metadata.google.internal/computeMetadata/v1/instance
meta() { curl -sf -H 'Metadata-Flavor: Google' "$MD/$1"; }
JOB=$(meta attributes/job)
BUCKET=$(meta attributes/bucket)
VM=$(meta name)
ZONE=$(meta zone | awk -F/ '{print $NF}')
J=gs://$BUCKET/jobs/$JOB
LOG=/var/log/trogs26-job.log
exec > >(tee -a "$LOG") 2>&1
START=$(date -u +%FT%TZ)
(while sleep 60; do gcloud storage cp -q "$LOG" "$J/log.txt"; done) &

spec() { python3 -c "import json,sys; v=json.load(open('/tmp/spec.json'))[sys.argv[1]]; print(' '.join(v) if isinstance(v, list) else v)" "$1"; }

finish() {
    local rc=$1 phase=$2
    if [ "$rc" = 0 ]; then
        for p in $(spec outputs); do
            if [ -d "/work/$p" ]; then gcloud storage rsync -r "/work/$p" "gs://$BUCKET/work/$p" || rc=97
            elif [ -f "/work/$p" ]; then gcloud storage cp "/work/$p" "gs://$BUCKET/work/$p" || rc=97
            else echo "missing output $p"; rc=98; fi
        done
    elif [ "$phase" = command ]; then
        for p in $(spec outputs); do
            if [ -d "/work/$p" ]; then gcloud storage rsync -r "/work/$p" "$J/salvage/$p"
            elif [ -f "/work/$p" ]; then gcloud storage cp "/work/$p" "$J/salvage/$p"; fi
        done
    fi
    printf '{"job": "%s", "vm": "%s", "zone": "%s", "start": "%s", "end": "%s", "rc": %s, "phase": "%s"}\n' \
        "$JOB" "$VM" "$ZONE" "$START" "$(date -u +%FT%TZ)" "$rc" "$phase" | gcloud storage cp - "$J/status.json"
    gcloud storage cp -q "$LOG" "$J/log.txt"
    gcloud compute instances delete "$VM" --zone "$ZONE" --quiet
    exit 0
}

step() { local phase=$1; shift; echo "== $phase $(date -u +%T)"; "$@" || finish $? "$phase"; }

step spec gcloud storage cp "$J/spec.json" /tmp/spec.json
COMMIT=$(spec commit)
mkdir -p /repo /work/images /data
step code bash -c "gcloud storage cat gs://$BUCKET/code/$COMMIT.tar.gz | tar -xz -C /repo"
export HOME=/root PATH=/root/.local/bin:$PATH
step uv bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh'
step venv uv venv /venv --python 3.11
source /venv/bin/activate
step deps uv pip install -r /repo/gcp/requirements-gpu.txt
step deps_nodeps uv pip install --no-deps -r /repo/gcp/requirements-gpu-nodeps.txt
step package uv pip install --no-deps -e /repo
nvidia-smi
if [ "$(spec public_images)" = True ]; then
    step public_images hf download papadimas/trogs-greek-squeezes --repo-type dataset --include 'data/*/*.png' \
        --local-dir /data/public
    find /data/public/data -name '*.png' -exec ln -sf {} /work/images/ \;
fi
if [ "$(spec trogs26_images)" = True ]; then
    step trogs26_images gcloud storage cp -r "gs://$BUCKET/images/trogs26/*" /work/images/
fi
for p in $(spec inputs); do
    mkdir -p "/work/$(dirname "$p")"
    step "input $p" gcloud storage cp -r "gs://$BUCKET/work/$p" "/work/$(dirname "$p")/"
done
cd /repo
export TROGS26_WORK=/work TROGS26_COMMIT=$COMMIT MPLBACKEND=Agg PYTHONUNBUFFERED=1
echo "== command $(date -u +%T)"
bash -c "$(spec cmd)"
finish $? command
