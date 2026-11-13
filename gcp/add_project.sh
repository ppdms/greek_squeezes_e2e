#!/bin/bash
# Make a project able to run trogs26 job VMs (launchers/gcp.py): created, linked to the billing account, Compute
# Engine enabled, a request for one GPU in all regions (and one flex-start A100 per A100 region), and the runner
# service account, which may manage the project's VMs (a job VM deletes itself) and read and write the job bucket.
#
#     gcp/add_project.sh trogs26-gpu3
#
# Prints the project's granted GPU quota at the end (the request is decided by Google, usually within minutes).
# Running it again on a project finishes what is missing.
set -euo pipefail
P=${1:?usage: gcp/add_project.sh <project id>}
BILLING=01FD30-EE2ACC-DC0946
BUCKET=gs://trogs26-541485613110
SA=trogs26-runner@$P.iam.gserviceaccount.com
gcloud projects describe "$P" >/dev/null 2>&1 || gcloud projects create "$P" --name="trogs26 ${P#trogs26-}"
[ "$(gcloud billing projects describe "$P" --format='value(billingAccountName)')" = "billingAccounts/$BILLING" ] \
    || gcloud billing projects link "$P" --billing-account=$BILLING >/dev/null
gcloud services enable compute.googleapis.com cloudquotas.googleapis.com iam.googleapis.com --project="$P"
request() {  # quota id, preference id, extra flags; a preference that exists is left alone
    gcloud beta quotas preferences describe "$2" --project="$P" >/dev/null 2>&1 && return
    gcloud beta quotas preferences create --service=compute.googleapis.com --quota-id="$1" --preferred-value=1 \
        --preference-id="$2" --project="$P" --email=me@ppdms.gr "${@:3}" \
        --justification="One GPU per training job of a research project on reading Greek squeezes." >/dev/null
}
request GPUS-ALL-REGIONS-per-project gpus-all-regions-1
for r in us-central1 us-east1 us-west1 europe-west4; do
    request PREEMPTIBLE-NVIDIA-A100-GPUS-per-project-region "preemptible-a100-$r" --dimensions=region=$r
done
gcloud iam service-accounts describe "$SA" --project="$P" >/dev/null 2>&1 \
    || gcloud iam service-accounts create trogs26-runner --project="$P" --display-name="trogs26 job runner"
gcloud projects add-iam-policy-binding "$P" --member="serviceAccount:$SA" --role=roles/compute.instanceAdmin.v1 \
    --condition=None >/dev/null
gcloud storage buckets add-iam-policy-binding $BUCKET --member="serviceAccount:$SA" --role=roles/storage.objectAdmin \
    >/dev/null
sleep 90
gcloud compute project-info describe --project="$P" --format=json \
    | python3 -c "import json,sys; print('$P', [(q['metric'], q['limit']) for q in json.load(sys.stdin)['quotas'] if q['metric'] == 'GPUS_ALL_REGIONS'])"
