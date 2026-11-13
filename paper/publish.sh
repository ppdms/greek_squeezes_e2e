#!/usr/bin/env bash
# Build the paper and publish it at https://data.ppdms.gr/report.pdf (R2 bucket `data`, rclone remote `r2`), then
# check that the URL serves exactly the local PDF. `Cache-Control: no-cache` makes Cloudflare's edge revalidate with R2
# on every request; browsers still get the zone's 4-hour browser TTL, so reload the tab to see a new version.
set -euo pipefail
cd "$(dirname "$0")"
URL=https://data.ppdms.gr/report.pdf

if ! latexmk -pdf -interaction=nonstopmode -halt-on-error report.tex >/dev/null 2>&1; then
    grep -n -A3 '^!' report.log >&2 || tail -n 30 report.log >&2
    exit 1
fi
overfull=$(grep -c '^Overfull' report.log || true)
[ "$overfull" = 0 ] || echo "warning: $overfull overfull boxes (report.log)" >&2

rclone copyto \
    --header-upload 'Content-Disposition: inline; filename="report.pdf"' \
    --header-upload 'Cache-Control: no-cache' \
    --no-check-dest \
    --s3-no-check-bucket \
    report.pdf r2:data/report.pdf

local_sha=$(shasum -a 256 report.pdf | cut -d' ' -f1)
served_sha=$(curl -fsS "$URL" | shasum -a 256 | cut -d' ' -f1)
if [ "$served_sha" != "$local_sha" ]; then
    echo "$URL serves sha256 $served_sha, not the local $local_sha" >&2
    exit 1
fi
echo "$URL: $(wc -c <report.pdf | tr -d ' ') bytes, sha256 ${local_sha:0:12}"
