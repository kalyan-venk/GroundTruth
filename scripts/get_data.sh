#!/usr/bin/env bash
# Fetch the Criteo Uplift v2.1 log.
#
# The URL printed in the original AdKDD paper (go.criteo.net/...) is dead as of
# 2026-07 — it 404s, and the S3 bucket behind it returns 403. The file is still
# published by Criteo AI Lab on HuggingFace, same bytes, same CC BY-NC-SA 4.0
# licence.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DIR="$REPO_ROOT/data/raw"
URL="https://huggingface.co/datasets/criteo/criteo-uplift/resolve/main/criteo-research-uplift-v2.1.csv.gz"
GZ="$RAW_DIR/criteo-research-uplift-v2.1.csv.gz"
CSV="$RAW_DIR/criteo-research-uplift-v2.1.csv"

mkdir -p "$RAW_DIR"

if [[ ! -f "$GZ" ]]; then
  echo "Downloading Criteo Uplift v2.1 (311 MB)..."
  curl -L --retry 3 -o "$GZ" "$URL"
else
  echo "Archive already present: $GZ"
fi

# Spark cannot split a gzip stream, so a .csv.gz is read by exactly one task no
# matter how many cores you have. Decompressing once buys real parallelism for
# every subsequent run. Costs ~3.5 GB of disk.
if [[ ! -f "$CSV" ]]; then
  echo "Decompressing to plain CSV (~3.5 GB, needed for parallel Spark reads)..."
  gunzip -k -c "$GZ" > "$CSV"
else
  echo "CSV already present: $CSV"
fi

echo
ls -lh "$RAW_DIR"
echo
echo "Header:"
head -1 "$CSV"
