#!/usr/bin/env bash
# Benchmark augrs vs Albumentations on the COCO val2017 subset in $AUGRS_DATA.
# Fetch the data first with scripts/fetch_coco_subset.sh.
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$PROJECT_DIR"
python benches/bench_vs_albumentations.py --data "$AUGRS_DATA" --threads "${THREADS:-4}" "$@"
