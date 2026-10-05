#!/usr/bin/env bash
# Run the Python tests (compat with Albumentations/OpenCV + API tests).
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$PROJECT_DIR"
export OMP_NUM_THREADS=4
python -m pytest compat-tests -q -p no:cacheprovider "$@"
