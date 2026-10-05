#!/usr/bin/env bash
# Run the PyTorch DataLoader example on the COCO subset (CPU-only torch in the project venv).
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$PROJECT_DIR"
python examples/torch_dataloader.py --data "$AUGRS_DATA" --workers "${THREADS:-4}" "$@"
