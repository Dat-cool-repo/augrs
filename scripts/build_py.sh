#!/usr/bin/env bash
# Build the Python extension (release, abi3) and install it into the project venv.
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$PROJECT_DIR/crates/augrs-py"
maturin develop --release --uv 2>&1 | grep -vE "^\s+(Compiling|Downloaded|Downloading)" | tail -n 25
python -c "import augrs; print('augrs', augrs.__version__, 'from', augrs.__file__)"
