#!/usr/bin/env bash
# Cross-build the abi3 Windows wheel from WSL/Linux with mingw-w64
# (needs: rustup target add x86_64-pc-windows-gnu; apt install gcc-mingw-w64-x86-64).
# PyO3's `generate-import-lib` feature creates python3.lib, so no Windows Python is needed.
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$PROJECT_DIR"
maturin build --release --target x86_64-pc-windows-gnu --out "$PROJECT_DIR/dist" "$@" 2>&1 \
  | grep -vE "^\s+(Compiling|Downloaded|Downloading)" | tail -n 15
ls -la "$PROJECT_DIR"/dist/*win*.whl
