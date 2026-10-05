#!/usr/bin/env bash
# Run the Rust unit + property tests (4 test threads).
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$PROJECT_DIR"
cargo test -p augrs-core --release "$@" -- --test-threads=4
