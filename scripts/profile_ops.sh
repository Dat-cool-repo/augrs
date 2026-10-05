#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$PROJECT_DIR"
cargo run --release -q -p augrs-core --example profile_ops
