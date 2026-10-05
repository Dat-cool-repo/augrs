#!/usr/bin/env bash
# cargo check the whole workspace (quick compile feedback).
set -uo pipefail
source "$(dirname "$0")/env.sh"
cd "$PROJECT_DIR"
cargo check --workspace --all-targets 2>&1 | grep -E "^(error|warning)|-->|^\s+\|" | head -${LINES_MAX:-150}
cargo check --workspace --all-targets --message-format short 2>&1 | tail -3
