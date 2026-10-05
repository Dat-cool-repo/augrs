# Common environment for the build/test/bench scripts (Linux, macOS, WSL). Source this file.
#
# Everything can be overridden from the environment:
#   AUGRS_VENV        Python virtualenv to activate (default: <repo>/.venv, if it exists;
#                     otherwise the scripts use whatever python/maturin is on PATH)
#   AUGRS_DATA        benchmark / example data directory (default: <repo>/data, git-ignored)
#   CARGO_TARGET_DIR  cargo build directory (default: cargo's own, <repo>/target). On WSL with
#                     the repo on a Windows drive, a Linux path such as ~/target/augrs is much faster.
#   CARGO_BUILD_JOBS, RAYON_NUM_THREADS
[ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"
export PATH=$HOME/.local/bin:$PATH
export PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-5}
export RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-4}
export AUGRS_VENV=${AUGRS_VENV:-$PROJECT_DIR/.venv}
export AUGRS_DATA=${AUGRS_DATA:-$PROJECT_DIR/data}
[ -f "$AUGRS_VENV/bin/activate" ] && source "$AUGRS_VENV/bin/activate"
true
