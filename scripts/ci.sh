#!/usr/bin/env bash
#
# Portable CI gate for recall.
#
# Runs the full quality bar — formatting, lints, and the test matrix (default + `test-support` +
# `static`) — with one command, fail-fast and NON-interactive. It resolves its own paths from
# ${BASH_SOURCE}, so it behaves identically whether invoked from a developer laptop or a CI job
# (no TTY needed). The process EXIT CODE is the gate result that CI reports.
#
# Usage:
#   scripts/ci.sh              # full gate: fmt + clippy (default & all-features) + tests
#   scripts/ci.sh --test-only  # tests only (skip fmt/clippy) — faster inner loop
#
set -euo pipefail

MODE="full"
case "${1:-}" in
  --test-only) MODE="test-only" ;;
  "") ;;
  *)
    echo "usage: $0 [--test-only]" >&2
    exit 2
    ;;
esac

# Resolve the module root from this script's own location so the current working directory does not
# matter (SSM Run Command executes from '/').
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="$SCRIPT_DIR/../Cargo.toml"

step() { printf '\n==> %s\n' "$*"; }
fail() {
  printf '\nFAILED: %s\n' "$*" >&2
  exit 1
}

step "toolchain"
command -v cargo >/dev/null 2>&1 || fail "cargo not on PATH (install Rust via rustup first)"
rustc --version
cargo --version

if [ "$MODE" = "full" ]; then
  step "rustfmt --check"
  cargo fmt --manifest-path "$MANIFEST" --all -- --check

  step "clippy (default features, all targets)"
  cargo clippy --manifest-path "$MANIFEST" --workspace --all-targets -- -D warnings

  step "clippy (all features, all targets)"
  cargo clippy --manifest-path "$MANIFEST" --workspace --all-targets --all-features -- -D warnings

  step "benches compile (criterion hot-path + ann; no run)"
  # Keep the benchmark harnesses building every CI run without paying the full measurement cost.
  # `cargo bench` reproduces the numbers locally (no published number ships unmeasured).
  cargo bench --manifest-path "$MANIFEST" --workspace --no-run
fi

step "test (default features, incl. doctests)"
cargo test --manifest-path "$MANIFEST" --workspace

step "test (recall-core test-support doubles)"
cargo test --manifest-path "$MANIFEST" -p recall-core --features test-support

step "test (static embedder, offline / local-load only)"
cargo test --manifest-path "$MANIFEST" -p recall-embed --features static

step "build (recall binary, static embedder)"
cargo build --manifest-path "$MANIFEST" -p recall --features static

step "test (durable redb store)"
cargo test --manifest-path "$MANIFEST" -p recall-store --features redb

step "build (recall binary, durable redb store)"
cargo build --manifest-path "$MANIFEST" -p recall --features store-redb

printf '\nALL CHECKS PASSED\n'
