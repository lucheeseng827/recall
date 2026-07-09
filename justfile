# recall — developer task runner.
#
# The CANONICAL gate is `scripts/ci.sh` (pure bash), so CI and EC2/SSM nodes need only bash —
# `just` is optional dev sugar here. Run `just ci` locally for the same checks CI/EC2 run.

# Run the full gate: fmt + clippy (default & all-features) + the test matrix.
ci:
    ./scripts/ci.sh

# Tests only (default + test-support + static), skipping fmt/clippy — faster inner loop.
test:
    ./scripts/ci.sh --test-only

# Auto-format the workspace.
fmt:
    cargo fmt --all

# Lints only, treating warnings as errors (matches the gate).
lint:
    cargo clippy --workspace --all-targets --all-features -- -D warnings
