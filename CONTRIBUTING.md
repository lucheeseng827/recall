# Contributing to recall

Thanks for your interest in improving recall — the self-hosted, Rust-native semantic cache.

## Developer Certificate of Origin (DCO)

We use the [DCO](https://developercertificate.org/) rather than a CLA. By signing
off on your commits you certify that you wrote the patch or otherwise have the
right to submit it under the project's Apache-2.0 license (inbound = outbound).

Add a sign-off line to every commit:

```bash
git commit -s -m "your message"
```

This appends `Signed-off-by: Your Name <you@example.com>` (using your `git`
identity). The patent grant comes from Apache-2.0 itself.

## Development

```bash
cargo build --workspace
cargo test  --workspace
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
```

The canonical quality gate is `scripts/ci.sh` (pure bash): formatting, lints, and
the test matrix across the `static` and `store-redb` features. Run `just ci`
(or `./scripts/ci.sh`) and make sure it passes before opening a PR — CI runs the
same checks on every PR.

## Architecture & extension points

recall is one cache behind four seams (traits in `recall-core`). Prefer adding
behind an existing seam rather than widening the core surface:

- `Embedder` — text → vectors (`recall-embed` for real backends)
- `AnnIndex` — nearest-neighbour search (`recall-index` for HNSW)
- `Store` — durable KV/blob (`recall-store` for redb)
- `ThresholdPolicy` — hit-vs-miss decision (`recall-calibrate` for the adaptive engine)

`recall-core` must stay network-free and dependency-light (only `blake3` +
`thiserror` in the default build) — that is the single-static-binary / air-gap
property. Heavy or optional dependencies live behind feature flags in the
backend crates, never in `recall-core`'s default build.

## Expectations

- New source files carry the SPDX header: `// SPDX-License-Identifier: Apache-2.0`.
- Add tests for new behaviour. Correctness-sensitive paths (cache keying,
  namespace isolation, threshold decisions, streaming replay) need explicit tests.
- Use clear, conventional commit messages (e.g. `feat:`, `fix:`, `docs:`).
- Keep the public surface small and legible.

## Code of Conduct

This project follows the [Contributor Covenant](https://www.contributor-covenant.org/).
Be respectful.
