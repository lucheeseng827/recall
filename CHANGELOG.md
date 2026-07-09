# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and this project adheres to
[Semantic Versioning](https://semver.org/) (pre-1.0: minor = breaking).

## [Unreleased]

## [0.1.1] — 2026-07-10

First tagged release after the public mirror went live.

### Added
- **Getting-started tutorial** (`docs/GETTING_STARTED.md`): bench → hand-driven
  loop → caching proxy against a mock upstream (miss at ~1.2 s, hit at ~2 ms,
  hits surviving a restart), with every output block captured from real runs.
- Reference docs shipped alongside the code: `docs/CONFIG.md`, `docs/API.md`,
  `docs/OPERATIONS.md`.
- Release scaffolding for the first public cut: `CONTRIBUTING.md` (DCO sign-off),
  `SECURITY.md` (disclosure policy + the structural-not-authenticated security
  model note), this `CHANGELOG.md`, and crates.io `keywords`/`categories` on the
  publishable crates.
- Vendored the upstream MIT license text for `model2vec-rs` into `NOTICE` (used by
  the optional, off-by-default `static` embedder backend).
- **Latency histograms** in `/metrics`: `recall_cache_get_duration_seconds`
  (end-to-end lookup p50/p99 — embed + ANN search + decide) and
  `recall_upstream_duration_seconds` (buffered upstream forward), hand-rolled over
  atomics with `LATENCY_BUCKETS` spanning 0.005–10 s.
- **Structured logging** via `tracing`: `recall serve` installs a `RUST_LOG`-aware
  subscriber and the proxy emits per-request hit/miss/bypass events (at `debug`)
  and upstream failures (at `warn`). Prompt and completion bodies are never logged
  — only structural fields (namespace, score, status) — so customer content cannot
  leak into server logs.
- **`recall calibrate --file <pairs.jsonl>`**: there is no portable magic threshold
  across embedders, so this measures the should-hit vs should-not cosine
  distributions of a labeled pair set under the configured embedder and prints the
  cutoff τ that best separates them (Youden's J), or — with `--target-fhr <r>` — the
  most permissive τ that holds the false-hit rate at or below `r`.
- **`recall-eval` crate** — the adaptive threshold's reproducible evidence harness. Runs the real
  shipped policies (`static@0.8`, a `static@best` baseline tuned to hold the budget,
  and the adaptive threshold feedback-off/on) over a deterministic, controllable-
  density synthetic workload and reports the hit-rate vs false-hit-rate comparison;
  `cargo run -p recall-eval`. Five library tests act as CI regression gates (the
  adaptive engine must beat the best single τ at the same false-hit budget, must
  match a static cutoff with no feedback, must degrade gracefully under noisy
  feedback, and `decide` must stay well under 50 µs).
- **Criterion benchmarks**: `recall-core/benches/hot_path.rs` isolates
  embed / search / decide / end-to-end lookup (M0 budgets embed < 1 ms, decide < 50 µs),
  and `recall-index/benches/ann.rs` benches flat vs HNSW search as the corpus grows
  (showing the empirical crossover — flat wins at small N, HNSW wins at scale). Run with
  `cargo bench`; CI compiles them every run (`cargo bench --no-run`) so the harness can't rot.
- **Config hot-reload** (`recall serve --watch-config`): edits to the
  threshold settings (`policy` / `tau` / `target_fhr`) in `--config` are applied in place
  via an `arc-swap` policy snapshot watched with `notify` — no restart, no dropped
  connections, and `decide` reads the live policy lock-free on the hot path. A field pinned
  by a flag or env keeps its startup value (precedence preserved); a parse error keeps the
  previous good policy so a typo can't take the cache down; changes to non-reloadable fields
  (listen / upstream / model / index / store) are logged with a restart hint and ignored.

### Changed
- Workspace version `0.1.0-alpha.1` → `0.1.1`.

### Fixed
- **Streaming cache no longer stores truncated completions.** A streamed miss is now cached only when
  the SSE stream terminates cleanly — it must carry the protocol terminator (`data: [DONE]` for
  OpenAI, `message_stop` for Anthropic) and must not report a truncation finish/stop reason
  (`length` / `max_tokens` / `content_filter` / tool calls). A graceful upstream half-close, a
  mid-stream error, or a non-UTF-8 body is streamed to the client but not stored, so a later identical
  request can never replay a partial answer as if whole. Dropped streams are counted by the new
  `recall_stream_not_stored_total` metric.

## [0.1.0-alpha.1] — first public alpha

The self-hosted semantic cache loop, end to end, as a single static binary or an
embeddable library.

### Added
- **Core cache loop** (`recall-core`): the four seams — `Embedder`, `AnnIndex`,
  `Store`, `ThresholdPolicy` — and the `SemanticCache` facade. Exact-hash shortcut
  → embed → nearest-neighbour search → threshold decision → hit/miss, with
  verify-on-hit and store-then-index ordering. The default build pulls only
  `blake3` + `thiserror` (no ML, no network, no C deps).
- **Embedders** (`recall-embed`): the deps-free `HashEmbedder` default, plus an
  optional `static` (model2vec/potion) backend behind a feature flag, local-load
  only (no network).
- **Durable store** (`recall-store`): an optional `redb` backend (pure-Rust ACID),
  with cross-restart rehydration that rebuilds the in-memory index + exact-map from
  persisted entries so cached *hits* survive a restart.
- **Index** (`recall-index`): a pure-Rust, zero-dependency HNSW graph for scale
  (recall@1 ≥ 0.98 vs the brute-force oracle).
- **Adaptive threshold engine** (`recall-calibrate`): a feedback-driven policy that
  targets an operator-chosen false-hit rate; rests at a cold-start cutoff until
  feedback arrives, so it is safe to enable.
- **Proxy** (`recall-proxy`, `recall serve`): OpenAI-compatible
  `/v1/chat/completions` and Anthropic-compatible `/v1/messages`, each in its own
  namespace partition, for both non-streaming and streaming (SSE) requests, plus a
  raw `/v1/cache/{lookup,insert,feedback}` sidecar, `/healthz`, and Prometheus
  `/metrics`.
- **CLI** (`recall`): `serve`, `ask`, `bench`, `ann-bench`, `replay`.

[Unreleased]: https://github.com/lucheeseng827/recall/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/lucheeseng827/recall/compare/v0.1.0-alpha.1...v0.1.1
[0.1.0-alpha.1]: https://github.com/lucheeseng827/recall/releases/tag/v0.1.0-alpha.1
