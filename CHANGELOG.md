# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and this project adheres to
[Semantic Versioning](https://semver.org/) (pre-1.0: minor = breaking).

## [Unreleased]

## [0.2.0] — 2026-09-16

### Added
- **The savings figure can outlive the process that measured it** (`--savings-flush-secs`, off by
  default, needs `--store redb`). The cumulative counters are snapshotted into the same database
  the entries live in and restored at start-up, so "this cache has saved $X" stops meaning "since
  the last deploy" — which is the only way it can be the sentence the cache is funded on. The
  snapshot is versioned and holds **counters only**, never prices, so changing `[pricing]`
  re-prices history for free and a snapshot can never disagree with the book that reads it. Every
  failure mode starts a fresh ledger with a warning (first run, corrupt blob, a version this build
  does not know): accounting must never be why a cache refuses to serve.
  **Off by default on purpose** — `/metrics` and `recall.export/v1` both document the process as
  the boundary their cumulative fields reset at, so counters that survive a restart change what
  every existing consumer is reading and are never switched on unasked. When it is on, the node
  publishes `recall_ledger_since_time_seconds` (and `node.ledger_since_unix` in the export,
  additive under the existing schema) so a consumer can tell the counters predate this process
  instead of inferring it; `recall status` labels the window in words. A fleet roll-up differencing
  successive documents needs no change either way: it keys on `process_start_unix`, so a restart
  still opens a fresh baseline rather than re-counting history the previous incarnation reported.
- **Savings, split by namespace** — `GET /v1/cache/namespaces`, plus the top rows in
  `recall status` and a `by_namespace` array in `recall export`. The node counters answer "is the
  cache worth running"; they cannot answer "worth it for whom", which is the question the moment
  more than one team or model shares a proxy. Rows are a **split of the same hits** the node
  counters already counted — never an addition — priced by the same `[pricing.<model>]` table,
  with an unpriced row carrying `cost_micros: 0` and its token count rather than a guess.
  Deliberately **not** a metric family: the namespace is derived from the request, so per-namespace
  series would be caller-controlled Prometheus cardinality (the same reason `/metrics` carries no
  per-model series). The document is bounded instead — at most 512 namespaces, the rest folded
  into one `(other namespaces)` row with `truncated` saying so. The `by_namespace` export field is
  additive under the existing `recall.export/v1` schema, and empty when the exporter could not
  read the breakdown, so every existing consumer is unaffected.
- **Second-neighbour ambiguity gate** (`--ambiguity-eps`, off by default): a would-be hit whose
  top-two neighbours sit within `eps` similarity of each other but store *different* answers is
  served as a miss. Near-tied lookalikes with disagreeing completions are exactly where false hits
  live; two near-duplicates with the *same* answer replay safely regardless of margin, and the
  exact-hash shortcut is never gated. A second neighbour that cannot be read misses conservatively.
  Demotions are counted by `recall_ambiguity_miss_total` — lost hits are lost savings, so the gate
  can never trade away hit-rate invisibly. Library builder: `SemanticCache::with_ambiguity_eps`.
- **HNSW tunables are configurable** rather than compiled in: `--hnsw-m`,
  `--hnsw-ef-construction`, `--hnsw-ef-search` on `serve` (plus `RECALL_HNSW_*` env and `hnsw_*`
  TOML keys), and matching `--m` / `--ef-construction` / `--ef-search` on `ann-bench` — defaulting
  to the same values `serve` uses, so a plain benchmark run measures the graph you actually deploy.
- **`recall export`** — the node's cumulative savings, fill cost, traffic, store facts, and
  feedback counts as one **versioned JSON document**, `recall.export/v1`. The shape is a
  documented compatibility surface (`docs/API.md`): `schema` first so consumers dispatch on a
  prefix read, additive-only evolution, integer micro-USD money, under-count flags riding
  alongside every figure, `node.process_start_unix` as the counter-reset boundary, and a
  content-derived `export_id` so an ingest ledger can dedupe a retried document.
- The metrics reader (`scrape`) now also parses the fill input/output split,
  `recall_process_start_time_seconds`, and the `recall_store_info` labels, so exports and any
  future consumer see the store's identity and the reset boundary.
- **Savings in dollars, from your own rates.** An optional `[pricing.<model>]` table in
  `recall.toml` (USD per 1M input / output tokens) makes `/metrics` export
  `recall_cost_saved_usd_total` alongside the raw token counters. There is no bundled price
  list and no network lookup — vendor prices change and differ by contract, so recall prices
  only with the numbers you supply, and an empty table (the default) is a supported state that
  `serve` reports at startup. Input and output are priced separately (output typically costs
  3–5× input), rates convert to integer micro-USD at load so millions of hits accumulate without
  drift, and a bad entry (negative / NaN / infinite / empty key) is warned about and skipped
  rather than stopping the proxy.
- **`recall_unpriced_tokens_saved_total`**: tokens saved for models with no matching price rule.
  recall never guesses a rate, so those tokens contribute `$0` and are flagged here instead —
  an incomplete or typo'd price table reads as *unpriced*, not as *the cache saved nothing*.
  Aggregate rather than per-model, because the model id comes off the request and a per-model
  label would be caller-controlled metric cardinality.
- Price-book keys match the model id **exactly** by default; a key ending in `*` is an opt-in
  prefix rule (so `"gpt-4o*"` covers dated snapshots), and the longest matching stem wins.
- **`recall replay` leads with the dollars.** The savings figure is now the first line of the
  report and comes from the target proxy's own per-model pricing, so a log spanning several
  models is priced with each model's rates instead of one blended rate. The flat
  `--price-input`/`--price-output` flags remain as a fallback for a target with no price book
  and are labeled as blended; with neither available the line says how to get a number rather
  than printing `$0.00`.
- **Savings reported gross *and* net of measured wrong hits.** With `--verify-sample`, `replay`
  prints the dollar figure discounted by the sampled false-hit rate alongside the gross one — a
  wrong hit is not a saving. Without sampling the figure is explicitly labeled un-discounted
  rather than passed off as verified.
- `docs/GETTING_STARTED.md` gains a "measure it on your own traffic — before you commit"
  walkthrough: replay a log of requests you already send at a throwaway proxy and read the
  savings, with no change to your production request path.
- **`recall status`** — one screen summarizing a running proxy: dollars saved, hit-rate, tokens,
  entries held, and the observed false-hit rate. Savings are discounted by that observed rate
  where outcomes have been reported and explicitly labelled un-discounted where none have.
  Under-counts (unpriced models, hits with no reported usage, dropped streamed misses) are
  printed as caveats rather than folded away, and a rate with no denominator abstains instead of
  printing `0.0%`. `--json` emits the raw snapshot; a non-zero exit distinguishes an unreachable
  proxy from an idle one.
- **`recall_feedback_agree_total` / `recall_feedback_wrong_total`** — served hits reported correct
  or wrong via `/v1/cache/feedback`. Counted at the proxy rather than inside the threshold policy,
  because the default `--policy static` discards observations: a wrong-hit report previously
  vanished behind `200 {"accepted": true}` with nothing anywhere recording that the cache had been
  caught serving a wrong answer, making the observed false-hit rate unmeasurable on the default
  configuration.
- **`recall_hits_usage_unknown_total`** — hits whose stored completion reported no token usage at
  all, so they credited nothing to the savings counters. The audit valve: it makes a savings
  figure read as under-counted rather than as an idle cache.
- **`recall_cache_entries`** — completions the store currently holds, read live at scrape time
  (cheap on both backends) so it stays correct across a restart that rehydrated a durable store.
- **What the cache cost to fill.** `recall_fill_cost_usd_total` and the `recall_fill_*` token
  counters are the mirror of the savings counters in the other direction: what this process paid
  the upstream for the entries it stored, priced from the same `[pricing.<model>]` table. Against
  `recall_cost_saved_usd_total` that is the instance's payback. Credited when a write **succeeds**,
  never on a miss that was not stored — a miss that bought the cache nothing would have cost the
  same with recall switched off. `recall_unpriced_fill_tokens_total` and
  `recall_fill_usage_unknown_total` mark the two ways it can under-count.
- **Start-up facts as metrics.** `recall_persisted_entries_at_start`, `recall_rehydrated_entries`,
  `recall_store_info{backend,durable}` and `recall_process_start_time_seconds`. "Did my cache
  survive the restart" was previously answerable only by whoever happened to be watching stderr at
  the time; it is now a ratio of two measured numbers. `recall status` prints it as
  "recovered N of M persisted entries rebuilt at start-up".
- The in-memory store now says at start-up that it begins empty and is not kept across a restart,
  so a low hit-rate in the first minutes reads as expected rather than as a fault.

### Fixed
- **HNSW was unusable at the embedding width recall actually ships.** Neighbour selection took the
  `M` closest candidates, which wires near-duplicate edges and starves the graph of the long-range
  links routing depends on at high dimensionality: recall@1 against the brute-force oracle measured
  **0.26 at 50k entries** on 256-dim vectors — the width of the bundled static embedder — so the
  documented `≥ 0.98` gate held only in the low-dimensional unit test. Insert and prune now use the
  paper's diversity heuristic (keep a candidate only if it is closer to the base point than to every
  neighbour already kept, then backfill with the nearest pruned ones), and the defaults move to
  `m=32`, `ef_construction=400`, `ef_search=256`. Re-measured on the same adversarial
  uniform-random workload: **0.998 at 10k** and **0.876 at 50k**. Recall still decays with N, so
  BENCHMARKS.md now publishes the numbers per scale and points 50k+ deployments at a higher
  `ef_search` (now configurable) or `--index brute`, rather than implying one number covers every
  corpus size. A missed neighbour was never a *wrong* answer — the threshold still gates what is
  served — it was a silently lost hit, and so silently lost savings.
- **`recall_misses_total` (and so `recall_hit_ratio`) now mean one thing on both delivery
  paths.** A miss is counted once, at lookup time, for every cacheable request the cache could
  not answer — regardless of what the upstream then does. Previously the buffered path counted a
  miss only when the upstream response was actually storable (2xx JSON), while the streamed path
  counted every forward, so the same failing upstream moved the hit rate on one path and not the
  other. **Dashboard note:** on buffered-path deployments with upstream failures, `misses` now
  also includes forwards that failed or returned non-cacheable content, so `recall_hit_ratio`
  can read slightly lower than before — it is now the honest rate over cacheable traffic.
- **`recall replay` reported the proxy's lifetime totals, not the run's.** Savings and token
  counters are now differenced across the run (snapshot before, snapshot after), so replaying
  twice against one long-lived `serve` measures the second run instead of reporting both runs
  added together. The hit-rate is likewise computed from the run's own responses rather than
  read from the proxy's lifetime `recall_hit_ratio` gauge.

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

[Unreleased]: https://github.com/lucheeseng827/recall/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/lucheeseng827/recall/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/lucheeseng827/recall/compare/v0.1.0-alpha.1...v0.1.1
[0.1.0-alpha.1]: https://github.com/lucheeseng827/recall/releases/tag/v0.1.0-alpha.1
