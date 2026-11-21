# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and this project adheres to
[Semantic Versioning](https://semver.org/) (pre-1.0: minor = breaking).

## [Unreleased]

### Fixed
- **Feedback for an unbounded number of namespaces could grow an adaptive policy's state without
  limit.** `AdaptiveThreshold` kept a `HashMap` of learned per-namespace bands with no cap, and
  `observe` is reachable from the network in exactly one place: `POST /v1/cache/feedback`, whose
  `namespace` is taken **verbatim from the request body**. A client posting feedback under distinct
  namespaces bought one permanent entry each, for as long as it kept posting — measured at 50,000
  entries after 50,000 posts. Only `--policy adaptive` was affected; the default `--policy static`
  keeps no per-namespace state (its `observe` is a no-op).

  Now bounded at **512 namespaces**, matching the per-namespace savings ledger, which was already
  bounded for exactly this reason. A newcomer past the cap **evicts a band rather than being turned
  away**: refusing it would mean whoever filled the map first held it forever, so a namespace that
  came up later — a new team, a new model, a service that simply started after the others — could
  never learn at all.

  Admission runs through a **doorkeeper**: a queue of at most 4096 namespaces the policy has heard
  from, each costing two integers and a key rather than a band. A namespace earns a band on its
  **second** observation — one is exactly what a namespace that posted once and never again looks
  like, and telling those apart is the doorkeeper's whole job. A flood of distinct namespaces
  therefore allocates **no bands at all** and evicts none, however long it runs: it only churns
  against itself inside the queue.

  The queue is eight times the band budget on purpose. A namespace posting feedback slowly needs
  its first observation still remembered when its second arrives, and a flood ages out one
  probationer per post — so what matters is how many posts it can land in between. The honest limit
  is that a flood longer than the queue does age out a single knock; the cost is one observation,
  never a band, and the namespace knocks again next time.

  Taking a band from a full budget is an **admission decision**, and a comparative one — the
  candidate must have knocked at least as many times as the band it would displace has been
  observed. Without that the doorkeeper is only a speed bump: a flood pays two posts per namespace
  and starts evicting bands with thousands of observations behind them. The displaced namespace
  returns to the queue **carrying what it earned**, one knock from winning its slot back rather
  than starting over; its moments do not survive, which is why that counts as an eviction.

  **A lookup counts as a use, not only feedback**, so a namespace serving production traffic with no
  feedback source attached keeps the cutoff its hits are served at. `decide` records that under the
  read lock it already takes, never the write lock the hot path cannot afford, and it reads only the
  band map — a namespace still at the door has learned nothing for traffic to protect. Reading a
  band through the introspection accessors is deliberately *not* a use, so a metrics scrape cannot
  perturb what it measures.

  Bands are never merged into a shared overflow row the way the savings ledger merges its own. That
  row only accounts, so merging there costs resolution and nothing else; this band *decides* — τ is
  the cutoff a future lookup is measured against — and pooling distinct namespaces' score
  distributions would let traffic that is not a namespace's own move that threshold. Evictions are
  counted on the new `recall_policy_namespaces_evicted_total`.

  `Namespace` also gained a **512-byte length cap**. Bounding how many keys those maps hold is only
  half the memory when a single key can be arbitrarily long, and the sidecar takes its namespace
  straight from the request body. The cap is far above anything the proxy derives
  (`{base}:{flavor}:{model}:{fingerprint}`); a longer one is now a `400`.

- **Rejected feedback was invisible.** All four `400` paths on `/v1/cache/feedback` returned without
  counting anything, so a client sending `"incorrect"` instead of `"wrong"`, or a score on a 0–100
  scale, left `recall_feedback_agree_total` and `recall_feedback_wrong_total` flat at zero — the
  same picture as no client at all — while the policy trained on nothing and the observed false-hit
  rate, the discount an honest savings figure is quoted net of, silently read as unmeasured. Now
  counted on `recall_feedback_rejected_total`.

- **The raw cache sidecar was invisible to `/metrics`.** `POST /v1/cache/lookup` counted nothing:
  a deployment using recall as a semantic cache over HTTP — a supported front door — served hits
  while `recall_hits_total`, `recall_misses_total` and `recall_requests_total` all read `0` and
  `recall_hit_ratio` read `0.0000`. A cache that looks idle is a cache that gets turned off.
  Sidecar lookups now count on the same hit/miss counters, the same
  `recall_cache_get_duration_seconds` histogram and the same per-namespace breakdown as proxied
  traffic, because they are the same question asked through a different door.

  A sidecar hit credits `recall_hits_usage_unknown_total` rather than any token counter: it really
  did avoid an upstream call, but the caller never said which model or how many tokens, so the
  saving is real and unmeasurable. Counting it as zero saved would be a number; counting it as
  unknown is the fact.

### Changed
- `recall_requests_total` now counts every request that **asked the cache for an answer** —
  proxied completions plus raw sidecar lookups — where it previously counted only chat
  completions. `/v1/cache/insert` and `/v1/cache/feedback` are still excluded: they do not ask the
  cache anything. A dashboard reading it as "chat completions received" should read it as "cache
  lookups served".

### Added
- **OpenTelemetry `gen_ai.*` spans for the cache decision** (`RECALL_OTEL_ENDPOINT`, unset by
  default). One span per proxied request, named `{operation} {model}` as the GenAI semantic
  conventions prescribe, so recall appears inside an existing tracing stack with nothing written to
  adapt it. `/metrics` answers "how is this node doing"; this answers "what happened to *that*
  request".

  **A hit does not report `gen_ai.usage.*`.** That is the decision the feature turns on, not an
  omission: no model call happened, so reporting usage tokens would have every tool that sums those
  attributes counting tokens nobody spent — a cache inflating the spend dashboard of the team that
  installed it to cut spend. The avoided tokens go to `recall.saved.input_tokens`,
  `recall.saved.output_tokens` and `recall.saved.cost_micros`, where no standard aggregation reads
  them as spend. A miss or bypass *did* call a model, and reports real usage — and no
  `recall.saved.*`, because nothing was saved and a zero would sum into a savings figure as though
  it were one. Usage the upstream never reported is **absent, not zero**: `0/0` would read as a
  call that cost nothing rather than one whose cost is unknown.

  `gen_ai.response.model` reports what the *response* named — not an echo of the request. Both
  providers resolve aliases to dated snapshots, and that resolution is the only reason the
  attribute is separate from `gen_ai.request.model`; an echo would make the two equal by
  construction and tell a reader nothing. Absent when the response named nothing.

  Every path emits, on both the OpenAI and the Anthropic front door: hit, buffered miss, streamed
  miss and bypass. The **raw cache sidecar** emits too, as a `cache.lookup` span carrying **no
  `gen_ai.*` attribute at all** — that caller supplies a namespace and a prompt and never names a
  model, so GenAI attributes there would be invented and would file non-LLM traffic into an LLM
  cost dashboard. The span type enforces it: the GenAI half is one optional field, so a sidecar
  span has nowhere to put a model. Two consequences a dashboard
  should know about. A **bypass has no `recall.cache.namespace`** — it never reached the cache, so
  there is no key, and an empty string would be something to group by that does not exist. And a
  **streamed span can disagree with the `x-recall-cache` header the caller got**: the header is
  written before the first byte of the body and cannot be taken back, so a stream that later turns
  out to be truncated ships `miss` and is then dropped. The span is emitted at the true end of the
  stream — the first moment the outcome is known — and says `miss-nostore`. Where they differ, the
  span is the accurate one.

  **Prompts and completions are never exported**, and cannot be: the span type has no field that
  could carry either. "Off by default" is not the same as impossible, and a cache whose argument is
  that prompts stay put cannot ship telemetry that ships prompts. A test asserts no prompt text
  appears in the encoded bytes.

  Unset `RECALL_OTEL_ENDPOINT` means no exporter is built, no task is spawned and no span leaves the
  process — the network-free default is untouched, and there is no separate enable flag to leave on
  with nowhere to send. Export cannot affect a request: a bounded queue (dropped and counted when
  full, never grown) feeds a background flush whose send is fire-and-forget, so a collector that is
  slow or gone cannot reach back into the request path. Written against semantic conventions
  **v1.40.0**; `gen_ai.provider.name` is replacing `gen_ai.system` and both are emitted during the
  overlap.

- **`recall_cache_put_duration_seconds`** — cache write latency (embed + store), the mirror of
  `recall_cache_get_duration_seconds` on the other side of the cache. The lookup side has always
  been timed; the write side never was, on any door. One histogram covers all three: a buffered
  miss stored before its response returns (so the latency is on the caller's critical path), a
  streamed miss stored after its last byte (so it is not), and `POST /v1/cache/insert`.

  Observed whether the write **succeeded or failed**. A store that takes two seconds and then
  errors is the case this exists to make visible, and timing only the successes would hide exactly
  the slow path worth finding. A request rejected before it reaches the store — a malformed
  namespace, say — is not observed: it never spent any time there.

- **`recall_sidecar_inserts_total` / `recall_sidecar_insert_errors_total`** — entries handed to
  the cache through `POST /v1/cache/insert` and successfully written, and those whose write failed.
  The last unmetered thing on that door. Kept off the fill counters deliberately: a *fill* is an
  entry recall bought with an upstream call it made and can price, while a sidecar insert was
  bought by the caller, who never says what it cost — folding them together would put entries
  recall never paid for into a figure that claims to be what filling this cache cost. Failures get
  their own counter for the same reason the hit counters exist at all: a success-only counter
  cannot tell an idle door from a broken one, and the caller's `500` never reaches the operator
  watching `/metrics`.

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
