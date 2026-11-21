# recall — configuration reference

Every knob in one place. Precedence for `serve` is **flag > env (`RECALL_*`) >
`--config` file (TOML) > built-in default** — resolved per field, so you can mix
sources. Source of truth: the hand-rolled arg parsing + `resolve` closure in
[`crates/recall/src/main.rs`](../crates/recall/src/main.rs) and the `Config` struct
in [`crates/recall-proxy/src/config.rs`](../crates/recall-proxy/src/config.rs);
regenerate these tables when those change. A commented reference config ships as
[`recall.toml`](../recall.toml).

- [`recall serve`](#recall-serve) — the proxy (flags, env, TOML keys)
- [Env-only settings](#env-only-settings) (API keys, logging)
- [Hot reload](#hot-reload---watch-config)
- [Other subcommands](#other-subcommands) — `ask` / `bench` / `replay` / `status` / `export` / `ann-bench` / `calibrate`
- [Cargo features](#cargo-features)
- [Fixed internals](#fixed-internals-library-defaults)
- [State & data formats](#state--data-formats)

## `recall serve`

| flag | env | TOML key | type | default | what it does | when to change |
|---|---|---|---|---|---|---|
| `--listen` | `RECALL_LISTEN` | `listen` | addr | `127.0.0.1:8080` | Bind address. Loopback by default — the proxy has no auth of its own (see [OPERATIONS §security](./OPERATIONS.md#security-posture)). | Sidecar on another interface / container. |
| `--upstream` | `RECALL_UPSTREAM` | `upstream` | URL | `https://api.openai.com` | OpenAI-compatible upstream base (no trailing `/v1`). Misses on `/v1/chat/completions` forward here. | Point at your own gateway/vLLM. |
| `--anthropic-upstream` | `RECALL_ANTHROPIC_UPSTREAM` | — | URL | `https://api.anthropic.com` | Anthropic Messages upstream base. Misses on `/v1/messages` forward here. | Anthropic-compatible gateway. |
| — | `RECALL_ANTHROPIC_VERSION` | — | string | `2023-06-01` | `anthropic-version` header sent upstream when the caller omits one. A caller-supplied *different* version bypasses the cache. | Upstream requires a newer version. |
| `--base-namespace` | `RECALL_BASE_NAMESPACE` | `base_namespace` | string | `default` | Outermost cache-namespace component; the full key is `base:flavor:model:param-fingerprint`. | Layer tenancy / segregate environments. |
| `--max-temperature` | `RECALL_MAX_TEMPERATURE` | `max_temperature` | f64 | `1.0` | Requests with `temperature` strictly above this **bypass** the cache (lookup *and* store) — replaying a highly-sampled completion is a silent behavior change. | Lower for strict determinism. |
| `--tau` | `RECALL_TAU` | `tau` | f32 | `0.9` | Similarity cutoff. With `--policy static` the fixed threshold; with `adaptive` the cold-start prior τ₀. | Calibrate per embedder with `recall calibrate`. |
| `--policy` | `RECALL_POLICY` | `policy` | `static` \| `adaptive` | `static` | Threshold policy. `adaptive` learns a per-namespace cutoff toward `--target-fhr` from `/v1/cache/feedback`; absent feedback it rests at τ₀ (safe to enable). | Turning on the adaptive engine. |
| `--target-fhr` | `RECALL_TARGET_FHR` | `target_fhr` | f64 | `0.02` | False-hit rate the adaptive policy aims to hold (0.02 = ≤2% wrong hits among served hits). | Your correctness budget. |
| `--model` | `RECALL_MODEL2VEC_PATH` | `model` | dir path | *(none → hash embedder)* | Local model2vec/potion model directory (offline; no download). Requires a `--features static` build — without it the flag is **ignored with a note** and the deterministic hash embedder is used. | Any production deploy (real paraphrase hits). |
| `--index` | `RECALL_INDEX` | `index` | `brute` \| `hnsw` | `brute` | ANN index. `brute` = exact oracle (O(N) per lookup); `hnsw` = pure-Rust approximate, sublinear at scale. Unknown value → exit 2. | Corpus ≳10k entries per namespace (validate with `recall ann-bench` first). |
| `--hnsw-m` | `RECALL_HNSW_M` | `hnsw_m` | usize | `32` | HNSW graph degree (`m0 = 2·m` on layer 0). Only with `--index hnsw`. Higher = better recall, more memory and insert cost. | `ann-bench` shows recall@1 short of the gate for your dims/N. |
| `--hnsw-ef-construction` | `RECALL_HNSW_EF_CONSTRUCTION` | `hnsw_ef_construction` | usize | `400` | Candidate breadth while building the graph. Higher = better graph, slower insert (a miss-path cost). | Same. |
| `--hnsw-ef-search` | `RECALL_HNSW_EF_SEARCH` | `hnsw_ef_search` | usize | `256` | Candidate breadth while searching — THE recall-vs-latency dial at query time. | Raise first when recall@1 falls short (see BENCHMARKS.md: 0.998 at 10k, 0.876 at 50k on the adversarial workload); lower to shave hit latency on small corpora. |
| `--ambiguity-eps` | `RECALL_AMBIGUITY_EPS` | `ambiguity_eps` | f32 | `0` (off) | Second-neighbour ambiguity gate: a would-be hit whose top-2 neighbours are within this similarity of each other AND store *different* answers is served as a miss (same-answer near-duplicates replay regardless). Demotions counted by `recall_ambiguity_miss_total`. | Trimming false hits on lookalike-heavy corpora; start ~0.01–0.05 and watch that counter against `recall_hits_total`. |
| `--savings-flush-secs` | `RECALL_SAVINGS_FLUSH_SECS` | `savings_flush_secs` | u64 | `0` (off) | Persist the savings counters into the durable store every N seconds, so the `$ saved` figure survives a restart rather than resetting each deploy. Requires `--store redb`. A clean shutdown flushes on the way out, so a planned deploy loses nothing; an unclean exit loses at most N seconds of accounting. When on, the node reports `recall_ledger_since_time_seconds` (and `recall export` carries `node.ledger_since_unix`) so a consumer can tell the counters predate this process. | You quote the savings figure across deploys. |
| `--store` | `RECALL_STORE` | `store` | `memory` \| `redb` | `memory` | Store backend. `memory` is ephemeral; `redb` is durable/ACID and **rehydrates on restart** (rebuilds the in-memory index + exact-map from persisted entries). `redb` on a build without `--features store-redb`, an unknown value, or a redb open failure → **exit 2**, never a silent downgrade. | Cache must survive restarts. |
| `--db-path` | `RECALL_DB_PATH` | `db_path` | file path | `recall.redb` | Where redb persists (only with `--store redb`). One live instance per file — redb is single-writer. | Put it on the instance's durable volume. |
| `--watch-config` | — | — | bool flag | off | Hot-reload the **threshold policy only** (`policy`/`tau`/`target_fhr`) from `--config` on file change; requires `--config`. See [Hot reload](#hot-reload---watch-config). | Live τ tuning without dropping connections. |
| `--config` | — | — | file path | *(none)* | The TOML file (lowest precedence). Every key optional. | — |

### `[pricing.<model>]` — report savings in dollars

File-only (a rate table is not a scalar, so there is no flag or env form), optional,
and **not** hot-reloadable. Each entry gives your **USD per million tokens** — the unit
providers publish — for one model:

```toml
[pricing."gpt-4o"]
input = 2.50      # USD per 1M input (prompt) tokens
output = 10.00    # USD per 1M output (completion) tokens

[pricing."gpt-4o-mini*"]   # prefix rule: also covers gpt-4o-mini-2024-07-18
input = 0.15
output = 0.60
```

With at least one rule configured, `/metrics` exports
`recall_cost_saved_usd_total` next to the raw token counters.

| rule | behavior |
|---|---|
| Your numbers only | recall bundles no price list and makes no network call. An empty table (the default) is a supported state: the dollar figure stays `0` and `serve` says so at startup. |
| Never guessed | A model with no matching rule contributes `$0` and its saved tokens are counted in `recall_unpriced_tokens_saved_total` — so an incomplete or typo'd table reads as *unpriced*, not as *no savings*. |
| Exact by default | A bare key matches the model id exactly; a key ending in `*` is a prefix rule, so `"gpt-4o*"` covers dated snapshots. When several match, the **longest stem** wins. |
| Case-sensitive | Keys match the model id byte-for-byte as the caller sends it: `"GPT-4o"` does not match `gpt-4o`. A mismatched-case key silently prices nothing — its tokens land in `recall_unpriced_tokens_saved_total`, which is where to look when the dollar figure is unexpectedly `0`. |
| Input ≠ output | The two tiers are priced separately (output typically costs 3–5× input), so a skewed mix is not mis-stated. |
| Integer money | Rates convert to integer micro-USD at load and accumulate as integers, so millions of hits don't drift; the float appears only in the exposition (6 dp). |
| Non-fatal | A negative, NaN, infinite, or empty-key entry is warned about and skipped — a typo in an optional reporting table never stops the proxy from serving. |

To price a workload *before* deploying, `recall replay` takes flat
`--price-input` / `--price-output` rates for a single run — see
[below](#recall-replay--drive-a-running-serve-with-a-request-log).

## Env-only settings

Secrets are **never flags** (they would leak into shell history and `ps`) and have
no TOML key:

| env | what it does |
|---|---|
| `RECALL_UPSTREAM_API_KEY` | `Authorization: Bearer` sent to the OpenAI upstream **when the incoming request carries no `Authorization` header** (a caller-supplied header is forwarded as-is). |
| `RECALL_ANTHROPIC_API_KEY` | `x-api-key` for the Anthropic upstream, same fallback-only semantics. |
| `RUST_LOG` | tracing filter, default `info`. `RUST_LOG=recall=debug` logs per-request hit/miss/bypass (structural fields only — prompts/completions are never logged). |
| `RECALL_OTEL_ENDPOINT` | OTLP/HTTP base for `gen_ai.*` span export, e.g. `http://collector:4318` (`/v1/traces` is appended). **Unset by default, and unset means no exporter is built and nothing leaves the process** — there is no separate enable flag to leave switched on with nowhere to send. |
| `RECALL_OTEL_SERVICE_NAME` | `service.name` on exported spans. Default `recall`. Only read when an endpoint is set. |

### `gen_ai.*` spans — what is and is not in them

One span per proxied request, named `{operation} {model}` (e.g. `chat gpt-4o`) as the OpenTelemetry
GenAI semantic conventions prescribe, carrying `gen_ai.provider.name`, `gen_ai.operation.name`,
`gen_ai.request.model`, plus recall's own `recall.cache.verdict`, `recall.cache.score` and
`recall.cache.namespace`. The verdict is the same word `x-recall-cache` carries — `hit`, `miss`,
`miss-nostore`, `bypass` — so no mapping is needed between a span and the response a caller saw.
`gen_ai.provider.name` is `openai` or `anthropic`, by which front door the request arrived at.

**A raw sidecar lookup (`POST /v1/cache/lookup`) emits a `cache.lookup` span with no `gen_ai.*`
attributes whatsoever** — only `recall.cache.verdict`, `.namespace` and `.score`. That caller
supplies a namespace and a prompt and nothing more: no model, no vendor, no token count. Every
GenAI attribute would therefore be invented, and would file non-LLM traffic into an LLM cost
dashboard. The span type enforces it rather than the emit site: the GenAI half lives in one
optional field, so a sidecar span has nowhere to put a model.

**`gen_ai.response.model` is what the response named, and is absent when it named nothing.** Both
providers resolve aliases to dated snapshots (`claude-sonnet-4-5` → `claude-sonnet-4-5-20250929`),
and that resolution is the whole reason the attribute is separate from `gen_ai.request.model`.
Echoing the request back would make the two equal by construction, so a reader comparing them would
conclude no alias is ever resolved. A hit has no response model at all — nothing responded.

**A hit does not report `gen_ai.usage.*`, and that is deliberate.** No model call happened, so
reporting usage tokens would have every tool that sums those attributes — which is most of them —
counting tokens nobody spent. A cache would inflate the spend dashboard of the team that installed
it to reduce spend. The avoided tokens are reported instead under `recall.saved.input_tokens`,
`recall.saved.output_tokens` and `recall.saved.cost_micros`, where no standard aggregation mistakes
them for spend. On a miss or a bypass a model really did run, and `gen_ai.usage.*` is present —
while `recall.saved.*` is absent, because nothing was saved and an explicit zero would sum into a
savings figure as a real (if unimpressive) one.

**Usage is absent rather than zero when the upstream did not report it.** A response carrying no
usage object parses as `0/0`; exporting that would read as a call that cost nothing rather than one
whose cost is unknown. This is the same distinction `recall_hits_usage_unknown_total` draws on the
metrics side.

**A bypass carries no `recall.cache.namespace`.** It never reached the cache, so there is no key —
and an empty string would give a consumer something to group by that does not exist.

**A streamed span may contradict the header the caller received.** `x-recall-cache` is written
before the first byte of the body, so a streamed miss commits to `miss` before anyone can know
whether the stream will terminate cleanly. A truncated stream is dropped rather than cached
(`recall_stream_not_stored_total` counts it), and its span — emitted at the true end of the stream,
which is the first moment the outcome is known — reports `miss-nostore`. The same applies to a
buffered miss whose cache write fails. Where the two disagree, the span is the accurate one.

**Prompts and completions are never exported.** The conventions define optional events carrying
message content; recall does not implement them, and the span type has no field that could hold
either — not an `Option`, not a redacted one. Same rule the request log already follows.

The conventions are not stable yet. These names were checked against **v1.40.0**;
`gen_ai.provider.name` is replacing `gen_ai.system`, and during the overlap both are emitted, so a
consumer pinned to either finds what it is looking for.

Export never affects a request: spans go to a bounded in-memory queue (dropped and counted when it
is full rather than queued), and the flush is a background task whose send is fire-and-forget. A
collector that is slow, wedged or gone cannot reach back into the request path.

## Hot reload (`--watch-config`)

Only the threshold policy (`policy`, `tau`, `target_fhr`) is hot-reloadable; the
listener, upstreams, embedder, index, and store own sockets/weights/files and need a
restart (a changed non-policy field is logged as
`config field changed but is not hot-reloadable` and ignored). A field pinned by a
flag or env var at startup keeps its startup value on reload. A read/parse error
keeps the previous good policy — a typo can never take the cache down. Source:
`apply_reload`/`effective_policy` in `crates/recall/src/main.rs` (unit-tested).

## Other subcommands

All in-process except `replay` (HTTP client). Source: `cmd_*` in
[`crates/recall/src/main.rs`](../crates/recall/src/main.rs).

### `recall ask` — drive the loop by hand (fresh in-memory cache per invocation)

| flag | type | default | what it does |
|---|---|---|---|
| `--ns` | string | **required** | Namespace. |
| `"<prompt>"` | positional | **required** | HIT prints the cached answer; MISS reads a completion from stdin and stores it. Fixed: hash embedder, τ = 0.9. |

### `recall bench` — synthetic FAQ workload (hit-rate + latency split)

| flag | type | default | what it does |
|---|---|---|---|
| `--tau` | f32 | `0.85` | Static threshold for the bench cache. |
| `--iters` | usize > 0 | `2000` | Lookup iterations (alternating exact / paraphrase queries). |
| `--model` | dir | `RECALL_MODEL2VEC_PATH`, else hash | Run with the real static embedder (needs `--features static`) so the numbers are a defensible baseline. |

### `recall replay` — drive a running `serve` with a request log

Measures what a log of real requests *would* have saved — run it against a throwaway
proxy before putting recall in your request path. Walkthrough:
[GETTING_STARTED §4](./GETTING_STARTED.md#4-measure-it-on-your-own-traffic--before-you-commit).

| flag | type | default | what it does |
|---|---|---|---|
| `--file` | path | **required** | JSONL request log, one request body per line (or `{"path":…,"body":…}` to mix endpoints). |
| `--target` | URL | `http://127.0.0.1:8080` | The running proxy. |
| `--path` | string | `/v1/chat/completions` | Default request path for lines that don't carry one. |
| `--verify-sample` | f64 in [0,1] | `0.0` | Fraction of hits re-asked to the upstream and compared, to sample the false-hit rate. Needs `--upstream`, and only means anything at `temperature: 0`. When set, the savings are reported **both** gross and net of the measured rate. |
| `--upstream` | URL | `RECALL_UPSTREAM` | Upstream for verification calls (key from `RECALL_UPSTREAM_API_KEY` / `RECALL_ANTHROPIC_API_KEY`). |
| `--price-input` / `--price-output` | f64 $/Mtok | — | **Fallback** pricing for a target with no price book: one blended rate across every model in the log. Ignored when the target reports a per-model figure. |

The headline `$ saved` comes from the target's own
[`[pricing.<model>]`](#pricingmodel--report-savings-in-dollars) table, so a log spanning
several models is priced per model rather than at one blended rate. Every figure is a
**delta** across the run (the proxy's counters are snapshotted before and after), so
replaying twice against one long-lived proxy measures the second run rather than both
added together; the reported hit-rate is likewise the run's own, not the proxy's
lifetime gauge.

Exit is non-zero when **every** request errored (proxy unreachable) so CI sees red
instead of a misleading 0% hit-rate.

### `recall status` — one screen on a running `serve`

Summarizes a live proxy: what it has saved, and what that figure does and does not
account for. Reads `/metrics` over HTTP and computes nothing the proxy did not
already measure.

| flag | type | default | what it does |
|---|---|---|---|
| `--target` | URL | `RECALL_TARGET`, else `http://127.0.0.1:8080` | The running proxy to summarize. |
| `--json` | bool flag | off | Emit the raw snapshot as JSON instead of the formatted summary — for piping into your own tooling. |

Every caveat it prints is measured, never assumed:

- Savings are discounted by the **observed** false-hit rate when outcomes have been
  reported to [`/v1/cache/feedback`](./API.md), and explicitly labelled
  un-discounted when none have. No feedback is not a 0% error rate.
- Saved tokens on models with no price rule, and hits whose stored completion
  carried no usage at all, are called out as **under-counts** rather than folded
  away — a figure that is quietly too low costs trust as fast as one that is too high.
- A rate with no denominator abstains ("no cacheable traffic yet", "unmeasured")
  rather than printing `0.0%`.

Exit is non-zero when the proxy cannot be reached, so a monitoring wrapper sees the
difference between "down" and "idle".

### `recall export` — the node's facts as one versioned JSON document

Emits everything `status` shows — savings, fill cost, traffic, store facts, feedback — as a
`recall.export/v1` document for machine consumption: roll-ups, reporting, chargeback. The shape is
a documented compatibility surface, evolved additively only — see
[API.md](./API.md#export-format-recall-export--recallexportv1).

| flag | type | default | what it does |
|---|---|---|---|
| `--target` | URL | `RECALL_TARGET`, else `http://127.0.0.1:8080` | The running proxy to export from. |

Exit is non-zero when the proxy cannot be reached. Money is integer micro-USD; under-count flags
ride alongside every figure; `export_id` is content-derived so an ingest ledger can dedupe a
retried document.

### `recall ann-bench` — brute vs HNSW as N grows

| flag | type | default | what it does |
|---|---|---|---|
| `--dims` | usize | `256` | Vector dimensionality. |
| `--queries` | usize ≥ 100 | `500` | Queries per corpus size. |
| `--sizes` | comma list | `100,1000,10000,50000` | Corpus sizes. |
| `--top-k` | usize > 0 | `1` | HNSW recall@1 is measured against the brute-force top-1 within this k. |
| `--m` / `--ef-construction` / `--ef-search` | usize | the `serve` defaults (32 / 400 / 256) | HNSW tunables for the benched graph — sweep these to find the recall/latency point for **your** dims/N, then set the matching `--hnsw-*` flags on `serve`. Defaulting to the serve values means a plain run benchmarks the config you actually deploy. |

### `recall calibrate` — pick τ for your embedder from labeled pairs

| flag | type | default | what it does |
|---|---|---|---|
| `--file` | path | **required** | JSONL of `{"a": …, "b": …, "should_hit": true\|false}` pairs. |
| `--model` | dir | `RECALL_MODEL2VEC_PATH`, else hash | Embedder to calibrate under (must match the serving embedder). |
| `--target-fhr` | f64 in [0,1] | — | Also report the most permissive τ holding the false-hit rate ≤ this. |

## Cargo features

Default build = zero ML / network / C dependencies (the air-gap property); escalate
only the seam you need.

| feature | crate(s) | what it adds |
|---|---|---|
| `static` | `recall` (via `recall-embed`) | The model2vec/potion static embedder (`--model` becomes functional; local safetensors load, no network). |
| `store-redb` | `recall` (via `recall-store`) | The durable, pure-Rust redb `Store` (`--store redb` becomes functional). |

## Fixed internals (library defaults)

Not configurable from the CLI today; embedders of `recall-core` set them in code.

| knob | value | source |
|---|---|---|
| Upstream connect / request timeout | 10 s / 120 s | `Config::default` in `crates/recall-proxy/src/config.rs` (struct fields exist; the binary does not expose flags for them yet) |
| ANN `top_k` per lookup | 5 (`SemanticCache::with_top_k`) | `crates/recall-core/src/cache.rs` |
| `verify_on_hit` (require exact prompt match on a hit) | `false` (`with_verify_on_hit`) | same |
| Hash embedder | 256-dim deterministic (`hash-v1@256`) — a **test/CI stub**: paraphrases rarely clear τ, so real deployments want `--features static --model …` | `crates/recall-core/src/embed.rs` |
| Feedback score bounds | must be finite and in [-1, 1] (cosine scale), else 400 | `cache_feedback` in `crates/recall-proxy/src/lib.rs` |

## State & data formats

| artifact | format | compatibility notes |
|---|---|---|
| `--db-path` (e.g. `recall.redb`) | a single [redb](https://github.com/cberner/redb) file (pure-Rust ACID KV) holding the cache entries (prompt, completion, namespace key) | Single-writer: exactly one live `serve` per file. On startup the entries are **rehydrated** — each persisted prompt is re-embedded to rebuild the in-memory index — so the file must be read with the **same embedder** it was written with (a different `--model` re-keys similarity; entries remain readable but hit behavior changes). No `format_version` field yet: treat the file as owned by the recall version + embedder that created it. Source: `crates/recall-store/src/redb_store.rs`. |
| The memory store | none (process-lifetime only) | Restart = cold cache. |
| Wire formats | OpenAI `/v1/chat/completions`, Anthropic `/v1/messages`, and the raw sidecar JSON — see [API.md](./API.md) | Streamed entries are stored as raw SSE text and replayed chunk-by-chunk; JSON and SSE live in disjoint namespaces and never cross-replay. |
