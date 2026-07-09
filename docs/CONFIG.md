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
- [Other subcommands](#other-subcommands) — `ask` / `bench` / `replay` / `ann-bench` / `calibrate`
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
| `--store` | `RECALL_STORE` | `store` | `memory` \| `redb` | `memory` | Store backend. `memory` is ephemeral; `redb` is durable/ACID and **rehydrates on restart** (rebuilds the in-memory index + exact-map from persisted entries). `redb` on a build without `--features store-redb`, an unknown value, or a redb open failure → **exit 2**, never a silent downgrade. | Cache must survive restarts. |
| `--db-path` | `RECALL_DB_PATH` | `db_path` | file path | `recall.redb` | Where redb persists (only with `--store redb`). One live instance per file — redb is single-writer. | Put it on the instance's durable volume. |
| `--watch-config` | — | — | bool flag | off | Hot-reload the **threshold policy only** (`policy`/`tau`/`target_fhr`) from `--config` on file change; requires `--config`. See [Hot reload](#hot-reload---watch-config). | Live τ tuning without dropping connections. |
| `--config` | — | — | file path | *(none)* | The TOML file (lowest precedence). Every key optional. | — |

## Env-only settings

Secrets are **never flags** (they would leak into shell history and `ps`) and have
no TOML key:

| env | what it does |
|---|---|
| `RECALL_UPSTREAM_API_KEY` | `Authorization: Bearer` sent to the OpenAI upstream **when the incoming request carries no `Authorization` header** (a caller-supplied header is forwarded as-is). |
| `RECALL_ANTHROPIC_API_KEY` | `x-api-key` for the Anthropic upstream, same fallback-only semantics. |
| `RUST_LOG` | tracing filter, default `info`. `RUST_LOG=recall=debug` logs per-request hit/miss/bypass (structural fields only — prompts/completions are never logged). |

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

| flag | type | default | what it does |
|---|---|---|---|
| `--file` | path | **required** | JSONL request log, one request body per line. |
| `--target` | URL | `http://127.0.0.1:8080` | The running proxy. |
| `--path` | string | `/v1/chat/completions` | Default request path for lines that don't carry one. |
| `--verify-sample` | f64 in [0,1] | `0.0` | Fraction of hits re-asked to the upstream to sample the candidate false-hit rate. Needs `--upstream`. |
| `--upstream` | URL | `RECALL_UPSTREAM` | Upstream for verification calls (key from `RECALL_UPSTREAM_API_KEY` / `RECALL_ANTHROPIC_API_KEY`). |
| `--price-input` / `--price-output` | f64 $/Mtok | — | Price the saved tokens into a dollar estimate. |

Exit is non-zero when **every** request errored (proxy unreachable) so CI sees red
instead of a misleading 0% hit-rate.

### `recall ann-bench` — brute vs HNSW as N grows

| flag | type | default | what it does |
|---|---|---|---|
| `--dims` | usize | `256` | Vector dimensionality. |
| `--queries` | usize ≥ 100 | `500` | Queries per corpus size. |
| `--sizes` | comma list | `100,1000,10000,50000` | Corpus sizes. |
| `--top-k` | usize > 0 | `1` | HNSW recall@1 is measured against the brute-force top-1 within this k. |

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
