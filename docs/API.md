# recall — API reference

Two surfaces: the [HTTP proxy](#http-api-recall-serve) (`recall serve`) and the
[library](#library-api-recall-core). Source of truth for the routes: `app()` in
[`crates/recall-proxy/src/lib.rs`](../crates/recall-proxy/src/lib.rs); regenerate
when it changes.

## HTTP API (`recall serve`)

```text
POST /v1/chat/completions    OpenAI-compatible caching proxy
POST /v1/messages            Anthropic-compatible caching proxy
POST /v1/cache/lookup        raw sidecar: lookup
POST /v1/cache/insert        raw sidecar: insert
POST /v1/cache/feedback      train the adaptive threshold
GET  /healthz                liveness
GET  /metrics                Prometheus text exposition
```

**Auth:** the proxy itself has **none** (bind loopback / your network boundary —
see [OPERATIONS §security](./OPERATIONS.md#security-posture)). Upstream credentials
are forwarded per flavor: a caller-supplied `Authorization` (OpenAI) or `x-api-key`
(Anthropic) header is passed through as-is; when absent, the configured
`RECALL_UPSTREAM_API_KEY` / `RECALL_ANTHROPIC_API_KEY` is used.

### The proxy surface — `POST /v1/chat/completions` and `POST /v1/messages`

Drop-in: point your OpenAI client's base URL (or Anthropic client's) at recall and
change nothing else. The request body is the provider's own schema, passed through
verbatim on a forward.

**Cache keying.** Namespace = `base_namespace:flavor:model:param-fingerprint`,
where the 16-hex fingerprint hashes the answer-affecting decode params (OpenAI:
`temperature`, `top_p`, `max_tokens`/`max_completion_tokens`, penalties, `stop`,
`seed`, `response_format`, `logit_bias`, `stream`, `stream_options`; Anthropic:
its own set incl. `top_k`, `stop_sequences`, required `max_tokens`, `thinking`).
The prompt = the `messages` array serialized as JSON (Anthropic also folds in the
top-level `system` field). `stream` is inside the fingerprint, so SSE and JSON
entries live in **disjoint namespaces** and never cross-replay. Source:
[`crates/recall-proxy/src/request.rs`](../crates/recall-proxy/src/request.rs).

**Response headers.** Every response carries `x-recall-cache`:

| value | meaning |
|---|---|
| `hit` | Replayed from cache; **the upstream was never called**. Also carries `x-recall-namespace` (the server-derived key) and `x-recall-score` (the similarity) — feed both straight into `/v1/cache/feedback`. |
| `miss` | Forwarded upstream and (for a clean 2xx JSON body, or a cleanly-terminated 2xx SSE stream) stored for next time. A *streamed* reply always says `miss` because the header precedes the body; if the stream then ends dirty it is simply not stored (`recall_stream_not_stored_total` counts it). |
| `miss-nostore` | Forwarded upstream but **not** cached at header time: non-2xx status, or a body that is not `application/json`/does not parse. |
| `bypass` | The cache was skipped entirely (no lookup, no store) and the request forwarded verbatim. |

**Bypass rules** (from `bypass_reason` in `request.rs`): `tools`/`functions`
present (reassembling tool-call JSON from cache is a severe failure mode); `n > 1`
(OpenAI only — one cached reply can't serve multiple samples); `temperature` above
`max_temperature`; a non-default `anthropic-version` or any `anthropic-beta` header
(Anthropic only); a namespace that cannot be constructed. Streaming is **not** a
bypass — streamed completions are cached as raw SSE and replayed as a stream.

Real round-trip (mock upstream; second request identical to the first):

```text
$ curl -si -X POST http://127.0.0.1:8080/v1/chat/completions -H "content-type: application/json" \
    -d '{"model":"Qwen/Qwen2.5-7B-Instruct","temperature":0,"messages":[{"role":"user","content":"Summarize: recall is a semantic cache."}]}' \
    | grep -iE "^HTTP|x-recall"
HTTP/1.1 200 OK
x-recall-cache: miss

# …same request again:
HTTP/1.1 200 OK
x-recall-cache: hit
x-recall-namespace: default:openai:Qwen/Qwen2.5-7B-Instruct:b4c28133aa120472
x-recall-score: 1
```

**Errors:** `400 {"error": "invalid JSON body: …"}` (unparseable request), `502
{"error": …}` (upstream transport failure), `500 {"error": "cache error: …"}`
(cache lookup failure). Upstream HTTP errors (e.g. 401/429 from OpenAI) are passed
through with their own status and `x-recall-cache: miss-nostore`.

### `POST /v1/cache/lookup` — raw sidecar lookup

For non-OpenAI/Anthropic callers (or library-over-HTTP use): you own the namespace
and prompt strings.

```text
$ curl -s -X POST http://127.0.0.1:8080/v1/cache/lookup -H "content-type: application/json" \
    -d '{"namespace":"demo","prompt":"how do i reset my password"}'
{"hit":true,"score":1.0,"completion":"Use the Forgot password link."}
# miss → {"hit":false,"score":null,"completion":null}
```

Errors: `400` invalid namespace, `500` cache error.

### `POST /v1/cache/insert` — raw sidecar insert

```text
$ curl -s -X POST http://127.0.0.1:8080/v1/cache/insert -H "content-type: application/json" \
    -d '{"namespace":"demo","prompt":"how do i reset my password","completion":"Use the Forgot password link."}'
{"stored":true,"key":"6b6eae0c5ffea81308fdac9f7a8282c289c3c9430bbc5338368442c2b41c9df8"}
```

Errors: `400` invalid namespace, `500` store/embed error.

### `POST /v1/cache/feedback` — train the adaptive threshold

Report a served hit's outcome: `"agree"` (correct) or `"wrong"` (a false hit —
raises that namespace's cutoff). `score` must be a finite cosine similarity in
[-1, 1] (the `x-recall-score` header / lookup `score` of the hit being judged).

```text
$ curl -s -X POST http://127.0.0.1:8080/v1/cache/feedback -H "content-type: application/json" \
    -d '{"namespace":"demo","score":0.97,"outcome":"agree"}'
{"accepted":true,"policy":"static@0.900"}
```

The `policy` field shows whether feedback actually trains: a `static@…` policy
accepts feedback but ignores it; only `adaptive@…` learns. Errors: `400` invalid
namespace / outcome / out-of-range score.

### `GET /v1/cache/namespaces` — savings, split by namespace

The node counters say whether the cache is worth running; these rows say *for
whom* — the question the moment more than one team or model shares a proxy.
Rows are a **split of the same hits** the node counters already counted, never
an addition, so they sum to at most the node totals.

```text
$ curl -s http://127.0.0.1:8080/v1/cache/namespaces
{
  "schema": "recall.namespaces/v1",
  "process_start_unix": 1787901084,
  "truncated": false,
  "namespaces": [
    {"namespace": "default:openai:gpt-4o:b4c28133aa120472", "hits": 5290,
     "input_tokens": 1861400, "output_tokens": 399872,
     "cost_micros": 8652220, "unpriced_tokens": 0}
  ]
}
```

Money is integer micro-USD, priced by the same `[pricing.<model>]` table as
everything else; a row whose model has no rule carries `cost_micros: 0` and a
non-zero `unpriced_tokens` rather than a guessed figure.

**Why this is not a metric family.** The namespace is derived from the request,
so per-namespace series would be caller-controlled Prometheus cardinality —
the same reason `/metrics` carries no per-model series. The breakdown is a
bounded JSON document instead: at most 512 namespaces are tracked and the rest
fold into one `(other namespaces)` row, with `truncated` saying when that
happened. `recall status` prints the top rows, and `recall export` carries them
as `by_namespace`.

### `GET /healthz` and `GET /metrics`

`/healthz` → `ok` (liveness for LB checks). `/metrics` → Prometheus text
(v0.0.4), hand-rolled over atomics. Source:
[`crates/recall-proxy/src/metrics.rs`](../crates/recall-proxy/src/metrics.rs).

| metric | type | meaning |
|---|---|---|
| `recall_requests_total` | counter | Proxy-flavor requests received (`/v1/chat/completions` + `/v1/messages`). |
| `recall_hits_total` / `recall_misses_total` / `recall_bypass_total` | counter | Disposition counts. A **miss** is counted at lookup time for every cacheable request the cache could not answer — including ones whose upstream forward then failed or returned a non-cacheable body — so `hits / (hits + misses)` is a rate over cacheable traffic, not over stored entries. |
| `recall_upstream_errors_total` | counter | Upstream forwarding failures (the 502s). |
| `recall_tokens_saved_total` | counter | Upstream tokens **not purchased** because a hit answered (from the stored body's `usage`). |
| `recall_input_tokens_saved_total` / `recall_output_tokens_saved_total` | counter | The saved tokens split by price tier (output costs 3–5× input; price them separately). |
| `recall_cost_saved_usd_total` | counter | Upstream spend avoided, priced with **your** `[pricing.<model>]` rates ([CONFIG](./CONFIG.md#pricingmodel--report-savings-in-dollars)). `0` when no rates are configured — recall ships no price list and makes no network call. |
| `recall_unpriced_tokens_saved_total` | counter | Tokens saved for models with **no** configured rule. They contribute `$0` above and are flagged here instead, so an incomplete price table reads as *unpriced* rather than as *no savings*. |
| `recall_hits_usage_unknown_total` | counter | Hits whose stored completion reported no token usage at all (e.g. an OpenAI stream cached without `stream_options.include_usage`). They credit nothing to the counters above, so a non-zero value means the savings are an **under-count**, not an idle cache. |
| `recall_feedback_agree_total` / `recall_feedback_wrong_total` | counter | Served hits reported correct / wrong via `/v1/cache/feedback`. `wrong / (agree + wrong)` is the **observed** false-hit rate. Counted at the proxy, so they record the judgement even under `--policy static`, which accepts feedback without learning from it. |
| `recall_cache_entries` | gauge | Completions the store currently holds, read live at scrape time (cheap on both backends — a map length or a B-tree metadata read). Correct across a restart that rehydrated a durable store. |
| `recall_ambiguity_miss_total` | counter | Would-be hits demoted to misses by the second-neighbour ambiguity gate (`--ambiguity-eps`). `0` while the gate is disarmed, which is the default. Read against `recall_hits_total` to see what the gate costs in hit-rate for the false hits it prevents. |
| `recall_ledger_since_time_seconds` | gauge | Unix time the node's **persisted** savings ledger began (`--savings-flush-secs`). `0` — the default — means the counters reset with this process, i.e. `recall_process_start_time_seconds` is the boundary. Non-zero means they predate it. |
| `recall_fill_entries_total` | counter | Entries this process bought from the upstream and stored. Credited when the write **succeeds** — a miss that was not stored bought nothing for the cache. |
| `recall_fill_tokens_total` | counter | Upstream tokens purchased on responses this process then stored, with `recall_fill_input_tokens_total` / `recall_fill_output_tokens_total` beside it. The mirror of the saved counters, in the other direction. |
| `recall_fill_cost_usd_total` | counter | What filling this cache cost, priced from the same `[pricing.<model>]` table as the savings — so `saved / fill` is this instance's payback. |
| `recall_unpriced_fill_tokens_total` / `recall_fill_usage_unknown_total` | counter | The two ways the fill cost can under-count: a model with no price rule, and a response that reported no usage. |
| `recall_persisted_entries_at_start` / `recall_rehydrated_entries` | gauge | Rows the store held at process start, and how many were rebuilt into the live index — "recovered N of M". Both 0 on an ephemeral backend, which has nothing to rebuild. |
| `recall_process_start_time_seconds` | gauge | Unix time this process started: the boundary its counters reset at, so `increase()` across a restart stays interpretable. |
| `recall_store_info{backend,durable}` | gauge | Constant 1; the labels carry the store backend and whether it survives a restart. Disambiguates `recall_rehydrated_entries 0` between "nothing to rebuild" and "recovered nothing". |
| `recall_stream_not_stored_total` | counter | Streamed misses forwarded but not cached (dirty stream end — no `[DONE]`/`message_stop`, truncation, non-UTF-8). |
| `recall_cache_get_duration_seconds` | histogram | End-to-end lookup latency (exact-hash → embed → ANN → decide) for cacheable requests. |
| `recall_upstream_duration_seconds` | histogram | Upstream forward latency for buffered (non-streamed) calls. |
| `recall_hit_ratio` | gauge | `hits / (hits + misses)` — over cacheable traffic only (bypasses excluded). |

## Export format (`recall export` — `recall.export/v1`)

One node's cumulative savings and usage as a single JSON document — the stable shape to build
roll-up, reporting, or chargeback tooling on. Where `/metrics` is for scrapers and `recall status`
is for humans, this is for **your** machines: pipe it into whatever aggregates your fleet.

```bash
recall export --target http://127.0.0.1:8080 > node-a.json
```

```json
{
  "schema": "recall.export/v1",
  "export_id": "7b796189124e1f88d6f883d25f845da4",
  "exported_at_unix_ms": 1787098498447,
  "target": "http://127.0.0.1:8080",
  "node":     { "process_start_unix": 1787098498, "store_backend": "memory", "store_durable": false,
                "cache_entries": 4, "persisted_entries_at_start": 0, "rehydrated_entries": 0 },
  "traffic":  { "requests": 200, "hits": 196, "misses": 4, "bypass": 0,
                "upstream_errors": 0, "stream_not_stored": 0 },
  "saved":    { "tokens": 5684, "input_tokens": 2352, "output_tokens": 3332,
                "cost_micros": 39200, "unpriced_tokens": 0, "hits_usage_unknown": 0 },
  "fill":     { "entries": 4, "tokens": 116, "input_tokens": 48, "output_tokens": 68,
                "cost_micros": 800, "unpriced_tokens": 0, "usage_unknown": 0 },
  "feedback": { "agree": 0, "wrong": 0 },
  "by_namespace": [
    { "namespace": "default:openai:gpt-4o:b4c28133aa120472", "hits": 196,
      "input_tokens": 2352, "output_tokens": 3332,
      "cost_micros": 39200, "unpriced_tokens": 0 }
  ]
}
```

`by_namespace` is `saved` split by cache namespace (see
[`GET /v1/cache/namespaces`](#get-v1cachenamespaces--savings-split-by-namespace)) — added under
the existing schema, so a consumer written before it simply ignores it. It is empty when the
exporter could not read that document (an older node, or one behind a router exposing only
`/metrics`), which is why the node-level `saved` stays authoritative and these rows are strictly
a breakdown of it.

**Contract rules** (source: `crates/recall-proxy/src/export.rs`):

- **Versioned, additive-only.** `schema` is always the first key, so a consumer can dispatch on a
  prefix read. New fields may appear within `recall.export/v1` and must be ignored if unknown; a
  shipped field never changes name, type, or meaning. A breaking change means a new schema string.
- **Integer money.** Every monetary field is integer **micro-USD** (`*_micros`) — the exact value
  recall accounts internally, no float in the document. Divide by 1e6 at your edge.
- **Under-counts are flagged, never folded away.** `unpriced_tokens` (models with no
  `[pricing.<model>]` rule) and `*usage_unknown` (responses that reported no usage) ride alongside
  every dollar figure; a `cost_micros` of `0` next to non-zero flags means "unpriced", not
  "nothing saved". `feedback` ships raw counts so `0/0` (unmeasured) stays distinct from a
  measured 0% false-hit rate.
- **Everything is cumulative since `node.process_start_unix`.** A consumer differencing successive
  exports uses it to tell a counter reset (restart) from a quiet period.
- **`export_id` is content-derived** (time + target + process identity + counters): an exact
  retransmission of the same document carries the same id, so an ingest ledger keyed on it can
  accept a fire-and-forget retry without double-counting, while any counter movement or later
  timestamp yields a fresh id.

## Library API (`recall-core`)

The cache is embeddable without the proxy — no extra process, no network hop. The
public surface is `SemanticCache` plus the four seams (traits) it is generic over.
Exports: `crates/recall-core/src/lib.rs`.

### `SemanticCache` — the loop

```rust
use recall_core::{SemanticCache, HashEmbedder, BruteForceIndex, MemKv, StaticThreshold,
                  Namespace, Lookup};

let cache = SemanticCache::new(
    HashEmbedder::default(),    // Embedder  (swap: recall_embed::Model2VecEmbedder, feature `static`)
    BruteForceIndex::new(),     // AnnIndex  (swap: recall_index::HnswIndex at scale)
    MemKv::new(),               // Store     (swap: recall_store::RedbStore, feature `redb`)
    StaticThreshold::new(0.9),  // ThresholdPolicy (swap: recall_calibrate::AdaptiveThreshold)
);
let ns = Namespace::new("tenant-a/chat")?;

match cache.get(&ns, "How do I reset my password?")? {
    Lookup::Hit { score, entry, .. } => println!("hit {score}: {}", entry.completion),
    Lookup::Miss { vector } => {
        // call the real LLM, then store — reusing the query vector (no second embed):
        cache.put(&ns, "How do I reset my password?", "Click 'Forgot password'.", &vector)?;
    }
}
# Ok::<(), recall_core::RecallError>(())
```

| entry point | what it does |
|---|---|
| `SemanticCache::new(embedder, index, store, policy)` | Build a cache over the four seams. Builders: `.with_top_k(k)` (default 5), `.with_verify_on_hit(bool)` (default false; require exact prompt equality on a hit). |
| `cache.get(&ns, prompt) -> Lookup` | Exact-hash shortcut → embed → ANN search → threshold decide. `Hit { score, entry, .. }` or `Miss { vector }`. |
| `cache.put(&ns, prompt, completion, &vector)` | Store a completion using the vector `get` returned (store → index, with rollback on index failure). |
| `cache.put_embedding(&ns, prompt, completion) -> Key` | Store with a fresh embed (when you don't have the miss vector). |
| `cache.rehydrate() -> usize` | Rebuild the in-memory index + exact-map from a durable store's persisted entries (what `serve --store redb` does at startup). |
| `cache.observe(&ns, score, outcome)` | Feed a served hit's `Outcome::{Agree, Wrong}` to the threshold policy (no-op for `StaticThreshold`). |
| `cache.entries()` / `cache.policy_id()` | Entry count / active policy id (`static@0.900`, `adaptive@…`). |

### The four seams (traits)

Implement any of them to swap a backend; `DynCache` (via
`recall_proxy::boxed_cache_full[_with_index]`) is the boxed form the binary uses
when the choice is made at runtime.

| trait | contract | shipped impls |
|---|---|---|
| `Embedder` | `embed_one(&str) -> Vector`, `id() -> ModelId` | `HashEmbedder` (deterministic 256-dim stub, `hash-v1@256`), `recall_embed::Model2VecEmbedder::load_local(path, id)` (feature `static`) |
| `AnnIndex` | `insert(ns, key, vec)`, `search(ns, vec, k) -> Vec<Scored>` over unit vectors (dot = cosine) | `BruteForceIndex` (exact oracle), `recall_index::HnswIndex` (pure-Rust HNSW; sublinear — validate recall@1 for your dims/N first with `recall ann-bench`) |
| `Store` | KV of `Key → Entry { prompt, completion, … }` + scan (for rehydrate) | `MemKv` (ephemeral), `recall_store::RedbStore::open(path)` (durable ACID, feature `redb`) |
| `ThresholdPolicy` | `decide(ns, score) -> Verdict`, `observe(ns, score, outcome)` | `StaticThreshold::new(tau)`, `recall_calibrate::AdaptiveThreshold::new(AdaptiveConfig::new(tau0, target_fhr))` |

Test doubles (`AlwaysHit`, `FailingIndex`) ship behind the `test-support` feature.
