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

### `GET /healthz` and `GET /metrics`

`/healthz` → `ok` (liveness for LB checks). `/metrics` → Prometheus text
(v0.0.4), hand-rolled over atomics. Source:
[`crates/recall-proxy/src/metrics.rs`](../crates/recall-proxy/src/metrics.rs).

| metric | type | meaning |
|---|---|---|
| `recall_requests_total` | counter | Proxy-flavor requests received (`/v1/chat/completions` + `/v1/messages`). |
| `recall_hits_total` / `recall_misses_total` / `recall_bypass_total` | counter | Disposition counts. |
| `recall_upstream_errors_total` | counter | Upstream forwarding failures (the 502s). |
| `recall_tokens_saved_total` | counter | Upstream tokens **not purchased** because a hit answered (from the stored body's `usage`). |
| `recall_input_tokens_saved_total` / `recall_output_tokens_saved_total` | counter | The saved tokens split by price tier (output costs 3–5× input; price them separately). |
| `recall_stream_not_stored_total` | counter | Streamed misses forwarded but not cached (dirty stream end — no `[DONE]`/`message_stop`, truncation, non-UTF-8). |
| `recall_cache_get_duration_seconds` | histogram | End-to-end lookup latency (exact-hash → embed → ANN → decide) for cacheable requests. |
| `recall_upstream_duration_seconds` | histogram | Upstream forward latency for buffered (non-streamed) calls. |
| `recall_hit_ratio` | gauge | `hits / (hits + misses)` — over cacheable traffic only (bypasses excluded). |

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
