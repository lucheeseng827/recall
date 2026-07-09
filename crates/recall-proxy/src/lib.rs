//! recall's OpenAI- and Anthropic-compatible drop-in proxy plus a raw cache sidecar. Caches both
//! non-streaming (JSON) and streaming (SSE) responses.
//!
//! Point an OpenAI client's base URL at this proxy for `POST /v1/chat/completions`, or an Anthropic
//! client's for `POST /v1/messages`. Either way the proxy derives a cache key from the request
//! (model + decode-param fingerprint as the namespace, the full conversation as the prompt — see
//! [`request`]; the two flavors live in disjoint namespace partitions, and `stream` is in the
//! fingerprint so SSE and JSON never cross-replay), looks it up, and:
//! - **hit** → replays the stored upstream response (`x-recall-cache: hit`), no upstream call; a hit
//!   also carries `x-recall-namespace` + `x-recall-score` so a caller can train the adaptive policy
//!   via `/v1/cache/feedback`;
//! - **miss** → forwards to the matching upstream and stores the response — a clean 2xx JSON body, or
//!   a 2xx SSE stream tee'd to the client as it arrives while the cache buffer fills
//!   (`x-recall-cache: miss`);
//! - **bypass** → tool-call / `n>1` (OpenAI) / high-temperature requests skip the cache entirely and
//!   forward verbatim (`x-recall-cache: bypass`).
//!
//! Auth is forwarded per flavor — `Authorization: Bearer` for OpenAI, `x-api-key` +
//! `anthropic-version` for Anthropic — with a configured key as the fallback. The cache itself never
//! calls the LLM; this proxy is the network shell around it.
//!
//! Observability: per-request hit/miss/bypass events are emitted via `tracing` (enable with
//! `RUST_LOG=recall=debug`; upstream failures log at `warn`), and latency + token-savings via the
//! hand-rolled Prometheus `/metrics`. **Prompt and completion bodies are never logged** — only
//! structural fields (namespace, score, status, latency) — so customer content cannot leak into
//! server logs (see SECURITY.md).

pub mod config;
pub mod metrics;
pub mod reload;
pub mod replay;
pub mod request;

use std::sync::Arc;

use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::Response;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::json;

use recall_core::{
    AnnIndex, BruteForceIndex, DynCache, Embedder, HashEmbedder, Key, Lookup, MemKv, Namespace,
    Outcome, RecallError, SemanticCache, StaticThreshold, Store, ThresholdPolicy,
};

pub use config::Config;
use metrics::Metrics;
pub use reload::{PolicyHandle, SwappablePolicy};
use request::{
    anthropic_stream_complete, anthropic_usage, openai_stream_complete, openai_usage, ChatRequest,
    MessagesRequest, TokenUsage,
};

/// Shared handler state. Held behind an `Arc`; all four cache seams are `Send + Sync + 'static`, so
/// the cache work can run on a blocking thread without lifetime gymnastics.
pub struct ProxyState {
    pub cache: Arc<DynCache>,
    pub http: reqwest::Client,
    pub config: Config,
    pub metrics: Metrics,
}

impl ProxyState {
    pub fn new(cache: Arc<DynCache>, config: Config) -> Arc<Self> {
        // Bound upstream calls: without timeouts a slow or wedged upstream would pin a proxy worker
        // (and the caller) indefinitely. Both knobs are configurable; fall back to a default client
        // only if the builder somehow rejects them.
        let http = reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(config.connect_timeout_secs))
            .timeout(std::time::Duration::from_secs(config.request_timeout_secs))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        Arc::new(Self {
            cache,
            http,
            config,
            metrics: Metrics::default(),
        })
    }
}

/// Build a runtime-boxed cache from all three runtime-chosen seams — embedder (hash vs static),
/// `Store` (in-memory vs durable redb), and `ThresholdPolicy` (static vs adaptive). The index is
/// still the in-memory brute-force oracle; it becomes a durable/HNSW backend later without touching
/// this proxy.
pub fn boxed_cache_full(
    embedder: Box<dyn Embedder>,
    store: Box<dyn Store>,
    policy: Box<dyn ThresholdPolicy>,
) -> DynCache {
    boxed_cache_full_with_index(
        embedder,
        Box::new(BruteForceIndex::new()) as Box<dyn AnnIndex>,
        store,
        policy,
    )
}

/// Like [`boxed_cache_full`] but with a caller-chosen ANN index (e.g. the at-scale HNSW from
/// `recall-index`). The proxy depends only on the `AnnIndex` seam, so the binary picks the concrete
/// index and passes it in — the proxy never needs to know `recall-index` exists.
pub fn boxed_cache_full_with_index(
    embedder: Box<dyn Embedder>,
    index: Box<dyn AnnIndex>,
    store: Box<dyn Store>,
    policy: Box<dyn ThresholdPolicy>,
) -> DynCache {
    SemanticCache::new(embedder, index, store, policy)
}

/// Build a cache around a chosen embedder and `Store` with a static threshold τ.
pub fn boxed_cache_with_store(
    embedder: Box<dyn Embedder>,
    store: Box<dyn Store>,
    tau: f32,
) -> DynCache {
    boxed_cache_full(
        embedder,
        store,
        Box::new(StaticThreshold::new(tau)) as Box<dyn ThresholdPolicy>,
    )
}

/// Build a cache around a chosen embedder with the in-memory `MemKv` store and a static threshold —
/// the M1 default and what the tests use.
pub fn boxed_cache_with(embedder: Box<dyn Embedder>, tau: f32) -> DynCache {
    boxed_cache_with_store(embedder, Box::new(MemKv::new()) as Box<dyn Store>, tau)
}

/// Convenience: an in-memory cache with the deterministic `HashEmbedder` — the M1 default and what
/// the tests use.
pub fn boxed_memory_cache(tau: f32) -> DynCache {
    boxed_cache_with(Box::new(HashEmbedder::default()) as Box<dyn Embedder>, tau)
}

/// Bind `state.config.listen` and serve the proxy until the process exits. The binary calls this
/// inside its own Tokio runtime.
pub async fn serve(state: Arc<ProxyState>) -> std::io::Result<()> {
    let listen = state.config.listen.clone();
    let listener = tokio::net::TcpListener::bind(&listen).await?;
    axum::serve(listener, app(state)).await
}

/// The router. Mount on any address with `axum::serve`.
pub fn app(state: Arc<ProxyState>) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/messages", post(messages))
        .route("/v1/cache/lookup", post(cache_lookup))
        .route("/v1/cache/insert", post(cache_insert))
        .route("/v1/cache/feedback", post(cache_feedback))
        .route("/healthz", get(healthz))
        .route("/metrics", get(render_metrics))
        .with_state(state)
}

// ---------------------------------------------------------------------------------------------
// OpenAI- and Anthropic-compatible endpoints
// ---------------------------------------------------------------------------------------------

/// OpenAI drop-in: `POST /v1/chat/completions`.
async fn chat_completions(
    State(state): State<Arc<ProxyState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    Metrics::inc(&state.metrics.requests);
    let parsed = match ChatRequest::parse(&body) {
        Ok(p) => p,
        Err(e) => return json_error(StatusCode::BAD_REQUEST, &format!("invalid JSON body: {e}")),
    };
    let mapped = Mapped {
        bypass: parsed.bypass_reason(state.config.max_temperature),
        namespace: Namespace::new(parsed.namespace_string(&state.config.base_namespace)).ok(),
        prompt: parsed.canonical_prompt(),
        stream: parsed.is_stream(),
    };
    let upstream = Upstream {
        base: &state.config.upstream_base,
        path: "/v1/chat/completions",
        auth: Auth::Bearer(state.config.upstream_api_key.as_deref()),
        usage_of: openai_usage,
        stream_complete: openai_stream_complete,
    };
    serve_cached(&state, &headers, &body, &upstream, mapped).await
}

/// Anthropic drop-in: `POST /v1/messages`.
async fn messages(
    State(state): State<Arc<ProxyState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    Metrics::inc(&state.metrics.requests);
    let parsed = match MessagesRequest::parse(&body) {
        Ok(p) => p,
        Err(e) => return json_error(StatusCode::BAD_REQUEST, &format!("invalid JSON body: {e}")),
    };
    // `anthropic-beta` and a non-default `anthropic-version` change upstream behavior but are not
    // part of the request body the cache key is derived from — so two identical bodies under
    // different variant headers must not share a cached answer. Conservatively bypass when such
    // headers are present (folding them into the key is a later refinement).
    let variant_headers = headers.contains_key("anthropic-beta")
        || headers
            .get("anthropic-version")
            .and_then(|v| v.to_str().ok())
            .is_some_and(|v| v != state.config.anthropic_version);
    let mapped = Mapped {
        bypass: parsed
            .bypass_reason(state.config.max_temperature)
            .or_else(|| variant_headers.then_some("anthropic-headers")),
        namespace: Namespace::new(parsed.namespace_string(&state.config.base_namespace)).ok(),
        prompt: parsed.canonical_prompt(),
        stream: parsed.is_stream(),
    };
    let upstream = Upstream {
        base: &state.config.anthropic_upstream_base,
        path: "/v1/messages",
        auth: Auth::Anthropic {
            key: state.config.anthropic_api_key.as_deref(),
            version: &state.config.anthropic_version,
        },
        usage_of: anthropic_usage,
        stream_complete: anthropic_stream_complete,
    };
    serve_cached(&state, &headers, &body, &upstream, mapped).await
}

/// A request mapped onto the cache: why to bypass (if at all), the namespace (`None` if it could not
/// be constructed — treated as a bypass), the canonical prompt, and whether the caller asked for a
/// streamed (SSE) response — which decides whether a hit/miss is served as `text/event-stream` or
/// `application/json`. `stream` is also folded into the namespace, so the two formats never collide.
struct Mapped {
    bypass: Option<&'static str>,
    namespace: Option<Namespace>,
    prompt: String,
    stream: bool,
}

/// THE shared cache flow, identical across API flavors: bypass → forward verbatim; otherwise
/// hit → replay the stored body; miss → forward, store a clean 2xx JSON response, return it. The
/// per-flavor differences (upstream URL, auth headers, token accounting) all live in `upstream`.
async fn serve_cached(
    state: &Arc<ProxyState>,
    headers: &HeaderMap,
    body: &Bytes,
    upstream: &Upstream<'_>,
    mapped: Mapped,
) -> Response {
    // Bypass: forward verbatim, never touch the cache.
    if let Some(reason) = mapped.bypass {
        Metrics::inc(&state.metrics.bypass);
        tracing::debug!(reason, "bypass");
        return forward(state, headers, body, upstream, "bypass").await;
    }
    // A non-constructible namespace is a safety stop, not an error: forward without caching.
    let Some(ns) = mapped.namespace else {
        Metrics::inc(&state.metrics.bypass);
        tracing::debug!(reason = "namespace", "bypass");
        return forward(state, headers, body, upstream, "bypass").await;
    };
    let prompt = mapped.prompt;
    let stream = mapped.stream;

    // Time the end-to-end lookup (exact → embed → ANN → decide) — the p50/p99 latency histogram.
    let lookup_start = std::time::Instant::now();
    let lookup = cache_get(state, ns.clone(), prompt.clone()).await;
    state
        .metrics
        .cache_get_seconds
        .observe(lookup_start.elapsed().as_secs_f64());

    match lookup {
        Ok(Lookup::Hit { score, entry, .. }) => {
            Metrics::inc(&state.metrics.hits);
            // Namespace + score only — never the prompt or completion (see the module redaction note).
            tracing::debug!(namespace = ns.as_str(), score, stream, "cache hit");
            // `usage_of` understands both a JSON body and a stored SSE stream (via sse_data_payloads),
            // so a streamed hit counts its saved tokens too.
            let usage = (upstream.usage_of)(&entry.completion);
            Metrics::add(&state.metrics.tokens_saved, usage.total());
            Metrics::add(&state.metrics.input_tokens_saved, usage.input);
            Metrics::add(&state.metrics.output_tokens_saved, usage.output);
            // The body was stored under a stream-specific namespace, so a streaming hit always holds
            // SSE (replayed chunk-by-chunk) and a non-streaming hit always holds JSON.
            let mut resp = if stream {
                Response::builder()
                    .status(StatusCode::OK)
                    .header("content-type", "text/event-stream")
                    .header("x-recall-cache", "hit")
                    .body(sse_body_from(entry.completion))
                    .expect("valid streaming response")
            } else {
                text_response(
                    StatusCode::OK,
                    "application/json",
                    Some("hit"),
                    entry.completion,
                )
            };
            // Surface what an OpenAI/Anthropic caller needs to train the adaptive policy via
            // `POST /v1/cache/feedback` without reconstructing the server-derived key: the namespace
            // this hit was served under and its similarity score. A caller only ever sees headers for
            // its own request, and the namespace is structural (not a secret) — recall does not
            // provide authenticated isolation.
            let h = resp.headers_mut();
            if let Ok(v) = HeaderValue::from_str(ns.as_str()) {
                h.insert("x-recall-namespace", v);
            }
            if let Ok(v) = HeaderValue::from_str(&score.to_string()) {
                h.insert("x-recall-score", v);
            }
            resp
        }
        Ok(Lookup::Miss { vector }) => {
            // A streamed miss is tee'd to the client as it arrives (and cached on a clean end); only
            // the non-streamed JSON path buffers the full upstream body before returning it.
            if stream {
                return forward_stream_and_cache(
                    state, headers, body, upstream, ns, prompt, vector,
                )
                .await;
            }
            match forward_collect(state, headers, body, upstream).await {
                Err(msg) => {
                    Metrics::inc(&state.metrics.upstream_errors);
                    tracing::warn!(error = %msg, "upstream forward failed");
                    json_error(StatusCode::BAD_GATEWAY, &msg)
                }
                Ok((status, ct, text)) => {
                    // Cache only a 2xx body the upstream labeled JSON *and* that actually parses
                    // (PLAN.md §3-OSS) — a brace-containing HTML/error body must never be stored and
                    // later replayed as `application/json`. Store-then-return so a concurrent
                    // identical request resolves the freshly stored entry.
                    let cacheable = status.is_success()
                        && ct.starts_with("application/json")
                        && serde_json::from_str::<serde_json::Value>(&text).is_ok();
                    if cacheable {
                        Metrics::inc(&state.metrics.misses);
                        // Log "stored" only after the write actually succeeds; a failed put must not
                        // claim the entry is cached (the next identical request would miss again).
                        let ns_label = ns.as_str().to_string();
                        match cache_put(state, ns, prompt, text.clone(), vector).await {
                            Ok(()) => tracing::debug!(namespace = %ns_label, "cache miss (stored)"),
                            Err(e) => tracing::warn!(
                                namespace = %ns_label,
                                error = %e,
                                "cache miss store failed"
                            ),
                        }
                        text_response(status, "application/json", Some("miss"), text)
                    } else {
                        // Non-2xx / wrong-format upstream → pass through, do not cache.
                        tracing::debug!(status = status.as_u16(), "cache miss (not stored)");
                        text_response(status, json_ct(&ct), Some("miss-nostore"), text)
                    }
                }
            }
        }
        Err(e) => {
            tracing::error!(error = %e, "cache lookup failed");
            json_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                &format!("cache error: {e}"),
            )
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Raw cache sidecar (for non-OpenAI callers / library-over-HTTP)
// ---------------------------------------------------------------------------------------------

#[derive(Deserialize)]
struct LookupReq {
    namespace: String,
    prompt: String,
}
#[derive(Serialize)]
struct LookupResp {
    hit: bool,
    score: Option<f32>,
    completion: Option<String>,
}

async fn cache_lookup(
    State(state): State<Arc<ProxyState>>,
    Json(req): Json<LookupReq>,
) -> Response {
    let ns = match Namespace::new(req.namespace) {
        Ok(ns) => ns,
        Err(_) => return json_error(StatusCode::BAD_REQUEST, "invalid namespace"),
    };
    match cache_get(&state, ns, req.prompt).await {
        Ok(Lookup::Hit { score, entry, .. }) => json_ok(&LookupResp {
            hit: true,
            score: Some(score),
            completion: Some(entry.completion),
        }),
        Ok(Lookup::Miss { .. }) => json_ok(&LookupResp {
            hit: false,
            score: None,
            completion: None,
        }),
        Err(e) => json_error(StatusCode::INTERNAL_SERVER_ERROR, &format!("{e}")),
    }
}

#[derive(Deserialize)]
struct InsertReq {
    namespace: String,
    prompt: String,
    completion: String,
}
#[derive(Serialize)]
struct InsertResp {
    stored: bool,
    key: String,
}

async fn cache_insert(
    State(state): State<Arc<ProxyState>>,
    Json(req): Json<InsertReq>,
) -> Response {
    let ns = match Namespace::new(req.namespace) {
        Ok(ns) => ns,
        Err(_) => return json_error(StatusCode::BAD_REQUEST, "invalid namespace"),
    };
    let cache = state.cache.clone();
    let res =
        tokio::task::spawn_blocking(move || cache.put_embedding(&ns, &req.prompt, &req.completion))
            .await;
    match res {
        Ok(Ok(key)) => json_ok(&InsertResp {
            stored: true,
            key: key_hex(&key),
        }),
        Ok(Err(e)) => json_error(StatusCode::INTERNAL_SERVER_ERROR, &format!("{e}")),
        Err(e) => json_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("join error: {e}"),
        ),
    }
}

#[derive(Deserialize)]
struct FeedbackReq {
    namespace: String,
    /// The similarity score of the served hit being judged — the `score` that `/v1/cache/lookup`
    /// returned on that hit.
    score: f32,
    /// `"agree"` (the hit was correct — reinforces the right distribution) or `"wrong"` (a false
    /// hit — raises the per-namespace cutoff).
    outcome: String,
}
#[derive(Serialize)]
struct FeedbackResp {
    accepted: bool,
    /// The active policy id, so a caller can see whether feedback actually trains: a `static@…`
    /// policy accepts feedback but does not learn from it.
    policy: String,
}

/// Feed a served hit's outcome back to the threshold policy so an adaptive cutoff retunes toward the
/// operator's false-hit target (PLAN.md §5) — the signal that activates the adaptive engine.
/// Explicit-namespace by design: a raw-sidecar caller owns its namespaces, so there is no
/// proxy-flavor namespace to reconstruct. A `StaticThreshold` accepts and ignores it (no-op
/// `observe`). Off the hot path; `observe` is a brief lock + moment update, so it runs inline.
async fn cache_feedback(
    State(state): State<Arc<ProxyState>>,
    Json(req): Json<FeedbackReq>,
) -> Response {
    let ns = match Namespace::new(req.namespace) {
        Ok(ns) => ns,
        Err(_) => return json_error(StatusCode::BAD_REQUEST, "invalid namespace"),
    };
    let outcome = match req.outcome.as_str() {
        "agree" => Outcome::Agree,
        "wrong" => Outcome::Wrong,
        other => {
            return json_error(
                StatusCode::BAD_REQUEST,
                &format!("invalid outcome '{other}' (expected: agree | wrong)"),
            )
        }
    };
    if !req.score.is_finite() {
        return json_error(StatusCode::BAD_REQUEST, "score must be a finite number");
    }
    // The score is a cosine similarity (a `/v1/cache/lookup` hit score), so it lives in [-1, 1].
    // Reject anything outside that scale before it reaches `observe` — an out-of-range value like
    // 999 would otherwise skew the adaptive policy's per-namespace moments.
    if !(-1.0..=1.0).contains(&req.score) {
        return json_error(
            StatusCode::BAD_REQUEST,
            "score must be a cosine similarity in [-1, 1]",
        );
    }
    state.cache.observe(&ns, req.score, outcome);
    json_ok(&FeedbackResp {
        accepted: true,
        policy: state.cache.policy_id().to_string(),
    })
}

async fn healthz() -> &'static str {
    "ok"
}

async fn render_metrics(State(state): State<Arc<ProxyState>>) -> Response {
    text_response(
        StatusCode::OK,
        "text/plain; version=0.0.4",
        None,
        state.metrics.render(),
    )
}

// ---------------------------------------------------------------------------------------------
// Cache calls off the reactor (embed + ANN search are CPU-bound; PLAN.md §3-OSS reactor rule)
// ---------------------------------------------------------------------------------------------

async fn cache_get(
    state: &Arc<ProxyState>,
    ns: Namespace,
    prompt: String,
) -> Result<Lookup, RecallError> {
    let cache = state.cache.clone();
    match tokio::task::spawn_blocking(move || cache.get(&ns, &prompt)).await {
        Ok(r) => r,
        Err(e) => Err(RecallError::Backend(format!("join error: {e}"))),
    }
}

async fn cache_put(
    state: &Arc<ProxyState>,
    ns: Namespace,
    prompt: String,
    completion: String,
    vector: Vec<f32>,
) -> Result<(), RecallError> {
    let cache = state.cache.clone();
    match tokio::task::spawn_blocking(move || cache.put(&ns, &prompt, &completion, &vector)).await {
        Ok(r) => r.map(|_| ()),
        Err(e) => Err(RecallError::Backend(format!("join error: {e}"))),
    }
}

// ---------------------------------------------------------------------------------------------
// Upstream forwarding
// ---------------------------------------------------------------------------------------------

/// Where and how to forward a request for a given API flavor.
struct Upstream<'a> {
    /// Upstream base URL (no trailing `/v1`).
    base: &'a str,
    /// The path appended to `base`, e.g. `/v1/chat/completions` or `/v1/messages`.
    path: &'static str,
    /// How to authenticate the forwarded request.
    auth: Auth<'a>,
    /// How to count tokens (input/output split) in a (cached or upstream) response body —
    /// flavor-specific. The hit path credits both tiers plus their sum to the savings metrics.
    usage_of: fn(&str) -> TokenUsage,
    /// Whether a captured SSE stream terminated *cleanly* and may be cached — flavor-specific
    /// (OpenAI's `[DONE]` vs Anthropic's `message_stop`, plus the truncation finish/stop reasons).
    /// A truncated or cut-off stream returns `false`, so it is never stored and replayed as a
    /// complete answer (PLAN.md §T4).
    stream_complete: fn(&str) -> bool,
}

/// Per-flavor upstream auth. The caller's own credentials are forwarded as-is when present; the
/// configured key is the fallback for requests that arrive without one.
enum Auth<'a> {
    /// OpenAI: `Authorization: Bearer <key>`.
    Bearer(Option<&'a str>),
    /// Anthropic: `x-api-key` + a required `anthropic-version` (and any `anthropic-beta` passthrough).
    Anthropic {
        key: Option<&'a str>,
        version: &'a str,
    },
}

/// Forward and return the upstream response verbatim, tagging it with an `x-recall-cache` label.
async fn forward(
    state: &Arc<ProxyState>,
    headers: &HeaderMap,
    body: &Bytes,
    upstream: &Upstream<'_>,
    label: &str,
) -> Response {
    match forward_collect(state, headers, body, upstream).await {
        Ok((status, ct, text)) => text_response(status, json_ct(&ct), Some(label), text),
        Err(msg) => {
            Metrics::inc(&state.metrics.upstream_errors);
            tracing::warn!(error = %msg, label, "upstream forward failed");
            json_error(StatusCode::BAD_GATEWAY, &msg)
        }
    }
}

/// Build the upstream `POST {base}{path}` request — JSON body + per-flavor auth headers (the caller's
/// own credentials win; the configured key is the fallback). Shared by the buffered and streamed
/// forwarding paths.
fn upstream_request(
    state: &Arc<ProxyState>,
    headers: &HeaderMap,
    body: &Bytes,
    upstream: &Upstream<'_>,
) -> reqwest::RequestBuilder {
    let url = format!("{}{}", upstream.base.trim_end_matches('/'), upstream.path);
    let mut rb = state
        .http
        .post(&url)
        .header("content-type", "application/json")
        .body(body.to_vec());
    match &upstream.auth {
        Auth::Bearer(fallback) => {
            if let Some(auth) = headers.get("authorization") {
                rb = rb.header("authorization", auth);
            } else if let Some(k) = fallback {
                rb = rb.header("authorization", format!("Bearer {k}"));
            }
        }
        Auth::Anthropic { key, version } => {
            if let Some(k) = headers.get("x-api-key") {
                rb = rb.header("x-api-key", k);
            } else if let Some(k) = key {
                rb = rb.header("x-api-key", *k);
            }
            // anthropic-version is required upstream — forward the caller's, else the configured default.
            match headers.get("anthropic-version") {
                Some(v) => rb = rb.header("anthropic-version", v),
                None => rb = rb.header("anthropic-version", *version),
            }
            // Pass the optional beta-features header through untouched.
            if let Some(beta) = headers.get("anthropic-beta") {
                rb = rb.header("anthropic-beta", beta);
            }
        }
    }
    rb
}

/// POST and collect the full response as `(status, content-type, body)` — the buffered path for
/// non-streamed responses and bypass forwarding.
async fn forward_collect(
    state: &Arc<ProxyState>,
    headers: &HeaderMap,
    body: &Bytes,
    upstream: &Upstream<'_>,
) -> Result<(StatusCode, String, String), String> {
    let started = std::time::Instant::now();
    let resp = upstream_request(state, headers, body, upstream)
        .send()
        .await
        .map_err(|e| format!("upstream request failed: {e}"))?;
    let status = StatusCode::from_u16(resp.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let ct = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("application/json")
        .to_string();
    let text = resp
        .text()
        .await
        .map_err(|e| format!("reading upstream body failed: {e}"))?;
    state
        .metrics
        .upstream_seconds
        .observe(started.elapsed().as_secs_f64());
    Ok((status, ct, text))
}

/// Streamed miss: tee the upstream SSE to the client as it arrives while accumulating the body, and
/// store it only if it is a cacheable 2xx event-stream that *terminated cleanly* — i.e. carries the
/// protocol terminator (`[DONE]`/`message_stop`) and no truncation finish/stop reason — so a later
/// identical request hits and replays the stream. A non-2xx / non-SSE upstream, a mid-stream upstream
/// error, or a truncated/cut-off stream is streamed through to the client but never cached (so a
/// partial completion can't be replayed as if whole). The cache write completes before the stream
/// ends, so a concurrent identical request resolves the freshly stored entry.
async fn forward_stream_and_cache(
    state: &Arc<ProxyState>,
    headers: &HeaderMap,
    body: &Bytes,
    upstream: &Upstream<'_>,
    ns: Namespace,
    prompt: String,
    vector: Vec<f32>,
) -> Response {
    use futures_util::StreamExt;

    let resp = match upstream_request(state, headers, body, upstream)
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => {
            Metrics::inc(&state.metrics.upstream_errors);
            tracing::warn!(error = %e, "upstream stream request failed");
            return json_error(
                StatusCode::BAD_GATEWAY,
                &format!("upstream request failed: {e}"),
            );
        }
    };
    let status = StatusCode::from_u16(resp.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let ct = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("application/json")
        .to_string();
    let cacheable = status.is_success() && ct.starts_with("text/event-stream");
    Metrics::inc(&state.metrics.misses);

    let state = state.clone();
    // Copy the flavor-specific completion check out of the borrowed `upstream` so the 'static stream
    // owns it (a `fn` pointer is `Copy`).
    let stream_complete = upstream.stream_complete;
    let stream = async_stream::stream! {
        let mut acc: Vec<u8> = Vec::new();
        let mut upstream_body = resp.bytes_stream();
        while let Some(item) = upstream_body.next().await {
            match item {
                Ok(chunk) => {
                    if cacheable {
                        acc.extend_from_slice(&chunk);
                    }
                    yield Ok::<Bytes, std::io::Error>(chunk);
                }
                Err(e) => {
                    // Surface the upstream error to the client and stop — nothing is cached.
                    tracing::warn!(error = %e, "upstream stream errored mid-flight; not caching");
                    yield Err(std::io::Error::other(e));
                    return;
                }
            }
        }
        // The byte loop ending is NOT proof the completion finished: a graceful upstream half-close
        // looks identical to a clean end here. Store ONLY if the assembled SSE carries its protocol
        // terminator (`[DONE]`/`message_stop`) and no truncation finish/stop reason — otherwise a
        // partial completion would poison the cache and be replayed as if whole (PLAN.md §T4).
        if cacheable {
            match String::from_utf8(acc) {
                Ok(text) if stream_complete(&text) => {
                    // Log "stored" only after the write succeeds — a failed put isn't a stored entry.
                    let ns_label = ns.as_str().to_string();
                    match cache_put(&state, ns, prompt, text, vector).await {
                        Ok(()) => tracing::debug!(namespace = %ns_label, "streamed miss (stored)"),
                        Err(e) => tracing::warn!(
                            namespace = %ns_label,
                            error = %e,
                            "streamed miss store failed"
                        ),
                    }
                }
                _ => {
                    // Streamed to the client but not cached; the `miss` header already went out
                    // (it precedes the body), so this counter is how the drop is observed.
                    Metrics::inc(&state.metrics.stream_not_stored);
                    tracing::debug!(
                        namespace = ns.as_str(),
                        "streamed miss (not stored — incomplete stream)"
                    );
                }
            }
        }
    };

    let label = if cacheable { "miss" } else { "miss-nostore" };
    let content_type = if cacheable {
        "text/event-stream".to_string()
    } else {
        json_ct(&ct).to_string()
    };
    Response::builder()
        .status(status)
        .header("content-type", content_type)
        .header("x-recall-cache", label)
        .body(Body::from_stream(stream))
        .expect("valid streaming response")
}

/// A streaming SSE [`Body`] from a stored SSE string — each event (split on the blank-line separator)
/// becomes its own chunk, so a hit replays as a stream rather than one buffered blob.
fn sse_body_from(full: String) -> Body {
    Body::from_stream(futures_util::stream::iter(
        sse_chunks(&full)
            .into_iter()
            .map(Ok::<Bytes, std::io::Error>),
    ))
}

/// Split a stored SSE body into per-event chunks for incremental replay. CRLF event separators
/// (`\r\n\r\n`) are normalized to `\n\n` first so a CRLF-stored body still chunks incrementally
/// instead of replaying as one blob; each `\n\n`-terminated event becomes one chunk.
fn sse_chunks(full: &str) -> Vec<Bytes> {
    full.replace("\r\n", "\n")
        .split_inclusive("\n\n")
        .map(|e| Bytes::from(e.to_owned()))
        .collect()
}

// ---------------------------------------------------------------------------------------------
// Response helpers
// ---------------------------------------------------------------------------------------------

fn text_response(
    status: StatusCode,
    content_type: &str,
    cache_tag: Option<&str>,
    body: String,
) -> Response {
    let mut b = Response::builder()
        .status(status)
        .header("content-type", content_type);
    if let Some(tag) = cache_tag {
        b = b.header("x-recall-cache", tag);
    }
    b.body(Body::from(body)).expect("valid response")
}

fn json_ok<T: Serialize>(v: &T) -> Response {
    let body = serde_json::to_string(v).unwrap_or_else(|_| "{}".into());
    text_response(StatusCode::OK, "application/json", None, body)
}

fn json_error(status: StatusCode, msg: &str) -> Response {
    let body = json!({"error": {"message": msg, "type": "recall_proxy_error"}}).to_string();
    text_response(status, "application/json", None, body)
}

/// Fall back to `application/json` when the upstream gives no content-type.
fn json_ct(ct: &str) -> &str {
    if ct.is_empty() {
        "application/json"
    } else {
        ct
    }
}

fn key_hex(k: &Key) -> String {
    let mut s = String::with_capacity(64);
    for b in k.as_bytes() {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::sse_chunks;

    #[test]
    fn sse_chunks_split_both_lf_and_crlf_events() {
        // Two LF-delimited events → two chunks.
        assert_eq!(sse_chunks("data: a\n\ndata: b\n\n").len(), 2);
        // The same body with CRLF separators must split too (not replay as one blob).
        let chunks = sse_chunks("data: a\r\n\r\ndata: b\r\n\r\n");
        assert_eq!(chunks.len(), 2, "CRLF events must chunk too");
        assert_eq!(&chunks[0][..], b"data: a\n\n");
    }
}
