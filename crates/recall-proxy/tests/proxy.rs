//! End-to-end proxy tests against a mock upstream. The mock embeds an incrementing call number in
//! every response, so a cache hit is proven by the *stored* body replaying (not a fresh upstream
//! number) and by the upstream call count not advancing.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::response::Response;
use axum::routing::post;
use axum::Router;
use tokio::net::TcpListener;

use recall_proxy::{app, boxed_memory_cache, Config, ProxyState};

#[derive(Clone)]
struct Mock {
    calls: Arc<AtomicU64>,
}

/// True if the (JSON) request body asked for a streamed response. Parse the body rather than
/// substring-match so the mock follows request semantics regardless of JSON whitespace/formatting.
fn is_stream_body(body: &Bytes) -> bool {
    serde_json::from_slice::<serde_json::Value>(body)
        .ok()
        .and_then(|v| v.get("stream").and_then(serde_json::Value::as_bool))
        .unwrap_or(false)
}

async fn mock_handler(State(st): State<Mock>, body: Bytes) -> Response {
    let n = st.calls.fetch_add(1, Ordering::SeqCst) + 1;
    if is_stream_body(&body) {
        // Minimal OpenAI streaming shape: one content chunk + a terminal [DONE].
        let sse = format!(
            "data: {{\"id\":\"chatcmpl-{n}\",\"object\":\"chat.completion.chunk\",\"choices\":[{{\"index\":0,\"delta\":{{\"content\":\"answer-{n}\"}}}}]}}\n\ndata: [DONE]\n\n"
        );
        return Response::builder()
            .header("content-type", "text/event-stream")
            .body(Body::from(sse))
            .unwrap();
    }
    // Real OpenAI usage carries the prompt/completion split AND total_tokens; the mock mirrors that so
    // the input/output savings metrics are exercised. prompt(4) + completion(6+n) = total(10+n).
    let body = format!(
        r#"{{"id":"chatcmpl-{n}","object":"chat.completion","choices":[{{"index":0,"message":{{"role":"assistant","content":"answer-{n}"}},"finish_reason":"stop"}}],"usage":{{"prompt_tokens":4,"completion_tokens":{out},"total_tokens":{tok}}}}}"#,
        n = n,
        out = 6 + n,
        tok = 10 + n
    );
    Response::builder()
        .header("content-type", "application/json")
        .body(Body::from(body))
        .unwrap()
}

/// Anthropic Messages-shaped mock: same incrementing call number, but the Anthropic response shape
/// (content blocks + `usage.input_tokens`/`output_tokens`, no `total_tokens`).
async fn anthropic_mock_handler(State(st): State<Mock>, body: Bytes) -> Response {
    let n = st.calls.fetch_add(1, Ordering::SeqCst) + 1;
    if is_stream_body(&body) {
        // Minimal Anthropic streaming shape: a text delta event + message_stop.
        let sse = format!(
            "event: content_block_delta\ndata: {{\"type\":\"content_block_delta\",\"index\":0,\"delta\":{{\"type\":\"text_delta\",\"text\":\"answer-{n}\"}}}}\n\nevent: message_stop\ndata: {{\"type\":\"message_stop\"}}\n\n"
        );
        return Response::builder()
            .header("content-type", "text/event-stream")
            .body(Body::from(sse))
            .unwrap();
    }
    let body = format!(
        r#"{{"id":"msg-{n}","type":"message","role":"assistant","model":"claude-x","content":[{{"type":"text","text":"answer-{n}"}}],"stop_reason":"end_turn","usage":{{"input_tokens":{inp},"output_tokens":{out}}}}}"#,
        n = n,
        inp = 5,
        out = 6 + n
    );
    Response::builder()
        .header("content-type", "application/json")
        .body(Body::from(body))
        .unwrap()
}

async fn serve(router: Router) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    format!("http://{addr}")
}

/// Spawn a mock upstream (both OpenAI `/v1/chat/completions` and Anthropic `/v1/messages`, sharing
/// the call counter) + a proxy pointed at it. Returns (proxy_url, upstream_call_counter).
async fn setup() -> (String, Arc<AtomicU64>) {
    let calls = Arc::new(AtomicU64::new(0));
    let mock = Router::new()
        .route("/v1/chat/completions", post(mock_handler))
        .route("/v1/messages", post(anthropic_mock_handler))
        .with_state(Mock {
            calls: calls.clone(),
        });
    let upstream = serve(mock).await;

    let cache = Arc::new(boxed_memory_cache(0.9));
    let config = Config {
        upstream_base: upstream.clone(),
        anthropic_upstream_base: upstream,
        base_namespace: "test".into(),
        max_temperature: 1.0,
        ..Config::default()
    };
    let proxy = serve(app(ProxyState::new(cache, config))).await;
    (proxy, calls)
}

const CHAT: &str =
    r#"{"model":"m","temperature":0,"messages":[{"role":"user","content":"hello there"}]}"#;

#[tokio::test]
async fn miss_then_hit_replays_stored_body_without_re_forwarding() {
    let (proxy, calls) = setup().await;
    let client = reqwest::Client::new();

    // First request: MISS → forwarded to upstream (call #1), stored, returned.
    let r1 = client
        .post(format!("{proxy}/v1/chat/completions"))
        .header("content-type", "application/json")
        .body(CHAT)
        .send()
        .await
        .unwrap();
    assert_eq!(r1.headers().get("x-recall-cache").unwrap(), "miss");
    let b1 = r1.text().await.unwrap();
    assert!(
        b1.contains("answer-1"),
        "miss serves upstream answer-1: {b1}"
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1);

    // Second identical request: HIT → replays answer-1, upstream NOT called again.
    let r2 = client
        .post(format!("{proxy}/v1/chat/completions"))
        .header("content-type", "application/json")
        .body(CHAT)
        .send()
        .await
        .unwrap();
    assert_eq!(r2.headers().get("x-recall-cache").unwrap(), "hit");
    let b2 = r2.text().await.unwrap();
    assert!(b2.contains("answer-1"), "hit replays stored body: {b2}");
    assert!(!b2.contains("answer-2"), "hit must not re-fetch upstream");
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "upstream not called on hit"
    );

    // Metrics reflect 1 hit and the saved tokens (total_tokens=11 from the stored answer-1).
    let m = client
        .get(format!("{proxy}/metrics"))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(m.contains("recall_hits_total 1"), "metrics: {m}");
    assert!(m.contains("recall_misses_total 1"), "metrics: {m}");
    assert!(m.contains("recall_tokens_saved_total 11"), "metrics: {m}");
    // Split: prompt_tokens(4) + completion_tokens(7) = 11.
    assert!(
        m.contains("recall_input_tokens_saved_total 4"),
        "metrics: {m}"
    );
    assert!(
        m.contains("recall_output_tokens_saved_total 7"),
        "metrics: {m}"
    );
}

#[tokio::test]
async fn streaming_request_miss_then_hit_replays_sse() {
    let (proxy, calls) = setup().await;
    let client = reqwest::Client::new();
    let stream_req = r#"{"model":"m","temperature":0,"stream":true,"messages":[{"role":"user","content":"hi"}]}"#;

    // MISS: forwarded (call #1), stored, returned as an event stream.
    let r1 = client
        .post(format!("{proxy}/v1/chat/completions"))
        .body(stream_req)
        .send()
        .await
        .unwrap();
    assert_eq!(r1.headers().get("x-recall-cache").unwrap(), "miss");
    assert_eq!(
        r1.headers().get("content-type").unwrap(),
        "text/event-stream"
    );
    let b1 = r1.text().await.unwrap();
    assert!(
        b1.contains("answer-1") && b1.contains("data:"),
        "miss replays the upstream SSE: {b1}"
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1);

    // HIT: replays the stored SSE as a stream, upstream NOT called again.
    let r2 = client
        .post(format!("{proxy}/v1/chat/completions"))
        .body(stream_req)
        .send()
        .await
        .unwrap();
    assert_eq!(r2.headers().get("x-recall-cache").unwrap(), "hit");
    assert_eq!(
        r2.headers().get("content-type").unwrap(),
        "text/event-stream"
    );
    let b2 = r2.text().await.unwrap();
    assert!(b2.contains("answer-1"), "hit replays the stored SSE: {b2}");
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "no upstream call on a streaming hit"
    );

    // A non-streaming request with otherwise-identical params must NOT collide with the streamed
    // entry — `stream` is in the namespace fingerprint, so this is a fresh miss (call #2).
    let nonstream = r#"{"model":"m","temperature":0,"messages":[{"role":"user","content":"hi"}]}"#;
    let r3 = client
        .post(format!("{proxy}/v1/chat/completions"))
        .body(nonstream)
        .send()
        .await
        .unwrap();
    assert_eq!(
        r3.headers().get("x-recall-cache").unwrap(),
        "miss",
        "streaming and non-streaming entries are isolated by the fingerprint"
    );
    assert_eq!(
        r3.headers().get("content-type").unwrap(),
        "application/json"
    );
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

// ----- Streaming-cache poisoning guard (PLAN.md §T4 / §6.4(2)) -----
// A streamed miss that does NOT terminate cleanly (no `[DONE]`/`message_stop`, or a truncation
// finish/stop reason) must never be stored — otherwise a later identical request would replay a
// partial completion as if it were whole.

#[derive(Clone)]
struct SseMock {
    calls: Arc<AtomicU64>,
    sse: &'static str,
}

/// Always streams the same fixed SSE body (counting the call), so a test controls exactly what the
/// "upstream" sent — a clean stream or a truncated one.
async fn fixed_sse_handler(State(st): State<SseMock>, _body: Bytes) -> Response {
    st.calls.fetch_add(1, Ordering::SeqCst);
    Response::builder()
        .header("content-type", "text/event-stream")
        .body(Body::from(st.sse))
        .unwrap()
}

/// A proxy whose upstream `path` always streams `sse`. Returns (proxy_url, upstream_call_counter).
async fn setup_fixed_sse(path: &str, sse: &'static str) -> (String, Arc<AtomicU64>) {
    let calls = Arc::new(AtomicU64::new(0));
    let mock = Router::new()
        .route(path, post(fixed_sse_handler))
        .with_state(SseMock {
            calls: calls.clone(),
            sse,
        });
    let upstream = serve(mock).await;
    let cache = Arc::new(boxed_memory_cache(0.9));
    let config = Config {
        upstream_base: upstream.clone(),
        anthropic_upstream_base: upstream,
        base_namespace: "test".into(),
        max_temperature: 1.0,
        ..Config::default()
    };
    let proxy = serve(app(ProxyState::new(cache, config))).await;
    (proxy, calls)
}

/// Read `recall_stream_not_stored_total` from /metrics.
async fn stream_not_stored(client: &reqwest::Client, proxy: &str) -> u64 {
    let m = client
        .get(format!("{proxy}/metrics"))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    m.lines()
        .find_map(|l| l.strip_prefix("recall_stream_not_stored_total "))
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(u64::MAX)
}

#[tokio::test]
async fn truncated_openai_stream_is_not_cached() {
    // A graceful half-close: a content delta but NO terminal `data: [DONE]`.
    let truncated = "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"partial\"}}]}\n\n";
    let (proxy, calls) = setup_fixed_sse("/v1/chat/completions", truncated).await;
    let client = reqwest::Client::new();
    let body = r#"{"model":"m","temperature":0,"stream":true,"messages":[{"role":"user","content":"hi"}]}"#;

    // First request streams back as a miss; draining the body runs the store/drop decision.
    let r1 = client
        .post(format!("{proxy}/v1/chat/completions"))
        .body(body)
        .send()
        .await
        .unwrap();
    assert_eq!(r1.headers().get("x-recall-cache").unwrap(), "miss");
    let _ = r1.text().await.unwrap();
    assert_eq!(
        stream_not_stored(&client, &proxy).await,
        1,
        "the truncated stream was dropped, not cached"
    );

    // Second identical request: nothing was cached, so it MISSES again (upstream called twice).
    let r2 = client
        .post(format!("{proxy}/v1/chat/completions"))
        .body(body)
        .send()
        .await
        .unwrap();
    assert_eq!(
        r2.headers().get("x-recall-cache").unwrap(),
        "miss",
        "a truncated stream must not produce a later hit"
    );
    let _ = r2.text().await.unwrap();
    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "upstream called again — the cache was not poisoned"
    );
}

#[tokio::test]
async fn openai_length_truncation_is_not_cached() {
    // A `[DONE]`-terminated stream whose finish_reason is `length` (cut off at max_tokens): the
    // terminator is present but the answer is truncated, so it must not be cached.
    let length = "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"partial\"}}]}\n\n\
                  data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"length\"}]}\n\n\
                  data: [DONE]\n\n";
    let (proxy, calls) = setup_fixed_sse("/v1/chat/completions", length).await;
    let client = reqwest::Client::new();
    let body = r#"{"model":"m","temperature":0,"stream":true,"messages":[{"role":"user","content":"hi"}]}"#;

    let r1 = client
        .post(format!("{proxy}/v1/chat/completions"))
        .body(body)
        .send()
        .await
        .unwrap();
    let _ = r1.text().await.unwrap();
    let r2 = client
        .post(format!("{proxy}/v1/chat/completions"))
        .body(body)
        .send()
        .await
        .unwrap();
    assert_eq!(
        r2.headers().get("x-recall-cache").unwrap(),
        "miss",
        "a length-truncated stream must not be replayed as a hit"
    );
    let _ = r2.text().await.unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn truncated_anthropic_stream_is_not_cached() {
    // An Anthropic text delta but NO terminal `message_stop` event.
    let truncated = "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"partial\"}}\n\n";
    let (proxy, calls) = setup_fixed_sse("/v1/messages", truncated).await;
    let client = reqwest::Client::new();
    let body = r#"{"model":"claude-x","max_tokens":16,"temperature":0,"stream":true,"messages":[{"role":"user","content":"hi"}]}"#;

    let r1 = client
        .post(format!("{proxy}/v1/messages"))
        .body(body)
        .send()
        .await
        .unwrap();
    assert_eq!(r1.headers().get("x-recall-cache").unwrap(), "miss");
    let _ = r1.text().await.unwrap();
    assert_eq!(stream_not_stored(&client, &proxy).await, 1);

    let r2 = client
        .post(format!("{proxy}/v1/messages"))
        .body(body)
        .send()
        .await
        .unwrap();
    assert_eq!(
        r2.headers().get("x-recall-cache").unwrap(),
        "miss",
        "a truncated Anthropic stream must not produce a later hit"
    );
    let _ = r2.text().await.unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn metrics_expose_latency_histograms() {
    let (proxy, _) = setup().await;
    let client = reqwest::Client::new();
    // One miss = one cache lookup + one upstream forward, so both histograms record one observation.
    client
        .post(format!("{proxy}/v1/chat/completions"))
        .body(CHAT)
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();

    let m = client
        .get(format!("{proxy}/metrics"))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    // The end-to-end cache-lookup histogram is exposed with the documented bucket bounds + one obs.
    assert!(
        m.contains("# TYPE recall_cache_get_duration_seconds histogram"),
        "{m}"
    );
    assert!(
        m.contains("recall_cache_get_duration_seconds_bucket{le=\"0.005\"}"),
        "{m}"
    );
    assert!(
        m.contains("recall_cache_get_duration_seconds_count 1"),
        "{m}"
    );
    // The upstream-forward histogram recorded the single miss's round-trip.
    assert!(
        m.contains("recall_upstream_duration_seconds_count 1"),
        "{m}"
    );
}

#[tokio::test]
async fn sidecar_insert_and_lookup() {
    let (proxy, _) = setup().await;
    let client = reqwest::Client::new();

    let ins = client
        .post(format!("{proxy}/v1/cache/insert"))
        .json(&serde_json::json!({"namespace":"ns1","prompt":"capital of france","completion":"Paris."}))
        .send()
        .await
        .unwrap();
    let ins: serde_json::Value = ins.json().await.unwrap();
    assert_eq!(ins["stored"], true);

    // Exact lookup hits.
    let hit: serde_json::Value = client
        .post(format!("{proxy}/v1/cache/lookup"))
        .json(&serde_json::json!({"namespace":"ns1","prompt":"capital of france"}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(hit["hit"], true);
    assert_eq!(hit["completion"], "Paris.");

    // A different namespace must not see it.
    let miss: serde_json::Value = client
        .post(format!("{proxy}/v1/cache/lookup"))
        .json(&serde_json::json!({"namespace":"ns2","prompt":"capital of france"}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(miss["hit"], false);
}

// The feedback endpoint activates the adaptive engine: it must accept a well-formed report (routing
// it to `cache.observe`) and reject malformed ones. Coverage is split by layer — the learning math
// is unit-tested in recall-calibrate, and the cache→policy forwarding in recall-core's spy test; so
// here we prove only the HTTP contract. The default test cache uses a static policy, which accepts
// feedback as a no-op and reports its id.
#[tokio::test]
async fn sidecar_feedback_validates_and_routes() {
    let (proxy, _) = setup().await;
    let client = reqwest::Client::new();

    // Well-formed feedback is accepted and echoes the active policy id.
    let ok = client
        .post(format!("{proxy}/v1/cache/feedback"))
        .json(&serde_json::json!({"namespace":"ns1","score":0.91,"outcome":"wrong"}))
        .send()
        .await
        .unwrap();
    assert_eq!(ok.status().as_u16(), 200);
    let ok: serde_json::Value = ok.json().await.unwrap();
    assert_eq!(ok["accepted"], true);
    assert!(
        ok["policy"].as_str().unwrap().starts_with("static@"),
        "echoes the active policy id: {}",
        ok["policy"]
    );

    // An unknown outcome is a 400 — not a silent accept.
    let bad_outcome = client
        .post(format!("{proxy}/v1/cache/feedback"))
        .json(&serde_json::json!({"namespace":"ns1","score":0.5,"outcome":"meh"}))
        .send()
        .await
        .unwrap();
    assert_eq!(bad_outcome.status().as_u16(), 400);

    // An invalid namespace is a 400.
    let bad_ns = client
        .post(format!("{proxy}/v1/cache/feedback"))
        .json(&serde_json::json!({"namespace":"","score":0.5,"outcome":"agree"}))
        .send()
        .await
        .unwrap();
    assert_eq!(bad_ns.status().as_u16(), 400);
}

// The proxy-flavor feedback loop: an OpenAI/Anthropic hit surfaces the server-derived namespace and
// the hit score as headers, so a caller can train the adaptive policy through /v1/cache/feedback
// without reconstructing the key. This proves that round-trip end to end.
#[tokio::test]
async fn proxy_hit_headers_feed_the_feedback_endpoint() {
    let (proxy, _) = setup().await;
    let client = reqwest::Client::new();
    let body =
        r#"{"model":"m","temperature":0,"messages":[{"role":"user","content":"feedback me"}]}"#;

    // Prime: a miss stores it, the second identical request is a hit.
    client
        .post(format!("{proxy}/v1/chat/completions"))
        .body(body)
        .send()
        .await
        .unwrap();
    let hit = client
        .post(format!("{proxy}/v1/chat/completions"))
        .body(body)
        .send()
        .await
        .unwrap();
    assert_eq!(hit.headers().get("x-recall-cache").unwrap(), "hit");

    // The hit exposes exactly what a caller needs to give feedback.
    let ns = hit
        .headers()
        .get("x-recall-namespace")
        .unwrap()
        .to_str()
        .unwrap()
        .to_string();
    let score: f32 = hit
        .headers()
        .get("x-recall-score")
        .unwrap()
        .to_str()
        .unwrap()
        .parse()
        .unwrap();
    assert!(
        ns.contains(":openai:"),
        "namespace header is the derived OpenAI key: {ns}"
    );
    assert!(
        (-1.0..=1.0).contains(&score),
        "score header is a cosine similarity: {score}"
    );

    // Those header values drive the existing feedback endpoint with no client-side key derivation.
    let fb = client
        .post(format!("{proxy}/v1/cache/feedback"))
        .json(&serde_json::json!({"namespace": ns, "score": score, "outcome": "wrong"}))
        .send()
        .await
        .unwrap();
    assert_eq!(fb.status().as_u16(), 200);
    let fb: serde_json::Value = fb.json().await.unwrap();
    assert_eq!(fb["accepted"], true);
}

#[tokio::test]
async fn healthz_ok() {
    let (proxy, _) = setup().await;
    let body = reqwest::get(format!("{proxy}/healthz"))
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert_eq!(body, "ok");
}

// ----- Anthropic Messages API (`/v1/messages`) -----

const MSG: &str = r#"{"model":"claude-x","max_tokens":1024,"temperature":0,"system":"be brief","messages":[{"role":"user","content":"hello there"}]}"#;

#[tokio::test]
async fn anthropic_messages_miss_then_hit_replays_stored_body_without_re_forwarding() {
    let (proxy, calls) = setup().await;
    let client = reqwest::Client::new();

    // First request: MISS → forwarded to the Anthropic mock (call #1), stored, returned.
    let r1 = client
        .post(format!("{proxy}/v1/messages"))
        .header("content-type", "application/json")
        .header("x-api-key", "test-key")
        .header("anthropic-version", "2023-06-01")
        .body(MSG)
        .send()
        .await
        .unwrap();
    assert_eq!(r1.headers().get("x-recall-cache").unwrap(), "miss");
    let b1 = r1.text().await.unwrap();
    assert!(
        b1.contains("answer-1"),
        "miss serves upstream answer-1: {b1}"
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1);

    // Second identical request: HIT → replays answer-1, upstream NOT called again.
    let r2 = client
        .post(format!("{proxy}/v1/messages"))
        .header("content-type", "application/json")
        .header("x-api-key", "test-key")
        .header("anthropic-version", "2023-06-01")
        .body(MSG)
        .send()
        .await
        .unwrap();
    assert_eq!(r2.headers().get("x-recall-cache").unwrap(), "hit");
    let b2 = r2.text().await.unwrap();
    assert!(b2.contains("answer-1"), "hit replays stored body: {b2}");
    assert!(!b2.contains("answer-2"), "hit must not re-fetch upstream");
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "upstream not called on hit"
    );

    // tokens_saved reflects the Anthropic usage shape: input_tokens(5) + output_tokens(7) = 12.
    let m = client
        .get(format!("{proxy}/metrics"))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(m.contains("recall_hits_total 1"), "metrics: {m}");
    assert!(m.contains("recall_tokens_saved_total 12"), "metrics: {m}");
    // Split: input_tokens(5) + output_tokens(7) = 12.
    assert!(
        m.contains("recall_input_tokens_saved_total 5"),
        "metrics: {m}"
    );
    assert!(
        m.contains("recall_output_tokens_saved_total 7"),
        "metrics: {m}"
    );
}

#[tokio::test]
async fn anthropic_streaming_request_miss_then_hit_replays_sse() {
    let (proxy, calls) = setup().await;
    let client = reqwest::Client::new();
    let stream_req = r#"{"model":"claude-x","max_tokens":1024,"temperature":0,"stream":true,"messages":[{"role":"user","content":"hi"}]}"#;

    // MISS: forwarded (call #1), stored, returned as an event stream.
    let r1 = client
        .post(format!("{proxy}/v1/messages"))
        .body(stream_req)
        .send()
        .await
        .unwrap();
    assert_eq!(r1.headers().get("x-recall-cache").unwrap(), "miss");
    assert_eq!(
        r1.headers().get("content-type").unwrap(),
        "text/event-stream"
    );
    let b1 = r1.text().await.unwrap();
    assert!(
        b1.contains("answer-1") && b1.contains("data:"),
        "miss replays the upstream SSE: {b1}"
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1);

    // HIT: replays the stored SSE, upstream NOT called again.
    let r2 = client
        .post(format!("{proxy}/v1/messages"))
        .body(stream_req)
        .send()
        .await
        .unwrap();
    assert_eq!(r2.headers().get("x-recall-cache").unwrap(), "hit");
    assert_eq!(
        r2.headers().get("content-type").unwrap(),
        "text/event-stream"
    );
    let b2 = r2.text().await.unwrap();
    assert!(b2.contains("answer-1"), "hit replays the stored SSE: {b2}");
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "no upstream call on a streaming hit"
    );
}
