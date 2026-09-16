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

use recall_proxy::pricing::{PriceBook, RateSpec};
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

/// Same as [`setup`], but with an operator price book covering the mock's model (`m`) only — so one
/// model is priced and any other is deliberately not.
async fn setup_priced() -> (String, Arc<AtomicU64>) {
    let calls = Arc::new(AtomicU64::new(0));
    let mock = Router::new()
        .route("/v1/chat/completions", post(mock_handler))
        .route("/v1/messages", post(anthropic_mock_handler))
        .with_state(Mock {
            calls: calls.clone(),
        });
    let upstream = serve(mock).await;

    // $2.50 per 1M input tokens, $10.00 per 1M output — the usual ~4× output premium, so the test
    // fails if the two tiers are ever priced at one blended rate.
    let table: std::collections::HashMap<String, RateSpec> = [(
        "m".to_string(),
        RateSpec {
            input: 2.50,
            output: 10.00,
        },
    )]
    .into_iter()
    .collect();
    let (pricing, rejected) = PriceBook::load(&table);
    assert!(rejected.is_empty());

    let cache = Arc::new(boxed_memory_cache(0.9));
    let config = Config {
        upstream_base: upstream.clone(),
        anthropic_upstream_base: upstream,
        base_namespace: "test".into(),
        max_temperature: 1.0,
        pricing,
        ..Config::default()
    };
    let proxy = serve(app(ProxyState::new(cache, config))).await;
    (proxy, calls)
}

/// A stored entry is credited to the fill counters exactly once, priced like the savings; a miss
/// that was NOT stored must credit nothing. A miss that bought nothing for the cache would have
/// cost the same with recall switched off, so charging it would inflate the fill cost.
#[tokio::test]
async fn fills_are_credited_once_per_stored_entry_and_never_on_an_unstored_miss() {
    let (proxy, calls) = setup_priced().await;
    let client = reqwest::Client::new();

    // One distinct prompt: miss (stores, priced) then hit (must not count as a fill).
    for _ in 0..2 {
        client
            .post(format!("{proxy}/v1/chat/completions"))
            .header("content-type", "application/json")
            .body(CHAT)
            .send()
            .await
            .unwrap();
    }
    // A high-temperature request bypasses the cache: forwarded upstream, never stored, so it buys
    // nothing for the cache and must not appear in the fill counters.
    let hot = r#"{"model":"m","temperature":2.0,"messages":[{"role":"user","content":"hot"}]}"#;
    client
        .post(format!("{proxy}/v1/chat/completions"))
        .header("content-type", "application/json")
        .body(hot)
        .send()
        .await
        .unwrap();
    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "one miss + one bypass forwarded"
    );

    let m = client
        .get(format!("{proxy}/metrics"))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    // Exactly one entry was bought and kept — the hit and the bypass are not fills.
    assert!(m.contains("recall_fill_entries_total 1"), "metrics: {m}");
    // The mock's first response: prompt 4 + completion 7 = 11 tokens purchased.
    assert!(m.contains("recall_fill_tokens_total 11"), "metrics: {m}");
    assert!(
        m.contains("recall_fill_input_tokens_total 4"),
        "metrics: {m}"
    );
    assert!(
        m.contains("recall_fill_output_tokens_total 7"),
        "metrics: {m}"
    );
    // Priced by the same table as the savings: 4 in @ $2.50/Mtok + 7 out @ $10.00/Mtok = $0.000080.
    assert!(
        m.contains("recall_fill_cost_usd_total 0.000080"),
        "metrics: {m}"
    );
    assert!(
        m.contains("recall_fill_usage_unknown_total 0"),
        "metrics: {m}"
    );
    // The memory backend reports itself as ephemeral, so `rehydrated 0` is unambiguous.
    assert!(
        m.contains("recall_store_info{backend=\"memory\",durable=\"false\"} 1"),
        "metrics: {m}"
    );
}

/// A hit whose stored completion reports NO usage must be flagged, not silently credited as zero.
/// This is the audit valve that tells "the provider did not report usage" apart from "the cache is
/// idle" — an OpenAI stream cached without `stream_options.include_usage` is exactly that case.
#[tokio::test]
async fn hits_with_no_reported_usage_are_flagged_not_silently_zero() {
    // A complete, cacheable stream that carries no usage chunk at all.
    let sse = concat!(
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"}}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
        "data: [DONE]\n\n",
    );
    let (proxy, calls) = setup_fixed_sse_priced("/v1/chat/completions", sse).await;
    let client = reqwest::Client::new();
    let body = r#"{"model":"m","temperature":0,"stream":true,"messages":[{"role":"user","content":"hi"}]}"#;
    for _ in 0..3 {
        let r = client
            .post(format!("{proxy}/v1/chat/completions"))
            .body(body)
            .send()
            .await
            .unwrap();
        let _ = r.text().await.unwrap();
    }
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "two of the three were hits"
    );

    let m = client
        .get(format!("{proxy}/metrics"))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(m.contains("recall_hits_total 2"), "metrics: {m}");
    // Nothing to credit — but the two hits are visibly flagged rather than reading as an idle cache.
    assert!(m.contains("recall_tokens_saved_total 0"), "metrics: {m}");
    assert!(
        m.contains("recall_hits_usage_unknown_total 2"),
        "metrics: {m}"
    );
    // The fill side has the same valve: the entry was stored, but its cost is unknown, not zero.
    assert!(m.contains("recall_fill_entries_total 1"), "metrics: {m}");
    assert!(
        m.contains("recall_fill_usage_unknown_total 1"),
        "metrics: {m}"
    );
}

/// `recall_misses_total` must mean the same thing however the response was delivered: one count per
/// cacheable request the cache could not answer. The two paths used to disagree — the buffered one
/// counted only responses it could cache, the streamed one counted every forward — so an upstream
/// failure moved the hit rate on one path and not the other.
#[tokio::test]
async fn a_failing_upstream_counts_the_same_miss_on_both_paths() {
    async fn failing(State(st): State<Mock>, _b: Bytes) -> Response {
        st.calls.fetch_add(1, Ordering::SeqCst);
        Response::builder()
            .status(503)
            .header("content-type", "application/json")
            .body(Body::from(r#"{"error":"upstream is down"}"#))
            .unwrap()
    }
    let misses_after = |stream: bool| async move {
        let calls = Arc::new(AtomicU64::new(0));
        let mock = Router::new()
            .route("/v1/chat/completions", post(failing))
            .with_state(Mock {
                calls: calls.clone(),
            });
        let upstream = serve(mock).await;
        let config = Config {
            upstream_base: upstream.clone(),
            anthropic_upstream_base: upstream,
            base_namespace: "test".into(),
            max_temperature: 1.0,
            ..Config::default()
        };
        let proxy = serve(app(ProxyState::new(
            Arc::new(boxed_memory_cache(0.9)),
            config,
        )))
        .await;
        let client = reqwest::Client::new();
        let body = format!(
            r#"{{"model":"m","temperature":0,"stream":{stream},"messages":[{{"role":"user","content":"hi"}}]}}"#
        );
        let r = client
            .post(format!("{proxy}/v1/chat/completions"))
            .header("content-type", "application/json")
            .body(body)
            .send()
            .await
            .unwrap();
        let _ = r.text().await.unwrap();
        let m = client
            .get(format!("{proxy}/metrics"))
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
        m.lines()
            .find_map(|l| l.strip_prefix("recall_misses_total "))
            .unwrap()
            .trim()
            .parse::<u64>()
            .unwrap()
    };
    // One cacheable request the cache could not answer, either way round.
    assert_eq!(misses_after(false).await, 1, "buffered path");
    assert_eq!(misses_after(true).await, 1, "streamed path");
}

/// A streamed miss that is stored is a fill too — and one that is NOT stored (a truncated stream)
/// must credit nothing, since the cache gained no entry from it.
#[tokio::test]
async fn streamed_fills_are_credited_only_when_the_stream_was_stored() {
    let sse = concat!(
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"}}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
        "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":1000,\"completion_tokens\":2000,\"total_tokens\":3000}}\n\n",
        "data: [DONE]\n\n",
    );
    let (proxy, _calls) = setup_fixed_sse_priced("/v1/chat/completions", sse).await;
    let client = reqwest::Client::new();
    let body = r#"{"model":"m","temperature":0,"stream":true,"messages":[{"role":"user","content":"hi"}]}"#;
    let r = client
        .post(format!("{proxy}/v1/chat/completions"))
        .body(body)
        .send()
        .await
        .unwrap();
    let _ = r.text().await.unwrap();

    let m = client
        .get(format!("{proxy}/metrics"))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(m.contains("recall_fill_entries_total 1"), "metrics: {m}");
    assert!(m.contains("recall_fill_tokens_total 3000"), "metrics: {m}");
    assert!(
        m.contains("recall_fill_cost_usd_total 0.022500"),
        "metrics: {m}"
    );

    // Now a stream with no terminator: forwarded to the client, never stored — so no fill.
    let truncated = "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"partial\"}}]}\n\n";
    let (proxy2, _c2) = setup_fixed_sse_priced("/v1/chat/completions", truncated).await;
    let r = client
        .post(format!("{proxy2}/v1/chat/completions"))
        .body(body)
        .send()
        .await
        .unwrap();
    let _ = r.text().await.unwrap();
    let m2 = client
        .get(format!("{proxy2}/metrics"))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(m2.contains("recall_fill_entries_total 0"), "metrics: {m2}");
    assert!(
        m2.contains("recall_fill_cost_usd_total 0.000000"),
        "metrics: {m2}"
    );
    assert!(
        m2.contains("recall_stream_not_stored_total 1"),
        "metrics: {m2}"
    );
}

/// Feedback outcomes must be counted at the proxy, and only for reports that were actually
/// accepted. The default `StaticThreshold` discards observations entirely, so without a
/// proxy-level counter a wrong-hit report vanishes behind `200 {"accepted": true}` and the
/// observed false-hit rate is unmeasurable on the default config.
#[tokio::test]
async fn feedback_outcomes_are_counted_and_rejects_are_not() {
    let (proxy, _calls) = setup().await;
    let client = reqwest::Client::new();

    // A hit, so there is something to judge — and so `x-recall-namespace` gives us the server's key.
    for _ in 0..2 {
        client
            .post(format!("{proxy}/v1/chat/completions"))
            .header("content-type", "application/json")
            .body(CHAT)
            .send()
            .await
            .unwrap();
    }
    let hit = client
        .post(format!("{proxy}/v1/chat/completions"))
        .header("content-type", "application/json")
        .body(CHAT)
        .send()
        .await
        .unwrap();
    let ns = hit
        .headers()
        .get("x-recall-namespace")
        .unwrap()
        .to_str()
        .unwrap()
        .to_string();

    let feedback = |outcome: &str, score: f64| {
        let (c, p, ns) = (client.clone(), proxy.clone(), ns.clone());
        let body = format!(r#"{{"namespace":"{ns}","score":{score},"outcome":"{outcome}"}}"#);
        async move {
            c.post(format!("{p}/v1/cache/feedback"))
                .header("content-type", "application/json")
                .body(body)
                .send()
                .await
                .unwrap()
                .status()
                .as_u16()
        }
    };

    assert_eq!(feedback("agree", 0.99).await, 200);
    assert_eq!(feedback("agree", 0.97).await, 200);
    assert_eq!(feedback("wrong", 0.91).await, 200);
    // Rejected reports must leave the counters alone: an out-of-range score never reached the
    // policy, so it must not appear in the observed rate either.
    assert_eq!(feedback("wrong", 999.0).await, 400);
    assert_eq!(feedback("sideways", 0.9).await, 400);

    let m = client
        .get(format!("{proxy}/metrics"))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(m.contains("recall_feedback_agree_total 2"), "metrics: {m}");
    assert!(m.contains("recall_feedback_wrong_total 1"), "metrics: {m}");
    // One distinct prompt was cached, so the store holds exactly one entry.
    assert!(m.contains("recall_cache_entries 1"), "metrics: {m}");
    // The mock always reports usage, so nothing is flagged as unknown.
    assert!(
        m.contains("recall_hits_usage_unknown_total 0"),
        "metrics: {m}"
    );
}

/// A **streamed** hit must credit its saved tokens and dollars, exactly like a non-streamed one.
/// A stored streamed response is raw SSE, not one JSON object, so extracting its usage needs the
/// SSE-aware path — without it every streaming deployment's savings figure reads zero.
#[tokio::test]
async fn streamed_hits_credit_saved_tokens_and_dollars() {
    // An OpenAI stream carrying its usage chunk, as `stream_options: {"include_usage": true}` yields:
    // one content delta, the usage chunk, then the terminator.
    let sse = concat!(
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"}}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
        "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":1000,\"completion_tokens\":2000,\"total_tokens\":3000}}\n\n",
        "data: [DONE]\n\n",
    );
    let (proxy, calls) = setup_fixed_sse_priced("/v1/chat/completions", sse).await;
    let client = reqwest::Client::new();
    let body = r#"{"model":"m","temperature":0,"stream":true,"messages":[{"role":"user","content":"hi"}]}"#;

    // Miss (stores the stream), then hit (replays it).
    for _ in 0..2 {
        let r = client
            .post(format!("{proxy}/v1/chat/completions"))
            .body(body)
            .send()
            .await
            .unwrap();
        let _ = r.text().await.unwrap();
    }
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "second request served from cache"
    );

    let m = client
        .get(format!("{proxy}/metrics"))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(m.contains("recall_hits_total 1"), "metrics: {m}");
    // The stored SSE says prompt 1000 / completion 2000 — the hit saved exactly those.
    assert!(m.contains("recall_tokens_saved_total 3000"), "metrics: {m}");
    assert!(
        m.contains("recall_input_tokens_saved_total 1000"),
        "metrics: {m}"
    );
    assert!(
        m.contains("recall_output_tokens_saved_total 2000"),
        "metrics: {m}"
    );
    // …and the dollar figure follows: 1000 in @ $2.50/Mtok + 2000 out @ $10.00/Mtok = $0.022500.
    assert!(
        m.contains("recall_cost_saved_usd_total 0.022500"),
        "metrics: {m}"
    );
}

/// [`setup_fixed_sse`] with a price book covering the test model, for asserting streamed dollars.
async fn setup_fixed_sse_priced(path: &str, sse: &'static str) -> (String, Arc<AtomicU64>) {
    let calls = Arc::new(AtomicU64::new(0));
    let mock = Router::new()
        .route(path, post(fixed_sse_handler))
        .with_state(SseMock {
            calls: calls.clone(),
            sse,
        });
    let upstream = serve(mock).await;
    let table: std::collections::HashMap<String, RateSpec> = [(
        "m".to_string(),
        RateSpec {
            input: 2.50,
            output: 10.00,
        },
    )]
    .into_iter()
    .collect();
    let cache = Arc::new(boxed_memory_cache(0.9));
    let config = Config {
        upstream_base: upstream.clone(),
        anthropic_upstream_base: upstream,
        base_namespace: "test".into(),
        max_temperature: 1.0,
        pricing: PriceBook::load(&table).0,
        ..Config::default()
    };
    let proxy = serve(app(ProxyState::new(cache, config))).await;
    (proxy, calls)
}

/// A hit on a priced model reports dollars saved; a hit on a model the book does not cover reports
/// its tokens as *unpriced* instead of quietly contributing $0 — the difference between "we don't
/// know this model's rate" and "the cache saved nothing".
#[tokio::test]
async fn cache_hits_are_priced_with_the_operator_rates() {
    let (proxy, calls) = setup_priced().await;
    let client = reqwest::Client::new();

    // Model `m` — priced. Miss (stores prompt_tokens=4, completion_tokens=7), then hit.
    for _ in 0..2 {
        client
            .post(format!("{proxy}/v1/chat/completions"))
            .header("content-type", "application/json")
            .body(CHAT)
            .send()
            .await
            .unwrap();
    }
    // Model `other` — same shape, no price rule. Miss then hit; the mock's second stored answer
    // carries prompt_tokens=4, completion_tokens=8 (call #2) = 12 tokens saved on its hit.
    let other =
        r#"{"model":"other","temperature":0,"messages":[{"role":"user","content":"hello there"}]}"#;
    for _ in 0..2 {
        client
            .post(format!("{proxy}/v1/chat/completions"))
            .header("content-type", "application/json")
            .body(other)
            .send()
            .await
            .unwrap();
    }
    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "one upstream call per model"
    );

    let m = client
        .get(format!("{proxy}/metrics"))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(m.contains("recall_hits_total 2"), "metrics: {m}");
    // Priced hit: 4 input @ $2.50/Mtok = 10 µUSD, 7 output @ $10.00/Mtok = 70 µUSD → $0.000080.
    // The unpriced model contributes nothing to this figure.
    assert!(
        m.contains("recall_cost_saved_usd_total 0.000080"),
        "metrics: {m}"
    );
    // …and its 12 saved tokens are visible as unpriced rather than lost.
    assert!(
        m.contains("recall_unpriced_tokens_saved_total 12"),
        "metrics: {m}"
    );
    // Token savings still count both models: 11 + 12.
    assert!(m.contains("recall_tokens_saved_total 23"), "metrics: {m}");
}

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

// ----- Per-namespace savings breakdown -----
// The node counters say the cache is worth running; these rows say for whom. They must be a split
// of the same hits — never an addition — and must stay out of `/metrics`, whose cardinality is not
// allowed to follow a caller-derived key.

#[tokio::test]
async fn namespaces_endpoint_splits_savings_and_keeps_metrics_cardinality_flat() {
    let (proxy, _calls) = setup_priced().await;
    let client = reqwest::Client::new();

    // Two hits under the priced model `m`, one under an unpriced model, each in its own namespace.
    for _ in 0..3 {
        client
            .post(format!("{proxy}/v1/chat/completions"))
            .header("content-type", "application/json")
            .body(CHAT)
            .send()
            .await
            .unwrap();
    }
    let unpriced = r#"{"model":"mystery","temperature":0,"messages":[{"role":"user","content":"hello there"}]}"#;
    for _ in 0..2 {
        client
            .post(format!("{proxy}/v1/chat/completions"))
            .header("content-type", "application/json")
            .body(unpriced)
            .send()
            .await
            .unwrap();
    }

    let doc: serde_json::Value = client
        .get(format!("{proxy}/v1/cache/namespaces"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(doc["schema"], "recall.namespaces/v1");
    assert_eq!(doc["truncated"], false);
    let rows = doc["namespaces"].as_array().unwrap();
    assert_eq!(rows.len(), 2, "one row per namespace: {doc}");

    // Biggest saver first, and it is the priced one.
    let priced = &rows[0];
    assert!(
        priced["namespace"]
            .as_str()
            .unwrap()
            .starts_with("test:openai:m:"),
        "{doc}"
    );
    assert_eq!(priced["hits"], 2, "3 requests = 1 miss + 2 hits: {doc}");
    // Each hit replays answer-1 (4 prompt / 7 completion tokens) at $2.50 in / $10.00 out per 1M:
    // 4×2.5 + 7×10 = 80 µ$ a hit, so 160 µ$ across the two. Priced per tier, never blended.
    assert_eq!(priced["cost_micros"], 160, "{doc}");
    assert_eq!(priced["unpriced_tokens"], 0);

    let unpriced_row = &rows[1];
    assert_eq!(
        unpriced_row["cost_micros"], 0,
        "no rate → no guessed dollars"
    );
    assert!(
        unpriced_row["unpriced_tokens"].as_u64().unwrap() > 0,
        "the row says how far its dollars under-count: {doc}"
    );

    // The rows are a split of the node totals, never an addition to them.
    let m = client
        .get(format!("{proxy}/metrics"))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    let node_hits: u64 = m
        .lines()
        .find_map(|l| l.strip_prefix("recall_hits_total "))
        .and_then(|v| v.trim().parse().ok())
        .unwrap();
    let row_hits: u64 = rows.iter().map(|r| r["hits"].as_u64().unwrap()).sum();
    assert_eq!(row_hits, node_hits, "rows sum to the node's hits: {m}");

    // And the namespace never becomes a metric label — that cardinality is caller-controlled.
    assert!(
        !m.contains("test:openai:m:"),
        "no per-namespace series in /metrics: {m}"
    );
}

// A scrape reads two documents, so it can never see one instant. The rows must therefore be a
// strict under-count of the totals they break down — never an over-count, which would let a
// namespace's dollars exceed the node's own `saved` in an export.
#[tokio::test]
async fn a_hit_landing_mid_scrape_never_makes_the_rows_exceed_the_totals() {
    let (proxy, _calls) = setup_priced().await;
    let client = reqwest::Client::new();

    // Warm one namespace so both documents have something to disagree about.
    for _ in 0..2 {
        client
            .post(format!("{proxy}/v1/chat/completions"))
            .header("content-type", "application/json")
            .body(CHAT)
            .send()
            .await
            .unwrap();
    }

    // Interleave traffic with scrapes: each scrape reads the namespaces document and the metrics
    // text at two different instants, with more hits landing throughout.
    for _ in 0..6 {
        let hitter = {
            let client = client.clone();
            let proxy = proxy.clone();
            tokio::spawn(async move {
                for _ in 0..4 {
                    let _ = client
                        .post(format!("{proxy}/v1/chat/completions"))
                        .header("content-type", "application/json")
                        .body(CHAT)
                        .send()
                        .await;
                }
            })
        };
        let snap = recall_proxy::scrape::fetch(&client, &proxy).await.unwrap();
        hitter.await.unwrap();

        let row_hits: u64 = snap.by_namespace.iter().map(|r| r.hits).sum();
        let row_cost: u64 = snap.by_namespace.iter().map(|r| r.cost_micros).sum();
        let row_tokens: u64 = snap
            .by_namespace
            .iter()
            .map(|r| r.input_tokens + r.output_tokens)
            .sum();
        assert!(
            row_hits <= snap.hits,
            "rows claimed {row_hits} hits against the node's {}",
            snap.hits
        );
        assert!(
            row_cost <= snap.cost_saved_micros,
            "rows claimed {row_cost} micro-USD against the node's {}",
            snap.cost_saved_micros
        );
        assert!(
            row_tokens <= snap.tokens_saved,
            "rows claimed {row_tokens} tokens against the node's {}",
            snap.tokens_saved
        );
    }
}
