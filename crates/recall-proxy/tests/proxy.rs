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
            "data: {{\"id\":\"chatcmpl-{n}\",\"object\":\"chat.completion.chunk\",\"model\":\"m-2026-01-01\",\"choices\":[{{\"index\":0,\"delta\":{{\"content\":\"answer-{n}\"}}}}]}}\n\ndata: [DONE]\n\n"
        );
        return Response::builder()
            .header("content-type", "text/event-stream")
            .body(Body::from(sse))
            .unwrap();
    }
    // Real OpenAI usage carries the prompt/completion split AND total_tokens; the mock mirrors that so
    // the input/output savings metrics are exercised. prompt(4) + completion(6+n) = total(10+n).
    let body = format!(
        r#"{{"id":"chatcmpl-{n}","object":"chat.completion","model":"m-2026-01-01","choices":[{{"index":0,"message":{{"role":"assistant","content":"answer-{n}"}},"finish_reason":"stop"}}],"usage":{{"prompt_tokens":4,"completion_tokens":{out},"total_tokens":{tok}}}}}"#,
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
        // The real Anthropic streaming shape, including where it puts usage: `message_start`
        // carries `input_tokens` and a *partial* `output_tokens`, and the final count arrives much
        // later on `message_delta`. A mock that omits both (as this one used to) makes every
        // streamed Anthropic response look like it cost nothing, which is exactly the case the
        // savings and span figures need to get right.
        let sse = format!(
            "event: message_start\ndata: {{\"type\":\"message_start\",\"message\":{{\"id\":\"msg-{n}\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-x-20260101\",\"content\":[],\"usage\":{{\"input_tokens\":5,\"output_tokens\":1}}}}}}\n\n\
             event: content_block_delta\ndata: {{\"type\":\"content_block_delta\",\"index\":0,\"delta\":{{\"type\":\"text_delta\",\"text\":\"answer-{n}\"}}}}\n\n\
             event: message_delta\ndata: {{\"type\":\"message_delta\",\"delta\":{{\"stop_reason\":\"end_turn\",\"stop_sequence\":null}},\"usage\":{{\"output_tokens\":{out}}}}}\n\n\
             event: message_stop\ndata: {{\"type\":\"message_stop\"}}\n\n",
            n = n,
            out = 6 + n
        );
        return Response::builder()
            .header("content-type", "text/event-stream")
            .body(Body::from(sse))
            .unwrap();
    }
    let body = format!(
        r#"{{"id":"msg-{n}","type":"message","role":"assistant","model":"claude-x-20260101","content":[{{"type":"text","text":"answer-{n}"}}],"stop_reason":"end_turn","usage":{{"input_tokens":{inp},"output_tokens":{out}}}}}"#,
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
    let (proxy, calls) = setup_fixed_sse_priced("/v1/chat/completions", sse, "m").await;
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
    let (proxy, _calls) = setup_fixed_sse_priced("/v1/chat/completions", sse, "m").await;
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
    let (proxy2, _c2) = setup_fixed_sse_priced("/v1/chat/completions", truncated, "m").await;
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
    // ...but they are not invisible. A client sending the wrong outcome word or a score on the
    // wrong scale otherwise looks exactly like a client sending nothing, while the policy trains on
    // nothing — and `wrong / (agree + wrong)` is the discount the savings figure is quoted net of.
    assert!(
        m.contains("recall_feedback_rejected_total 2"),
        "metrics: {m}"
    );
    // The default policy keeps no per-namespace state, so it has nothing to evict.
    assert!(
        m.contains("recall_policy_namespaces_evicted_total 0"),
        "metrics: {m}"
    );
}

/// Every shape of malformed feedback is counted, not just the two the happy-path test happens to
/// send. Each of these leaves `agree`/`wrong` untouched, which is exactly why they need a counter.
#[tokio::test]
async fn every_malformed_feedback_shape_is_counted_as_a_rejection() {
    let (proxy, _calls) = setup().await;
    let client = reqwest::Client::new();
    let post = |body: String| {
        let (c, p) = (client.clone(), proxy.clone());
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

    // An empty namespace is not a namespace...
    assert_eq!(
        post(r#"{"namespace":"","score":0.9,"outcome":"wrong"}"#.into()).await,
        400
    );
    // ...nor is one past the length cap, which exists because this value is retained as a map key.
    let too_long = "n".repeat(600);
    assert_eq!(
        post(format!(
            r#"{{"namespace":"{too_long}","score":0.9,"outcome":"wrong"}}"#
        ))
        .await,
        400
    );
    // The wrong word for the right idea — the most likely real integration bug.
    assert_eq!(
        post(r#"{"namespace":"ns","score":0.9,"outcome":"incorrect"}"#.into()).await,
        400
    );
    // A score on a 0–100 scale rather than a cosine similarity.
    assert_eq!(
        post(r#"{"namespace":"ns","score":91.0,"outcome":"wrong"}"#.into()).await,
        400
    );

    let m = client
        .get(format!("{proxy}/metrics"))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(
        m.contains("recall_feedback_rejected_total 4"),
        "metrics: {m}"
    );
    // None of them reached the policy, so the observed false-hit rate is untouched — which is the
    // whole reason they must show up somewhere else.
    assert!(m.contains("recall_feedback_agree_total 0"), "metrics: {m}");
    assert!(m.contains("recall_feedback_wrong_total 0"), "metrics: {m}");
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
    let (proxy, calls) = setup_fixed_sse_priced("/v1/chat/completions", sse, "m").await;
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
    // The streamed door writes too — once, at the end of the stream that was stored. The hit that
    // followed replayed it and wrote nothing.
    assert!(
        m.contains("recall_cache_put_duration_seconds_count 1"),
        "metrics: {m}"
    );
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
async fn setup_fixed_sse_priced(
    path: &str,
    sse: &'static str,
    priced_model: &str,
) -> (String, Arc<AtomicU64>) {
    let calls = Arc::new(AtomicU64::new(0));
    let mock = Router::new()
        .route(path, post(fixed_sse_handler))
        .with_state(SseMock {
            calls: calls.clone(),
            sse,
        });
    let upstream = serve(mock).await;
    let table: std::collections::HashMap<String, RateSpec> = [(
        priced_model.to_string(),
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
    // ...and the write that stored it. On a buffered miss the store is on the request's critical
    // path — it completes before the response returns — so this is latency the caller paid for.
    assert!(
        m.contains("# TYPE recall_cache_put_duration_seconds histogram"),
        "{m}"
    );
    assert!(
        m.contains("recall_cache_put_duration_seconds_bucket{le=\"0.005\"}"),
        "{m}"
    );
    assert!(
        m.contains("recall_cache_put_duration_seconds_count 1"),
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

/// The whole point of the `gen_ai.*` export, proven through the real request path rather than on a
/// hand-built span: a **hit made no model call**, so it must not report `gen_ai.usage.*`. Anything
/// that sums those attributes — which is every LLM observability tool — would otherwise count
/// tokens nobody spent, and a cache would inflate the spend dashboard of the team that installed it
/// to cut spend.
///
/// The miss in the same test is the control: there a model really did run, so the usage attributes
/// must be present.
#[tokio::test]
async fn a_hit_span_reports_saved_tokens_and_never_gen_ai_usage() {
    use recall_proxy::otel;

    let (proxy, queue, calls) = setup_exporting().await;

    let body = serde_json::json!({
        "model": "m",
        "messages": [{ "role": "user", "content": "what is a semantic cache" }],
    });
    let client = reqwest::Client::new();
    let post = || {
        client
            .post(format!("{proxy}/v1/chat/completions"))
            .json(&body)
            .send()
    };

    // First call misses and populates; the second is served from cache.
    assert_eq!(post().await.unwrap().status(), 200);
    let hit = post().await.unwrap();
    assert_eq!(hit.headers()["x-recall-cache"], "hit");
    assert_eq!(
        calls.load(Ordering::Relaxed),
        1,
        "the hit called no upstream"
    );

    // Drain and encode exactly as the flush task would.
    let (doc, spans) = exported(&queue);
    let hit_span = with_verdict(&spans, "hit");

    let k = span_keys(hit_span);
    assert!(
        !k.iter().any(|x| x == otel::ATTR_USAGE_INPUT_TOKENS),
        "a hit must not report gen_ai usage — no model ran: {k:?}"
    );
    assert!(
        !k.iter().any(|x| x == otel::ATTR_USAGE_OUTPUT_TOKENS),
        "{k:?}"
    );
    // The avoided tokens ARE reported, under recall's own namespace.
    assert!(
        k.iter().any(|x| x == otel::ATTR_SAVED_INPUT_TOKENS),
        "{k:?}"
    );
    assert!(k.iter().any(|x| x == otel::ATTR_SAVED_COST_MICROS), "{k:?}");
    assert!(
        k.iter().any(|x| x == otel::ATTR_NAMESPACE),
        "a hit was served under a key: {k:?}"
    );
    // Convention basics that a collector keys on.
    assert_eq!(hit_span["name"], "chat m");
    assert!(k.iter().any(|x| x == otel::ATTR_PROVIDER_NAME));
    assert!(
        k.iter().any(|x| x == otel::ATTR_SYSTEM),
        "migration overlap"
    );

    // No prompt or completion text anywhere in the document — the redaction rule, checked against
    // the bytes that would actually be shipped rather than against the struct.
    let wire = serde_json::to_string(&doc).unwrap();
    assert!(
        !wire.contains("what is a semantic cache"),
        "a prompt reached the exported span"
    );
}

// ----- OTLP span export: shared inspection helpers -----

/// Drain the queue and encode it exactly as the flush task would, returning the OTLP document and
/// its spans. Nothing connects to a collector in these tests — the queue is the seam.
fn exported(queue: &recall_proxy::otel::SpanQueue) -> (serde_json::Value, Vec<serde_json::Value>) {
    let doc = recall_proxy::otel::encode_otlp("recall-test", &queue.drain());
    let spans = doc["resourceSpans"][0]["scopeSpans"][0]["spans"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    (doc, spans)
}

/// The attribute keys of one exported span.
fn span_keys(s: &serde_json::Value) -> Vec<String> {
    s["attributes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|a| a["key"].as_str().unwrap().to_string())
        .collect()
}

/// One attribute's value object, or `None` when the span does not carry that key.
fn attr<'a>(s: &'a serde_json::Value, key: &str) -> Option<&'a serde_json::Value> {
    s["attributes"]
        .as_array()
        .unwrap()
        .iter()
        .find(|a| a["key"] == key)
        .map(|a| &a["value"])
}

/// The single exported span carrying `verdict`. Panics if there is not exactly one, so a test
/// cannot silently pass by matching a span from some other request.
fn with_verdict<'a>(spans: &'a [serde_json::Value], verdict: &str) -> &'a serde_json::Value {
    let mut found = spans.iter().filter(|s| {
        attr(s, recall_proxy::otel::ATTR_VERDICT).map(|v| v["stringValue"] == verdict) == Some(true)
    });
    let first = found
        .next()
        .unwrap_or_else(|| panic!("no span with verdict {verdict}: {spans:#?}"));
    assert!(
        found.next().is_none(),
        "more than one {verdict} span: {spans:#?}"
    );
    first
}

/// A proxy over a fixed-SSE upstream with span export switched on. Returns (proxy_url, queue).
async fn setup_fixed_sse_exporting(
    path: &str,
    sse: &'static str,
) -> (String, Arc<recall_proxy::otel::SpanQueue>) {
    let calls = Arc::new(AtomicU64::new(0));
    let mock = Router::new()
        .route(path, post(fixed_sse_handler))
        .with_state(SseMock { calls, sse });
    let upstream = serve(mock).await;
    let cache = Arc::new(boxed_memory_cache(0.9));
    let config = Config {
        upstream_base: upstream.clone(),
        anthropic_upstream_base: upstream,
        base_namespace: "test".into(),
        max_temperature: 1.0,
        otel_endpoint: Some("http://127.0.0.1:1/unused".into()),
        otel_service_name: "recall-test".into(),
        ..Config::default()
    };
    let state = ProxyState::new(cache, config);
    let queue = state.otel.clone().expect("export is on");
    (serve(app(state)).await, queue)
}

/// The mirror of the hit-span rule. A miss and a bypass both really ran a model, so the tokens they
/// burned are the request's own spend and belong in `gen_ai.usage.*` — and neither may claim a
/// saving, because neither saved anything. A bypass additionally has no cache key to report.
#[tokio::test]
async fn a_miss_and_a_bypass_report_the_tokens_a_model_actually_burned() {
    use recall_proxy::otel;

    let (proxy, queue, calls) = setup_exporting().await;
    let client = reqwest::Client::new();

    // (1) A cacheable request the cache cannot answer: forwarded, stored, `miss`.
    let miss = client
        .post(format!("{proxy}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": "m",
            "messages": [{ "role": "user", "content": "what is a semantic cache" }],
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(miss.headers()["x-recall-cache"], "miss");

    // (2) A request above the temperature cap: never eligible, so it never touches the cache.
    let bypass = client
        .post(format!("{proxy}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": "m",
            "temperature": 1.5,
            "messages": [{ "role": "user", "content": "roll me a dice" }],
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(bypass.headers()["x-recall-cache"], "bypass");
    assert_eq!(calls.load(Ordering::Relaxed), 2, "both went upstream");

    let (doc, spans) = exported(&queue);
    let miss_span = with_verdict(&spans, "miss");
    let bypass_span = with_verdict(&spans, "bypass");

    for (label, span) in [("miss", miss_span), ("bypass", bypass_span)] {
        let k = span_keys(span);
        // A model ran, so its usage is reported as usage — the attribute a cost tool sums as spend.
        assert!(
            k.iter().any(|x| x == otel::ATTR_USAGE_INPUT_TOKENS),
            "{label}: {k:?}"
        );
        assert!(
            k.iter().any(|x| x == otel::ATTR_USAGE_OUTPUT_TOKENS),
            "{label}: {k:?}"
        );
        // ...and nothing was saved, so nothing is claimed. An explicit zero would read as a real
        // (if unimpressive) saving and would sum into a savings dashboard as one.
        assert!(
            !k.iter().any(|x| x == otel::ATTR_SAVED_INPUT_TOKENS),
            "{label} claimed a saving: {k:?}"
        );
        assert!(
            !k.iter().any(|x| x == otel::ATTR_SAVED_COST_MICROS),
            "{label} claimed a saving: {k:?}"
        );
        // A score is a hit's similarity to a stored entry; neither of these compared anything.
        assert!(!k.iter().any(|x| x == otel::ATTR_SCORE), "{label}: {k:?}");
        assert_eq!(span["name"], "chat m");
    }

    // The figures are the mock's own (`prompt_tokens: 4`, `completion_tokens: 6 + call number`),
    // which is what makes this a test of parsing the real response rather than of plumbing.
    // Encoded as strings because OTLP maps a 64-bit int to JSON that way.
    assert_eq!(
        attr(miss_span, otel::ATTR_USAGE_INPUT_TOKENS).unwrap()["intValue"],
        "4"
    );
    assert_eq!(
        attr(miss_span, otel::ATTR_USAGE_OUTPUT_TOKENS).unwrap()["intValue"],
        "7"
    );
    assert_eq!(
        attr(bypass_span, otel::ATTR_USAGE_OUTPUT_TOKENS).unwrap()["intValue"],
        "8"
    );

    // The alias the caller asked for is not the model that answered. Both mocks resolve one, as
    // every real provider does; an attribute echoing the request back would hide that entirely.
    for (label, span) in [("miss", miss_span), ("bypass", bypass_span)] {
        assert_eq!(
            attr(span, otel::ATTR_REQUEST_MODEL).unwrap()["stringValue"],
            "m"
        );
        assert_eq!(
            attr(span, otel::ATTR_RESPONSE_MODEL).unwrap()["stringValue"],
            "m-2026-01-01",
            "{label}: the responding model, not the requested one"
        );
    }

    // The one structural difference: a miss was looked up under a key, a bypass never had one.
    assert!(attr(miss_span, otel::ATTR_NAMESPACE).is_some());
    assert!(
        attr(bypass_span, otel::ATTR_NAMESPACE).is_none(),
        "a bypass never reached the cache, so it has no namespace to group by"
    );

    // Redaction holds on every path, not just the hit one.
    let wire = serde_json::to_string(&doc).unwrap();
    assert!(!wire.contains("what is a semantic cache"), "prompt leaked");
    assert!(!wire.contains("roll me a dice"), "prompt leaked");
}

/// The streamed miss, where the span is the *only* honest record. `x-recall-cache` is written
/// before the first byte of the body and can never be corrected, so a stream that turns out to be
/// truncated ships a `miss` header and is dropped. The span is emitted at the true end of the
/// stream and says `miss-nostore`.
#[tokio::test]
async fn a_streamed_span_outranks_the_header_it_could_not_take_back() {
    use recall_proxy::otel;

    let body = r#"{"model":"m","temperature":0,"stream":true,"messages":[{"role":"user","content":"hi"}]}"#;
    let client = reqwest::Client::new();

    // A clean stream, with the usage chunk OpenAI sends under `stream_options.include_usage`.
    let whole = "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"whole\"}}]}\n\n\
                 data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n\
                 data: {\"choices\":[],\"usage\":{\"prompt_tokens\":4,\"completion_tokens\":9,\"total_tokens\":13}}\n\n\
                 data: [DONE]\n\n";
    let (proxy, queue) = setup_fixed_sse_exporting("/v1/chat/completions", whole).await;
    let r = client
        .post(format!("{proxy}/v1/chat/completions"))
        .body(body)
        .send()
        .await
        .unwrap();
    assert_eq!(r.headers()["x-recall-cache"], "miss");
    // Draining the body is what runs the store decision — and therefore the span.
    let _ = r.text().await.unwrap();

    let (_, spans) = exported(&queue);
    let stored = with_verdict(&spans, "miss");
    assert_eq!(
        attr(stored, otel::ATTR_USAGE_OUTPUT_TOKENS).unwrap()["intValue"],
        "9",
        "usage recovered from the assembled SSE, not from a JSON body"
    );
    assert!(attr(stored, otel::ATTR_NAMESPACE).is_some());

    // The same request shape against an upstream that half-closes without `[DONE]`.
    let truncated = "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"partial\"}}]}\n\n";
    let (proxy, queue) = setup_fixed_sse_exporting("/v1/chat/completions", truncated).await;
    let r = client
        .post(format!("{proxy}/v1/chat/completions"))
        .body(body)
        .send()
        .await
        .unwrap();
    assert_eq!(
        r.headers()["x-recall-cache"],
        "miss",
        "the header is committed before the body — it cannot know yet"
    );
    let _ = r.text().await.unwrap();

    let (_, spans) = exported(&queue);
    let dropped = with_verdict(&spans, "miss-nostore");
    // The namespace is still known (the lookup happened); the usage is not, because the completion
    // never arrived whole — absent rather than a confident zero.
    assert!(attr(dropped, otel::ATTR_NAMESPACE).is_some());
    assert!(
        attr(dropped, otel::ATTR_USAGE_INPUT_TOKENS).is_none(),
        "a truncated stream reported no usage to stand behind"
    );
    assert!(
        attr(dropped, otel::ATTR_SAVED_INPUT_TOKENS).is_none(),
        "nothing was stored, so nothing was saved"
    );
}

/// A proxy over the shared OpenAI + Anthropic mock with span export on and both mock models priced,
/// so a span test can assert `recall.saved.cost_micros` on either flavour. Returns
/// (proxy_url, span_queue, upstream_call_counter).
async fn setup_exporting() -> (String, Arc<recall_proxy::otel::SpanQueue>, Arc<AtomicU64>) {
    let calls = Arc::new(AtomicU64::new(0));
    let mock = Router::new()
        .route("/v1/chat/completions", post(mock_handler))
        .route("/v1/messages", post(anthropic_mock_handler))
        .with_state(Mock {
            calls: calls.clone(),
        });
    let upstream = serve(mock).await;

    let rate = RateSpec {
        input: 2.50,
        output: 10.00,
    };
    let table: std::collections::HashMap<String, RateSpec> =
        [("m".to_string(), rate), ("claude-x".to_string(), rate)]
            .into_iter()
            .collect();

    let cache = Arc::new(boxed_memory_cache(0.9));
    let config = Config {
        upstream_base: upstream.clone(),
        anthropic_upstream_base: upstream,
        base_namespace: "test".into(),
        max_temperature: 1.0,
        pricing: PriceBook::load(&table).0,
        // An endpoint is what turns export on. Nothing ever connects to it here — the queue is
        // drained directly, so the test exercises the span contents without a collector.
        otel_endpoint: Some("http://127.0.0.1:1/unused".into()),
        otel_service_name: "recall-test".into(),
        ..Config::default()
    };
    let state = ProxyState::new(cache, config);
    let queue = state.otel.clone().expect("export is on");
    (serve(app(state)).await, queue, calls)
}

/// The Anthropic flavour reports itself as Anthropic, on every verdict, with usage read out of the
/// Anthropic response shape (`usage.input_tokens` / `output_tokens` — not OpenAI's `prompt_tokens`).
/// The two flavours share one span path, so what is worth checking here is the flavour-specific
/// parts: the provider name, the usage keys, and the bypass reason only Anthropic has.
#[tokio::test]
async fn the_anthropic_flavour_reports_itself_in_every_verdict() {
    use recall_proxy::otel;

    let (proxy, queue, calls) = setup_exporting().await;
    let client = reqwest::Client::new();
    let body = r#"{"model":"claude-x","max_tokens":1024,"temperature":0,"messages":[{"role":"user","content":"what is a semantic cache"}]}"#;

    // Miss (stores), then hit (replays) — the upstream is called exactly once.
    let miss = client
        .post(format!("{proxy}/v1/messages"))
        .body(body)
        .send()
        .await
        .unwrap();
    assert_eq!(miss.headers()["x-recall-cache"], "miss");
    let hit = client
        .post(format!("{proxy}/v1/messages"))
        .body(body)
        .send()
        .await
        .unwrap();
    assert_eq!(hit.headers()["x-recall-cache"], "hit");

    // The bypass reason only this flavour has: `anthropic-beta` changes upstream behaviour but is
    // not part of the body the key is derived from, so the request is never eligible.
    let bypass = client
        .post(format!("{proxy}/v1/messages"))
        .header("anthropic-beta", "some-feature-2024-01-01")
        .body(body)
        .send()
        .await
        .unwrap();
    assert_eq!(
        bypass.headers()["x-recall-cache"],
        "bypass",
        "a variant header must not share a cached answer"
    );
    assert_eq!(
        calls.load(Ordering::Relaxed),
        2,
        "the hit called no upstream"
    );

    let (doc, spans) = exported(&queue);
    let miss_span = with_verdict(&spans, "miss");
    let hit_span = with_verdict(&spans, "hit");
    let bypass_span = with_verdict(&spans, "bypass");

    for (label, span) in [
        ("miss", miss_span),
        ("hit", hit_span),
        ("bypass", bypass_span),
    ] {
        // Both names during the semconv migration overlap, and both say `anthropic` — an OpenAI
        // label here would file every Claude call under the wrong vendor in a cost report.
        for key in [otel::ATTR_PROVIDER_NAME, otel::ATTR_SYSTEM] {
            assert_eq!(
                attr(span, key).unwrap()["stringValue"],
                "anthropic",
                "{label}/{key}"
            );
        }
        assert_eq!(span["name"], "chat claude-x", "{label}");
    }

    // The mock's Anthropic-shaped usage: `input_tokens: 5`, `output_tokens: 6 + call number`. The
    // first call is the one that missed, so the miss reports 5/7.
    assert_eq!(
        attr(miss_span, otel::ATTR_USAGE_INPUT_TOKENS).unwrap()["intValue"],
        "5"
    );
    assert_eq!(
        attr(miss_span, otel::ATTR_USAGE_OUTPUT_TOKENS).unwrap()["intValue"],
        "7"
    );
    // The bypass was the second upstream call.
    assert_eq!(
        attr(bypass_span, otel::ATTR_USAGE_OUTPUT_TOKENS).unwrap()["intValue"],
        "8"
    );
    assert!(
        attr(bypass_span, otel::ATTR_NAMESPACE).is_none(),
        "a bypassed Anthropic request has no cache key either"
    );

    // Anthropic resolves the alias too, and the span reports what answered rather than what was
    // asked for. A hit has no response model at all — nothing responded to it.
    for (label, span) in [("miss", miss_span), ("bypass", bypass_span)] {
        assert_eq!(
            attr(span, otel::ATTR_REQUEST_MODEL).unwrap()["stringValue"],
            "claude-x"
        );
        assert_eq!(
            attr(span, otel::ATTR_RESPONSE_MODEL).unwrap()["stringValue"],
            "claude-x-20260101",
            "{label}: the responding model, not the requested one"
        );
    }
    assert!(attr(hit_span, otel::ATTR_RESPONSE_MODEL).is_none());

    // The hit replayed the stored body, so it saved exactly what that body reported — reported as a
    // saving, never as usage, and priced because `claude-x` is in the book.
    assert!(attr(hit_span, otel::ATTR_USAGE_INPUT_TOKENS).is_none());
    assert_eq!(
        attr(hit_span, otel::ATTR_SAVED_INPUT_TOKENS).unwrap()["intValue"],
        "5"
    );
    assert_eq!(
        attr(hit_span, otel::ATTR_SAVED_OUTPUT_TOKENS).unwrap()["intValue"],
        "7"
    );
    assert!(attr(hit_span, otel::ATTR_SAVED_COST_MICROS).is_some());

    let wire = serde_json::to_string(&doc).unwrap();
    assert!(!wire.contains("what is a semantic cache"), "prompt leaked");
}

/// Anthropic splits usage across two events that can be thousands of tokens apart in the stream:
/// `message_start` carries `input_tokens` and a placeholder `output_tokens`, and the real output
/// count only arrives on `message_delta`. Reading "the last usage object" — which is right for
/// OpenAI — would report an input of zero here. Both the span and the savings ledger depend on the
/// two being recombined, so this asserts it end to end rather than over a string fixture.
#[tokio::test]
async fn an_anthropic_streamed_span_reassembles_usage_split_across_two_events() {
    use recall_proxy::otel;

    let (proxy, queue, calls) = setup_exporting().await;
    let client = reqwest::Client::new();
    let body = r#"{"model":"claude-x","max_tokens":1024,"temperature":0,"stream":true,"messages":[{"role":"user","content":"hi"}]}"#;

    let miss = client
        .post(format!("{proxy}/v1/messages"))
        .body(body)
        .send()
        .await
        .unwrap();
    assert_eq!(miss.headers()["x-recall-cache"], "miss");
    // Draining the body is what runs the store decision, and therefore the span.
    let _ = miss.text().await.unwrap();

    let hit = client
        .post(format!("{proxy}/v1/messages"))
        .body(body)
        .send()
        .await
        .unwrap();
    assert_eq!(hit.headers()["x-recall-cache"], "hit");
    let _ = hit.text().await.unwrap();
    assert_eq!(calls.load(Ordering::Relaxed), 1);

    let (_, spans) = exported(&queue);
    let miss_span = with_verdict(&spans, "miss");
    let hit_span = with_verdict(&spans, "hit");

    // `input_tokens: 5` came from `message_start`; `output_tokens: 7` from `message_delta`, which
    // also had to beat the placeholder `1` that `message_start` carried.
    assert_eq!(
        attr(miss_span, otel::ATTR_USAGE_INPUT_TOKENS).unwrap()["intValue"],
        "5",
        "input came from message_start and was not overwritten by the later event"
    );
    assert_eq!(
        attr(miss_span, otel::ATTR_USAGE_OUTPUT_TOKENS).unwrap()["intValue"],
        "7",
        "output came from message_delta, not the message_start placeholder"
    );
    // Read out of `message_start`, the only event in an Anthropic stream that names the model.
    assert_eq!(
        attr(miss_span, otel::ATTR_RESPONSE_MODEL).unwrap()["stringValue"],
        "claude-x-20260101"
    );

    // The hit re-reads the same figures out of the *stored SSE* — a second, independent exercise of
    // the same reassembly, this time against the cache rather than the wire.
    assert_eq!(
        attr(hit_span, otel::ATTR_SAVED_INPUT_TOKENS).unwrap()["intValue"],
        "5"
    );
    assert_eq!(
        attr(hit_span, otel::ATTR_SAVED_OUTPUT_TOKENS).unwrap()["intValue"],
        "7"
    );
}

/// The Anthropic mirror of `streamed_hits_credit_saved_tokens_and_dollars`: the same reassembly,
/// read through `/metrics` rather than a span, so the savings figure the product is sold on is
/// covered on both flavours.
#[tokio::test]
async fn anthropic_streamed_hits_credit_saved_tokens_and_dollars() {
    // A full Anthropic stream: usage opens on `message_start` and closes on `message_delta`.
    let sse = concat!(
        "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-x\",\"content\":[],\"usage\":{\"input_tokens\":1000,\"output_tokens\":1}}}\n\n",
        "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n",
        "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":2000}}\n\n",
        "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
    );
    let (proxy, calls) = setup_fixed_sse_priced("/v1/messages", sse, "claude-x").await;
    let client = reqwest::Client::new();
    let body = r#"{"model":"claude-x","max_tokens":1024,"temperature":0,"stream":true,"messages":[{"role":"user","content":"hi"}]}"#;

    // Miss (stores the stream), then hit (replays it).
    for _ in 0..2 {
        let r = client
            .post(format!("{proxy}/v1/messages"))
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
    // 1000 input + 2000 output, recombined from the two events the stream split them across.
    assert!(m.contains("recall_tokens_saved_total 3000"), "metrics: {m}");
    assert!(
        m.contains("recall_input_tokens_saved_total 1000"),
        "metrics: {m}"
    );
    assert!(
        m.contains("recall_output_tokens_saved_total 2000"),
        "metrics: {m}"
    );
    // 1000 @ $2.50/1M + 2000 @ $10.00/1M = $0.0225.
    assert!(
        m.contains("recall_cost_saved_usd_total 0.0225"),
        "metrics: {m}"
    );
    assert!(
        !m.contains("recall_hits_usage_unknown_total 1"),
        "usage was known — the split must not read as absent: {m}"
    );
}

/// The sidecar is a front door, and it used to be invisible from one. A deployment using recall as
/// a raw semantic cache over HTTP served hits while `/metrics` reported `recall_hits_total 0` and a
/// hit ratio of zero — a cache that looks idle is a cache that gets turned off.
#[tokio::test]
async fn sidecar_lookups_are_visible_in_the_same_counters_as_proxied_ones() {
    let (proxy, calls) = setup().await;
    let client = reqwest::Client::new();
    let lookup = |prompt: &'static str| {
        client
            .post(format!("{proxy}/v1/cache/lookup"))
            .json(&serde_json::json!({ "namespace": "raw", "prompt": prompt }))
            .send()
    };

    // Two misses, an insert, then a lookup that hits the inserted entry.
    assert_eq!(lookup("hello").await.unwrap().status(), 200);
    assert_eq!(lookup("hello").await.unwrap().status(), 200);
    client
        .post(format!("{proxy}/v1/cache/insert"))
        .json(&serde_json::json!({"namespace":"raw","prompt":"hello","completion":"world"}))
        .send()
        .await
        .unwrap();
    let hit: serde_json::Value = lookup("hello").await.unwrap().json().await.unwrap();
    assert_eq!(hit["hit"], true, "{hit}");
    assert_eq!(hit["completion"], "world");
    assert_eq!(
        calls.load(Ordering::SeqCst),
        0,
        "the sidecar never forwards anywhere"
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
    assert!(m.contains("recall_misses_total 2"), "metrics: {m}");
    // Three lookups asked the cache a question; the insert did not.
    assert!(m.contains("recall_requests_total 3"), "metrics: {m}");
    assert!(m.contains("recall_hit_ratio 0.3333"), "metrics: {m}");
    // The latency histogram covers sidecar lookups too — it is the same lookup.
    assert!(
        m.contains("recall_cache_get_duration_seconds_count 3"),
        "metrics: {m}"
    );
    // The hit avoided an upstream call, but the caller never said which model or how many tokens.
    // Counted as a saving of unknown size, never as a saving of zero.
    assert!(
        m.contains("recall_hits_usage_unknown_total 1"),
        "metrics: {m}"
    );
    assert!(m.contains("recall_tokens_saved_total 0"), "metrics: {m}");
    assert!(
        m.contains("recall_cost_saved_usd_total 0.000000"),
        "metrics: {m}"
    );

    // The insert landed, on its own counter. Never on the fill counters: a fill is an entry recall
    // bought with an upstream call it made and can price, and this one the caller paid for.
    assert!(m.contains("recall_sidecar_inserts_total 1"), "metrics: {m}");
    assert!(
        m.contains("recall_sidecar_insert_errors_total 0"),
        "metrics: {m}"
    );
    assert!(m.contains("recall_fill_entries_total 0"), "metrics: {m}");
    // The write is timed on the same histogram as a proxied one — same work, different door. The
    // three lookups wrote nothing, so the insert is the only observation.
    assert!(
        m.contains("recall_cache_put_duration_seconds_count 1"),
        "metrics: {m}"
    );

    // And the namespace shows up in the per-namespace breakdown, serving hits it cannot price.
    let ns: serde_json::Value = client
        .get(format!("{proxy}/v1/cache/namespaces"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let row = ns["namespaces"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["namespace"] == "raw")
        .unwrap_or_else(|| panic!("no row for the sidecar namespace: {ns}"));
    assert_eq!(row["hits"], 1, "{row}");
}

/// A sidecar span says what recall knows and nothing else. The caller handed over a namespace and a
/// prompt; a `gen_ai.request.model` here would be a model nobody named.
#[tokio::test]
async fn a_sidecar_span_carries_the_cache_verdict_and_no_gen_ai_attributes() {
    use recall_proxy::otel;

    let (proxy, queue, _calls) = setup_exporting().await;
    let client = reqwest::Client::new();
    let lookup = || {
        client
            .post(format!("{proxy}/v1/cache/lookup"))
            .json(&serde_json::json!({"namespace":"raw","prompt":"what is a semantic cache"}))
            .send()
    };

    assert_eq!(lookup().await.unwrap().status(), 200);
    client
        .post(format!("{proxy}/v1/cache/insert"))
        .json(&serde_json::json!({
            "namespace": "raw",
            "prompt": "what is a semantic cache",
            "completion": "a cache keyed by meaning",
        }))
        .send()
        .await
        .unwrap();
    let hit: serde_json::Value = lookup().await.unwrap().json().await.unwrap();
    assert_eq!(hit["hit"], true, "{hit}");

    let (doc, spans) = exported(&queue);
    let miss_span = with_verdict(&spans, "miss");
    let hit_span = with_verdict(&spans, "hit");

    for (label, span) in [("miss", miss_span), ("hit", hit_span)] {
        let k = span_keys(span);
        assert!(
            k.iter().all(|key| !key.starts_with("gen_ai.")),
            "{label} carried a gen_ai attribute: {k:?}"
        );
        assert_eq!(
            span["name"], "cache.lookup",
            "{label} must not be named as a model call"
        );
        assert_eq!(
            attr(span, otel::ATTR_NAMESPACE).unwrap()["stringValue"],
            "raw"
        );
        // Nothing is claimed as saved — the caller knows what it saved, recall does not.
        assert!(
            !k.iter().any(|x| x == otel::ATTR_SAVED_INPUT_TOKENS),
            "{label}: {k:?}"
        );
    }
    // The score is the one thing a sidecar hit can report that a miss cannot.
    assert!(attr(hit_span, otel::ATTR_SCORE).is_some());
    assert!(attr(miss_span, otel::ATTR_SCORE).is_none());

    // The redaction rule holds on this door too — the prompt is the request body here.
    let wire = serde_json::to_string(&doc).unwrap();
    assert!(!wire.contains("what is a semantic cache"), "prompt leaked");
    assert!(
        !wire.contains("a cache keyed by meaning"),
        "completion leaked"
    );
}

// ----- A store whose writes fail, so the "the write did not happen" paths are reachable -----

/// Reads like an empty cache and refuses every write. The in-memory store has no failure mode, so
/// without this the store-failed branches — which are the ones that decide whether a counter or a
/// span tells the truth — could only be reasoned about, never run.
struct FailingStore;

impl recall_core::Store for FailingStore {
    fn get(
        &self,
        _key: recall_core::Key,
    ) -> Result<Option<recall_core::Entry>, recall_core::RecallError> {
        Ok(None)
    }
    fn put(
        &self,
        _key: recall_core::Key,
        _entry: &recall_core::Entry,
    ) -> Result<(), recall_core::RecallError> {
        Err(recall_core::RecallError::Backend("store is full".into()))
    }
    fn remove(&self, _key: recall_core::Key) -> Result<(), recall_core::RecallError> {
        Ok(())
    }
    fn len(&self) -> usize {
        0
    }
}

/// A proxy over the shared mock whose cache cannot be written to, with span export on.
async fn setup_failing_store() -> (String, Arc<recall_proxy::otel::SpanQueue>) {
    let calls = Arc::new(AtomicU64::new(0));
    let mock = Router::new()
        .route("/v1/chat/completions", post(mock_handler))
        .with_state(Mock { calls });
    let upstream = serve(mock).await;
    let cache = Arc::new(recall_proxy::boxed_cache_full(
        Box::new(recall_core::HashEmbedder::default()),
        Box::new(FailingStore),
        Box::new(recall_core::StaticThreshold::new(0.9)),
    ));
    let config = Config {
        upstream_base: upstream.clone(),
        anthropic_upstream_base: upstream,
        base_namespace: "test".into(),
        max_temperature: 1.0,
        otel_endpoint: Some("http://127.0.0.1:1/unused".into()),
        otel_service_name: "recall-test".into(),
        ..Config::default()
    };
    let state = ProxyState::new(cache, config);
    let queue = state.otel.clone().expect("export is on");
    (serve(app(state)).await, queue)
}

/// A sidecar insert that does not land must say so on a counter, not only in the caller's `500`.
/// A success-only counter cannot tell an idle door from a broken one — both leave it flat — which
/// is the same blind spot that made the whole sidecar invisible.
#[tokio::test]
async fn a_sidecar_insert_that_fails_to_store_is_counted_as_a_failure() {
    let (proxy, _queue) = setup_failing_store().await;
    let client = reqwest::Client::new();

    let r = client
        .post(format!("{proxy}/v1/cache/insert"))
        .json(&serde_json::json!({"namespace":"raw","prompt":"hi","completion":"there"}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 500, "the caller is told");

    // A malformed request is rejected before any write is attempted, so it is neither a stored
    // entry nor a failed one — it never reached the store.
    let bad = client
        .post(format!("{proxy}/v1/cache/insert"))
        .json(&serde_json::json!({"namespace":"","prompt":"hi","completion":"there"}))
        .send()
        .await
        .unwrap();
    assert_eq!(bad.status(), 400);

    let m = client
        .get(format!("{proxy}/metrics"))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(
        m.contains("recall_sidecar_insert_errors_total 1"),
        "metrics: {m}"
    );
    assert!(m.contains("recall_sidecar_inserts_total 0"), "metrics: {m}");
    // The failed write is still timed. A store that takes two seconds and *then* errors is the
    // case the histogram matters most for; timing only successes would hide it. The rejected
    // request is not counted — it never reached the store to take any time in it.
    assert!(
        m.contains("recall_cache_put_duration_seconds_count 1"),
        "metrics: {m}"
    );
}

/// The claim the buffered miss path makes in a comment, actually run: `x-recall-cache` is chosen
/// before the write is attempted, so a failed put still ships `miss` — and the span, emitted after,
/// says `miss-nostore`. Until a store could fail, only the truncated-stream case exercised this.
#[tokio::test]
async fn a_miss_whose_write_fails_ships_a_miss_header_and_a_miss_nostore_span() {
    use recall_proxy::otel;

    let (proxy, queue) = setup_failing_store().await;
    let client = reqwest::Client::new();

    let r = client
        .post(format!("{proxy}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": "m",
            "messages": [{ "role": "user", "content": "hello" }],
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200, "the caller still gets its answer");
    assert_eq!(
        r.headers()["x-recall-cache"],
        "miss",
        "the header is committed before the write is attempted"
    );

    let (_, spans) = exported(&queue);
    let span = with_verdict(&spans, "miss-nostore");
    assert!(attr(span, otel::ATTR_NAMESPACE).is_some());
    // A model really ran, so the tokens are still the request's own spend.
    assert_eq!(
        attr(span, otel::ATTR_USAGE_INPUT_TOKENS).unwrap()["intValue"],
        "4"
    );
    assert!(
        attr(span, otel::ATTR_SAVED_INPUT_TOKENS).is_none(),
        "nothing was stored, so nothing was saved"
    );
}
