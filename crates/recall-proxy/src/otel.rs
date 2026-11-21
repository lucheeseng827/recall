//! OpenTelemetry GenAI (`gen_ai.*`) spans for the cache decision.
//!
//! Prometheus (`/metrics`) answers "how is this node doing". This answers "what happened to *that*
//! request", in the vocabulary every LLM observability tool already reads — so recall shows up
//! inside an existing tracing stack with no adapter written for it.
//!
//! ## The decision this feature turns on
//!
//! recall is not a GenAI client. On a **hit it does not call a model at all**, and the semantic
//! conventions have no concept of "I declined to make the call". That leaves a trap:
//!
//! > If a hit emitted `gen_ai.usage.input_tokens`, every tool that sums that attribute would count
//! > tokens **nobody ever spent**. A cache would inflate the spend dashboard of the very team that
//! > installed it to reduce spend.
//!
//! So `gen_ai.usage.*` is populated **only when an upstream call actually happened** — a miss or a
//! bypass. On a hit the attributes are absent, and the avoided tokens are reported under recall's
//! own `recall.saved.*` namespace, where no standard aggregation will mistake them for spend. Two
//! different facts, two different names.
//!
//! ## Prompts and completions are structurally impossible to emit
//!
//! The conventions define optional events carrying prompt and completion content. recall does not
//! implement them, and [`CacheSpan`] has no field that could hold either — not an `Option`, not a
//! redacted one. A product whose argument is that prompts never leave the box cannot ship a
//! telemetry feature that ships prompts, and "off by default" is not the same as "impossible".
//! This matches the existing redaction rule on the request path: namespace and score only.
//!
//! ## Which version of an unstable spec
//!
//! The GenAI conventions are **not stable**. Written against **v{SEMCONV_VERSION}**; the core
//! attributes (operation, provider, model, token usage) have been stable in shape since v1.37.0,
//! and `gen_ai.system` is migrating to [`ATTR_PROVIDER_NAME`]. During that overlap real
//! instrumentation emits **both**, so this does too — a consumer pinned to either name finds what
//! it is looking for, and dropping the old one is a later, deliberate change rather than a silent
//! break. `SEMCONV_VERSION` is here so a future reader knows what this was checked against rather
//! than guessing.
//!
//! ## Shape
//!
//! [`CacheSpan`] is the fact. [`encode_otlp`] renders a batch as OTLP/HTTP **JSON** — pure, no
//! network, testable. [`SpanSink`] is the seam that actually ships it, the same stance as every
//! other outbound call in this project: with no endpoint configured there is no exporter and no
//! network, and the air-gap default is untouched.

use std::sync::atomic::{AtomicU64, Ordering};

use serde_json::{json, Value};

/// The semantic-convention version these attribute names were checked against.
pub const SEMCONV_VERSION: &str = "1.40.0";

// --- gen_ai.*, verbatim from the registry ---

/// The current provider discriminator.
pub const ATTR_PROVIDER_NAME: &str = "gen_ai.provider.name";
/// The name it is migrating from. Emitted alongside during the overlap — see the module docs.
pub const ATTR_SYSTEM: &str = "gen_ai.system";
pub const ATTR_OPERATION_NAME: &str = "gen_ai.operation.name";
pub const ATTR_REQUEST_MODEL: &str = "gen_ai.request.model";
pub const ATTR_RESPONSE_MODEL: &str = "gen_ai.response.model";
pub const ATTR_USAGE_INPUT_TOKENS: &str = "gen_ai.usage.input_tokens";
pub const ATTR_USAGE_OUTPUT_TOKENS: &str = "gen_ai.usage.output_tokens";

// --- recall's own, deliberately NOT under gen_ai.* ---
//
// These describe the cache, not the model call, and three of them are tokens that were *avoided*.
// Putting avoided tokens under `gen_ai.usage.*` is the trap the module docs open with.

pub const ATTR_VERDICT: &str = "recall.cache.verdict";
pub const ATTR_SCORE: &str = "recall.cache.score";
pub const ATTR_NAMESPACE: &str = "recall.cache.namespace";
pub const ATTR_SAVED_INPUT_TOKENS: &str = "recall.saved.input_tokens";
pub const ATTR_SAVED_OUTPUT_TOKENS: &str = "recall.saved.output_tokens";
pub const ATTR_SAVED_COST_MICROS: &str = "recall.saved.cost_micros";

/// What recall did with the request. The same four labels the `x-recall-cache` response header
/// carries, so a trace and a response agree by construction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// Served from cache. **No upstream call was made** — so no `gen_ai.usage.*`.
    Hit,
    /// Forwarded upstream and stored.
    Miss,
    /// Forwarded upstream, not stored (non-2xx, unparseable body, truncated stream).
    MissNoStore,
    /// Not eligible for caching (tool calls, `n > 1`, temperature above the cap).
    Bypass,
}

impl Verdict {
    pub fn as_str(self) -> &'static str {
        match self {
            Verdict::Hit => "hit",
            Verdict::Miss => "miss",
            Verdict::MissNoStore => "miss-nostore",
            Verdict::Bypass => "bypass",
        }
    }

    /// Whether an upstream model call actually happened. The one predicate that decides whether
    /// `gen_ai.usage.*` may appear.
    pub fn called_upstream(self) -> bool {
        !matches!(self, Verdict::Hit)
    }
}

/// One cache decision, as telemetry.
///
/// Note what is **not** here and cannot be added without changing this type: the prompt, the
/// completion, or any message content. See the module docs.
/// The `gen_ai.*` half of a span: everything that only means anything when a **model** was
/// involved.
///
/// Grouped into one optional field on [`CacheSpan`] rather than five independent `Option`s, so that
/// "the raw cache sidecar cannot report a model, a vendor or a token count" is enforced by the type
/// instead of by remembering — the same reason the span has no prompt field. A `/v1/cache/lookup`
/// caller never tells recall which model (or which vendor) the entry is for, so every one of these
/// is unanswerable there and none is emitted.
#[derive(Clone, Debug, PartialEq)]
pub struct GenAi {
    /// `openai` or `anthropic` — the wire flavour this request arrived in.
    pub provider: &'static str,
    /// The GenAI operation. `chat` for both wires recall proxies today.
    pub operation: &'static str,
    pub model: String,
    /// The model the *response* named, when it named one — which differs from `model` whenever the
    /// provider resolved an alias to a dated snapshot. `None` where nothing said, rather than an
    /// echo of the request that would make the attribute true by construction and useless.
    pub response_model: Option<String>,
    /// Real usage, and ONLY when `verdict.called_upstream()`. Enforced in
    /// [`CacheSpan::attributes`].
    pub usage_input_tokens: Option<u64>,
    pub usage_output_tokens: Option<u64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CacheSpan {
    /// `None` for a raw-sidecar lookup, which is a cache decision but **not** a GenAI operation:
    /// no model ran, none was named, and no `gen_ai.*` attribute is emitted at all.
    pub gen_ai: Option<GenAi>,
    pub verdict: Verdict,
    /// The server-derived cache key. Structural, not a secret — the same value the
    /// `x-recall-namespace` header already returns to the caller.
    ///
    /// `None` on a **bypass**, which never reached the cache and therefore has no key. An empty
    /// string here would be a lie a consumer could group by; an absent attribute is the fact.
    pub namespace: Option<String>,
    /// Similarity of the served entry, on a hit only.
    pub score: Option<f64>,
    /// Tokens the upstream was not asked for, on a hit. Never under `gen_ai.usage.*`. Stays `0` on
    /// a sidecar hit: the caller never said what the entry cost, so the saving is real but
    /// unmeasurable, and `0` here means the attributes are omitted rather than reported as nothing
    /// saved (`recall_hits_usage_unknown_total` is where that shows up).
    pub saved_input_tokens: u64,
    pub saved_output_tokens: u64,
    /// Priced with the operator's own rate table, integer micro-USD. `None` when the model has no
    /// configured price — absent rather than zero, so an incomplete price book reads as unknown
    /// instead of as nothing saved.
    pub saved_cost_micros: Option<u64>,
    pub start_unix_nanos: u128,
    pub end_unix_nanos: u128,
}

impl CacheSpan {
    /// The span name the conventions prescribe: `{gen_ai.operation.name} {gen_ai.request.model}`.
    /// A sidecar lookup has neither, and is not a GenAI operation, so it gets a plain cache name
    /// rather than a GenAI-shaped one with blanks in it.
    pub fn name(&self) -> String {
        match &self.gen_ai {
            Some(g) => format!("{} {}", g.operation, g.model),
            None => "cache.lookup".to_string(),
        }
    }

    /// The attribute set, as `(key, value)` pairs.
    ///
    /// The usage rule is enforced HERE rather than trusted to callers: a hit drops
    /// `gen_ai.usage.*` even if a caller populated it, because the cost of one caller getting that
    /// wrong is every downstream dashboard over-counting spend, silently.
    pub fn attributes(&self) -> Vec<(&'static str, Value)> {
        // recall's own vocabulary applies to every cache decision, GenAI or not.
        let mut attrs: Vec<(&'static str, Value)> =
            vec![(ATTR_VERDICT, json!(self.verdict.as_str()))];
        if let Some(ns) = &self.namespace {
            attrs.push((ATTR_NAMESPACE, json!(ns)));
        }
        if let Some(score) = self.score {
            attrs.push((ATTR_SCORE, json!(score)));
        }
        // Everything below is `gen_ai.*`, and none of it exists for a sidecar lookup.
        if let Some(g) = &self.gen_ai {
            attrs.push((ATTR_PROVIDER_NAME, json!(g.provider)));
            attrs.push((ATTR_SYSTEM, json!(g.provider)));
            attrs.push((ATTR_OPERATION_NAME, json!(g.operation)));
            attrs.push((ATTR_REQUEST_MODEL, json!(g.model)));
            if self.verdict.called_upstream() {
                // Only meaningful when a model actually responded, and only when that response
                // said which model it was.
                if let Some(m) = &g.response_model {
                    attrs.push((ATTR_RESPONSE_MODEL, json!(m)));
                }
                if let Some(v) = g.usage_input_tokens {
                    attrs.push((ATTR_USAGE_INPUT_TOKENS, json!(v)));
                }
                if let Some(v) = g.usage_output_tokens {
                    attrs.push((ATTR_USAGE_OUTPUT_TOKENS, json!(v)));
                }
            }
        }
        if self.saved_input_tokens > 0 || self.saved_output_tokens > 0 {
            attrs.push((ATTR_SAVED_INPUT_TOKENS, json!(self.saved_input_tokens)));
            attrs.push((ATTR_SAVED_OUTPUT_TOKENS, json!(self.saved_output_tokens)));
        }
        if let Some(micros) = self.saved_cost_micros {
            attrs.push((ATTR_SAVED_COST_MICROS, json!(micros)));
        }
        attrs
    }
}

/// Trace and span ids.
///
/// Derived from a per-process nonce and a monotonic counter rather than a CSPRNG, which would mean
/// a new dependency in this crate. That is sound here because trace ids need to be **unique**, not
/// unpredictable — they are correlation handles, carried in plaintext through every collector, and
/// nothing authenticates on them. blake3 over (nonce, counter) gives uniqueness across processes
/// and restarts without adding a crate.
pub struct IdGen {
    nonce: [u8; 16],
    counter: AtomicU64,
}

impl IdGen {
    pub fn new() -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let pid = std::process::id() as u128;
        let mut nonce = [0u8; 16];
        nonce.copy_from_slice(&(now ^ (pid << 64)).to_be_bytes());
        Self {
            nonce,
            counter: AtomicU64::new(0),
        }
    }

    /// A `(trace_id, span_id)` pair as lowercase hex, 32 and 16 characters — the OTLP/JSON form.
    pub fn next(&self) -> (String, String) {
        let n = self.counter.fetch_add(1, Ordering::Relaxed);
        let mut h = blake3::Hasher::new();
        h.update(&self.nonce);
        h.update(&n.to_be_bytes());
        let d = h.finalize();
        let b = d.as_bytes();
        (hex(&b[0..16]), hex(&b[16..24]))
    }
}

impl Default for IdGen {
    fn default() -> Self {
        Self::new()
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// One attribute in OTLP/JSON form.
///
/// **64-bit integers are strings.** That is the proto3 JSON mapping, not a quirk of this encoder:
/// `int64`/`uint64` serialise as strings so JavaScript consumers do not silently lose precision
/// past 2^53. An encoder that emits them as numbers produces a document some collectors accept and
/// others reject, which is the worst of both.
fn attr(key: &str, value: &Value) -> Value {
    let v = match value {
        Value::String(s) => json!({ "stringValue": s }),
        Value::Bool(b) => json!({ "boolValue": b }),
        Value::Number(n) if n.is_f64() => json!({ "doubleValue": n.as_f64().unwrap_or(0.0) }),
        Value::Number(n) => json!({ "intValue": n.to_string() }),
        other => json!({ "stringValue": other.to_string() }),
    };
    json!({ "key": key, "value": v })
}

/// Render a batch as an OTLP/HTTP JSON `ExportTraceServiceRequest` body.
///
/// Pure: it makes no network call and takes no clock. POST the bytes to `<endpoint>/v1/traces` with
/// `content-type: application/json`.
pub fn encode_otlp(service_name: &str, spans: &[(CacheSpan, String, String)]) -> Value {
    let encoded: Vec<Value> = spans
        .iter()
        .map(|(s, trace_id, span_id)| {
            json!({
                "traceId": trace_id,
                "spanId": span_id,
                "name": s.name(),
                // 3 = SPAN_KIND_CLIENT. recall stands where the client to the model would be, which
                // is true on a miss and is the most useful lie on a hit — the alternative, INTERNAL,
                // hides the span from every "what did we call" view a user would look in.
                "kind": 3,
                "startTimeUnixNano": s.start_unix_nanos.to_string(),
                "endTimeUnixNano": s.end_unix_nanos.to_string(),
                "attributes": s.attributes().iter().map(|(k, v)| attr(k, v)).collect::<Vec<_>>(),
                "status": {},
            })
        })
        .collect();

    json!({
        "resourceSpans": [{
            "resource": {
                "attributes": [
                    attr("service.name", &json!(service_name)),
                    attr("telemetry.sdk.name", &json!("recall")),
                    attr("telemetry.sdk.language", &json!("rust")),
                ]
            },
            "scopeSpans": [{
                "scope": { "name": "recall", "version": env!("CARGO_PKG_VERSION") },
                "spans": encoded,
            }]
        }]
    })
}

/// Where an encoded batch goes. A seam, so the default build ships no exporter and makes no network
/// call — the same stance as every other outbound edge in this project.
pub trait SpanSink: Send + Sync {
    /// Ship one encoded OTLP batch. Errors are the sink's business; the caller drops them.
    fn send(&self, body: Value);
}

/// A bounded, non-blocking span queue.
///
/// Two rules, both of which exist because telemetry must never be able to hurt the thing it
/// observes:
///
/// - **Bounded.** At capacity a span is DROPPED and counted, never queued. An unbounded queue in
///   front of a collector that has gone away turns a telemetry outage into an out-of-memory kill of
///   the cache.
/// - **Non-blocking.** `record` takes the lock, pushes, and returns. It never waits on a collector,
///   because the request path is waiting on it.
///
/// `dropped()` is deliberately readable: a consumer that cannot tell it is missing spans will
/// happily report a hit rate computed from a fraction of the traffic.
pub struct SpanQueue {
    capacity: usize,
    queued: std::sync::Mutex<Vec<(CacheSpan, String, String)>>,
    dropped: AtomicU64,
    ids: IdGen,
}

impl SpanQueue {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            queued: std::sync::Mutex::new(Vec::new()),
            dropped: AtomicU64::new(0),
            ids: IdGen::new(),
        }
    }

    /// Enqueue a span, or drop it if the queue is full. Never blocks on I/O.
    pub fn record(&self, span: CacheSpan) {
        let (trace_id, span_id) = self.ids.next();
        // A poisoned lock means another thread panicked while holding it. Telemetry is not worth
        // propagating that into the request path, so the span is dropped and counted like any other.
        let Ok(mut q) = self.queued.lock() else {
            self.dropped.fetch_add(1, Ordering::Relaxed);
            return;
        };
        if q.len() >= self.capacity {
            self.dropped.fetch_add(1, Ordering::Relaxed);
            return;
        }
        q.push((span, trace_id, span_id));
    }

    /// Take everything queued, leaving the queue empty. For the flush task.
    pub fn drain(&self) -> Vec<(CacheSpan, String, String)> {
        match self.queued.lock() {
            Ok(mut q) => std::mem::take(&mut *q),
            Err(_) => Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.queued.lock().map(|q| q.len()).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Spans dropped for want of queue space. Cumulative.
    pub fn dropped(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }

    /// Drain, encode and hand to the sink. A no-op when nothing is queued — an empty batch is a
    /// request a collector has to parse to learn nothing.
    pub fn flush(&self, service_name: &str, sink: &dyn SpanSink) {
        let batch = self.drain();
        if batch.is_empty() {
            return;
        }
        sink.send(encode_otlp(service_name, &batch));
    }
}

/// An OTLP/HTTP JSON sink over the `reqwest` client the proxy already holds for upstream
/// forwarding — so shipping spans adds **no dependency**: the crate is already an HTTP client by
/// nature, which is exactly why this exporter lives here and not in `recall-core`.
///
/// Fire-and-forget, deliberately. A collector that is slow, wedged or gone must not be able to
/// delay a flush, let alone a request: the send is spawned and its result is dropped. Losing spans
/// during a collector outage is the correct trade — the alternative is a cache that stalls because
/// its telemetry backend did.
pub struct HttpSink {
    client: reqwest::Client,
    /// The full traces URL, e.g. `http://collector:4318/v1/traces`.
    url: String,
    /// Optional headers, for a collector behind an auth proxy.
    headers: Vec<(String, String)>,
}

impl HttpSink {
    /// `endpoint` is the OTLP HTTP base (e.g. `http://collector:4318`); `/v1/traces` is appended,
    /// which is the path the protocol fixes rather than a choice.
    pub fn new(client: reqwest::Client, endpoint: &str, headers: Vec<(String, String)>) -> Self {
        Self {
            client,
            url: format!("{}/v1/traces", endpoint.trim_end_matches('/')),
            headers,
        }
    }
}

impl SpanSink for HttpSink {
    fn send(&self, body: Value) {
        let mut req = self.client.post(&self.url).json(&body);
        for (k, v) in &self.headers {
            req = req.header(k, v);
        }
        tokio::spawn(async move {
            // The result is dropped on purpose. There is nobody to tell and nothing to retry into —
            // a retry queue in front of a dead collector is the unbounded buffer the queue above
            // exists to avoid.
            let _ = req.send().await;
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn span(verdict: Verdict) -> CacheSpan {
        CacheSpan {
            gen_ai: Some(GenAi {
                provider: "openai",
                operation: "chat",
                model: "gpt-4o".into(),
                // The alias the caller asked for resolves to a dated snapshot upstream — the case
                // the attribute exists for.
                response_model: Some("gpt-4o-2024-08-06".into()),
                usage_input_tokens: Some(1_000),
                usage_output_tokens: Some(250),
            }),
            verdict,
            namespace: Some("acme/proj/openai:gpt-4o:abc".into()),
            score: matches!(verdict, Verdict::Hit).then_some(0.94),
            saved_input_tokens: if matches!(verdict, Verdict::Hit) {
                1_000
            } else {
                0
            },
            saved_output_tokens: if matches!(verdict, Verdict::Hit) {
                250
            } else {
                0
            },
            saved_cost_micros: matches!(verdict, Verdict::Hit).then_some(4_500),
            start_unix_nanos: 1_700_000_000_000_000_000,
            end_unix_nanos: 1_700_000_000_002_000_000,
        }
    }

    fn keys(s: &CacheSpan) -> Vec<&'static str> {
        s.attributes().into_iter().map(|(k, _)| k).collect()
    }

    /// THE test for this module. A hit made no model call, so emitting `gen_ai.usage.*` would have
    /// every tool that sums those attributes counting tokens nobody spent — a cache inflating the
    /// spend dashboard of the team that installed it to cut spend.
    #[test]
    fn a_hit_never_reports_gen_ai_usage_even_when_the_caller_supplies_it() {
        let s = span(Verdict::Hit);
        assert_eq!(
            s.gen_ai.as_ref().unwrap().usage_input_tokens,
            Some(1_000),
            "the caller did populate it"
        );
        let k = keys(&s);
        assert!(!k.contains(&ATTR_USAGE_INPUT_TOKENS), "{k:?}");
        assert!(!k.contains(&ATTR_USAGE_OUTPUT_TOKENS), "{k:?}");
        // The avoided tokens are still reported — under a name no standard aggregation sums as spend.
        assert!(k.contains(&ATTR_SAVED_INPUT_TOKENS));
        assert!(k.contains(&ATTR_SAVED_OUTPUT_TOKENS));
        assert!(k.contains(&ATTR_SAVED_COST_MICROS));
    }

    #[test]
    fn a_miss_reports_real_usage_because_a_model_actually_ran() {
        for v in [Verdict::Miss, Verdict::MissNoStore, Verdict::Bypass] {
            let k = keys(&span(v));
            assert!(k.contains(&ATTR_USAGE_INPUT_TOKENS), "{v:?}: {k:?}");
            assert!(k.contains(&ATTR_USAGE_OUTPUT_TOKENS), "{v:?}");
            assert!(k.contains(&ATTR_RESPONSE_MODEL), "{v:?}");
            // ...and nothing was saved, so no saved-token attributes at all.
            assert!(!k.contains(&ATTR_SAVED_INPUT_TOKENS), "{v:?}");
        }
    }

    #[test]
    fn both_provider_names_are_emitted_during_the_migration_overlap() {
        let k = keys(&span(Verdict::Miss));
        assert!(k.contains(&ATTR_PROVIDER_NAME), "the current name");
        assert!(k.contains(&ATTR_SYSTEM), "the name being migrated from");
    }

    #[test]
    fn the_span_name_follows_the_convention() {
        assert_eq!(span(Verdict::Hit).name(), "chat gpt-4o");
    }

    /// A bypass never reached the cache, so it has no key. Reporting an empty namespace would give
    /// a consumer something to group by that does not exist.
    #[test]
    fn a_bypass_reports_no_namespace_at_all() {
        let mut s = span(Verdict::Bypass);
        s.namespace = None;
        let k = keys(&s);
        assert!(!k.contains(&ATTR_NAMESPACE), "{k:?}");
        // ...and the rest of the span is still a complete record of a real upstream call.
        assert!(k.contains(&ATTR_USAGE_INPUT_TOKENS));
        assert!(k.contains(&ATTR_REQUEST_MODEL));
    }

    /// `gen_ai.response.model` must carry what the response said, not a copy of the request. If it
    /// echoed `gen_ai.request.model` the two would be equal by construction, and a reader comparing
    /// them would conclude no provider ever resolves an alias.
    #[test]
    fn the_response_model_is_the_responders_not_the_requested_one() {
        let s = span(Verdict::Miss);
        let v = s.attributes();
        let get = |key: &str| {
            v.iter()
                .find(|(k, _)| *k == key)
                .map(|(_, val)| val.as_str().unwrap().to_string())
        };
        assert_eq!(get(ATTR_REQUEST_MODEL).as_deref(), Some("gpt-4o"));
        assert_eq!(
            get(ATTR_RESPONSE_MODEL).as_deref(),
            Some("gpt-4o-2024-08-06")
        );

        // And absent, not echoed, when the response did not name a model.
        let mut quiet = span(Verdict::Miss);
        quiet.gen_ai.as_mut().unwrap().response_model = None;
        assert!(!keys(&quiet).contains(&ATTR_RESPONSE_MODEL), "{quiet:?}");
        // A hit never has one at all — nothing responded.
        assert!(!keys(&span(Verdict::Hit)).contains(&ATTR_RESPONSE_MODEL));
    }

    /// A raw-sidecar lookup is a cache decision, not a GenAI operation. The caller supplied a
    /// namespace and a prompt — no model, no vendor, no tokens — so a `gen_ai.*` attribute here
    /// would be invented, and would file non-LLM traffic into an LLM cost dashboard.
    #[test]
    fn a_sidecar_lookup_is_not_dressed_up_as_a_gen_ai_call() {
        let s = CacheSpan {
            gen_ai: None,
            verdict: Verdict::Hit,
            namespace: Some("raw".into()),
            score: Some(0.97),
            saved_input_tokens: 0,
            saved_output_tokens: 0,
            saved_cost_micros: None,
            start_unix_nanos: 10,
            end_unix_nanos: 20,
        };
        let k = keys(&s);
        assert!(
            k.iter().all(|key| !key.starts_with("gen_ai.")),
            "a sidecar span must carry no gen_ai attribute at all: {k:?}"
        );
        // recall's own vocabulary still applies — this is what makes the span worth emitting.
        assert!(k.contains(&ATTR_VERDICT), "{k:?}");
        assert!(k.contains(&ATTR_NAMESPACE), "{k:?}");
        assert!(k.contains(&ATTR_SCORE), "{k:?}");
        // No saving is claimed: the caller never said what the entry cost.
        assert!(!k.contains(&ATTR_SAVED_INPUT_TOKENS), "{k:?}");
        assert!(!k.contains(&ATTR_SAVED_COST_MICROS), "{k:?}");
        // And it is not named as though a model were involved.
        assert_eq!(s.name(), "cache.lookup");
    }

    #[test]
    fn a_score_appears_only_where_one_exists() {
        assert!(keys(&span(Verdict::Hit)).contains(&ATTR_SCORE));
        assert!(!keys(&span(Verdict::Miss)).contains(&ATTR_SCORE));
    }

    /// proto3 JSON maps 64-bit integers to STRINGS. An encoder that emits them as numbers produces
    /// a document some collectors take and others reject.
    #[test]
    fn otlp_encodes_64_bit_integers_as_strings_and_doubles_as_numbers() {
        let doc = encode_otlp(
            "recall",
            &[(span(Verdict::Hit), "a".repeat(32), "b".repeat(16))],
        );
        let s = &doc["resourceSpans"][0]["scopeSpans"][0]["spans"][0];
        assert!(s["startTimeUnixNano"].is_string(), "timestamps are strings");
        assert_eq!(s["endTimeUnixNano"], "1700000000002000000");

        let attrs = s["attributes"].as_array().unwrap();
        let find = |k: &str| attrs.iter().find(|a| a["key"] == k).cloned().unwrap();
        assert!(find(ATTR_SAVED_INPUT_TOKENS)["value"]["intValue"].is_string());
        assert_eq!(find(ATTR_SAVED_INPUT_TOKENS)["value"]["intValue"], "1000");
        // A similarity is a double, and doubles are numbers in proto3 JSON.
        assert!(find(ATTR_SCORE)["value"]["doubleValue"].is_number());
        assert_eq!(find(ATTR_VERDICT)["value"]["stringValue"], "hit");
    }

    #[test]
    fn otlp_carries_the_resource_and_scope_a_collector_expects() {
        let doc = encode_otlp(
            "my-gateway",
            &[(span(Verdict::Miss), "t".into(), "s".into())],
        );
        let res = &doc["resourceSpans"][0]["resource"]["attributes"];
        let name = res
            .as_array()
            .unwrap()
            .iter()
            .find(|a| a["key"] == "service.name")
            .unwrap();
        assert_eq!(name["value"]["stringValue"], "my-gateway");
        assert_eq!(
            doc["resourceSpans"][0]["scopeSpans"][0]["scope"]["name"],
            "recall"
        );
        assert_eq!(
            doc["resourceSpans"][0]["scopeSpans"][0]["spans"][0]["kind"],
            3
        );
    }

    #[test]
    fn ids_are_unique_and_the_right_width() {
        let g = IdGen::new();
        let mut seen = std::collections::HashSet::new();
        for _ in 0..1_000 {
            let (t, s) = g.next();
            assert_eq!(t.len(), 32, "trace ids are 16 bytes of hex");
            assert_eq!(s.len(), 16, "span ids are 8 bytes of hex");
            assert!(t.chars().all(|c| c.is_ascii_hexdigit()));
            assert!(s.chars().all(|c| c.is_ascii_hexdigit()));
            assert!(seen.insert(t), "trace ids must not repeat");
        }
    }

    /// Telemetry must never be able to hurt the thing it observes. At capacity a span is dropped and
    /// counted — an unbounded queue in front of a collector that went away turns a telemetry outage
    /// into an OOM kill of the cache.
    #[test]
    fn the_queue_is_bounded_and_reports_what_it_dropped() {
        let q = SpanQueue::new(4);
        for _ in 0..10 {
            q.record(span(Verdict::Hit));
        }
        assert_eq!(q.len(), 4);
        assert_eq!(q.dropped(), 6);

        // Draining makes room again, and the drop count is cumulative rather than reset — a gauge
        // that forgets is a gauge that says everything is fine.
        let batch = q.drain();
        assert_eq!(batch.len(), 4);
        assert!(q.is_empty());
        q.record(span(Verdict::Miss));
        assert_eq!(q.len(), 1);
        assert_eq!(q.dropped(), 6);
    }

    #[test]
    fn flushing_nothing_does_not_call_the_sink() {
        struct Counting(std::sync::atomic::AtomicUsize);
        impl SpanSink for Counting {
            fn send(&self, _: Value) {
                self.0.fetch_add(1, Ordering::Relaxed);
            }
        }
        let sink = Counting(std::sync::atomic::AtomicUsize::new(0));
        let q = SpanQueue::new(8);
        q.flush("recall", &sink);
        assert_eq!(
            sink.0.load(Ordering::Relaxed),
            0,
            "an empty batch is not worth a request"
        );
        q.record(span(Verdict::Hit));
        q.flush("recall", &sink);
        assert_eq!(sink.0.load(Ordering::Relaxed), 1);
        assert!(q.is_empty(), "a flush drains");
    }
}
