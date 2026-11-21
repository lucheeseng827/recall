//! Mapping an OpenAI `/v1/chat/completions` request onto recall's `(Namespace, prompt)` cache key,
//! plus the bypass policy. Value-based (not a rigid struct) so the proxy tolerates OpenAI's evolving
//! request schema and forwards unknown fields untouched.
//!
//! Key design (PLAN.md §3-OSS "cache-key correctness"):
//! - The **namespace** encodes everything answer-affecting *except the conversation text*: the model
//!   and a fingerprint of the decode params (temperature, top_p, max_tokens, penalties, stop, seed,
//!   response_format). Different params ⇒ different namespace ⇒ fully isolated — "a different
//!   temperature is a different request" falls out of namespace isolation for free.
//! - The **prompt** is the canonical full message array (every turn, every role), so a paraphrase of
//!   the *same* conversation under the *same* params can semantically hit, but a different system
//!   prompt or an extra turn cannot.

use serde_json::{json, Value};

/// A parsed chat-completion request. Holds the raw JSON for verbatim upstream forwarding.
pub struct ChatRequest {
    pub raw: Value,
}

impl ChatRequest {
    pub fn parse(bytes: &[u8]) -> Result<Self, serde_json::Error> {
        Ok(Self {
            raw: serde_json::from_slice(bytes)?,
        })
    }

    pub fn model(&self) -> &str {
        self.raw
            .get("model")
            .and_then(Value::as_str)
            .unwrap_or("unknown")
    }

    pub fn is_stream(&self) -> bool {
        self.raw
            .get("stream")
            .and_then(Value::as_bool)
            .unwrap_or(false)
    }

    pub fn n(&self) -> u64 {
        self.raw.get("n").and_then(Value::as_u64).unwrap_or(1)
    }

    /// OpenAI's documented default temperature is 1.0 when the field is absent.
    pub fn temperature(&self) -> f64 {
        self.raw
            .get("temperature")
            .and_then(Value::as_f64)
            .unwrap_or(1.0)
    }

    /// True if the request defines tools/functions. Caching tool-call requests is bypassed in v1 —
    /// reassembling tool-argument JSON from a cached reply is a severe failure mode (PLAN.md §3-OSS).
    pub fn has_tools(&self) -> bool {
        let non_empty = |v: &Value| v.as_array().map_or(!v.is_null(), |a| !a.is_empty());
        self.raw.get("tools").is_some_and(non_empty)
            || self.raw.get("functions").is_some_and(non_empty)
    }

    /// Why this request must skip the cache entirely (no lookup, no store), or `None` to cache it.
    /// Streaming is *not* a bypass: a streamed completion is cached as its raw SSE body and replayed
    /// as a stream on a hit (it is folded into the namespace fingerprint so it can never be served to
    /// a non-streaming caller as JSON).
    pub fn bypass_reason(&self, max_temperature: f64) -> Option<&'static str> {
        if self.has_tools() {
            return Some("tools");
        }
        if self.n() > 1 {
            return Some("n>1"); // multiple samples requested; a single cached reply is wrong
        }
        if self.temperature() > max_temperature {
            return Some("temperature");
        }
        None
    }

    /// The canonical conversation string that gets embedded and exact-hashed: the full `messages`
    /// array serialized as JSON. Serializing (rather than flattening to `"{role}: {content}\n"`)
    /// keeps message boundaries unambiguous, so content with embedded newlines or `role:`-like text
    /// can never collide into a *wrong* exact-hash hit.
    pub fn canonical_prompt(&self) -> String {
        self.raw
            .get("messages")
            .map(|msgs| serde_json::to_string(msgs).unwrap_or_default())
            .unwrap_or_default()
    }

    /// 16-hex-char fingerprint of the answer-affecting decode params (NOT the messages). Built from a
    /// canonical JSON object; serde_json serializes object keys in sorted order by default, so this is
    /// deterministic across requests.
    fn param_fingerprint(&self) -> String {
        let g = |k: &str| self.raw.get(k).cloned().unwrap_or(Value::Null);
        // An omitted temperature is OpenAI's effective default of 1.0 (see `temperature()`); hash it
        // as 1.0 — not `null` — so "temperature absent" and "temperature: 1.0" share a namespace
        // instead of missing each other.
        let temperature = self.raw.get("temperature").cloned().unwrap_or(json!(1.0));
        let params = json!({
            "temperature": temperature,
            "top_p": g("top_p"),
            "max_tokens": g("max_tokens"),
            "max_completion_tokens": g("max_completion_tokens"),
            "frequency_penalty": g("frequency_penalty"),
            "presence_penalty": g("presence_penalty"),
            "stop": g("stop"),
            "seed": g("seed"),
            "response_format": g("response_format"),
            "logit_bias": g("logit_bias"),
            // Streaming responses are stored as raw SSE and non-streaming as JSON; folding `stream`
            // into the key keeps the two formats in disjoint namespaces so neither replays the other.
            "stream": self.is_stream(),
            // `stream_options` (e.g. `include_usage`, `include_obfuscation`) changes the SSE payload,
            // so two streamed requests differing only here must not share a cached reply.
            "stream_options": g("stream_options"),
        });
        let bytes = serde_json::to_vec(&params).unwrap_or_default();
        let hash = blake3::hash(&bytes);
        hex16(hash.as_bytes())
    }

    /// The cache namespace string: `base ⊕ model ⊕ param-fingerprint`. Uses `:` separators (never the
    /// reserved `\u{1f}`), so `Namespace::new` always accepts it.
    pub fn namespace_string(&self, base: &str) -> String {
        format!(
            "{base}:openai:{}:{}",
            self.model(),
            self.param_fingerprint()
        )
    }
}

/// A parsed Anthropic [Messages API](https://docs.anthropic.com/en/api/messages) request. Same
/// value-based, schema-tolerant approach as [`ChatRequest`], but Anthropic differs from OpenAI in
/// ways that matter to the cache key:
/// - the **system prompt is a top-level `system` field**, not a `messages` entry — it is folded into
///   the canonical prompt so a different system prompt cannot return a wrong cached answer;
/// - decode params are a different set (`top_k`, `stop_sequences` not `stop`, required `max_tokens`,
///   `thinking`), and there is no `n`;
/// - usage is reported as `input_tokens` + `output_tokens`, with no `total_tokens`.
pub struct MessagesRequest {
    pub raw: Value,
}

impl MessagesRequest {
    pub fn parse(bytes: &[u8]) -> Result<Self, serde_json::Error> {
        Ok(Self {
            raw: serde_json::from_slice(bytes)?,
        })
    }

    pub fn model(&self) -> &str {
        self.raw
            .get("model")
            .and_then(Value::as_str)
            .unwrap_or("unknown")
    }

    pub fn is_stream(&self) -> bool {
        self.raw
            .get("stream")
            .and_then(Value::as_bool)
            .unwrap_or(false)
    }

    /// Anthropic's documented default temperature is 1.0 when the field is absent.
    pub fn temperature(&self) -> f64 {
        self.raw
            .get("temperature")
            .and_then(Value::as_f64)
            .unwrap_or(1.0)
    }

    /// True if the request defines tools. Caching tool-call requests is bypassed in v1 — reassembling
    /// tool-argument JSON from a cached reply is a severe failure mode (PLAN.md §3-OSS).
    pub fn has_tools(&self) -> bool {
        self.raw
            .get("tools")
            .is_some_and(|v| v.as_array().map_or(!v.is_null(), |a| !a.is_empty()))
    }

    /// Why this request must skip the cache entirely, or `None` to cache it. Anthropic has no `n`
    /// (one completion per request), so the OpenAI `n>1` bypass does not apply here. Streaming is not
    /// a bypass — it is cached as raw SSE and replayed as a stream (see `ChatRequest::bypass_reason`).
    pub fn bypass_reason(&self, max_temperature: f64) -> Option<&'static str> {
        if self.has_tools() {
            return Some("tools");
        }
        if self.temperature() > max_temperature {
            return Some("temperature");
        }
        None
    }

    /// The canonical conversation string that gets embedded and exact-hashed: the top-level `system`
    /// field *and* the full `messages` array, serialized together as JSON. Folding `system` in (it is
    /// not a `messages` entry on Anthropic) means a different system prompt yields a different key, so
    /// it can never return a wrong cached answer; JSON serialization keeps message boundaries
    /// unambiguous.
    pub fn canonical_prompt(&self) -> String {
        let payload = json!({
            "system": self.raw.get("system").cloned().unwrap_or(Value::Null),
            "messages": self.raw.get("messages").cloned().unwrap_or(Value::Null),
        });
        serde_json::to_string(&payload).unwrap_or_default()
    }

    /// 16-hex-char fingerprint of the answer-affecting decode params (NOT the messages). Anthropic's
    /// set differs from OpenAI's: `top_k` and `stop_sequences` exist, `max_tokens` is required and
    /// caps output length (so it is answer-affecting), and `thinking` changes the output.
    fn param_fingerprint(&self) -> String {
        let g = |k: &str| self.raw.get(k).cloned().unwrap_or(Value::Null);
        // An omitted temperature is Anthropic's effective default of 1.0 (see `temperature()`); hash
        // it as 1.0 — not `null` — so "absent" and "1.0" share a namespace instead of missing.
        let temperature = self.raw.get("temperature").cloned().unwrap_or(json!(1.0));
        let params = json!({
            "temperature": temperature,
            "top_p": g("top_p"),
            "top_k": g("top_k"),
            "max_tokens": g("max_tokens"),
            "stop_sequences": g("stop_sequences"),
            "tool_choice": g("tool_choice"),
            "service_tier": g("service_tier"),
            "thinking": g("thinking"),
            // See ChatRequest::param_fingerprint — streamed (SSE) and non-streamed (JSON) responses
            // are kept in disjoint namespaces.
            "stream": self.is_stream(),
        });
        let bytes = serde_json::to_vec(&params).unwrap_or_default();
        hex16(blake3::hash(&bytes).as_bytes())
    }

    /// The cache namespace string: `base ⊕ anthropic ⊕ model ⊕ param-fingerprint`. The `anthropic`
    /// segment keeps Anthropic traffic in a partition disjoint from OpenAI's, even if a model name
    /// ever collided.
    pub fn namespace_string(&self, base: &str) -> String {
        format!(
            "{base}:anthropic:{}:{}",
            self.model(),
            self.param_fingerprint()
        )
    }
}

/// Tokens a cache hit did NOT buy from the upstream, split by price tier. Input (prompt) and output
/// (completion) tokens are priced differently by every provider — typically output ≈ 3–5× input — so
/// the savings metric keeps them apart; `total()` re-sums them for the headline figure.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TokenUsage {
    pub input: u64,
    pub output: u64,
}

impl TokenUsage {
    pub fn total(self) -> u64 {
        self.input.saturating_add(self.output)
    }
}

/// The JSON payloads of an SSE body's `data:` events, skipping blanks and the `[DONE]` sentinel.
/// Used to recover usage from a *stored streamed* response, which is raw SSE rather than one JSON
/// object.
fn sse_data_payloads(body: &str) -> impl Iterator<Item = &str> {
    body.lines()
        .filter_map(|l| l.strip_prefix("data:"))
        .map(str::trim)
        .filter(|p| !p.is_empty() && *p != "[DONE]")
}

/// A `usage.<key>` u64 out of an already-parsed usage object, defaulting to 0.
fn usage_u64(usage: &Value, key: &str) -> u64 {
    usage.get(key).and_then(Value::as_u64).unwrap_or(0)
}

/// OpenAI input/output split: `usage.prompt_tokens` and `usage.completion_tokens`.
///
/// Handles **both** shapes a stored completion can take: one JSON object (non-streaming) and raw SSE
/// (a stored streamed response). The SSE arm matters as much as the JSON one — a streamed hit is
/// still a hit, and reading its usage as zero would silently zero the savings figure for every
/// streaming deployment.
///
/// In an OpenAI stream the usage arrives on a single late chunk (present when the request set
/// `stream_options: {"include_usage": true}`), so the last chunk carrying usage wins. A stream
/// without that option has no usage to read at all and correctly yields zero — the caller
/// distinguishes "no usage reported" from "nothing saved", this function does not.
pub fn openai_usage(body: &str) -> TokenUsage {
    if let Ok(v) = serde_json::from_str::<Value>(body) {
        if let Some(u) = v.get("usage") {
            return TokenUsage {
                input: usage_u64(u, "prompt_tokens"),
                output: usage_u64(u, "completion_tokens"),
            };
        }
    }
    let mut usage = TokenUsage::default();
    for payload in sse_data_payloads(body) {
        if let Ok(v) = serde_json::from_str::<Value>(payload) {
            if let Some(u) = v.get("usage") {
                let chunk = TokenUsage {
                    input: usage_u64(u, "prompt_tokens"),
                    output: usage_u64(u, "completion_tokens"),
                };
                // A `"usage": null` chunk (OpenAI sends one per content delta under
                // include_usage) must not wipe a real reading that already arrived.
                if chunk != TokenUsage::default() {
                    usage = chunk;
                }
            }
        }
    }
    usage
}

/// Anthropic input/output split: `usage.input_tokens` and `usage.output_tokens`.
///
/// Handles both a non-streaming JSON object and a stored streamed (SSE) body. Anthropic splits its
/// stream usage across events — `input_tokens` arrives once on `message_start` (nested under
/// `message.usage`) while the cumulative `output_tokens` grows across `message_delta` events (usage
/// top-level) — so each tier takes the maximum seen rather than the last, which is what makes the
/// running total correct no matter which event the stream ended on.
/// The model that actually answered, as the *response* reported it. This is not the requested
/// model whenever the provider resolves an alias to a dated snapshot (`claude-sonnet-4-5` →
/// `claude-sonnet-4-5-20250929`, `gpt-4o` → `gpt-4o-2024-08-06`) — which is precisely the fact
/// `gen_ai.response.model` exists to carry.
///
/// Both flavours put it at the top level of a JSON response. In a stream OpenAI repeats it on every
/// chunk, and Anthropic carries it once, nested under `message_start`'s `message`.
///
/// `None` when the body never said. Echoing the request back instead would make
/// `gen_ai.response.model` equal to `gen_ai.request.model` by construction, so a reader comparing
/// the two would learn nothing and would conclude no alias was ever resolved.
pub fn response_model(body: &str) -> Option<String> {
    fn pick(v: &Value) -> Option<String> {
        v.get("model")
            .or_else(|| v.get("message").and_then(|m| m.get("model")))
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
    }
    if let Ok(v) = serde_json::from_str::<Value>(body) {
        if let Some(m) = pick(&v) {
            return Some(m);
        }
    }
    sse_data_payloads(body)
        .filter_map(|p| serde_json::from_str::<Value>(p).ok())
        .find_map(|v| pick(&v))
}

pub fn anthropic_usage(body: &str) -> TokenUsage {
    if let Ok(v) = serde_json::from_str::<Value>(body) {
        if let Some(u) = v.get("usage") {
            return TokenUsage {
                input: usage_u64(u, "input_tokens"),
                output: usage_u64(u, "output_tokens"),
            };
        }
    }
    let (mut input, mut output) = (0u64, 0u64);
    for payload in sse_data_payloads(body) {
        let Ok(v) = serde_json::from_str::<Value>(payload) else {
            continue;
        };
        for usage in [
            v.get("usage"),
            v.get("message").and_then(|m| m.get("usage")),
        ]
        .into_iter()
        .flatten()
        {
            input = input.max(usage_u64(usage, "input_tokens"));
            output = output.max(usage_u64(usage, "output_tokens"));
        }
    }
    TokenUsage { input, output }
}

/// Total tokens in an OpenAI-shaped body (JSON or stored SSE), for callers that do not need the
/// price-tier split.
///
/// The provider's own `usage.total_tokens` wins wherever the body states one — a body may report a
/// total without the prompt/completion split, and summing a split that is not there would read as
/// zero. Only when no total is stated does this fall back to summing [`openai_usage`].
pub fn total_tokens(body: &str) -> u64 {
    let stated = |v: &Value| {
        v.get("usage")
            .and_then(|u| u.get("total_tokens"))
            .and_then(Value::as_u64)
    };
    if let Ok(v) = serde_json::from_str::<Value>(body) {
        if let Some(t) = stated(&v) {
            return t;
        }
    }
    // Streamed: the last chunk stating a total wins, matching openai_usage's last-chunk rule.
    let mut total = 0;
    for payload in sse_data_payloads(body) {
        if let Ok(v) = serde_json::from_str::<Value>(payload) {
            if let Some(t) = stated(&v) {
                total = t;
            }
        }
    }
    if total > 0 {
        return total;
    }
    openai_usage(body).total()
}

/// Total tokens in an Anthropic-shaped body (JSON or stored SSE), for callers that do not need the
/// price-tier split. Anthropic reports `input_tokens`/`output_tokens` separately with no
/// `total_tokens`, so the total is simply their sum — see [`anthropic_usage`] for how each tier is
/// recovered from the two events that carry it.
pub fn anthropic_total_tokens(body: &str) -> u64 {
    anthropic_usage(body).total()
}

/// True iff a captured OpenAI SSE stream completed *cleanly* and is therefore safe to cache
/// (PLAN.md §T4 / §3-OSS streaming rules). The binding signal is the protocol terminal sentinel
/// `data: [DONE]`: a graceful upstream half-close that truncates the stream mid-flight never sends
/// it, so its absence means "do not store". As a second guard we reject any `finish_reason` other
/// than `"stop"` — a `length` (max-tokens) truncation, a `content_filter` cut, or a `tool_calls`
/// turn must not be replayed as a complete answer. A missing `finish_reason` chunk is allowed (the
/// `[DONE]` is the authoritative terminator and a `usage` chunk is not assumed to be present).
pub fn openai_stream_complete(body: &str) -> bool {
    let saw_done = body
        .lines()
        .filter_map(|l| l.strip_prefix("data:"))
        .any(|p| p.trim() == "[DONE]");
    if !saw_done {
        return false;
    }
    for payload in sse_data_payloads(body) {
        if let Ok(v) = serde_json::from_str::<Value>(payload) {
            for choice in v
                .get("choices")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                if let Some(reason) = choice.get("finish_reason").and_then(Value::as_str) {
                    if reason != "stop" {
                        return false;
                    }
                }
            }
        }
    }
    true
}

/// True iff a captured Anthropic SSE stream completed *cleanly* and is safe to cache. The terminal
/// signal is the `message_stop` event; without it the stream was cut off mid-flight. As a second
/// guard we reject a `message_delta` whose `stop_reason` is anything other than a natural stop
/// (`end_turn`/`stop_sequence`) — a `max_tokens` truncation or a `tool_use` turn must not be cached.
pub fn anthropic_stream_complete(body: &str) -> bool {
    let mut saw_stop = false;
    for payload in sse_data_payloads(body) {
        let Ok(v) = serde_json::from_str::<Value>(payload) else {
            continue;
        };
        match v.get("type").and_then(Value::as_str) {
            Some("message_stop") => saw_stop = true,
            Some("message_delta") => {
                if let Some(reason) = v
                    .get("delta")
                    .and_then(|d| d.get("stop_reason"))
                    .and_then(Value::as_str)
                {
                    if reason != "end_turn" && reason != "stop_sequence" {
                        return false;
                    }
                }
            }
            _ => {}
        }
    }
    saw_stop
}

fn hex16(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(16);
    for b in bytes.iter().take(8) {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(s: &str) -> ChatRequest {
        ChatRequest::parse(s.as_bytes()).unwrap()
    }

    #[test]
    fn bypasses_tools_and_high_temp_but_caches_stream() {
        // Streaming is cached now (folded into the namespace), not bypassed.
        assert_eq!(
            req(r#"{"model":"m","stream":true,"temperature":0,"messages":[]}"#).bypass_reason(1.0),
            None
        );
        assert_eq!(
            req(r#"{"model":"m","tools":[{"x":1}],"messages":[]}"#).bypass_reason(1.0),
            Some("tools")
        );
        assert_eq!(
            req(r#"{"model":"m","n":3,"messages":[]}"#).bypass_reason(1.0),
            Some("n>1")
        );
        assert_eq!(
            req(r#"{"model":"m","temperature":1.5,"messages":[]}"#).bypass_reason(1.0),
            Some("temperature")
        );
        assert_eq!(
            req(r#"{"model":"m","temperature":0.2,"messages":[]}"#).bypass_reason(1.0),
            None
        );
        // Empty tools array is not a tool request.
        assert_eq!(
            req(r#"{"model":"m","tools":[],"messages":[]}"#).bypass_reason(1.0),
            None
        );
    }

    #[test]
    fn stream_is_folded_into_the_namespace() {
        let base = "default";
        let streamed = req(
            r#"{"model":"m","temperature":0,"stream":true,"messages":[{"role":"user","content":"hi"}]}"#,
        );
        let plain =
            req(r#"{"model":"m","temperature":0,"messages":[{"role":"user","content":"hi"}]}"#);
        // Same model/params/messages but different delivery → different namespace, so a streamed
        // (SSE) entry can never be replayed to a non-streaming caller as JSON, or vice versa.
        assert_ne!(
            streamed.namespace_string(base),
            plain.namespace_string(base)
        );
    }

    #[test]
    fn stream_options_are_folded_into_the_namespace() {
        let base = "default";
        // Two streamed requests differing only in `stream_options` (which changes the SSE payload,
        // e.g. include_usage) must not share a cached reply.
        let with_usage = req(
            r#"{"model":"m","temperature":0,"stream":true,"stream_options":{"include_usage":true},"messages":[{"role":"user","content":"hi"}]}"#,
        );
        let without = req(
            r#"{"model":"m","temperature":0,"stream":true,"messages":[{"role":"user","content":"hi"}]}"#,
        );
        assert_ne!(
            with_usage.namespace_string(base),
            without.namespace_string(base)
        );
    }

    #[test]
    fn namespace_separates_model_and_params_but_not_message_text() {
        let base = "default";
        let a = req(
            r#"{"model":"gpt-4o","temperature":0,"messages":[{"role":"user","content":"hi there"}]}"#,
        );
        let b = req(
            r#"{"model":"gpt-4o","temperature":0,"messages":[{"role":"user","content":"hello!"}]}"#,
        );
        let c = req(
            r#"{"model":"gpt-4o","temperature":0.7,"messages":[{"role":"user","content":"hi there"}]}"#,
        );
        let d = req(
            r#"{"model":"gpt-3.5","temperature":0,"messages":[{"role":"user","content":"hi there"}]}"#,
        );

        // Same model+params → same namespace regardless of message text (semantic match is possible).
        assert_eq!(a.namespace_string(base), b.namespace_string(base));
        // Different temperature → different namespace (isolated).
        assert_ne!(a.namespace_string(base), c.namespace_string(base));
        // Different model → different namespace (isolated).
        assert_ne!(a.namespace_string(base), d.namespace_string(base));
    }

    #[test]
    fn canonical_prompt_includes_all_turns_and_roles() {
        let r = req(
            r#"{"model":"m","messages":[{"role":"system","content":"be brief"},{"role":"user","content":"hi"}]}"#,
        );
        // Canonical JSON of the messages array: parse it back to prove every role/turn survives,
        // independent of how serialization happens to order object keys.
        let cp = r.canonical_prompt();
        let parsed: Value = serde_json::from_str(&cp).expect("canonical prompt is valid JSON");
        assert_eq!(parsed, *r.raw.get("messages").unwrap());
        assert_eq!(parsed.as_array().unwrap().len(), 2);
    }

    // Two different message boundaries that flatten to the SAME "{role}: {content}\n" string must now
    // produce different canonical prompts — the collision the JSON form exists to prevent.
    #[test]
    fn canonical_prompt_resists_delimiter_collision() {
        let a = req(r#"{"messages":[{"role":"user","content":"hi\nuser: bye"}]}"#);
        let b =
            req(r#"{"messages":[{"role":"user","content":"hi"},{"role":"user","content":"bye"}]}"#);
        assert_ne!(a.canonical_prompt(), b.canonical_prompt());
    }

    #[test]
    fn total_tokens_extracted() {
        assert_eq!(total_tokens(r#"{"usage":{"total_tokens":42}}"#), 42);
        assert_eq!(total_tokens(r#"{"no":"usage"}"#), 0);
    }

    #[test]
    fn usage_split_recovered_from_a_stored_openai_stream() {
        // A stored streamed response is raw SSE, not one JSON object. Reading it as JSON yields
        // nothing, which would zero the savings figure for every streaming deployment.
        let sse = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}],\"usage\":null}\n\n",
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":1000,\"completion_tokens\":2000,\"total_tokens\":3000}}\n\n",
            "data: [DONE]\n\n",
        );
        assert_eq!(
            openai_usage(sse),
            TokenUsage {
                input: 1000,
                output: 2000
            }
        );
        // The per-delta `"usage": null` chunks must not wipe the real reading.
        assert_eq!(total_tokens(sse), 3000);
        // The non-streaming shape still works unchanged.
        assert_eq!(
            openai_usage(r#"{"usage":{"prompt_tokens":4,"completion_tokens":7}}"#),
            TokenUsage {
                input: 4,
                output: 7
            }
        );
        // A stream with no usage at all reports zero — "not reported", which the caller flags.
        let none = "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\ndata: [DONE]\n\n";
        assert_eq!(openai_usage(none), TokenUsage::default());
    }

    #[test]
    fn usage_split_recovered_from_a_stored_anthropic_stream() {
        // Anthropic splits usage across events: input_tokens once on message_start (nested under
        // `message`), cumulative output_tokens across message_delta events (top-level).
        let sse = concat!(
            "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":1000,\"output_tokens\":1}}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"delta\":{\"text\":\"hi\"}}\n\n",
            "data: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":1200}}\n\n",
            "data: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":2000}}\n\n",
            "data: {\"type\":\"message_stop\"}\n\n",
        );
        // Max per tier, not last: input survives the later events that omit it, and the cumulative
        // output takes its highest reading regardless of which event the stream ended on.
        assert_eq!(
            anthropic_usage(sse),
            TokenUsage {
                input: 1000,
                output: 2000
            }
        );
        assert_eq!(anthropic_total_tokens(sse), 3000);
    }

    #[test]
    fn total_tokens_from_streamed_sse() {
        // `stream_options.include_usage` puts a usage chunk before `[DONE]`.
        let sse = "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n\
                   data: {\"choices\":[],\"usage\":{\"total_tokens\":42}}\n\n\
                   data: [DONE]\n\n";
        assert_eq!(total_tokens(sse), 42);
        // A stream without usage has nothing to count.
        let no_usage = "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\ndata: [DONE]\n\n";
        assert_eq!(total_tokens(no_usage), 0);
    }

    // ----- Anthropic Messages API mapping -----

    fn amsg(s: &str) -> MessagesRequest {
        MessagesRequest::parse(s.as_bytes()).unwrap()
    }

    #[test]
    fn anthropic_bypasses_tools_and_high_temp_but_caches_stream() {
        // Streaming is cached (folded into the namespace), not bypassed.
        assert_eq!(
            amsg(r#"{"model":"m","max_tokens":16,"stream":true,"temperature":0,"messages":[]}"#)
                .bypass_reason(1.0),
            None
        );
        assert_eq!(
            amsg(r#"{"model":"m","max_tokens":16,"tools":[{"name":"x","input_schema":{}}],"messages":[]}"#)
                .bypass_reason(1.0),
            Some("tools")
        );
        assert_eq!(
            amsg(r#"{"model":"m","max_tokens":16,"temperature":1.5,"messages":[]}"#)
                .bypass_reason(1.0),
            Some("temperature")
        );
        assert_eq!(
            amsg(r#"{"model":"m","max_tokens":16,"temperature":0.2,"messages":[]}"#)
                .bypass_reason(1.0),
            None
        );
        // Empty tools array is not a tool request.
        assert_eq!(
            amsg(r#"{"model":"m","max_tokens":16,"tools":[],"messages":[]}"#).bypass_reason(1.0),
            None
        );
    }

    #[test]
    fn anthropic_namespace_separates_model_and_params_but_not_message_text() {
        let base = "default";
        let a = amsg(
            r#"{"model":"claude","max_tokens":16,"temperature":0,"messages":[{"role":"user","content":"hi there"}]}"#,
        );
        let b = amsg(
            r#"{"model":"claude","max_tokens":16,"temperature":0,"messages":[{"role":"user","content":"hello!"}]}"#,
        );
        let c = amsg(
            r#"{"model":"claude","max_tokens":16,"temperature":0.7,"messages":[{"role":"user","content":"hi there"}]}"#,
        );
        let d = amsg(
            r#"{"model":"claude-other","max_tokens":16,"temperature":0,"messages":[{"role":"user","content":"hi there"}]}"#,
        );
        assert_eq!(a.namespace_string(base), b.namespace_string(base));
        assert_ne!(a.namespace_string(base), c.namespace_string(base));
        assert_ne!(a.namespace_string(base), d.namespace_string(base));
        // And never collides with the OpenAI partition for the same model/params.
        let openai = req(
            r#"{"model":"claude","temperature":0,"messages":[{"role":"user","content":"hi there"}]}"#,
        );
        assert_ne!(a.namespace_string(base), openai.namespace_string(base));
    }

    #[test]
    fn anthropic_stream_is_folded_into_the_namespace() {
        let base = "default";
        let streamed = amsg(
            r#"{"model":"claude","max_tokens":16,"temperature":0,"stream":true,"messages":[{"role":"user","content":"hi"}]}"#,
        );
        let plain = amsg(
            r#"{"model":"claude","max_tokens":16,"temperature":0,"messages":[{"role":"user","content":"hi"}]}"#,
        );
        assert_ne!(
            streamed.namespace_string(base),
            plain.namespace_string(base)
        );
    }

    #[test]
    fn anthropic_system_changes_prompt_but_not_namespace() {
        let base = "default";
        let a = amsg(
            r#"{"model":"claude","max_tokens":16,"temperature":0,"system":"be terse","messages":[{"role":"user","content":"hi"}]}"#,
        );
        let b = amsg(
            r#"{"model":"claude","max_tokens":16,"temperature":0,"system":"be verbose","messages":[{"role":"user","content":"hi"}]}"#,
        );
        // System is answer-affecting content, not a decode param: the namespace is unchanged, but the
        // canonical prompt differs so a different system prompt can never exact-hit the wrong answer.
        assert_eq!(a.namespace_string(base), b.namespace_string(base));
        assert_ne!(a.canonical_prompt(), b.canonical_prompt());
    }

    #[test]
    fn anthropic_canonical_prompt_carries_system_and_messages() {
        let r = amsg(
            r#"{"model":"m","max_tokens":16,"system":"be brief","messages":[{"role":"user","content":"hi"}]}"#,
        );
        let cp: Value = serde_json::from_str(&r.canonical_prompt()).expect("valid JSON");
        assert_eq!(cp.get("system").unwrap(), "be brief");
        assert_eq!(cp.get("messages").unwrap(), r.raw.get("messages").unwrap());
    }

    #[test]
    fn openai_stream_complete_requires_done_and_clean_finish() {
        // A clean stream: content delta + [DONE] (no finish_reason chunk — allowed).
        let clean = "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\ndata: [DONE]\n\n";
        assert!(openai_stream_complete(clean));
        // Explicit finish_reason "stop" before [DONE] is also clean.
        let clean_stop = "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n\
                          data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";
        assert!(openai_stream_complete(clean_stop));
        // Truncated: a graceful half-close that never sent [DONE] must NOT be cached.
        let truncated = "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n";
        assert!(!openai_stream_complete(truncated));
        // max_tokens truncation: [DONE] is present but finish_reason is "length" → not cacheable.
        let length = "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n\
                      data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"length\"}]}\n\ndata: [DONE]\n\n";
        assert!(!openai_stream_complete(length));
        // Empty body is never a complete stream.
        assert!(!openai_stream_complete(""));
    }

    #[test]
    fn anthropic_stream_complete_requires_message_stop_and_clean_reason() {
        // Clean: a text delta then message_stop (no stop_reason delta — allowed).
        let clean = "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n\
                     event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n";
        assert!(anthropic_stream_complete(clean));
        // Clean with an explicit end_turn stop_reason on message_delta.
        let clean_end = "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"}}\n\n\
                         event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n";
        assert!(anthropic_stream_complete(clean_end));
        // Truncated: no message_stop → not cacheable.
        let truncated = "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n";
        assert!(!anthropic_stream_complete(truncated));
        // max_tokens truncation: message_stop present but stop_reason is max_tokens → not cacheable.
        let max_tokens = "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"max_tokens\"}}\n\n\
                          event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n";
        assert!(!anthropic_stream_complete(max_tokens));
    }

    #[test]
    fn anthropic_total_tokens_sums_input_and_output() {
        assert_eq!(
            anthropic_total_tokens(r#"{"usage":{"input_tokens":5,"output_tokens":7}}"#),
            12
        );
        assert_eq!(anthropic_total_tokens(r#"{"no":"usage"}"#), 0);
    }

    #[test]
    fn the_response_model_is_read_from_the_response_not_echoed_back() {
        // The case the attribute exists for: an alias resolves upstream to a dated snapshot.
        assert_eq!(
            response_model(r#"{"id":"c1","model":"gpt-4o-2024-08-06","choices":[]}"#).as_deref(),
            Some("gpt-4o-2024-08-06")
        );
        assert_eq!(
            response_model(r#"{"id":"m1","type":"message","model":"claude-x-20260101"}"#)
                .as_deref(),
            Some("claude-x-20260101")
        );
        // Streamed: OpenAI repeats it on each chunk...
        assert_eq!(
            response_model(
                "data: {\"model\":\"gpt-4o-2024-08-06\",\"choices\":[]}\n\ndata: [DONE]\n\n"
            )
            .as_deref(),
            Some("gpt-4o-2024-08-06")
        );
        // ...while Anthropic carries it once, nested inside `message_start`.
        assert_eq!(
            response_model(concat!(
                "event: message_start\n",
                "data: {\"type\":\"message_start\",\"message\":{\"model\":\"claude-x-20260101\"}}\n\n",
                "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
            ))
            .as_deref(),
            Some("claude-x-20260101")
        );
        // Absent rather than guessed when the body never said, and an empty string is not a name.
        assert_eq!(response_model(r#"{"id":"c1","choices":[]}"#), None);
        assert_eq!(response_model(r#"{"model":""}"#), None);
        assert_eq!(response_model("not json at all"), None);
    }

    #[test]
    fn anthropic_total_tokens_from_streamed_sse() {
        // input_tokens on message_start (under `message`), cumulative output_tokens on message_delta.
        let sse = "event: message_start\n\
                   data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":5,\"output_tokens\":1}}}\n\n\
                   event: message_delta\n\
                   data: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":7}}\n\n\
                   event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n";
        // input=5, output=max(1,7)=7 → 12
        assert_eq!(anthropic_total_tokens(sse), 12);
    }
}
