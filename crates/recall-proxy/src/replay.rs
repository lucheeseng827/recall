//! Traffic-replay harness: drive a running `recall serve` with a request log and report the savings
//! it would have produced. This is how the "we saved $X" claim is *validated* rather than asserted —
//! synthetic FAQ benches prove the loop works; replaying real (or representative) traffic measures the
//! hit-rate and token savings on a workload that matters.
//!
//! Input is JSONL, one request per line, in either shape:
//!   - a raw request body: `{"model":"gpt-4o","messages":[...]}`
//!   - a wrapper choosing the endpoint: `{"path":"/v1/messages","body":{...}}`
//!
//! Each line is POSTed to the target proxy; the `x-recall-cache` response header classifies it
//! hit/miss/bypass. After the run the authoritative token counters are read back from `/metrics`.
//!
//! **False-hit sampling** (`verify_rate > 0`): for a sampled fraction of *hits*, the same body is sent
//! straight to the upstream and the served (cached) answer is compared to a fresh one. A mismatch is a
//! candidate false hit — a wrong answer the cache served. This only means anything at `temperature: 0`
//! (a sampled model disagrees with itself), so the caller is responsible for pinning it.

use serde_json::Value;

/// Knobs for one replay run.
pub struct ReplayOpts {
    /// Base URL of the running `recall serve`, e.g. `http://127.0.0.1:8080`.
    pub target: String,
    /// Endpoint used for lines that don't carry their own `path`.
    pub default_path: String,
    /// Fraction of hits to verify against the upstream (0.0 = off).
    pub verify_rate: f64,
    /// Upstream base URL for verification (required when `verify_rate > 0`).
    pub upstream: Option<String>,
    /// Upstream API key for verification (Bearer for OpenAI, `x-api-key` for Anthropic).
    pub upstream_key: Option<String>,
    /// `anthropic-version` header sent on Anthropic verification calls.
    pub anthropic_version: String,
}

/// What a replay run measured. Counts are from the response headers; the `*_tokens_saved` and
/// `hit_ratio` fields are read back from the proxy's `/metrics` (the authoritative accounting).
#[derive(Debug, Default, Clone)]
pub struct ReplayReport {
    pub requests: u64,
    pub hits: u64,
    pub misses: u64,
    pub bypass: u64,
    pub errors: u64,
    pub tokens_saved: u64,
    pub input_tokens_saved: u64,
    pub output_tokens_saved: u64,
    pub hit_ratio: f64,
    /// Hits sampled for upstream verification.
    pub verified: u64,
    /// Sampled hits whose cached answer differed from a fresh upstream answer (candidate false hits).
    pub verify_mismatch: u64,
    /// Sampled hits the verifier could not fetch an upstream answer for (not counted as mismatches).
    pub verify_unchecked: u64,
}

/// Replay `lines` (JSONL request bodies) against `opts.target`, returning the measured report.
pub async fn run_replay(lines: Vec<String>, opts: ReplayOpts) -> Result<ReplayReport, String> {
    let client = reqwest::Client::new();
    let base = opts.target.trim_end_matches('/').to_string();
    let mut rep = ReplayReport::default();

    // Deterministic sampling: verify every Nth hit, N = round(1/rate). No RNG dependency, and a fixed
    // stride is reproducible across runs of the same log.
    let verify_every = if opts.verify_rate > 0.0 {
        (1.0 / opts.verify_rate).round().max(1.0) as u64
    } else {
        0
    };

    for line in lines {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Ok(parsed) = serde_json::from_str::<Value>(line) else {
            rep.errors += 1;
            continue;
        };
        // Wrapper {path, body} chooses the endpoint; otherwise the whole line is the body.
        let (path, body) = match parsed.get("body") {
            Some(b) => (
                parsed
                    .get("path")
                    .and_then(Value::as_str)
                    .unwrap_or(&opts.default_path)
                    .to_string(),
                b.clone(),
            ),
            None => (opts.default_path.clone(), parsed.clone()),
        };

        if !path.starts_with('/') || path.contains("..") {
            rep.errors += 1;
            continue;
        }
        rep.requests += 1;
        let url = format!("{base}{path}");
        let resp = match client.post(&url).json(&body).send().await {
            Ok(r) => r,
            Err(_) => {
                rep.errors += 1;
                continue;
            }
        };
        let status = resp.status();
        let cache_hdr = resp
            .headers()
            .get("x-recall-cache")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();
        // Cap body to 1 MiB — large upstream error bodies must not OOM the replayer.
        let served = resp
            .bytes()
            .await
            .map(|b| {
                let cap = b.len().min(1 << 20);
                String::from_utf8_lossy(&b[..cap]).into_owned()
            })
            .unwrap_or_default();
        if !status.is_success() {
            rep.errors += 1;
            continue;
        }

        match cache_hdr.as_str() {
            "hit" => {
                rep.hits += 1;
                if verify_every > 0 && rep.hits % verify_every == 0 {
                    rep.verified += 1;
                    match verify_against_upstream(&client, &path, &body, &served, &opts).await {
                        Some(true) => {}                         // matched fresh upstream
                        Some(false) => rep.verify_mismatch += 1, // candidate false hit
                        None => rep.verify_unchecked += 1,       // couldn't fetch upstream
                    }
                }
            }
            "miss" => rep.misses += 1,
            "bypass" => rep.bypass += 1,
            _ => rep.errors += 1,
        }
    }

    // Authoritative token accounting: read the counters the proxy itself kept.
    if let Ok(m) = client.get(format!("{base}/metrics")).send().await {
        if let Ok(body) = m.text().await {
            rep.tokens_saved = metric_u64(&body, "recall_tokens_saved_total");
            rep.input_tokens_saved = metric_u64(&body, "recall_input_tokens_saved_total");
            rep.output_tokens_saved = metric_u64(&body, "recall_output_tokens_saved_total");
            rep.hit_ratio = metric_f64(&body, "recall_hit_ratio");
        }
    }
    Ok(rep)
}

/// Fetch a fresh upstream answer for `body` and compare its assistant text to the `served` (cached)
/// one. `Some(true)` = match, `Some(false)` = mismatch (candidate false hit), `None` = could not
/// fetch an upstream answer to compare against.
async fn verify_against_upstream(
    client: &reqwest::Client,
    path: &str,
    body: &Value,
    served: &str,
    opts: &ReplayOpts,
) -> Option<bool> {
    let base = opts.upstream.as_ref()?.trim_end_matches('/');
    let url = format!("{base}{path}");
    let mut req = client.post(&url).json(body);
    // Anthropic endpoints authenticate with x-api-key + a version header; OpenAI with a Bearer token.
    if path.contains("messages") {
        if let Some(k) = &opts.upstream_key {
            req = req.header("x-api-key", k);
        }
        req = req.header("anthropic-version", &opts.anthropic_version);
    } else if let Some(k) = &opts.upstream_key {
        req = req.header("authorization", format!("Bearer {k}"));
    }
    let fresh = req.send().await.ok()?.text().await.ok()?;
    Some(assistant_text(served) == assistant_text(&fresh))
}

/// Extract the assistant's text from an OpenAI- or Anthropic-shaped response body, for comparison.
fn assistant_text(body: &str) -> String {
    let v: Value = serde_json::from_str(body).unwrap_or(Value::Null);
    if let Some(c) = v
        .pointer("/choices/0/message/content")
        .and_then(Value::as_str)
    {
        return c.to_string();
    }
    if let Some(arr) = v.get("content").and_then(Value::as_array) {
        return arr
            .iter()
            .filter_map(|b| b.get("text").and_then(Value::as_str))
            .collect::<String>();
    }
    String::new()
}

/// Read the value from a `<name> <value>` or `<name>{...} <value>` Prometheus text line.
/// The label-aware variant handles future metric label additions without silently returning 0.
fn metric_line<'a>(body: &'a str, name: &str) -> Option<&'a str> {
    body.lines()
        .find_map(|l| {
            // No-label form: "recall_tokens_saved_total 1100"
            l.strip_prefix(&format!("{name} ")).or_else(|| {
                // Labeled form: "recall_tokens_saved_total{host=\"x\"} 1100"
                l.strip_prefix(&format!("{name}{{"))
                    .and_then(|rest| rest.split_once("} ").map(|(_, v)| v))
            })
        })
        .map(str::trim)
}

fn metric_u64(body: &str, name: &str) -> u64 {
    metric_line(body, name)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0)
}

fn metric_f64(body: &str, name: &str) -> f64 {
    metric_line(body, name)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assistant_text_handles_both_shapes() {
        let openai = r#"{"choices":[{"message":{"role":"assistant","content":"hi there"}}]}"#;
        assert_eq!(assistant_text(openai), "hi there");
        let anthropic =
            r#"{"content":[{"type":"text","text":"hi "},{"type":"text","text":"there"}]}"#;
        assert_eq!(assistant_text(anthropic), "hi there");
        assert_eq!(assistant_text("not json"), "");
    }

    #[test]
    fn metric_parse_is_exact() {
        let m = "recall_tokens_saved_total 1100000\nrecall_input_tokens_saved_total 1000000\nrecall_hit_ratio 0.4200\n";
        // The shorter name must not match the longer line.
        assert_eq!(metric_u64(m, "recall_tokens_saved_total"), 1_100_000);
        assert_eq!(metric_u64(m, "recall_input_tokens_saved_total"), 1_000_000);
        assert_eq!(metric_u64(m, "recall_missing_total"), 0);
        assert!((metric_f64(m, "recall_hit_ratio") - 0.42).abs() < 1e-9);
    }

    #[test]
    fn metric_parse_handles_labels() {
        // Labeled form must still extract the value correctly.
        let m = "recall_tokens_saved_total{host=\"proxy-1\"} 42\nrecall_input_tokens_saved_total{host=\"proxy-1\"} 30\n";
        assert_eq!(metric_u64(m, "recall_tokens_saved_total"), 42);
        assert_eq!(metric_u64(m, "recall_input_tokens_saved_total"), 30);
        assert_eq!(metric_u64(m, "recall_missing_total"), 0);
    }
}
