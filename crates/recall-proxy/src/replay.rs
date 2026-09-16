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
//! hit/miss/bypass. The authoritative savings accounting is read back from the proxy's `/metrics`
//! as a **delta** — a snapshot before the run subtracted from one after it — so replaying twice
//! against the same long-lived proxy measures the second run, not both runs added together.
//!
//! The dollar figure comes from the proxy's own `recall_cost_saved_usd_total`, which prices each hit
//! per model against the operator's `[pricing.<model>]` table. Replay deliberately does not re-price
//! anything itself: it only ever sees *aggregate* token counts, so any pricing it did would be one
//! blended rate across every model in the log — precisely the mis-statement the per-model table
//! exists to avoid.
//!
//! **False-hit sampling** (`verify_rate > 0`): for a sampled fraction of *hits*, the same body is sent
//! straight to the upstream and the served (cached) answer is compared to a fresh one. A mismatch is a
//! candidate false hit — a wrong answer the cache served. This only means anything at `temperature: 0`
//! (a sampled model disagrees with itself), so the caller is responsible for pinning it.

use serde_json::Value;

use crate::scrape;

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
/// `cost_saved_*` fields are the proxy's own `/metrics` counters, differenced across the run.
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
    /// Upstream spend this run avoided, in **integer micro-USD**, priced per model by the proxy from
    /// its `[pricing.<model>]` table. `0` when the proxy has no price book configured — read it
    /// together with `unpriced_tokens_saved` to tell "no rates configured" from "nothing saved".
    pub cost_saved_micros: u64,
    /// Tokens this run saved on models the proxy had no price rule for. They contribute `$0` to
    /// `cost_saved_micros`; a non-zero value here means the dollar figure understates the run.
    pub unpriced_tokens_saved: u64,
    /// Hits in this run whose stored completion reported no token usage at all, so they credited
    /// nothing to any counter above. The other way the savings figure can understate a run.
    pub hits_usage_unknown: u64,
    /// Hit rate over this run's own cacheable requests (`hits / (hits + misses)`), computed from the
    /// responses this run saw rather than read from the proxy's lifetime `recall_hit_ratio` gauge —
    /// which would fold in every request the proxy served before the replay started.
    pub hit_ratio: f64,
    /// Hits sampled for upstream verification.
    pub verified: u64,
    /// Sampled hits whose cached answer differed from a fresh upstream answer (candidate false hits).
    pub verify_mismatch: u64,
    /// Sampled hits the verifier could not fetch an upstream answer for (not counted as mismatches).
    pub verify_unchecked: u64,
}

impl ReplayReport {
    /// Dollars saved, gross — the run's `cost_saved_micros` as USD. Float only here, at the edge.
    pub fn cost_saved_usd(&self) -> f64 {
        self.cost_saved_micros as f64 / 1e6
    }

    /// The measured share of sampled hits whose cached answer disagreed with a fresh upstream one,
    /// or `None` when nothing was sampled (`--verify-sample 0`). This is a *sample*, not the true
    /// false-hit rate, and only means anything at `temperature: 0`.
    pub fn sampled_false_hit_rate(&self) -> Option<f64> {
        (self.verified > 0).then(|| self.verify_mismatch as f64 / self.verified as f64)
    }

    /// Dollars saved, **discounted by the sampled false-hit rate** — the savings figure minus the
    /// share of hits that were measurably wrong. `None` when nothing was sampled, because there is
    /// then no measured error rate to discount by and inventing one would be worse than abstaining.
    ///
    /// A wrong hit is not a saving: it bought a wrong answer with the money it did not spend. An ROI
    /// number quoted gross when a measured error rate exists is the kind that gets a cache ripped
    /// back out a quarter later.
    pub fn cost_saved_usd_discounted(&self) -> Option<f64> {
        self.sampled_false_hit_rate()
            .map(|fhr| self.cost_saved_usd() * (1.0 - fhr))
    }
}

/// Replay `lines` (JSONL request bodies) against `opts.target`, returning the measured report.
pub async fn run_replay(lines: Vec<String>, opts: ReplayOpts) -> Result<ReplayReport, String> {
    let client = reqwest::Client::new();
    let base = opts.target.trim_end_matches('/').to_string();
    let mut rep = ReplayReport::default();

    // Baseline the proxy's lifetime counters so the report measures THIS run. Without it, a second
    // replay against the same `serve` reports both runs' savings and reads as a doubling.
    let before = scrape::fetch(&client, &base).await.unwrap_or_default();

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

    // Authoritative accounting: the proxy's own counters, differenced across the run. `saturating_sub`
    // so a proxy restarted mid-replay (counters back to 0) reports 0 rather than an underflowed
    // astronomical saving.
    if let Ok(after) = scrape::fetch(&client, &base).await {
        let run = after.since(&before);
        rep.tokens_saved = run.tokens_saved;
        rep.input_tokens_saved = run.input_tokens_saved;
        rep.output_tokens_saved = run.output_tokens_saved;
        rep.cost_saved_micros = run.cost_saved_micros;
        rep.unpriced_tokens_saved = run.unpriced_tokens_saved;
        rep.hits_usage_unknown = run.hits_usage_unknown;
    }
    // This run's own hit rate over its own cacheable traffic — not the proxy's lifetime gauge.
    let cacheable = rep.hits + rep.misses;
    rep.hit_ratio = if cacheable > 0 {
        rep.hits as f64 / cacheable as f64
    } else {
        0.0
    };
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
    fn discounted_savings_needs_a_measured_error_rate() {
        let mut rep = ReplayReport {
            cost_saved_micros: 10_000_000, // $10.00
            ..Default::default()
        };
        // Nothing sampled → no measured false-hit rate → abstain rather than quote a gross number
        // as if it were verified.
        assert_eq!(rep.sampled_false_hit_rate(), None);
        assert_eq!(rep.cost_saved_usd_discounted(), None);
        assert!((rep.cost_saved_usd() - 10.0).abs() < 1e-9);

        // 2 of 100 sampled hits disagreed with a fresh upstream answer → discount by 2%.
        rep.verified = 100;
        rep.verify_mismatch = 2;
        assert!((rep.sampled_false_hit_rate().unwrap() - 0.02).abs() < 1e-9);
        assert!((rep.cost_saved_usd_discounted().unwrap() - 9.80).abs() < 1e-9);
    }
}
