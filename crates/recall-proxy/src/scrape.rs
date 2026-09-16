//! Reading a running proxy's `/metrics` back as numbers.
//!
//! `replay` and `status` both need the same thing: fetch the Prometheus text exposition from a live
//! `recall serve` and pull named series out of it. This is that parsing, in one place, so the two
//! cannot drift into disagreeing about what a metric means.
//!
//! It is deliberately a *reader*, not a Prometheus client: no scrape scheduling, no label matching
//! beyond skipping a label set, no histograms. Anything more belongs in the operator's own
//! monitoring stack, which is where these numbers are meant to end up.

use serde::Serialize;

/// The savings and traffic counters as of one scrape. Two of these bracket a `replay` run; one is
/// enough for a `status` snapshot.
///
/// Money is carried as integer micro-USD, matching the rest of recall — the exposition renders USD
/// at six decimals, which round-trips exactly.
#[derive(Debug, Default, Clone, Serialize)]
pub struct Snapshot {
    pub requests: u64,
    pub hits: u64,
    pub misses: u64,
    pub bypass: u64,
    pub upstream_errors: u64,
    pub tokens_saved: u64,
    pub input_tokens_saved: u64,
    pub output_tokens_saved: u64,
    pub cost_saved_micros: u64,
    pub unpriced_tokens_saved: u64,
    pub hits_usage_unknown: u64,
    pub feedback_agree: u64,
    pub feedback_wrong: u64,
    pub stream_not_stored: u64,
    pub cache_entries: u64,
    pub fill_entries: u64,
    pub fill_tokens: u64,
    pub fill_input_tokens: u64,
    pub fill_output_tokens: u64,
    pub fill_cost_micros: u64,
    pub unpriced_fill_tokens: u64,
    pub fill_usage_unknown: u64,
    pub persisted_entries_at_start: u64,
    pub rehydrated_entries: u64,
    /// Unix seconds this proxy process started — the boundary every counter above reset at. What
    /// lets a consumer differencing two snapshots tell a restart from a quiet period.
    pub process_start_unix: u64,
    /// Store backend name from `recall_store_info{backend=…}` (`memory`, `redb`), or empty when the
    /// target predates that series.
    pub store_backend: String,
    /// Whether the store survives a restart, from `recall_store_info{durable=…}`. `false` also for
    /// a target that predates the series — callers wanting to distinguish should check
    /// `store_backend.is_empty()`.
    pub store_durable: bool,
    /// Savings split by cache namespace, from the separate `/v1/cache/namespaces` document (see
    /// [`crate::namespaces`] for why it is not a metric family). Empty when the target predates the
    /// endpoint or it could not be read — the node-level fields above stand on their own, so an
    /// absent breakdown is a missing *detail*, never a failed read.
    pub by_namespace: Vec<crate::namespaces::NamespaceRow>,
    /// Whether [`Self::by_namespace`] was actually read, as opposed to unavailable. An empty
    /// breakdown is ambiguous on its own — a node with no hits yet and a node whose namespaces
    /// document could not be read both show no rows — and the two differ completely when
    /// differencing: rows absent because nothing happened make the delta zero, rows absent because
    /// nobody could read them make every later row look like it arrived in one window. Only
    /// [`Self::since`] needs the distinction, but only a read knows it, so it is recorded here.
    pub by_namespace_available: bool,
    /// Unix seconds the node's persisted ledger began, from `recall_ledger_since_time_seconds`.
    /// `0` — the default, and what an older node reports by omission — means the counters reset
    /// with the process, i.e. `process_start_unix` is the boundary.
    pub ledger_since_unix: u64,
}

impl Snapshot {
    /// Parse a Prometheus text exposition body.
    ///
    /// A series the body does not carry reads 0. That is right for a counter (absent means nothing
    /// happened) but it also means an *older* proxy, which never emitted a series this build knows
    /// about, is indistinguishable from a new one at rest — callers that care should say "0" rather
    /// than assert what it implies.
    pub fn parse(body: &str) -> Self {
        Self {
            // Filled by `fetch` from the separate namespaces document; parsing the metrics text
            // alone never yields rows, and never knows whether any were readable.
            by_namespace: Vec::new(),
            by_namespace_available: false,
            requests: metric_u64(body, "recall_requests_total"),
            hits: metric_u64(body, "recall_hits_total"),
            misses: metric_u64(body, "recall_misses_total"),
            bypass: metric_u64(body, "recall_bypass_total"),
            upstream_errors: metric_u64(body, "recall_upstream_errors_total"),
            tokens_saved: metric_u64(body, "recall_tokens_saved_total"),
            input_tokens_saved: metric_u64(body, "recall_input_tokens_saved_total"),
            output_tokens_saved: metric_u64(body, "recall_output_tokens_saved_total"),
            cost_saved_micros: usd_to_micros(metric_f64(body, "recall_cost_saved_usd_total")),
            unpriced_tokens_saved: metric_u64(body, "recall_unpriced_tokens_saved_total"),
            hits_usage_unknown: metric_u64(body, "recall_hits_usage_unknown_total"),
            feedback_agree: metric_u64(body, "recall_feedback_agree_total"),
            feedback_wrong: metric_u64(body, "recall_feedback_wrong_total"),
            stream_not_stored: metric_u64(body, "recall_stream_not_stored_total"),
            cache_entries: metric_u64(body, "recall_cache_entries"),
            fill_entries: metric_u64(body, "recall_fill_entries_total"),
            fill_tokens: metric_u64(body, "recall_fill_tokens_total"),
            fill_input_tokens: metric_u64(body, "recall_fill_input_tokens_total"),
            fill_output_tokens: metric_u64(body, "recall_fill_output_tokens_total"),
            fill_cost_micros: usd_to_micros(metric_f64(body, "recall_fill_cost_usd_total")),
            unpriced_fill_tokens: metric_u64(body, "recall_unpriced_fill_tokens_total"),
            fill_usage_unknown: metric_u64(body, "recall_fill_usage_unknown_total"),
            persisted_entries_at_start: metric_u64(body, "recall_persisted_entries_at_start"),
            rehydrated_entries: metric_u64(body, "recall_rehydrated_entries"),
            process_start_unix: metric_u64(body, "recall_process_start_time_seconds"),
            ledger_since_unix: metric_u64(body, "recall_ledger_since_time_seconds"),
            store_backend: metric_label(body, "recall_store_info", "backend").unwrap_or_default(),
            store_durable: metric_label(body, "recall_store_info", "durable").as_deref()
                == Some("true"),
        }
    }

    /// The epoch this snapshot's cumulative counters are measured from — what makes two snapshots
    /// differenceable at all. Normally the process start, since the counters reset with the
    /// process; on a node running a durable savings ledger they instead began when that ledger did
    /// and survive restarts, so the ledger epoch is the boundary and a restart mid-window is not a
    /// discontinuity. Two snapshots with different epochs are counting from different zeros.
    fn counter_epoch(&self) -> u64 {
        if self.ledger_since_unix != 0 {
            self.ledger_since_unix
        } else {
            self.process_start_unix
        }
    }

    /// This snapshot minus an earlier one, field by field — what happened between the two scrapes.
    ///
    /// Saturating: a proxy restarted between the scrapes has counters back at zero, and reporting
    /// that as 0 is right where an underflowed astronomical figure would be catastrophic.
    /// `cache_entries` is a gauge, not a counter, so it is taken from the later snapshot as-is
    /// rather than differenced.
    ///
    /// The namespace breakdown differences only when *both* snapshots carry one read from the same
    /// counter epoch. Saturation makes a missing row safe in the node-level fields — they fall to
    /// zero — but a namespace row has no `before` to saturate against, so treating an absent
    /// breakdown as all-zeros would publish each row's whole cumulative history as this window's
    /// work: 100 prior hits, one scrape that could not read the breakdown, then rows at 110 would
    /// claim 110 namespace hits inside a 10-hit window. When the breakdown is not comparable the
    /// delta carries no rows and says so, because a missing split is a gap and a wrong one is a lie.
    pub fn since(&self, before: &Snapshot) -> Snapshot {
        let namespaces_comparable = self.by_namespace_available
            && before.by_namespace_available
            && self.counter_epoch() == before.counter_epoch();
        Snapshot {
            by_namespace: if namespaces_comparable {
                diff_namespaces(&self.by_namespace, &before.by_namespace)
            } else {
                Vec::new()
            },
            by_namespace_available: namespaces_comparable,
            // An identity, not a counter: a window inherits the ledger epoch it was measured in.
            ledger_since_unix: self.ledger_since_unix,
            requests: self.requests.saturating_sub(before.requests),
            hits: self.hits.saturating_sub(before.hits),
            misses: self.misses.saturating_sub(before.misses),
            bypass: self.bypass.saturating_sub(before.bypass),
            upstream_errors: self.upstream_errors.saturating_sub(before.upstream_errors),
            tokens_saved: self.tokens_saved.saturating_sub(before.tokens_saved),
            input_tokens_saved: self
                .input_tokens_saved
                .saturating_sub(before.input_tokens_saved),
            output_tokens_saved: self
                .output_tokens_saved
                .saturating_sub(before.output_tokens_saved),
            cost_saved_micros: self
                .cost_saved_micros
                .saturating_sub(before.cost_saved_micros),
            unpriced_tokens_saved: self
                .unpriced_tokens_saved
                .saturating_sub(before.unpriced_tokens_saved),
            hits_usage_unknown: self
                .hits_usage_unknown
                .saturating_sub(before.hits_usage_unknown),
            feedback_agree: self.feedback_agree.saturating_sub(before.feedback_agree),
            feedback_wrong: self.feedback_wrong.saturating_sub(before.feedback_wrong),
            stream_not_stored: self
                .stream_not_stored
                .saturating_sub(before.stream_not_stored),
            cache_entries: self.cache_entries,
            fill_entries: self.fill_entries.saturating_sub(before.fill_entries),
            fill_tokens: self.fill_tokens.saturating_sub(before.fill_tokens),
            fill_input_tokens: self
                .fill_input_tokens
                .saturating_sub(before.fill_input_tokens),
            fill_output_tokens: self
                .fill_output_tokens
                .saturating_sub(before.fill_output_tokens),
            fill_cost_micros: self
                .fill_cost_micros
                .saturating_sub(before.fill_cost_micros),
            unpriced_fill_tokens: self
                .unpriced_fill_tokens
                .saturating_sub(before.unpriced_fill_tokens),
            fill_usage_unknown: self
                .fill_usage_unknown
                .saturating_sub(before.fill_usage_unknown),
            // Start-up facts are levels, not flows: differencing them would be meaningless.
            persisted_entries_at_start: self.persisted_entries_at_start,
            rehydrated_entries: self.rehydrated_entries,
            process_start_unix: self.process_start_unix,
            store_backend: self.store_backend.clone(),
            store_durable: self.store_durable,
        }
    }

    /// Dollars saved, gross. Float only here, at the edge.
    pub fn cost_saved_usd(&self) -> f64 {
        self.cost_saved_micros as f64 / 1e6
    }

    /// What this process paid the upstream for the entries it stored.
    pub fn fill_cost_usd(&self) -> f64 {
        self.fill_cost_micros as f64 / 1e6
    }

    /// Hit rate over cacheable (non-bypassed) traffic, or `None` when nothing cacheable was seen —
    /// which is not the same as a 0% hit rate, and reads very differently to whoever is deciding
    /// whether the cache earns its place.
    pub fn hit_rate(&self) -> Option<f64> {
        let cacheable = self.hits + self.misses;
        (cacheable > 0).then(|| self.hits as f64 / cacheable as f64)
    }

    /// The **observed** false-hit rate — wrong reports over judged reports — or `None` when no
    /// feedback was reported at all. `None` is not zero: nobody looking is not the same as nobody
    /// finding anything.
    pub fn observed_false_hit_rate(&self) -> Option<f64> {
        let judged = self.feedback_agree + self.feedback_wrong;
        (judged > 0).then(|| self.feedback_wrong as f64 / judged as f64)
    }

    /// Dollars saved discounted by the observed false-hit rate, or `None` when none was observed.
    /// A wrong hit is not a saving — it bought a wrong answer with the money it did not spend.
    pub fn cost_saved_usd_discounted(&self) -> Option<f64> {
        self.observed_false_hit_rate()
            .map(|fhr| self.cost_saved_usd() * (1.0 - fhr))
    }
}

/// Fetch and parse `<base>/metrics` with a client of this module's own, so a caller that only wants
/// one snapshot does not have to take on an HTTP client dependency to get it. Bounded by a short
/// timeout: a status read that hangs is worse than one that fails.
pub async fn fetch_url(base: &str) -> Result<Snapshot, String> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .map_err(|e| format!("cannot build http client: {e}"))?;
    fetch(&client, base).await
}

/// Fetch and parse `<base>/metrics`. `Err` carries a message fit to print — an unreachable proxy is
/// the most common way this is used wrong, and "connection refused" is more useful than "0 hits".
pub async fn fetch(client: &reqwest::Client, base: &str) -> Result<Snapshot, String> {
    let base = base.trim_end_matches('/');
    // Namespaces before metrics — see the reconciliation note at the bottom of this function.
    let ns_first = fetch_namespaces(client, base).await;
    let resp = client
        .get(format!("{base}/metrics"))
        .send()
        .await
        .map_err(|e| format!("cannot reach {base}/metrics: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("{base}/metrics returned {}", resp.status()));
    }
    let body = resp
        .text()
        .await
        .map_err(|e| format!("cannot read {base}/metrics: {e}"))?;
    let mut snap = Snapshot::parse(&body);
    // Attach the breakdown only when it cannot overstate the totals it breaks down.
    //
    // Two reads of a live node can never be one instant, so the ordering above decides which way
    // the skew falls. Namespaces are read FIRST: a hit landing between the two reads is then in
    // the node totals but not yet in the rows, leaving them a strict under-count — the direction
    // the report already promises ("rows sum to at most `saved`"). Reading metrics first would
    // invert it and let a row's hits or dollars exceed the total they are a share of, which is the
    // one thing this document must never do.
    //
    // The process boundary is the second guard: a node that restarted between the reads would pair
    // one incarnation's rows with another's counters, and those are unrelated numbers. When the
    // stamps disagree — or the node is too old to report one — the rows are dropped rather than
    // reconciled, because a missing breakdown is a gap and a wrong one is a lie.
    if let Some((ns_start, rows)) = ns_first {
        if ns_start != 0 && ns_start == snap.process_start_unix {
            snap.by_namespace = rows;
            // Distinct from "no rows": a node that has served nothing yet reports an empty
            // breakdown, and that empty is a fact about the node rather than a hole in the read.
            snap.by_namespace_available = true;
        }
    }
    Ok(snap)
}

/// Read the per-namespace savings rows from `<base>/v1/cache/namespaces`.
///
/// Best-effort by design: this is a *second* document, and every reason it might not arrive — a
/// target running a version that predates the endpoint (404), a proxy behind a router that only
/// exposes `/metrics`, a transport hiccup — leaves the node-level figures completely valid. So a
/// failure yields no rows rather than failing the whole read: an export without the breakdown is
/// still a correct export, while a `recall export` that fails because an *optional* breakdown was
/// unavailable would be a regression for every existing consumer.
async fn fetch_namespaces(
    client: &reqwest::Client,
    base: &str,
) -> Option<(u64, Vec<crate::namespaces::NamespaceRow>)> {
    #[derive(serde::Deserialize)]
    struct Body {
        #[serde(default)]
        process_start_unix: u64,
        #[serde(default)]
        namespaces: Vec<crate::namespaces::NamespaceRow>,
    }
    let resp = client
        .get(format!("{base}/v1/cache/namespaces"))
        .send()
        .await
        .ok()?;
    if !resp.status().is_success() {
        return None;
    }
    let body = resp.json::<Body>().await.ok()?;
    Some((body.process_start_unix, body.namespaces))
}

/// Per-namespace rows of `after` minus those of `before`, matched by namespace.
///
/// Same rule as every other counter here: saturating, so a restart between the two scrapes reports
/// zero rather than an underflowed astronomical figure. A row present only in `after` is a
/// namespace first seen during the window and carries its full total; a row that moved nothing is
/// dropped, so the delta lists what the run actually touched rather than the node's whole history.
///
/// That reading of an absent `before` row — "this namespace is new, so all of it is the window's" —
/// is only true when `before` is a breakdown that was actually read. Callers must establish that;
/// [`Snapshot::since`] is the one that does.
fn diff_namespaces(
    after: &[crate::namespaces::NamespaceRow],
    before: &[crate::namespaces::NamespaceRow],
) -> Vec<crate::namespaces::NamespaceRow> {
    after
        .iter()
        .filter_map(|now| {
            let was = before.iter().find(|b| b.namespace == now.namespace);
            let sub = |a: u64, b: fn(&crate::namespaces::NamespaceRow) -> u64| {
                a.saturating_sub(was.map_or(0, b))
            };
            let row = crate::namespaces::NamespaceRow {
                namespace: now.namespace.clone(),
                hits: sub(now.hits, |r| r.hits),
                input_tokens: sub(now.input_tokens, |r| r.input_tokens),
                output_tokens: sub(now.output_tokens, |r| r.output_tokens),
                cost_micros: sub(now.cost_micros, |r| r.cost_micros),
                unpriced_tokens: sub(now.unpriced_tokens, |r| r.unpriced_tokens),
            };
            (row.hits > 0).then_some(row)
        })
        .collect()
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

/// Read one label's value off a labelled series line, e.g. `backend` from
/// `recall_store_info{backend="redb",durable="true"} 1`. Quoted values only, no escape handling —
/// recall's own exposition never emits an escaped label, and this reader is for recall's series.
fn metric_label(body: &str, name: &str, label: &str) -> Option<String> {
    let line = body.lines().find(|l| {
        l.strip_prefix(name)
            .is_some_and(|rest| rest.starts_with('{'))
    })?;
    let labels = &line[line.find('{')? + 1..line.find('}')?];
    labels.split(',').find_map(|pair| {
        let (k, v) = pair.split_once('=')?;
        (k.trim() == label).then(|| v.trim().trim_matches('"').to_string())
    })
}

/// Read an integer-valued series. Prometheus exposition permits a float rendering of an integral
/// value (`1787000000` and `1787000000.0` are the same sample), so fall back to a float parse and
/// truncate rather than reading 0 — recall renders these as integers today, but a 0 here is not a
/// harmless default: `process_start_unix` is half of a node's identity on the paid side, and a 0
/// would collapse every replica and every restart into one node whose counters then
/// cross-difference. A series that is genuinely absent still reads 0, which is right for a counter.
pub(crate) fn metric_u64(body: &str, name: &str) -> u64 {
    let Some(raw) = metric_line(body, name) else {
        return 0;
    };
    if let Ok(v) = raw.parse::<u64>() {
        return v;
    }
    // The fallback accepts exactly one thing: an in-range non-negative integer that happens to be
    // written with a decimal point. Everything else reads 0, the same as an absent series.
    //
    // Both extra guards are load-bearing. Without the integral check a genuinely fractional sample
    // would truncate, inventing a plausible counter out of a value that was never a `u64`. Without
    // the range check the `as` cast SATURATES (defined behaviour since Rust 1.45), so a garbage
    // sample far above the range would read `u64::MAX` — worse than the 0 it replaces, because an
    // astronomical counter looks like real traffic while a 0 looks like nothing happened. Note
    // `u64::MAX as f64` rounds up to 2^64, so `<` is the correct comparison for the boundary.
    match raw.parse::<f64>() {
        Ok(v) if v.is_finite() && v >= 0.0 && v.fract() == 0.0 && v < u64::MAX as f64 => v as u64,
        _ => 0,
    }
}

pub(crate) fn metric_f64(body: &str, name: &str) -> f64 {
    metric_line(body, name)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0)
}

/// USD (as exposed) → integer micro-USD. Rounds, so the 6-decimal exposition round-trips exactly.
///
/// The money counters are integer micro-USD inside the proxy and become a float only in the
/// `{:.6}` rendering, which carries the full micro precision; this multiplies it straight back. The
/// round-trip is therefore exact while the micro-USD total stays under `2^53` — about **$9.0
/// billion** of savings on one node between restarts — which is where an f64 stops representing
/// consecutive integers. Past that the value drifts by micro-USD, never by a displayed cent.
/// Anything absent, negative, or non-finite is 0 — a corrupt metric line must read as "no measured
/// saving", never as a wrapped or astronomically large one.
pub(crate) fn usd_to_micros(usd: f64) -> u64 {
    if !usd.is_finite() || usd <= 0.0 {
        return 0;
    }
    let micros = (usd * 1_000_000.0).round();
    if micros > u64::MAX as f64 {
        u64::MAX
    } else {
        micros as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The float fallback exists for one shape — an integral value written with a decimal point —
    /// and must reject everything else to 0. The saturating case is the sharp one: `as u64` on an
    /// out-of-range float yields `u64::MAX`, and an astronomical counter reads as real traffic
    /// where a 0 reads as nothing happened.
    #[test]
    fn metric_u64_accepts_only_in_range_integers() {
        let body = |v: &str| format!("recall_probe {v}\n");
        let probe = |v: &str| metric_u64(&body(v), "recall_probe");

        // Plain integers, and the float rendering of one.
        assert_eq!(probe("1787000000"), 1_787_000_000);
        assert_eq!(probe("1787000000.0"), 1_787_000_000);
        assert_eq!(probe("0.0"), 0);

        // Fractional: never truncated into a counter it was not.
        assert_eq!(probe("1.9"), 0);
        assert_eq!(probe("0.5"), 0);

        // Out of range: 0, NOT the u64::MAX the `as` cast would saturate to.
        assert_eq!(probe("99999999999999999999999"), 0);
        assert_eq!(probe("18446744073709551616.0"), 0);
        assert_eq!(probe("1e300"), 0);

        // Negative, non-finite, and non-numeric.
        assert_eq!(probe("-5"), 0);
        assert_eq!(probe("-0.0"), 0);
        assert_eq!(probe("inf"), 0);
        assert_eq!(probe("NaN"), 0);
        assert_eq!(probe("abc"), 0);

        // An absent series is 0 too — right for a counter.
        assert_eq!(metric_u64("", "recall_probe"), 0);
    }

    const BODY: &str = "\
# HELP recall_hits_total Requests served from the semantic cache.
# TYPE recall_hits_total counter
recall_hits_total 42
recall_misses_total 8
recall_tokens_saved_total 1100000
recall_input_tokens_saved_total 1000000
recall_cost_saved_usd_total 12.483012
recall_feedback_agree_total 96
recall_feedback_wrong_total 4
recall_cache_entries 17
";

    #[test]
    fn parses_the_exposition_without_confusing_prefixes_or_help_lines() {
        let s = Snapshot::parse(BODY);
        assert_eq!(s.hits, 42);
        assert_eq!(s.misses, 8);
        // The shorter name must not match the longer line.
        assert_eq!(s.tokens_saved, 1_100_000);
        assert_eq!(s.input_tokens_saved, 1_000_000);
        // A "# HELP <name> ..." line must never be read as that metric's value.
        assert_eq!(s.output_tokens_saved, 0);
        assert_eq!(s.cache_entries, 17);
        // USD round-trips through integer micro-USD exactly.
        assert_eq!(s.cost_saved_micros, 12_483_012);
        assert!((s.cost_saved_usd() - 12.483012).abs() < 1e-9);
    }

    #[test]
    fn labelled_series_are_read_too() {
        let m = "recall_hits_total{instance=\"proxy-1\"} 7\nrecall_misses_total{instance=\"proxy-1\"} 3\n";
        let s = Snapshot::parse(m);
        assert_eq!((s.hits, s.misses), (7, 3));
    }

    #[test]
    fn rates_abstain_rather_than_report_zero() {
        let empty = Snapshot::default();
        // No cacheable traffic is not a 0% hit rate, and no feedback is not a 0% error rate.
        assert_eq!(empty.hit_rate(), None);
        assert_eq!(empty.observed_false_hit_rate(), None);
        assert_eq!(empty.cost_saved_usd_discounted(), None);

        let s = Snapshot::parse(BODY);
        assert!((s.hit_rate().unwrap() - 0.84).abs() < 1e-9);
        // 4 wrong of 100 judged = 4%.
        assert!((s.observed_false_hit_rate().unwrap() - 0.04).abs() < 1e-9);
        assert!((s.cost_saved_usd_discounted().unwrap() - 11.98369152).abs() < 1e-6);
    }

    #[test]
    fn since_differences_counters_and_takes_the_gauge_as_is() {
        let before = Snapshot {
            hits: 40,
            tokens_saved: 900,
            cost_saved_micros: 5_000_000,
            cache_entries: 3,
            ..Default::default()
        };
        let after = Snapshot {
            hits: 100,
            tokens_saved: 2_400,
            cost_saved_micros: 12_000_000,
            cache_entries: 9,
            ..Default::default()
        };
        let d = after.since(&before);
        assert_eq!(d.hits, 60);
        assert_eq!(d.tokens_saved, 1_500);
        assert_eq!(d.cost_saved_micros, 7_000_000);
        // A gauge is a level, not a flow — differencing it would be meaningless.
        assert_eq!(d.cache_entries, 9);
    }

    #[test]
    fn a_restart_between_scrapes_reports_zero_not_an_underflow() {
        let before = Snapshot {
            hits: 1_000,
            cost_saved_micros: 9_000_000,
            ..Default::default()
        };
        let after = Snapshot {
            hits: 2,
            cost_saved_micros: 10,
            ..Default::default()
        };
        let d = after.since(&before);
        assert_eq!(d.hits, 0);
        assert_eq!(d.cost_saved_micros, 0);
    }

    fn ns_row(namespace: &str, hits: u64) -> crate::namespaces::NamespaceRow {
        crate::namespaces::NamespaceRow {
            namespace: namespace.to_string(),
            hits,
            input_tokens: hits * 10,
            output_tokens: hits * 30,
            cost_micros: hits * 80,
            unpriced_tokens: 0,
        }
    }

    #[test]
    fn namespace_rows_difference_when_both_scrapes_read_the_breakdown() {
        let before = Snapshot {
            hits: 100,
            process_start_unix: 1_700_000_000,
            by_namespace: vec![ns_row("app:chat", 100)],
            by_namespace_available: true,
            ..Default::default()
        };
        let after = Snapshot {
            hits: 120,
            process_start_unix: 1_700_000_000,
            by_namespace: vec![ns_row("app:chat", 115), ns_row("app:batch", 5)],
            by_namespace_available: true,
            ..Default::default()
        };
        let d = after.since(&before);
        assert!(d.by_namespace_available);
        assert_eq!(d.hits, 20);
        // 15 of the window's 20 hits went to the namespace that already existed, 5 to one that
        // appeared during it. The rows are a split of the node delta, so they sum to it.
        let by_hits: Vec<(&str, u64)> = d
            .by_namespace
            .iter()
            .map(|r| (r.namespace.as_str(), r.hits))
            .collect();
        assert_eq!(by_hits, vec![("app:chat", 15), ("app:batch", 5)]);
        assert_eq!(d.by_namespace.iter().map(|r| r.hits).sum::<u64>(), d.hits);
    }

    #[test]
    fn a_scrape_that_could_not_read_the_breakdown_does_not_backdate_the_next_one() {
        // The node has served 100 hits before anyone starts watching. The first scrape of the
        // window reaches /metrics but not the namespaces document — an older target, a router that
        // only exposes /metrics, a transport hiccup — so it carries node counters and no rows. By
        // the second scrape the breakdown is readable again and reports the node's whole history.
        let unavailable = Snapshot {
            hits: 110,
            process_start_unix: 1_700_000_000,
            ..Default::default()
        };
        assert!(!unavailable.by_namespace_available);
        let after = Snapshot {
            hits: 120,
            process_start_unix: 1_700_000_000,
            by_namespace: vec![ns_row("app:chat", 120)],
            by_namespace_available: true,
            ..Default::default()
        };
        let d = after.since(&unavailable);
        assert_eq!(d.hits, 10);
        // Diffing against an unread breakdown would treat the missing row as zero and bill all 120
        // cumulative hits to a 10-hit window. Report no split instead, and say it is missing.
        assert!(d.by_namespace.is_empty());
        assert!(!d.by_namespace_available);
    }

    #[test]
    fn an_empty_breakdown_that_was_read_still_differences() {
        // A node with no hits yet reports an empty namespaces document. That empty is a fact about
        // the node, not a failed read, so the next window is a legitimate diff from zero.
        let before = Snapshot {
            process_start_unix: 1_700_000_000,
            by_namespace_available: true,
            ..Default::default()
        };
        let after = Snapshot {
            hits: 7,
            process_start_unix: 1_700_000_000,
            by_namespace: vec![ns_row("app:chat", 7)],
            by_namespace_available: true,
            ..Default::default()
        };
        let d = after.since(&before);
        assert!(d.by_namespace_available);
        assert_eq!(d.by_namespace.len(), 1);
        assert_eq!(d.by_namespace[0].hits, 7);
    }

    #[test]
    fn a_restart_mid_window_drops_the_split_unless_a_ledger_spans_it() {
        let before = Snapshot {
            hits: 100,
            process_start_unix: 1_700_000_000,
            by_namespace: vec![ns_row("app:chat", 100)],
            by_namespace_available: true,
            ..Default::default()
        };
        // Restarted, counters back near zero. The node fields saturate to 0; the rows would not,
        // so a namespace first seen after the restart would be the only thing in a "zero" window.
        let restarted = Snapshot {
            hits: 4,
            process_start_unix: 1_700_009_999,
            by_namespace: vec![ns_row("app:batch", 4)],
            by_namespace_available: true,
            ..Default::default()
        };
        let d = restarted.since(&before);
        assert_eq!(d.hits, 0);
        assert!(d.by_namespace.is_empty());
        assert!(!d.by_namespace_available);

        // With a durable savings ledger the counters did NOT reset — they are measured from the
        // ledger epoch, which both scrapes share — so the same restart is not a discontinuity and
        // the split stays comparable.
        let before = Snapshot {
            ledger_since_unix: 1_699_000_000,
            ..before
        };
        let after = Snapshot {
            hits: 130,
            ledger_since_unix: 1_699_000_000,
            process_start_unix: 1_700_009_999,
            by_namespace: vec![ns_row("app:chat", 130)],
            by_namespace_available: true,
            ..Default::default()
        };
        let d = after.since(&before);
        assert_eq!(d.hits, 30);
        assert!(d.by_namespace_available);
        assert_eq!(d.by_namespace[0].hits, 30);
    }

    #[test]
    fn usd_conversion_is_defensive() {
        assert_eq!(usd_to_micros(0.022500), 22_500);
        assert_eq!(usd_to_micros(0.000001), 1);
        assert_eq!(usd_to_micros(0.0), 0);
        assert_eq!(usd_to_micros(-5.0), 0);
        assert_eq!(usd_to_micros(f64::NAN), 0);
        assert_eq!(usd_to_micros(f64::INFINITY), 0);
    }
}
