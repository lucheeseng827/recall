//! Hand-rolled Prometheus counters over atomics — no metrics framework, matching the house style
//! (PLAN.md §3-OSS observability). These are the numbers that back the "we saved you $X" story: hit
//! rate, tokens saved, and — when the operator configures a `[pricing.<model>]` table — the dollars
//! those tokens would have cost. recall bundles no price list and makes no network call: the dollar
//! figure is the operator's own rates applied locally, or it is absent (see [`crate::pricing`]).

use std::sync::atomic::{AtomicU64, Ordering};

use crate::config::StoreFacts;
use crate::pricing::PriceBook;

/// The store-side facts `/metrics` reports alongside the request-path counters: what the store holds
/// now, what it started with, and when this process began. Assembled per scrape.
pub struct RuntimeFacts<'a> {
    /// Entries the store holds right now, read live.
    pub entries: usize,
    /// What the store was at start-up.
    pub store: &'a StoreFacts,
    /// Unix seconds at which this process started — the boundary every counter above resets at, and
    /// therefore what makes `increase()` across a restart interpretable without recall having to
    /// invent, name, or bound a "cold window" of its own.
    pub started_at_unix: u64,
    /// Would-be hits the second-neighbour ambiguity gate demoted to misses. Lives on the cache (the
    /// facade owns the gate), so it is read at scrape time like `entries` rather than mirrored into
    /// a counter here. `0` whenever the gate is disarmed, which is the default.
    pub ambiguity_misses: u64,
    /// Unix seconds this node's ledger began accumulating, when counter persistence is on
    /// (`--savings-flush-secs`). `0` means the counters reset with this process — the default, and
    /// what every consumer assumed before persistence existed — so the two cases are told apart
    /// here rather than guessed at from `started_at_unix` alone.
    pub ledger_since_unix: u64,
}

/// Prometheus histogram bucket upper bounds in seconds (PLAN.md §3-OSS / §7: `LATENCY_BUCKETS`
/// spanning 0.005..10). Cumulative `le` buckets are emitted at render time; the implicit `+Inf`
/// bucket equals the total observation count (so an observation above 10 s still shows up there).
const LATENCY_BUCKETS: [f64; 11] = [
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
];

/// A hand-rolled Prometheus histogram over atomics — same no-framework style as the counters. Each
/// `observe` records into exactly one bucket (the smallest whose upper bound it fits under) plus the
/// running sum and count; `render` turns the per-bucket counts into the cumulative `le` series
/// Prometheus expects. The sum is accumulated in microseconds (an integer) to keep it lock-free, and
/// converted back to seconds on render.
pub struct Histogram {
    buckets: [AtomicU64; LATENCY_BUCKETS.len()],
    sum_micros: AtomicU64,
    count: AtomicU64,
}

impl Default for Histogram {
    fn default() -> Self {
        Self {
            buckets: std::array::from_fn(|_| AtomicU64::new(0)),
            sum_micros: AtomicU64::new(0),
            count: AtomicU64::new(0),
        }
    }
}

impl Histogram {
    /// Record one observation in seconds. Lock-free: a counted-once bucket increment + sum + count.
    pub fn observe(&self, seconds: f64) {
        let seconds = if seconds.is_finite() && seconds >= 0.0 {
            seconds
        } else {
            0.0
        };
        self.count.fetch_add(1, Ordering::Relaxed);
        self.sum_micros
            .fetch_add((seconds * 1e6) as u64, Ordering::Relaxed);
        // Record in the first (smallest) bucket whose upper bound contains the observation. An
        // observation larger than the last bound falls only into the implicit +Inf bucket (= count).
        for (i, ub) in LATENCY_BUCKETS.iter().enumerate() {
            if seconds <= *ub {
                self.buckets[i].fetch_add(1, Ordering::Relaxed);
                break;
            }
        }
    }

    /// Append the Prometheus `histogram` exposition for this metric (cumulative `_bucket{le=…}`
    /// series, then `_sum` in seconds and `_count`).
    fn render(&self, out: &mut String, name: &str, help: &str) {
        out.push_str(&format!("# HELP {name} {help}\n# TYPE {name} histogram\n"));
        let mut cumulative = 0u64;
        for (i, ub) in LATENCY_BUCKETS.iter().enumerate() {
            cumulative += self.buckets[i].load(Ordering::Relaxed);
            out.push_str(&format!("{name}_bucket{{le=\"{ub}\"}} {cumulative}\n"));
        }
        let count = self.count.load(Ordering::Relaxed);
        out.push_str(&format!("{name}_bucket{{le=\"+Inf\"}} {count}\n"));
        let sum_seconds = self.sum_micros.load(Ordering::Relaxed) as f64 / 1e6;
        out.push_str(&format!("{name}_sum {sum_seconds}\n"));
        out.push_str(&format!("{name}_count {count}\n"));
    }
}

#[derive(Default)]
pub struct Metrics {
    pub requests: AtomicU64,
    pub hits: AtomicU64,
    pub misses: AtomicU64,
    pub bypass: AtomicU64,
    pub upstream_errors: AtomicU64,
    /// Sum of `usage.total_tokens` from completions served from cache — i.e. tokens NOT bought from
    /// the upstream model because a hit answered the request.
    pub tokens_saved: AtomicU64,
    /// `tokens_saved` split by price tier: input (prompt) and output (completion) tokens not bought.
    /// Kept apart because providers price them differently (output ≈ 3–5× input), so an accurate
    /// dollar figure needs the split — the lumped `tokens_saved` mis-prices when the mix is skewed
    /// (as a 1M-token-context workload is: almost all input).
    /// Input (prompt) tokens saved — exposed as `recall_input_tokens_saved_total`.
    pub input_tokens_saved: AtomicU64,
    /// Output (completion) tokens saved — exposed as `recall_output_tokens_saved_total`.
    pub output_tokens_saved: AtomicU64,
    /// What the saved tokens would have cost, in **integer micro-USD**, priced by the operator's
    /// `[pricing.<model>]` table. Integer so millions of hits accumulate exactly; rendered as
    /// `recall_cost_saved_usd_total` (a float, at render only). Stays 0 when no price book is
    /// configured — which is why `unpriced_tokens_saved` exists to tell that apart from a cache
    /// that genuinely saved nothing.
    pub cost_saved_micros: AtomicU64,
    /// Tokens saved on hits whose model had **no matching price rule** — exposed as
    /// `recall_unpriced_tokens_saved_total`. The visible "unpriced" flag: recall never guesses a
    /// rate, so an incomplete or typo'd price book shows up here instead of quietly deflating
    /// `recall_cost_saved_usd_total`. Aggregate, not per-model: the model id comes off the request,
    /// so a per-model label would be caller-controlled (unbounded) metric cardinality.
    pub unpriced_tokens_saved: AtomicU64,
    /// Hits whose stored completion reported **no** token usage at all, so they credited nothing to
    /// any savings counter. Exposed as `recall_hits_usage_unknown_total`. The audit valve on the
    /// savings figure: a provider that omits usage (an OpenAI stream without
    /// `stream_options.include_usage`, say) makes the savings an under-count rather than a wrong
    /// count, and this is how an operator sees that rather than concluding the cache is idle.
    pub hits_usage_unknown: AtomicU64,
    /// Served hits an operator later judged **correct** via `POST /v1/cache/feedback`. Exposed as
    /// `recall_feedback_agree_total`.
    pub feedback_agree: AtomicU64,
    /// Entries this process wrote into its own cache, each one paid for with an upstream call.
    /// Exposed as `recall_fill_entries_total`. The primitive of the fill side: exact, backend- and
    /// path-agnostic, and needing no usage parsing, so it stays right even where the provider
    /// reported no token counts.
    pub fill_entries: AtomicU64,
    /// Upstream tokens purchased on requests whose response this process then stored — the exact
    /// mirror of `tokens_saved`, in the other direction. Exposed as `recall_fill_tokens_total`,
    /// with the same input/output split beside it.
    pub fill_tokens: AtomicU64,
    /// Input (prompt) tokens purchased on stored fills — `recall_fill_input_tokens_total`.
    pub fill_input_tokens: AtomicU64,
    /// Output (completion) tokens purchased on stored fills — `recall_fill_output_tokens_total`.
    pub fill_output_tokens: AtomicU64,
    /// What filling this cache cost, in **integer micro-USD**, priced by the same
    /// `[pricing.<model>]` table as the savings. Exposed as `recall_fill_cost_usd_total`.
    pub fill_cost_micros: AtomicU64,
    /// Tokens on stored fills whose model matched no price rule — `recall_unpriced_fill_tokens_total`.
    /// Excluded from the cost above rather than guessed at, exactly as on the savings side.
    pub unpriced_fill_tokens: AtomicU64,
    /// Stored fills whose upstream response reported no token usage at all, so their cost is
    /// unknown rather than zero. Exposed as `recall_fill_usage_unknown_total`.
    pub fill_usage_unknown: AtomicU64,
    /// Served hits an operator later judged **wrong** via `POST /v1/cache/feedback` — a false hit.
    /// Exposed as `recall_feedback_wrong_total`.
    ///
    /// Counted here at the proxy, not inside the policy, precisely because the default
    /// `--policy static` discards observations entirely: without this counter a wrong-hit report
    /// vanishes with a `200 {"accepted": true}` and nothing anywhere records that the cache was
    /// caught serving a wrong answer. `wrong / (agree + wrong)` is the observed false-hit rate —
    /// the discount an honest savings figure is quoted net of.
    pub feedback_wrong: AtomicU64,
    /// Streamed misses that were forwarded to the client but NOT stored because the stream did not
    /// terminate cleanly (no `[DONE]`/`message_stop`, a `length`/`max_tokens` truncation, or a
    /// non-UTF-8 body). A streamed reply's `x-recall-cache` header is sent before its body, so such a
    /// response still carries `miss`; this counter is how an operator sees the drop. A non-zero rate
    /// means upstream streams are being cut off (timeouts, `max_tokens`, disconnects).
    pub stream_not_stored: AtomicU64,
    /// End-to-end cache-lookup latency (exact-hash shortcut → embed → ANN search → threshold decide),
    /// for every cacheable request (hit or miss) — the p50/p99 the README headlines. Bypassed
    /// requests are excluded (they never looked up). Upstream-forward time on a miss is separate.
    pub cache_get_seconds: Histogram,
    /// Upstream forward latency for buffered (non-streamed) requests — a miss or a bypass round-trip.
    /// Streamed upstream calls are intentionally not timed here (their duration is entangled with the
    /// client's read pace, so it is not a clean server-side latency).
    pub upstream_seconds: Histogram,
}

impl Metrics {
    pub fn inc(counter: &AtomicU64) {
        counter.fetch_add(1, Ordering::Relaxed);
    }
    pub fn add(counter: &AtomicU64, n: u64) {
        counter.fetch_add(n, Ordering::Relaxed);
    }

    /// The observed false-hit rate — the share of *judged* hits an operator marked wrong — or
    /// `None` when no feedback has been reported. `None` is not zero: nobody looking is not the
    /// same as nobody finding anything, and a savings figure discounted by an assumed-zero error
    /// rate would be the gross figure wearing a disguise.
    pub fn observed_false_hit_rate(&self) -> Option<f64> {
        let wrong = self.feedback_wrong.load(Ordering::Relaxed);
        let judged = wrong + self.feedback_agree.load(Ordering::Relaxed);
        (judged > 0).then(|| wrong as f64 / judged as f64)
    }

    /// Credit a hit's saved tokens to the dollar figure, using the operator's price book.
    ///
    /// A model the book has no rule for contributes **nothing** to `cost_saved_micros` and its
    /// tokens land in `unpriced_tokens_saved` instead — the difference between "we don't know what
    /// this model costs you" and "this model is free", which a savings number nobody can audit
    /// would otherwise blur. Saturating: the counter clamps rather than wrapping.
    pub fn record_savings(&self, prices: &PriceBook, model: &str, input: u64, output: u64) {
        Self::credit(
            prices,
            model,
            input,
            output,
            &self.cost_saved_micros,
            &self.unpriced_tokens_saved,
        );
    }

    /// Credit an entry this process just bought and stored — the mirror of [`record_savings`].
    ///
    /// Called at the moment a write **succeeds**, not when a miss occurs: a miss that was not
    /// stored bought nothing for the cache, and counting it would inflate the fill cost with money
    /// that would have been spent with recall switched off entirely.
    ///
    /// A response that reported no usage still counts as an entry and is flagged, so the fill cost
    /// reads as incomplete rather than as free.
    pub fn record_fill(&self, prices: &PriceBook, model: &str, input: u64, output: u64) {
        Self::inc(&self.fill_entries);
        if input == 0 && output == 0 {
            Self::inc(&self.fill_usage_unknown);
            return;
        }
        Self::add(&self.fill_tokens, input.saturating_add(output));
        Self::add(&self.fill_input_tokens, input);
        Self::add(&self.fill_output_tokens, output);
        Self::credit(
            prices,
            model,
            input,
            output,
            &self.fill_cost_micros,
            &self.unpriced_fill_tokens,
        );
    }

    /// Price `input`/`output` for `model` into `cost_micros`, or — when the book has no rule —
    /// charge the tokens to `unpriced` instead. The one place the never-guess-a-model rule lives,
    /// shared by both directions so they cannot drift apart.
    fn credit(
        prices: &PriceBook,
        model: &str,
        input: u64,
        output: u64,
        cost_micros: &AtomicU64,
        unpriced: &AtomicU64,
    ) {
        match prices.cost_micros(model, input, output) {
            Some(micros) => {
                // fetch_update rather than fetch_add: a wrapped total would read as a plausible
                // small number, and a money figure that silently resets is worse than one pinned
                // at the ceiling.
                let _ = cost_micros.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |cur| {
                    Some(cur.saturating_add(micros))
                });
            }
            // Plain `fetch_add` is enough for the token counter: wrapping needs ~1.8e19 tokens
            // (millennia of traffic), whereas the money counters saturate because a *rate* is
            // operator-supplied and a fat-fingered one multiplies every hit.
            None => Self::add(unpriced, input.saturating_add(output)),
        }
    }

    /// Prometheus text exposition (v0.0.4), plus the gauges that describe the *store* rather than
    /// the request path.
    ///
    /// These are passed in rather than held on `Metrics` because they do not belong to the request
    /// path. The live entry count in particular is read from the store at scrape time, so it stays
    /// correct across a restart that rehydrated a durable store — which re-indexes without going
    /// through `put`, and would leave any process-local counter reading zero.
    ///
    /// Nothing here is editorialized. A node reports what it recovered and when it started; it
    /// never claims anything about other nodes, because it cannot see them. Whatever scrapes every
    /// node can do that arithmetic — and unlike the node, it is entitled to.
    pub fn render_with(&self, facts: &RuntimeFacts) -> String {
        let mut s = self.render();
        let gauge = |s: &mut String, name: &str, help: &str, v: String| {
            s.push_str(&format!(
                "# HELP {name} {help}\n# TYPE {name} gauge\n{name} {v}\n"
            ));
        };
        gauge(
            &mut s,
            "recall_cache_entries",
            "Completions currently held by the store.",
            facts.entries.to_string(),
        );
        // A counter, but sourced from the cache at scrape time (like `entries`) rather than from a
        // field on `self`, so it is emitted here rather than in the counter block below.
        s.push_str(
            "# HELP recall_ambiguity_miss_total Would-be hits demoted to misses by the second-neighbour ambiguity gate (--ambiguity-eps; 0 while disarmed).\n\
             # TYPE recall_ambiguity_miss_total counter\n",
        );
        s.push_str(&format!(
            "recall_ambiguity_miss_total {}\n",
            facts.ambiguity_misses
        ));
        // 0 = counters reset with the process (the default). Non-zero = they were restored from a
        // durable ledger that began at this stamp, so they predate `recall_process_start_time_seconds`.
        gauge(
            &mut s,
            "recall_ledger_since_time_seconds",
            "Unix time the persisted savings ledger began; 0 when counters reset with the process.",
            facts.ledger_since_unix.to_string(),
        );
        gauge(
            &mut s,
            "recall_persisted_entries_at_start",
            "Rows the store held at process start, before rehydration ran.",
            facts.store.persisted_at_start.to_string(),
        );
        gauge(
            &mut s,
            "recall_rehydrated_entries",
            "Entries rebuilt into the live index at process start (cache not re-purchased).",
            facts.store.rehydrated.to_string(),
        );
        gauge(
            &mut s,
            "recall_process_start_time_seconds",
            "Unix time this process started — the boundary its counters reset at.",
            facts.started_at_unix.to_string(),
        );
        // An info gauge: the value is always 1, the labels carry the facts. Bounded cardinality —
        // there are exactly two backends. Without it, `recall_rehydrated_entries 0` is ambiguous
        // between "ephemeral backend, nothing to rebuild, working as designed" and "durable
        // backend, recovered nothing".
        s.push_str(
            "# HELP recall_store_info The store backend in use, and whether it survives a restart.\n# TYPE recall_store_info gauge\n",
        );
        s.push_str(&format!(
            "recall_store_info{{backend=\"{}\",durable=\"{}\"}} 1\n",
            facts.store.backend, facts.store.durable
        ));
        s
    }

    /// Prometheus text exposition (v0.0.4). Counters as `counter`, the derived hit ratio as `gauge`.
    pub fn render(&self) -> String {
        let r = self.requests.load(Ordering::Relaxed);
        let h = self.hits.load(Ordering::Relaxed);
        let m = self.misses.load(Ordering::Relaxed);
        let b = self.bypass.load(Ordering::Relaxed);
        let e = self.upstream_errors.load(Ordering::Relaxed);
        let ts = self.tokens_saved.load(Ordering::Relaxed);
        let its = self.input_tokens_saved.load(Ordering::Relaxed);
        let ots = self.output_tokens_saved.load(Ordering::Relaxed);
        let sns = self.stream_not_stored.load(Ordering::Relaxed);
        let ups = self.unpriced_tokens_saved.load(Ordering::Relaxed);
        let huu = self.hits_usage_unknown.load(Ordering::Relaxed);
        let fba = self.feedback_agree.load(Ordering::Relaxed);
        let fbw = self.feedback_wrong.load(Ordering::Relaxed);
        let fe = self.fill_entries.load(Ordering::Relaxed);
        let ft = self.fill_tokens.load(Ordering::Relaxed);
        let fit = self.fill_input_tokens.load(Ordering::Relaxed);
        let fot = self.fill_output_tokens.load(Ordering::Relaxed);
        let uft = self.unpriced_fill_tokens.load(Ordering::Relaxed);
        let fuu = self.fill_usage_unknown.load(Ordering::Relaxed);
        let fill_cost_usd = self.fill_cost_micros.load(Ordering::Relaxed) as f64 / 1e6;
        // Integer micro-USD → USD, at render only. 6 decimals is exactly the stored resolution, so
        // the exposition neither invents precision nor drops a single micro-dollar.
        let cost_saved_usd = self.cost_saved_micros.load(Ordering::Relaxed) as f64 / 1e6;
        // Hit ratio over cacheable (non-bypassed) requests — bypassed traffic was never a cache
        // candidate, so including it would understate cache effectiveness.
        let cacheable = h + m;
        let ratio = if cacheable > 0 {
            h as f64 / cacheable as f64
        } else {
            0.0
        };

        let mut s = String::new();
        let counter = |s: &mut String, name: &str, help: &str, v: u64| {
            s.push_str(&format!(
                "# HELP {name} {help}\n# TYPE {name} counter\n{name} {v}\n"
            ));
        };
        counter(
            &mut s,
            "recall_requests_total",
            "Chat-completion requests received.",
            r,
        );
        counter(
            &mut s,
            "recall_hits_total",
            "Requests served from the semantic cache.",
            h,
        );
        counter(
            &mut s,
            "recall_misses_total",
            "Cacheable requests forwarded upstream.",
            m,
        );
        counter(
            &mut s,
            "recall_bypass_total",
            "Requests that bypassed the cache (stream/tools/temperature/n>1).",
            b,
        );
        counter(
            &mut s,
            "recall_upstream_errors_total",
            "Upstream forwarding failures.",
            e,
        );
        counter(
            &mut s,
            "recall_tokens_saved_total",
            "Upstream tokens not purchased due to cache hits.",
            ts,
        );
        counter(
            &mut s,
            "recall_input_tokens_saved_total",
            "Upstream input (prompt) tokens not purchased due to cache hits.",
            its,
        );
        counter(
            &mut s,
            "recall_output_tokens_saved_total",
            "Upstream output (completion) tokens not purchased due to cache hits.",
            ots,
        );
        counter(
            &mut s,
            "recall_unpriced_tokens_saved_total",
            "Tokens saved for models with no configured price rule (excluded from cost_saved).",
            ups,
        );
        counter(
            &mut s,
            "recall_hits_usage_unknown_total",
            "Hits whose stored completion reported no token usage (savings under-counted, not zero).",
            huu,
        );
        counter(
            &mut s,
            "recall_feedback_agree_total",
            "Served hits reported correct via /v1/cache/feedback.",
            fba,
        );
        counter(
            &mut s,
            "recall_feedback_wrong_total",
            "Served hits reported wrong (false hits) via /v1/cache/feedback.",
            fbw,
        );
        counter(
            &mut s,
            "recall_fill_entries_total",
            "Entries this process bought from the upstream and stored.",
            fe,
        );
        counter(
            &mut s,
            "recall_fill_tokens_total",
            "Upstream tokens purchased on responses this process then stored.",
            ft,
        );
        counter(
            &mut s,
            "recall_fill_input_tokens_total",
            "Input (prompt) tokens purchased on stored fills.",
            fit,
        );
        counter(
            &mut s,
            "recall_fill_output_tokens_total",
            "Output (completion) tokens purchased on stored fills.",
            fot,
        );
        counter(
            &mut s,
            "recall_unpriced_fill_tokens_total",
            "Fill tokens for models with no configured price rule (excluded from fill cost).",
            uft,
        );
        counter(
            &mut s,
            "recall_fill_usage_unknown_total",
            "Stored fills whose upstream response reported no token usage (cost unknown, not zero).",
            fuu,
        );
        counter(
            &mut s,
            "recall_stream_not_stored_total",
            "Streamed misses not cached because the stream did not terminate cleanly.",
            sns,
        );
        self.cache_get_seconds.render(
            &mut s,
            "recall_cache_get_duration_seconds",
            "End-to-end cache lookup latency (embed + ANN search + decide) for cacheable requests.",
        );
        self.upstream_seconds.render(
            &mut s,
            "recall_upstream_duration_seconds",
            "Upstream forward latency for buffered (non-streamed) misses and bypasses.",
        );
        // Not emitted via `counter` (which takes a u64): money is stored as integer micro-USD and
        // only becomes a float here.
        s.push_str(
            "# HELP recall_cost_saved_usd_total Upstream spend avoided by cache hits, priced with the operator's configured rates.\n# TYPE recall_cost_saved_usd_total counter\n",
        );
        s.push_str(&format!(
            "recall_cost_saved_usd_total {cost_saved_usd:.6}\n"
        ));
        s.push_str(
            "# HELP recall_fill_cost_usd_total What this process paid the upstream for the entries it stored, at the operator's configured rates.\n# TYPE recall_fill_cost_usd_total counter\n",
        );
        s.push_str(&format!("recall_fill_cost_usd_total {fill_cost_usd:.6}\n"));
        s.push_str("# HELP recall_hit_ratio Hit ratio over cacheable requests.\n# TYPE recall_hit_ratio gauge\n");
        s.push_str(&format!("recall_hit_ratio {ratio:.4}\n"));
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pricing::RateSpec;
    use std::collections::HashMap;

    fn book(entries: &[(&str, f64, f64)]) -> PriceBook {
        let table: HashMap<String, RateSpec> = entries
            .iter()
            .map(|(k, i, o)| {
                (
                    (*k).to_string(),
                    RateSpec {
                        input: *i,
                        output: *o,
                    },
                )
            })
            .collect();
        PriceBook::load(&table).0
    }

    #[test]
    fn priced_hits_accumulate_dollars_and_render_at_micro_resolution() {
        let m = Metrics::default();
        let prices = book(&[("gpt-4o", 2.50, 10.00)]);
        // 1M input @ $2.50 + 1M output @ $10.00 = $12.50, twice = $25.00.
        m.record_savings(&prices, "gpt-4o", 1_000_000, 1_000_000);
        m.record_savings(&prices, "gpt-4o", 1_000_000, 1_000_000);
        let out = m.render();
        assert!(
            out.contains("recall_cost_saved_usd_total 25.000000"),
            "{out}"
        );
        assert!(
            out.contains("recall_unpriced_tokens_saved_total 0"),
            "{out}"
        );
    }

    #[test]
    fn unpriced_model_is_flagged_not_counted_as_free() {
        let m = Metrics::default();
        let prices = book(&[("gpt-4o", 2.50, 10.00)]);
        m.record_savings(&prices, "some-local-model", 700, 300);
        let out = m.render();
        // Zero dollars, but the tokens are visibly unpriced — an operator can tell "no price rule"
        // apart from "the cache saved nothing".
        assert!(
            out.contains("recall_cost_saved_usd_total 0.000000"),
            "{out}"
        );
        assert!(
            out.contains("recall_unpriced_tokens_saved_total 1000"),
            "{out}"
        );
    }

    #[test]
    fn no_price_book_prices_nothing_and_flags_everything() {
        let m = Metrics::default();
        m.record_savings(&PriceBook::default(), "gpt-4o", 10, 5);
        let out = m.render();
        assert!(
            out.contains("recall_cost_saved_usd_total 0.000000"),
            "{out}"
        );
        assert!(
            out.contains("recall_unpriced_tokens_saved_total 15"),
            "{out}"
        );
    }

    #[test]
    fn cost_saved_clamps_instead_of_wrapping() {
        let m = Metrics::default();
        let prices = book(&[("m", 1000.0, 1000.0)]);
        m.record_savings(&prices, "m", u64::MAX, u64::MAX);
        m.record_savings(&prices, "m", u64::MAX, u64::MAX);
        // Pinned at the ceiling, not wrapped back to a small plausible-looking figure.
        assert_eq!(m.cost_saved_micros.load(Ordering::Relaxed), u64::MAX);
    }

    #[test]
    fn histogram_renders_cumulative_buckets_sum_and_count() {
        let h = Histogram::default();
        h.observe(0.001); // ≤ 0.005
        h.observe(0.02); // ≤ 0.025
        h.observe(100.0); // above the last bound → only +Inf
        let mut out = String::new();
        h.render(&mut out, "x_seconds", "help");
        // Cumulative: the 0.005 bucket has 1, and it stays ≥1 monotonically up the buckets.
        assert!(out.contains("x_seconds_bucket{le=\"0.005\"} 1"), "{out}");
        assert!(out.contains("x_seconds_bucket{le=\"0.025\"} 2"), "{out}");
        assert!(out.contains("x_seconds_bucket{le=\"10\"} 2"), "{out}");
        // The +Inf bucket counts every observation, including the 100 s outlier.
        assert!(out.contains("x_seconds_bucket{le=\"+Inf\"} 3"), "{out}");
        assert!(out.contains("x_seconds_count 3"), "{out}");
        // Sum is in seconds (~100.021), proving micros→seconds conversion.
        assert!(out.contains("x_seconds_sum 100.021"), "{out}");
    }
}
