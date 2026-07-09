//! Hand-rolled Prometheus counters over atomics — no metrics framework, matching the house style
//! (PLAN.md §3-OSS observability). These are the numbers that back the "we saved you $X" story: hit
//! rate and tokens saved. The dollar figure itself is the paid control plane's job (it needs a price
//! book); the OSS proxy exposes the raw token savings the operator can price with their own numbers.

use std::sync::atomic::{AtomicU64, Ordering};

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
        s.push_str("# HELP recall_hit_ratio Hit ratio over cacheable requests.\n# TYPE recall_hit_ratio gauge\n");
        s.push_str(&format!("recall_hit_ratio {ratio:.4}\n"));
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
