//! Making the savings figure outlive the process that measured it.
//!
//! Every counter in [`crate::metrics`] resets when `serve` restarts. For latency that is correct —
//! a histogram of a process describes that process. For **money** it is not: "this cache has saved
//! $18,400" is the sentence the cache is funded on, and a number that silently returns to zero on
//! every deploy can never be that sentence. It answers "since the last restart", which nobody
//! budgets against.
//!
//! This module is the codec for that: a versioned snapshot of the cumulative counters (and the
//! per-namespace rows beside them), written to whatever durable store the node already runs. The
//! store persists opaque bytes; the shape lives here, with the counters it describes.
//!
//! **Opt-in, and off by default** (`--savings-flush-secs 0`). `recall.export/v1` documents
//! `node.process_start_unix` as the boundary its cumulative fields reset at, and `/metrics` is read
//! by consumers that assume the same. Persisting counters changes that for every one of them, so it
//! is never switched on behind an operator's back — and when it *is* on, the node publishes
//! `recall_ledger_since_time_seconds` so a consumer can tell that the counters predate this process
//! rather than having to infer it.
//!
//! Restore is best-effort by construction: a snapshot that is missing, unreadable, or written by a
//! newer version starts a fresh ledger with a warning. Accounting must never be the reason a cache
//! refuses to serve.

use serde::{Deserialize, Serialize};

use crate::metrics::Metrics;
use crate::namespaces::{NamespaceLedger, NamespaceRow};

/// The persisted form of a node's cumulative accounting.
///
/// Deliberately **counters only** — no prices, no derived dollars beyond the integer micro-USD the
/// counters already hold. Re-pricing history is then just a matter of changing `[pricing]`, and a
/// snapshot never disagrees with the price book that reads it.
/// `serde(default)` is on the STRUCT, not just the newest field: within a version, evolution is
/// additive, so a blob written by an older build must still restore. Per-field defaults would make
/// the next added counter turn every stored blob into a deserialize error — which `restore_ledger`
/// reports as "corrupt" and answers by starting over, silently discarding the whole history this
/// module exists to keep.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct LedgerSnapshot {
    /// Format version. [`restore`] refuses anything it does not recognise rather than
    /// half-reading a future shape.
    pub version: u32,
    /// Unix seconds this ledger first began accumulating — carried across every restart, so the
    /// figure can say *since when*. This is what `recall_ledger_since_time_seconds` reports.
    pub since_unix: u64,
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
    pub fill_entries: u64,
    pub fill_tokens: u64,
    pub fill_input_tokens: u64,
    pub fill_output_tokens: u64,
    pub fill_cost_micros: u64,
    pub unpriced_fill_tokens: u64,
    pub fill_usage_unknown: u64,
    /// The per-namespace breakdown, so "worth it for whom" survives a restart too.
    pub by_namespace: Vec<NamespaceRow>,
}

/// Current snapshot format version.
pub const VERSION: u32 = 1;

/// Capture the current counters. `since_unix` is the ledger's own epoch — the caller passes the one
/// it restored (or `now` for a fresh ledger), never the process start, which is a different fact.
pub fn snapshot(
    metrics: &Metrics,
    namespaces: &NamespaceLedger,
    since_unix: u64,
) -> LedgerSnapshot {
    use std::sync::atomic::Ordering::Relaxed;
    LedgerSnapshot {
        version: VERSION,
        since_unix,
        requests: metrics.requests.load(Relaxed),
        hits: metrics.hits.load(Relaxed),
        misses: metrics.misses.load(Relaxed),
        bypass: metrics.bypass.load(Relaxed),
        upstream_errors: metrics.upstream_errors.load(Relaxed),
        tokens_saved: metrics.tokens_saved.load(Relaxed),
        input_tokens_saved: metrics.input_tokens_saved.load(Relaxed),
        output_tokens_saved: metrics.output_tokens_saved.load(Relaxed),
        cost_saved_micros: metrics.cost_saved_micros.load(Relaxed),
        unpriced_tokens_saved: metrics.unpriced_tokens_saved.load(Relaxed),
        hits_usage_unknown: metrics.hits_usage_unknown.load(Relaxed),
        feedback_agree: metrics.feedback_agree.load(Relaxed),
        feedback_wrong: metrics.feedback_wrong.load(Relaxed),
        stream_not_stored: metrics.stream_not_stored.load(Relaxed),
        fill_entries: metrics.fill_entries.load(Relaxed),
        fill_tokens: metrics.fill_tokens.load(Relaxed),
        fill_input_tokens: metrics.fill_input_tokens.load(Relaxed),
        fill_output_tokens: metrics.fill_output_tokens.load(Relaxed),
        fill_cost_micros: metrics.fill_cost_micros.load(Relaxed),
        unpriced_fill_tokens: metrics.unpriced_fill_tokens.load(Relaxed),
        fill_usage_unknown: metrics.fill_usage_unknown.load(Relaxed),
        by_namespace: namespaces.rows(),
    }
}

/// Load a snapshot's counters into a fresh node.
///
/// Returns `false` — leaving both the metrics and the namespace rows untouched — for a version this
/// build does not know, so an older binary rolled back onto a newer ledger starts clean rather than
/// reporting a number it cannot vouch for.
///
/// The latency histograms are intentionally *not* restored: they describe this process's behaviour,
/// and merging a previous process's percentiles into them would make both meaningless.
#[must_use]
pub fn restore(snap: &LedgerSnapshot, metrics: &Metrics, namespaces: &NamespaceLedger) -> bool {
    use std::sync::atomic::Ordering::Relaxed;
    if snap.version != VERSION {
        return false;
    }
    metrics.requests.store(snap.requests, Relaxed);
    metrics.hits.store(snap.hits, Relaxed);
    metrics.misses.store(snap.misses, Relaxed);
    metrics.bypass.store(snap.bypass, Relaxed);
    metrics.upstream_errors.store(snap.upstream_errors, Relaxed);
    metrics.tokens_saved.store(snap.tokens_saved, Relaxed);
    metrics
        .input_tokens_saved
        .store(snap.input_tokens_saved, Relaxed);
    metrics
        .output_tokens_saved
        .store(snap.output_tokens_saved, Relaxed);
    metrics
        .cost_saved_micros
        .store(snap.cost_saved_micros, Relaxed);
    metrics
        .unpriced_tokens_saved
        .store(snap.unpriced_tokens_saved, Relaxed);
    metrics
        .hits_usage_unknown
        .store(snap.hits_usage_unknown, Relaxed);
    metrics.feedback_agree.store(snap.feedback_agree, Relaxed);
    metrics.feedback_wrong.store(snap.feedback_wrong, Relaxed);
    metrics
        .stream_not_stored
        .store(snap.stream_not_stored, Relaxed);
    metrics.fill_entries.store(snap.fill_entries, Relaxed);
    metrics.fill_tokens.store(snap.fill_tokens, Relaxed);
    metrics
        .fill_input_tokens
        .store(snap.fill_input_tokens, Relaxed);
    metrics
        .fill_output_tokens
        .store(snap.fill_output_tokens, Relaxed);
    metrics
        .fill_cost_micros
        .store(snap.fill_cost_micros, Relaxed);
    metrics
        .unpriced_fill_tokens
        .store(snap.unpriced_fill_tokens, Relaxed);
    metrics
        .fill_usage_unknown
        .store(snap.fill_usage_unknown, Relaxed);
    namespaces.restore(&snap.by_namespace);
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pricing::PriceBook;

    fn busy_node() -> (Metrics, NamespaceLedger) {
        let m = Metrics::default();
        let ns = NamespaceLedger::new();
        Metrics::inc(&m.requests);
        Metrics::inc(&m.hits);
        Metrics::add(&m.tokens_saved, 1_500);
        Metrics::add(&m.input_tokens_saved, 1_000);
        Metrics::add(&m.output_tokens_saved, 500);
        Metrics::add(&m.cost_saved_micros, 7_500);
        Metrics::inc(&m.feedback_wrong);
        Metrics::add(&m.fill_cost_micros, 900);
        ns.record_hit("t:openai:m:aa", 1_000, 500, 7_500, true);
        (m, ns)
    }

    #[test]
    fn snapshot_restore_round_trips_counters_and_namespaces() {
        let (m, ns) = busy_node();
        let snap = snapshot(&m, &ns, 1_700_000_000);
        assert_eq!(snap.version, VERSION);

        // "Restart": a fresh node restored from the snapshot reports identical counters.
        let (m2, ns2) = (Metrics::default(), NamespaceLedger::new());
        assert!(restore(&snap, &m2, &ns2));
        assert_eq!(snapshot(&m2, &ns2, snap.since_unix), snap);

        // And it keeps accumulating on top of the restored history rather than from zero.
        Metrics::add(&m2.cost_saved_micros, 2_500);
        assert_eq!(
            snapshot(&m2, &ns2, snap.since_unix).cost_saved_micros,
            10_000
        );
    }

    #[test]
    fn restore_refuses_an_unknown_version_and_leaves_the_node_untouched() {
        let (m, ns) = busy_node();
        let mut snap = snapshot(&m, &ns, 1_700_000_000);
        snap.version = VERSION + 1;

        let (fresh, fresh_ns) = (Metrics::default(), NamespaceLedger::new());
        Metrics::add(&fresh.cost_saved_micros, 42);
        assert!(
            !restore(&snap, &fresh, &fresh_ns),
            "unknown version refused"
        );
        assert_eq!(
            snapshot(&fresh, &fresh_ns, 0).cost_saved_micros,
            42,
            "the running node is left exactly as it was"
        );
        assert!(fresh_ns.is_empty());
    }

    // The point of the whole module: a restored node's savings figure covers the ledger's life,
    // not the process's — which is what makes it quotable after a deploy.
    #[test]
    fn restored_savings_survive_a_simulated_restart() {
        let (m, ns) = busy_node();
        let ledger_epoch = 1_700_000_000;
        let blob = serde_json::to_vec(&snapshot(&m, &ns, ledger_epoch)).unwrap();

        // A new process: fresh counters, then the blob read back off disk.
        let (m2, ns2) = (Metrics::default(), NamespaceLedger::new());
        assert_eq!(snapshot(&m2, &ns2, 0).cost_saved_micros, 0, "starts cold");
        let decoded: LedgerSnapshot = serde_json::from_slice(&blob).unwrap();
        assert!(restore(&decoded, &m2, &ns2));

        assert_eq!(
            snapshot(&m2, &ns2, decoded.since_unix).cost_saved_micros,
            7_500
        );
        assert_eq!(
            decoded.since_unix, ledger_epoch,
            "the epoch travels with it"
        );
        let rows = ns2.rows();
        assert_eq!(rows.len(), 1, "the namespace breakdown survives too");
        assert_eq!(rows[0].cost_micros, 7_500);
    }

    // A snapshot written before `by_namespace` existed must still restore its node-level counters
    // (the serde(default) contract), because a ledger is read by binaries newer than the one that
    // wrote it far more often than the reverse.
    #[test]
    fn older_snapshot_without_namespace_rows_still_restores() {
        let json = r#"{"version":1,"since_unix":1700000000,"requests":5,"hits":4,"misses":1,
            "bypass":0,"upstream_errors":0,"tokens_saved":100,"input_tokens_saved":60,
            "output_tokens_saved":40,"cost_saved_micros":250,"unpriced_tokens_saved":0,
            "hits_usage_unknown":0,"feedback_agree":0,"feedback_wrong":0,"stream_not_stored":0,
            "fill_entries":1,"fill_tokens":10,"fill_input_tokens":6,"fill_output_tokens":4,
            "fill_cost_micros":25,"unpriced_fill_tokens":0,"fill_usage_unknown":0}"#;
        let snap: LedgerSnapshot = serde_json::from_str(json).unwrap();
        assert!(snap.by_namespace.is_empty());

        let (m, ns) = (Metrics::default(), NamespaceLedger::new());
        assert!(restore(&snap, &m, &ns));
        assert_eq!(snapshot(&m, &ns, snap.since_unix).cost_saved_micros, 250);
        assert!(ns.is_empty(), "no rows to restore, and none invented");
    }

    // Restoring must not disturb pricing: the ledger carries counters, so a price-book change
    // applies to new traffic without rewriting history that was already priced.
    #[test]
    fn restore_is_price_book_agnostic() {
        let (m, ns) = busy_node();
        let snap = snapshot(&m, &ns, 1_700_000_000);
        let (m2, ns2) = (Metrics::default(), NamespaceLedger::new());
        assert!(restore(&snap, &m2, &ns2));
        // Nothing in the snapshot references a rate; an empty book prices new traffic at nothing
        // while the restored dollars stand.
        assert!(PriceBook::default().is_empty());
        assert_eq!(snapshot(&m2, &ns2, 0).cost_saved_micros, 7_500);
    }
}
