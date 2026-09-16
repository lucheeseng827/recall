//! Per-namespace savings: *which* traffic the cache is paying off for.
//!
//! The node-level counters in [`crate::metrics`] answer "is the cache worth it". They cannot answer
//! "worth it for whom" — and that is the question asked the moment more than one team, tenant, or
//! model shares a proxy, because that is when someone has to decide where the cache budget goes.
//! This module keeps the same saved-token accounting split by cache namespace (`base:flavor:model:
//! param-fingerprint`), which is exactly the axis those questions are asked along.
//!
//! **Why this is not a metric label.** `/metrics` deliberately keeps no per-model or per-namespace
//! series: the namespace is derived from the request, so its cardinality is caller-controlled, and
//! an unbounded label set is how a Prometheus install gets taken down. The breakdown is served as a
//! **bounded JSON document** instead (`GET /v1/cache/namespaces`), which a scraper reads once per
//! interval rather than fanning out into series.
//!
//! Three rules, matching the ones the dollar figures already obey:
//!
//! 1. **Bounded.** At most [`NamespaceLedger::MAX`] namespaces are tracked; everything after that
//!    folds into a single [`OVERFLOW`] row. A client inventing namespaces cannot grow this map.
//! 2. **A view, never an addition.** Every row is a slice of the same hits the node counters
//!    already counted. Rows sum to *at most* the node totals — never past them.
//! 3. **Integer money, never guessed.** Costs are the same integer micro-USD the price book
//!    produced for that hit; a row whose model has no rate carries `0` and says so via
//!    `unpriced_tokens`, exactly as the node-level figure does.

use std::collections::HashMap;
use std::sync::Mutex;

use serde::{Deserialize, Serialize};

/// The row every namespace past the cap folds into. Named so it cannot collide with a real
/// namespace: `Namespace::new` accepts arbitrary text, but the proxy derives its namespaces as
/// `base:flavor:model:fingerprint`, and this contains characters that shape never produces.
pub const OVERFLOW: &str = "(other namespaces)";

/// What one namespace's hits avoided buying.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct NamespaceRow {
    pub namespace: String,
    /// Hits served under this namespace.
    pub hits: u64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    /// Integer micro-USD, priced by the node's own `[pricing.<model>]` table. `0` alongside a
    /// non-zero `unpriced_tokens` means "no rate for this model", not "saved nothing".
    pub cost_micros: u64,
    /// Tokens in this row that no price rule matched — how far `cost_micros` under-counts.
    pub unpriced_tokens: u64,
}

/// Bounded per-namespace accumulator. Interior-mutable so it sits in shared handler state; a hit
/// takes one short lock, off the embed/ANN hot path (it runs beside response serialisation).
#[derive(Default)]
pub struct NamespaceLedger {
    rows: Mutex<HashMap<String, NamespaceRow>>,
}

impl NamespaceLedger {
    /// How many distinct namespaces are tracked before the rest fold into [`OVERFLOW`]. Sized so a
    /// real deployment (a handful of models × a handful of parameter shapes per team) fits with
    /// room to spare, while a hostile or buggy client cannot grow the map.
    pub const MAX: usize = 512;

    pub fn new() -> Self {
        Self::default()
    }

    /// Credit one served hit. `cost_micros` is what the price book charged for these tokens (`0`
    /// when the model is unpriced, in which case the tokens are recorded as unpriced instead).
    pub fn record_hit(
        &self,
        namespace: &str,
        input: u64,
        output: u64,
        cost_micros: u64,
        priced: bool,
    ) {
        let mut rows = self.rows.lock().unwrap();
        // An unknown namespace once the map is full folds into the overflow row rather than
        // growing it. A namespace already tracked keeps accumulating under its own name.
        let key = if rows.contains_key(namespace) || rows.len() < Self::MAX {
            namespace
        } else {
            OVERFLOW
        };
        let row = rows.entry(key.to_string()).or_default();
        if row.namespace.is_empty() {
            row.namespace = key.to_string();
        }
        row.hits = row.hits.saturating_add(1);
        row.input_tokens = row.input_tokens.saturating_add(input);
        row.output_tokens = row.output_tokens.saturating_add(output);
        row.cost_micros = row.cost_micros.saturating_add(cost_micros);
        if !priced {
            row.unpriced_tokens = row
                .unpriced_tokens
                .saturating_add(input.saturating_add(output));
        }
    }

    /// Every row, biggest saver first (then most hits, then name) so the order is stable and the
    /// interesting rows are the ones a truncated read sees.
    pub fn rows(&self) -> Vec<NamespaceRow> {
        let mut out: Vec<NamespaceRow> = self.rows.lock().unwrap().values().cloned().collect();
        out.sort_by(|a, b| {
            b.cost_micros
                .cmp(&a.cost_micros)
                .then_with(|| b.hits.cmp(&a.hits))
                .then_with(|| a.namespace.cmp(&b.namespace))
        });
        out
    }

    /// Replace the tracked rows wholesale — the restore half of ledger persistence, so a restarted
    /// node's breakdown covers the ledger's life rather than restarting from empty beside
    /// node-level counters that did not. Rows past the cap are dropped rather than folded, since a
    /// snapshot written by this build is already capped.
    pub fn restore(&self, rows: &[NamespaceRow]) {
        let mut map = self.rows.lock().unwrap();
        map.clear();
        for row in rows.iter().take(Self::MAX + 1) {
            map.insert(row.namespace.clone(), row.clone());
        }
    }

    /// Number of namespaces currently tracked (including the overflow row, if any).
    pub fn len(&self) -> usize {
        self.rows.lock().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_accumulate_and_sort_by_savings() {
        let led = NamespaceLedger::new();
        led.record_hit("t:openai:gpt-4o:aa", 1000, 100, 3_500, true);
        led.record_hit("t:openai:gpt-4o:aa", 1000, 100, 3_500, true);
        led.record_hit("t:openai:small:bb", 10, 1, 7, true);

        let rows = led.rows();
        assert_eq!(rows.len(), 2);
        assert_eq!(
            rows[0].namespace, "t:openai:gpt-4o:aa",
            "biggest saver first"
        );
        assert_eq!(rows[0].hits, 2);
        assert_eq!(rows[0].cost_micros, 7_000);
        assert_eq!(rows[0].input_tokens, 2000);
        assert_eq!(rows[1].cost_micros, 7);
    }

    #[test]
    fn unpriced_hits_count_tokens_but_no_dollars() {
        let led = NamespaceLedger::new();
        led.record_hit("t:openai:mystery:cc", 40, 2, 0, false);
        let rows = led.rows();
        assert_eq!(rows[0].cost_micros, 0);
        assert_eq!(
            rows[0].unpriced_tokens, 42,
            "the row says how far its dollars under-count"
        );
    }

    #[test]
    fn cardinality_is_bounded_by_the_overflow_row() {
        let led = NamespaceLedger::new();
        for i in 0..(NamespaceLedger::MAX + 25) {
            led.record_hit(&format!("ns-{i}"), 1, 1, 1, true);
        }
        assert_eq!(
            led.len(),
            NamespaceLedger::MAX + 1,
            "cap plus exactly one overflow row"
        );
        let rows = led.rows();
        let overflow = rows
            .iter()
            .find(|r| r.namespace == OVERFLOW)
            .expect("overflow row exists");
        assert_eq!(overflow.hits, 25);

        // An already-tracked namespace keeps its own identity after the cap is reached.
        led.record_hit("ns-3", 1, 1, 1, true);
        let rows = led.rows();
        assert_eq!(rows.iter().find(|r| r.namespace == "ns-3").unwrap().hits, 2);
        assert_eq!(led.len(), NamespaceLedger::MAX + 1, "still capped");
    }

    // The invariant the report leans on: rows are a split of the node's hits, never more.
    #[test]
    fn rows_sum_to_the_hits_recorded() {
        let led = NamespaceLedger::new();
        for i in 0..50 {
            led.record_hit(&format!("ns-{}", i % 7), 3, 2, 5, true);
        }
        let rows = led.rows();
        assert_eq!(rows.iter().map(|r| r.hits).sum::<u64>(), 50);
        assert_eq!(rows.iter().map(|r| r.cost_micros).sum::<u64>(), 250);
    }
}
