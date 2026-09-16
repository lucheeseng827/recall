//! `recall export` — one node's savings and usage as a **versioned JSON document**.
//!
//! This shape is a compatibility surface, not a debug dump. Whatever ingests it — a script, a
//! spreadsheet, a fleet roll-up service — gets to rely on it, so it obeys wire-contract rules:
//!
//! - **Versioned from day one.** `schema` is the first field. Evolution is **additive only**: a
//!   consumer of `recall.export/v1` may see fields it does not know (and must ignore them), but a
//!   field, once shipped, never changes name, type, or meaning. A breaking change is a new schema
//!   string, emitted only by an operator's explicit choice.
//! - **Integer money.** Every monetary field is integer micro-USD (`*_micros`), exactly as recall
//!   accounts internally — the document never carries a float that could disagree with the counter
//!   it came from. Consumers divide by 1e6 at their own edge.
//! - **Facts only, flagged where incomplete.** The same honesty rules as `/metrics`: unpriced and
//!   usage-unknown counts ride along with every dollar figure, so a consumer can always tell "the
//!   number is complete" from "the number is a floor".
//! - **Idempotency-ready.** `export_id` is a content-derived id: two exports of the same state at
//!   the same instant collide (that is the point — a fire-and-forget retry of the same document
//!   should dedupe), while any change in state or time yields a fresh id. An ingest ledger keyed
//!   on it can accept the same document twice without double-counting.
//!
//! Everything here is cumulative **since this process started** (`node.process_start_unix` — a
//! consumer differencing successive exports uses it to tell a counter reset from a quiet period).

use serde::Serialize;

use crate::scrape::Snapshot;

/// The `schema` value this module emits. Bump only for a breaking change; additions keep it.
pub const SCHEMA: &str = "recall.export/v1";

/// The `recall.export/v1` document. Field order is stable (serde emits declaration order), with
/// `schema` first so a consumer can dispatch before parsing the rest.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ExportV1 {
    /// Always [`SCHEMA`] for this shape.
    pub schema: &'static str,
    /// Content-derived id for idempotent ingestion — see the module doc.
    pub export_id: String,
    /// When this export was taken, Unix milliseconds.
    pub exported_at_unix_ms: u64,
    /// The proxy this was read from, as the exporter addressed it.
    pub target: String,
    pub node: Node,
    pub traffic: Traffic,
    pub saved: Saved,
    pub fill: Fill,
    pub feedback: Feedback,
    /// `saved`, split by cache namespace — added under this schema, so an older consumer ignores
    /// it. A split of the same hits, never an addition: the rows sum to at most `saved`, and past
    /// the node's cap the tail folds into one `(other namespaces)` row. Empty when nothing has
    /// been served yet, or when the exporter could not read the node's namespaces document —
    /// either way `saved` stays the authoritative total.
    pub by_namespace: Vec<NamespaceSaved>,
}

/// One namespace's share of what the cache saved.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct NamespaceSaved {
    /// The server-derived cache namespace (`base:flavor:model:param-fingerprint`), or
    /// `(other namespaces)` for the folded tail.
    pub namespace: String,
    pub hits: u64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    /// Integer micro-USD, priced by the node's own table — same rules as `saved.cost_micros`.
    pub cost_micros: u64,
    /// Tokens in this row no price rule matched: how far its `cost_micros` under-counts.
    pub unpriced_tokens: u64,
}

/// The process and its store: what this node is, and what survived its last restart.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Node {
    /// Unix seconds the proxy process started — the boundary all cumulative fields reset at.
    pub process_start_unix: u64,
    /// Store backend (`memory`, `redb`); empty when the target predates `recall_store_info`.
    pub store_backend: String,
    /// Whether the store survives a restart.
    pub store_durable: bool,
    /// Entries the store held at the moment of export.
    pub cache_entries: u64,
    /// Rows the store held at process start, before rehydration.
    pub persisted_entries_at_start: u64,
    /// Entries rebuilt into the live index at process start.
    pub rehydrated_entries: u64,
    /// Unix seconds the node's **persisted** savings ledger began, or `0` when the counters reset
    /// with the process (the default, and what every document before this field implied). It says
    /// what window the cumulative fields cover, which `process_start_unix` alone stops answering
    /// once a node persists them. Differencing consumers are unaffected either way: they key on
    /// `process_start_unix`, so a restart still opens a fresh baseline.
    pub ledger_since_unix: u64,
}

/// Request dispositions since process start.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Traffic {
    pub requests: u64,
    pub hits: u64,
    pub misses: u64,
    pub bypass: u64,
    pub upstream_errors: u64,
    pub stream_not_stored: u64,
}

/// What cache hits avoided buying. `cost_micros` is priced by the node's own `[pricing.<model>]`
/// table; `unpriced_tokens` and `hits_usage_unknown` are how far it under-counts.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Saved {
    pub tokens: u64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    /// Integer micro-USD. `0` with `unpriced_tokens > 0` means "no rates", not "no savings".
    pub cost_micros: u64,
    pub unpriced_tokens: u64,
    pub hits_usage_unknown: u64,
}

/// What this node paid to populate its own cache — the mirror of [`Saved`], and the denominator of
/// its payback.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Fill {
    pub entries: u64,
    pub tokens: u64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    /// Integer micro-USD, same pricing rules as `saved.cost_micros`.
    pub cost_micros: u64,
    pub unpriced_tokens: u64,
    /// Stored entries whose upstream response reported no usage — their cost is unknown, not zero.
    pub usage_unknown: u64,
}

/// Hit outcomes reported to `/v1/cache/feedback`. `wrong / (agree + wrong)` is the observed
/// false-hit rate; zero judged reports means *unmeasured*, never a 0% error rate.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Feedback {
    pub agree: u64,
    pub wrong: u64,
}

/// Assemble the document from a scraped [`Snapshot`]. Pure — time comes in from the caller, so
/// the same inputs always yield the same document (and the same `export_id`).
pub fn build(snapshot: &Snapshot, target: &str, exported_at_unix_ms: u64) -> ExportV1 {
    let mut doc = ExportV1 {
        schema: SCHEMA,
        // Filled in below, once there is a document to hash.
        export_id: String::new(),
        exported_at_unix_ms,
        target: target.to_string(),
        node: Node {
            process_start_unix: snapshot.process_start_unix,
            store_backend: snapshot.store_backend.clone(),
            store_durable: snapshot.store_durable,
            cache_entries: snapshot.cache_entries,
            ledger_since_unix: snapshot.ledger_since_unix,
            persisted_entries_at_start: snapshot.persisted_entries_at_start,
            rehydrated_entries: snapshot.rehydrated_entries,
        },
        traffic: Traffic {
            requests: snapshot.requests,
            hits: snapshot.hits,
            misses: snapshot.misses,
            bypass: snapshot.bypass,
            upstream_errors: snapshot.upstream_errors,
            stream_not_stored: snapshot.stream_not_stored,
        },
        saved: Saved {
            tokens: snapshot.tokens_saved,
            input_tokens: snapshot.input_tokens_saved,
            output_tokens: snapshot.output_tokens_saved,
            cost_micros: snapshot.cost_saved_micros,
            unpriced_tokens: snapshot.unpriced_tokens_saved,
            hits_usage_unknown: snapshot.hits_usage_unknown,
        },
        fill: Fill {
            entries: snapshot.fill_entries,
            tokens: snapshot.fill_tokens,
            input_tokens: snapshot.fill_input_tokens,
            output_tokens: snapshot.fill_output_tokens,
            cost_micros: snapshot.fill_cost_micros,
            unpriced_tokens: snapshot.unpriced_fill_tokens,
            usage_unknown: snapshot.fill_usage_unknown,
        },
        feedback: Feedback {
            agree: snapshot.feedback_agree,
            wrong: snapshot.feedback_wrong,
        },
        by_namespace: snapshot
            .by_namespace
            .iter()
            .map(|r| NamespaceSaved {
                namespace: r.namespace.clone(),
                hits: r.hits,
                input_tokens: r.input_tokens,
                output_tokens: r.output_tokens,
                cost_micros: r.cost_micros,
                unpriced_tokens: r.unpriced_tokens,
            })
            .collect(),
    };
    doc.export_id = export_id(&doc);
    doc
}

/// Content-derived export id: a blake3 over the **whole serialized document** (with the id field
/// itself empty). Deliberately deterministic — an exact retransmission of the same document must
/// carry the same id so an ingest ledger can dedupe it, while any movement in any field yields a
/// fresh one.
///
/// Hashing the serialization rather than a hand-listed subset of counters is the point: a field
/// added to the schema is covered the moment it is added, so no future export can silently share
/// an id with a document it differs from.
fn export_id(doc: &ExportV1) -> String {
    debug_assert!(
        doc.export_id.is_empty(),
        "export_id must be hashed over the document with the id field still empty"
    );
    // Struct serialization is field-declaration ordered, so this is stable across runs and builds
    // for a given schema version — and the schema string is itself the first field hashed.
    let body =
        serde_json::to_vec(doc).expect("the export document is plain data and always serializes");
    let mut h = blake3::Hasher::new();
    h.update(&body);
    // 16 bytes / 32 hex chars: far below any birthday-collision concern at fleet-export volumes,
    // short enough to live in a database key.
    h.finalize().to_hex()[..32].to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snapshot() -> Snapshot {
        Snapshot {
            requests: 201,
            hits: 197,
            misses: 4,
            tokens_saved: 5713,
            input_tokens_saved: 2364,
            output_tokens_saved: 3349,
            cost_saved_micros: 39_400,
            fill_entries: 4,
            fill_tokens: 116,
            fill_input_tokens: 48,
            fill_output_tokens: 68,
            fill_cost_micros: 800,
            feedback_agree: 58,
            feedback_wrong: 2,
            cache_entries: 4,
            process_start_unix: 1_787_000_000,
            store_backend: "redb".to_string(),
            store_durable: true,
            ..Default::default()
        }
    }

    /// The contract's load-bearing invariants: schema first, integer money, the audit flags
    /// riding alongside every dollar figure. A rename or type change here is a breaking change to
    /// a published wire format — this test failing means bump [`SCHEMA`], not edit the test.
    #[test]
    fn v1_shape_is_the_published_contract() {
        let doc = build(&snapshot(), "http://127.0.0.1:8080", 1_787_100_000_000);
        let json = serde_json::to_value(&doc).unwrap();

        assert_eq!(json["schema"], "recall.export/v1");
        // `schema` must be the FIRST key, so consumers can dispatch on a prefix read.
        let text = serde_json::to_string(&doc).unwrap();
        assert!(
            text.starts_with(r#"{"schema":"recall.export/v1""#),
            "{text}"
        );

        // Money is integer micros, and the under-count flags sit next to each figure.
        assert_eq!(json["saved"]["cost_micros"], 39_400);
        assert_eq!(json["saved"]["unpriced_tokens"], 0);
        assert_eq!(json["saved"]["hits_usage_unknown"], 0);
        assert_eq!(json["fill"]["cost_micros"], 800);
        assert_eq!(json["fill"]["usage_unknown"], 0);

        // The counter-reset boundary and store identity ship with every document.
        assert_eq!(json["node"]["process_start_unix"], 1_787_000_000);
        assert_eq!(json["node"]["store_backend"], "redb");
        assert_eq!(json["node"]["store_durable"], true);

        // Feedback is raw counts — the consumer computes the rate, and can tell 0/0 (unmeasured)
        // from a measured 0%.
        assert_eq!(json["feedback"]["agree"], 58);
        assert_eq!(json["feedback"]["wrong"], 2);
    }

    #[test]
    fn export_id_dedupes_retransmissions_and_separates_everything_else() {
        let snap = snapshot();
        let a = build(&snap, "http://127.0.0.1:8080", 1_787_100_000_000);
        let b = build(&snap, "http://127.0.0.1:8080", 1_787_100_000_000);
        // Same state, same instant → same id: an exact retry dedupes at the ingest ledger.
        assert_eq!(a.export_id, b.export_id);
        assert_eq!(a.export_id.len(), 32);

        // A later export, a different node, or any counter movement is a different document.
        let later = build(&snap, "http://127.0.0.1:8080", 1_787_100_000_001);
        assert_ne!(a.export_id, later.export_id);
        let other_node = build(&snap, "http://10.0.0.2:8080", 1_787_100_000_000);
        assert_ne!(a.export_id, other_node.export_id);
        let mut moved = snap.clone();
        moved.hits += 1;
        let moved = build(&moved, "http://127.0.0.1:8080", 1_787_100_000_000);
        assert_ne!(a.export_id, moved.export_id);

        // Every field the document carries participates — including the ones a hand-listed hash
        // would plausibly forget: the caveat counters, the store gauges, and the node's store
        // identity. Two documents that differ anywhere must never share an id, because an ingest
        // ledger keyed on that id would drop the second as a retry of the first.
        for mutate in [
            (|s: &mut Snapshot| s.unpriced_tokens_saved += 1) as fn(&mut Snapshot),
            |s: &mut Snapshot| s.hits_usage_unknown += 1,
            |s: &mut Snapshot| s.unpriced_fill_tokens += 1,
            |s: &mut Snapshot| s.fill_usage_unknown += 1,
            |s: &mut Snapshot| s.input_tokens_saved += 1,
            |s: &mut Snapshot| s.output_tokens_saved += 1,
            |s: &mut Snapshot| s.fill_input_tokens += 1,
            |s: &mut Snapshot| s.fill_output_tokens += 1,
            |s: &mut Snapshot| s.fill_tokens += 1,
            |s: &mut Snapshot| s.cache_entries += 1,
            |s: &mut Snapshot| s.upstream_errors += 1,
            |s: &mut Snapshot| s.stream_not_stored += 1,
            |s: &mut Snapshot| s.persisted_entries_at_start += 1,
            |s: &mut Snapshot| s.rehydrated_entries += 1,
            |s: &mut Snapshot| s.store_backend = "memory".to_string(),
            |s: &mut Snapshot| s.store_durable = !s.store_durable,
        ] {
            let mut changed = snap.clone();
            mutate(&mut changed);
            let changed = build(&changed, "http://127.0.0.1:8080", 1_787_100_000_000);
            assert_ne!(
                a.export_id, changed.export_id,
                "a document differing in {changed:?} must not reuse an id"
            );
        }
    }
}
