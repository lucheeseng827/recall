//! The ANN seam and its MVP implementation. `BruteForceIndex` is exact cosine via dot product over
//! pre-normalized vectors — correct by construction, and therefore the **recall oracle** every
//! future HNSW backend (`recall-index`, `hnsw_rs`/`instant-distance`) is benchmarked against
//! (PLAN.md §3-MVP). Because stored and query vectors are both L2-normalized by the facade, a dot
//! product *is* the cosine similarity.

use std::collections::HashMap;
use std::sync::RwLock;

use crate::error::RecallError;
use crate::math::dot;
use crate::types::{Key, Scored};

/// Object-safe ANN seam. MVP = exact brute force; scale = HNSW behind it with the same signatures.
/// `search` is CPU-bound and on a large graph can take low-single-digit ms, so a server runs it off
/// the reactor (`spawn_blocking`); that is a server concern, not a seam concern (PLAN.md §3-OSS).
pub trait AnnIndex: Send + Sync {
    fn insert(&self, ns: &str, key: Key, vector: &[f32]) -> Result<(), RecallError>;
    fn search(&self, ns: &str, query: &[f32], k: usize) -> Result<Vec<Scored>, RecallError>;
    fn remove(&self, ns: &str, key: Key) -> Result<(), RecallError>;
    fn len(&self, ns: &str) -> usize;
    fn is_empty(&self, ns: &str) -> bool {
        self.len(ns) == 0
    }
}

/// One namespace's vectors: a flat `(Key, vector)` list, scanned exactly on search.
type Partition = Vec<(Key, Vec<f32>)>;

/// Exact brute-force cosine index. One `Partition` per namespace string — fine to the low thousands
/// of entries per namespace, which is the MVP/small-N regime. Larger N is the `index-hnsw`
/// escalation, gated on `recall@1 ≥ 0.98` vs this oracle (PLAN.md §3-OSS, T3).
#[derive(Default)]
pub struct BruteForceIndex {
    ns: RwLock<HashMap<String, Partition>>,
}

impl BruteForceIndex {
    pub fn new() -> Self {
        Self::default()
    }
}

impl AnnIndex for BruteForceIndex {
    fn insert(&self, ns: &str, key: Key, vector: &[f32]) -> Result<(), RecallError> {
        let mut g = self.ns.write().unwrap();
        let bucket = g.entry(ns.to_string()).or_default();
        // Replace any prior vector for this key so a re-insert updates rather than duplicates.
        if let Some(slot) = bucket.iter_mut().find(|(k, _)| *k == key) {
            slot.1 = vector.to_vec();
        } else {
            bucket.push((key, vector.to_vec()));
        }
        Ok(())
    }

    fn search(&self, ns: &str, query: &[f32], k: usize) -> Result<Vec<Scored>, RecallError> {
        let g = self.ns.read().unwrap();
        let Some(bucket) = g.get(ns) else {
            return Ok(Vec::new());
        };
        let mut scored: Vec<Scored> = Vec::with_capacity(bucket.len());
        for (key, v) in bucket.iter() {
            if v.len() != query.len() {
                // Same-embedder namespaces never hit this; if they do, it's a wiring bug worth
                // surfacing loudly rather than silently zip-truncating (PLAN.md T2).
                return Err(RecallError::DimensionMismatch {
                    expected: query.len(),
                    got: v.len(),
                });
            }
            scored.push(Scored {
                key: *key,
                score: dot(query, v),
            });
        }
        // Descending by score; NaN sorts last so a degenerate vector can't win a hit.
        scored.sort_by(|a, b| match (a.score.is_nan(), b.score.is_nan()) {
            (true, true) => std::cmp::Ordering::Equal,
            (true, false) => std::cmp::Ordering::Greater, // NaN sorts last
            (false, true) => std::cmp::Ordering::Less,    // NaN sorts last
            (false, false) => b
                .score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal),
        });
        scored.truncate(k);
        Ok(scored)
    }

    fn remove(&self, ns: &str, key: Key) -> Result<(), RecallError> {
        if let Some(bucket) = self.ns.write().unwrap().get_mut(ns) {
            bucket.retain(|(k, _)| *k != key);
        }
        Ok(())
    }

    fn len(&self, ns: &str) -> usize {
        self.ns.read().unwrap().get(ns).map_or(0, |b| b.len())
    }
}
