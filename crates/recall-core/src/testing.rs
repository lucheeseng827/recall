//! Test-support seam doubles for exercising `SemanticCache`'s failure and edge paths that the
//! in-memory MVP backends never trigger on their own — e.g. a `Store::put` that succeeds followed by
//! an `AnnIndex::insert` that *fails*, or a `ThresholdPolicy` that returns `Hit` with no neighbor.
//!
//! Gated behind `cfg(any(test, feature = "test-support"))`: available to `recall-core`'s own unit
//! tests automatically, and to downstream crates (the proxy, future M2 backends) when they enable
//! `features = ["test-support"]` in their `[dev-dependencies]`. NEVER part of the default build, so
//! the zero-dependency / air-gap property is untouched.

use std::sync::atomic::{AtomicUsize, Ordering};

use crate::error::RecallError;
use crate::index::AnnIndex;
use crate::policy::ThresholdPolicy;
use crate::types::{Key, Scored, Verdict};

/// An [`AnnIndex`] whose `insert` always fails. Proves [`crate::SemanticCache::put`] rolls back the
/// preceding KV write when indexing fails, so a durable store is never left holding an unreachable
/// orphan. `search` yields no neighbors; `remove`/`len` are inert.
#[derive(Default)]
pub struct FailingIndex {
    /// Count of `insert` attempts — lets a test assert the failing path was actually reached.
    inserts: AtomicUsize,
}

impl FailingIndex {
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of `insert` attempts observed so far.
    pub fn insert_attempts(&self) -> usize {
        self.inserts.load(Ordering::SeqCst)
    }
}

impl AnnIndex for FailingIndex {
    fn insert(&self, _ns: &str, _key: Key, _vector: &[f32]) -> Result<(), RecallError> {
        self.inserts.fetch_add(1, Ordering::SeqCst);
        Err(RecallError::Backend("injected index insert failure".into()))
    }
    fn search(&self, _ns: &str, _query: &[f32], _k: usize) -> Result<Vec<Scored>, RecallError> {
        Ok(Vec::new())
    }
    fn remove(&self, _ns: &str, _key: Key) -> Result<(), RecallError> {
        Ok(())
    }
    fn len(&self, _ns: &str) -> usize {
        0
    }
}

/// A [`ThresholdPolicy`] that returns [`Verdict::Hit`] regardless of the top score — including when
/// there is no neighbor (`top == None`). Proves the cache treats `(Hit, None)` as a miss rather than
/// unwrapping and panicking on a misbehaving custom policy.
#[derive(Default)]
pub struct AlwaysHit;

impl ThresholdPolicy for AlwaysHit {
    fn id(&self) -> &str {
        "test:always-hit"
    }
    fn decide(&self, _ns: &str, _top: Option<f32>) -> Verdict {
        Verdict::Hit
    }
}
