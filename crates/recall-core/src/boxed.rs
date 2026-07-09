//! Runtime backend selection. The MVP wires concrete seam impls as generic type parameters, but a
//! server picks its embedder/index/store/policy from config at startup — so it needs to hold them
//! as trait objects. These blanket impls let `Box<dyn Embedder>` (etc.) satisfy the very trait it
//! boxes, so `SemanticCache<Box<dyn Embedder>, …>` — aliased [`DynCache`] — just works. The traits
//! are object-safe by design (PLAN.md §2.2), which is what makes this sound.

use crate::cache::SemanticCache;
use crate::embed::Embedder;
use crate::error::RecallError;
use crate::index::AnnIndex;
use crate::kv::Store;
use crate::policy::ThresholdPolicy;
use crate::types::{Entry, Key, ModelId, Outcome, Scored, Vector, Verdict};

impl<T: Embedder + ?Sized> Embedder for Box<T> {
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vector>, RecallError> {
        (**self).embed(texts)
    }
    fn dim(&self) -> usize {
        (**self).dim()
    }
    fn id(&self) -> ModelId {
        (**self).id()
    }
    fn embed_one(&self, text: &str) -> Result<Vector, RecallError> {
        (**self).embed_one(text)
    }
}

impl<T: AnnIndex + ?Sized> AnnIndex for Box<T> {
    fn insert(&self, ns: &str, key: Key, vector: &[f32]) -> Result<(), RecallError> {
        (**self).insert(ns, key, vector)
    }
    fn search(&self, ns: &str, query: &[f32], k: usize) -> Result<Vec<Scored>, RecallError> {
        (**self).search(ns, query, k)
    }
    fn remove(&self, ns: &str, key: Key) -> Result<(), RecallError> {
        (**self).remove(ns, key)
    }
    fn len(&self, ns: &str) -> usize {
        (**self).len(ns)
    }
}

impl<T: Store + ?Sized> Store for Box<T> {
    fn get(&self, key: Key) -> Result<Option<Entry>, RecallError> {
        (**self).get(key)
    }
    fn put(&self, key: Key, entry: &Entry) -> Result<(), RecallError> {
        (**self).put(key, entry)
    }
    fn remove(&self, key: Key) -> Result<(), RecallError> {
        (**self).remove(key)
    }
    fn len(&self) -> usize {
        (**self).len()
    }
    fn scan(&self) -> Result<Vec<(Key, Entry)>, RecallError> {
        (**self).scan()
    }
}

impl<T: ThresholdPolicy + ?Sized> ThresholdPolicy for Box<T> {
    fn id(&self) -> &str {
        (**self).id()
    }
    fn decide(&self, ns: &str, top: Option<f32>) -> Verdict {
        (**self).decide(ns, top)
    }
    fn observe(&self, ns: &str, score: f32, outcome: Outcome) {
        (**self).observe(ns, score, outcome)
    }
}

/// A cache whose seams are chosen at runtime — what a server holds behind an `Arc`. All four boxes
/// are `Send + Sync` (their traits require it), so `Arc<DynCache>` is a sound shared handler state.
pub type DynCache =
    SemanticCache<Box<dyn Embedder>, Box<dyn AnnIndex>, Box<dyn Store>, Box<dyn ThresholdPolicy>>;
