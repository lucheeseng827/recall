//! The vocabulary the seams speak. Every type here is intentionally cheap to pass across a trait
//! boundary (`Copy` where it can be, owned returns otherwise) so the binary can hold the seams as
//! `Box<dyn _>` and still keep the hot path allocation-light. See PLAN.md §2.2.

use crate::error::RecallError;

/// A dense embedding. Aliased rather than newtyped so `&[f32]` slices flow into the ANN seam with
/// no wrapping; the index never needs to know which embedder produced them.
pub type Vector = Vec<f32>;

/// Stable `"model/name@version"` (plus weight dtype/dim in a real backend). Folded into the
/// namespace so swapping embedders — or an f32↔i8 swap of "the same" model — can never return a
/// stale cross-model hit (PLAN.md §2.3, T2).
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ModelId(pub String);

impl ModelId {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for ModelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// The isolation + correctness boundary. The caller passes a correctly tenant-scoped namespace;
/// the cache enforces partition isolation but does not infer tenancy (PLAN.md §2.3). Validated so
/// `namespace ⊕ embedder-id` can never collide.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Namespace(String);

impl Namespace {
    pub fn new(s: impl Into<String>) -> Result<Self, RecallError> {
        let s = s.into();
        // Reject the join delimiter and empty strings: `ns_key = format!("{ns}\u{1f}{id}")` is only
        // collision-free if `ns` cannot itself contain `\u{1f}` (PLAN.md §2.3).
        if s.is_empty() || s.contains('\u{1f}') {
            return Err(RecallError::InvalidNamespace);
        }
        Ok(Self(s))
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Content-addressed handle to a cached entry: `blake3(len-prefixed ns_key ‖ prompt)`. Length
/// prefixing means `(ns="a", prompt="bc")` and `(ns="ab", prompt="c")` can never collide — a
/// collision here would be a cross-prompt (or cross-tenant) leak (PLAN.md §3-OSS).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Key([u8; 32]);

impl Key {
    /// Derive the stable key for a `(ns_key, prompt)` pair. Length-prefixed to be unambiguous.
    pub fn derive(ns_key: &str, prompt: &str) -> Self {
        let mut h = blake3::Hasher::new();
        h.update(&(ns_key.len() as u64).to_le_bytes());
        h.update(ns_key.as_bytes());
        h.update(&(prompt.len() as u64).to_le_bytes());
        h.update(prompt.as_bytes());
        Self(*h.finalize().as_bytes())
    }

    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Reconstruct a key from its 32 raw bytes. `derive` is the forward direction (compute the key
    /// for a `(ns_key, prompt)` pair); this is the inverse used by a durable `Store` that enumerates
    /// already-stored keys to rehydrate the in-memory index after a restart.
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }
}

/// A scored neighbor from the ANN seam. `Copy` so `hits.first().copied()` and the `Hit` destructure
/// on the hot path stay allocation-free.
#[derive(Clone, Copy, Debug)]
pub struct Scored {
    pub key: Key,
    pub score: f32,
}

/// A cached prompt→completion pair. `prompt` is retained so the optional verify-on-hit guard can
/// reject an obvious false hit before serving (PLAN.md §2.3). `serde`-optional so a durable `Store`
/// can blob it without `recall-core` taking serde unless asked.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct Entry {
    pub prompt: String,
    pub completion: String,
    pub model_id: ModelId,
    /// Unix-epoch SECONDS — never an `Instant`, which is monotonic-with-no-epoch and meaningless
    /// across a restart (PLAN.md §3-OSS "never persist `Instant` for TTL").
    pub created_at_unix: u64,
    /// The raw [`Namespace`] string this entry was stored under. Persisted so a durable `Store` can
    /// be enumerated on startup to rebuild the in-memory ANN index + exact-map (see
    /// `SemanticCache::rehydrate`): the partition key `ns_key = namespace ⊕ embedder.id()` is
    /// otherwise unrecoverable from a `(Key, Entry)` pair. `#[serde(default)]` so a blob written
    /// before this field still deserialises — it simply won't rehydrate (and is re-learned on first
    /// miss).
    #[cfg_attr(feature = "serde", serde(default))]
    pub namespace: String,
}

/// The threshold seam's per-query verdict.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    Hit,
    Miss,
}

/// Feedback an adaptive policy learns from. A no-op for `StaticThreshold`; the seam exists so the
/// adaptive engine (PLAN.md §5) is a pure impl swap with zero facade changes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// A served hit was later judged correct (reinforces the "right" distribution).
    Agree,
    /// A served hit was later judged wrong — a false hit (raises the threshold).
    Wrong,
}

/// The result of a `get()`. On a `Miss` the already-computed query vector travels back so `put()`
/// reuses it — one embed per request (PLAN.md §2.3, step 7).
#[derive(Clone, Debug)]
pub enum Lookup {
    Hit { key: Key, score: f32, entry: Entry },
    Miss { vector: Vector },
}
