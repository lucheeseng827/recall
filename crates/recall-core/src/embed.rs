//! The default embedder. `HashEmbedder` is deterministic, dependency-free, and network-free, so the
//! whole loop is testable and the default build pulls zero ML/network deps (PLAN.md §3-MVP). It is
//! *not* semantic — it is a feature-hashed bag of word + char-3gram features. Identical prompts map
//! to identical vectors (cosine 1.0), which exercises the exact-hit and threshold paths. Real
//! semantics arrive later via `recall-embed`'s `Model2VecEmbedder` (model2vec-rs / potion) behind a
//! feature flag — without touching this crate.

use crate::error::RecallError;
use crate::types::{ModelId, Vector};

/// Object-safe text→vector seam. The ONLY seam allowed a heavy/optional backend, and even then only
/// in a separate feature-gated crate, never compiled into `recall-core` by default (PLAN.md §2.2).
pub trait Embedder: Send + Sync {
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vector>, RecallError>;
    fn dim(&self) -> usize;
    /// Stable model identity, folded into the namespace so a backend swap never returns stale
    /// cross-model hits.
    fn id(&self) -> ModelId;
    /// Convenience for the single-query hot path; defaulted in terms of `embed`.
    fn embed_one(&self, text: &str) -> Result<Vector, RecallError> {
        self.embed(&[text])?
            .into_iter()
            .next()
            .ok_or(RecallError::EmptyEmbedding)
    }
}

/// Deterministic feature-hashing embedder. Buckets word tokens and char 3-grams into a fixed-width
/// vector via a stable blake3-derived hash — no learned weights, no allocation beyond the output.
pub struct HashEmbedder {
    dim: usize,
}

impl HashEmbedder {
    /// `dim` is the feature-hash width. 256 is plenty to keep collisions low for short prompts while
    /// staying cheap; the value is part of the model identity so it can't silently change.
    pub fn new(dim: usize) -> Self {
        assert!(dim > 0, "embedder dim must be > 0");
        Self { dim }
    }

    fn bucket(&self, feature: &str) -> usize {
        // First 8 bytes of blake3(feature) as a u64, reduced into the vector width. Stable across
        // runs and platforms (unlike `DefaultHasher`), which a cache key demands.
        let h = blake3::hash(feature.as_bytes());
        let b = h.as_bytes();
        let v = u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]);
        (v % self.dim as u64) as usize
    }
}

impl Default for HashEmbedder {
    fn default() -> Self {
        Self::new(256)
    }
}

impl Embedder for HashEmbedder {
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vector>, RecallError> {
        let mut out = Vec::with_capacity(texts.len());
        for text in texts {
            let mut v = vec![0.0f32; self.dim];
            let lower = text.to_lowercase();
            // Word features.
            for word in lower.split_whitespace() {
                v[self.bucket(word)] += 1.0;
            }
            // Char 3-gram features — give the embedder a little sub-word/typo robustness so a near
            // paraphrase isn't perfectly orthogonal to its source.
            let chars: Vec<char> = lower.chars().collect();
            for w in chars.windows(3) {
                let gram: String = w.iter().collect();
                v[self.bucket(&gram)] += 1.0;
            }
            out.push(v);
        }
        Ok(out)
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn id(&self) -> ModelId {
        ModelId::new(format!("hash-v1@{}", self.dim))
    }
}
