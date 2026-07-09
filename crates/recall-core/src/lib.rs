//! # recall-core
//!
//! The seams and the in-memory MVP of **recall**, a self-hosted, Rust-native semantic cache for
//! LLM proxies. This crate is the product: a `SemanticCache` facade over four object-safe seams —
//! [`Embedder`], [`AnnIndex`], [`Store`], [`ThresholdPolicy`] — wired in the MVP to deterministic,
//! network-free, dependency-light implementations. Every "real" backend (model2vec/potion
//! embeddings, HNSW, redb/fjall persistence, the adaptive per-region threshold) is a swap-an-impl
//! task behind these seams, not a rewrite. See `PLAN.md` for the full MVP → OSS → paid plan.
//!
//! The default build pulls **zero** ML/network dependencies (`thiserror` + `blake3` only) — that is
//! the single-static-binary, air-gap property recall is built around.
//!
//! ```
//! use recall_core::{SemanticCache, HashEmbedder, BruteForceIndex, MemKv, StaticThreshold,
//!                   Namespace, Lookup};
//!
//! let cache = SemanticCache::new(
//!     HashEmbedder::default(),
//!     BruteForceIndex::new(),
//!     MemKv::new(),
//!     StaticThreshold::new(0.9),
//! );
//! let ns = Namespace::new("tenant-a/chat").unwrap();
//!
//! // First call misses; the caller fills it from the real LLM, then stores it.
//! let prompt = "How do I reset my password?";
//! if let Lookup::Miss { vector } = cache.get(&ns, prompt).unwrap() {
//!     cache.put(&ns, prompt, "Click 'Forgot password'.", &vector).unwrap();
//! }
//!
//! // Next identical call is an exact hit — no LLM, no embed.
//! assert!(matches!(cache.get(&ns, prompt).unwrap(), Lookup::Hit { .. }));
//! ```

#![forbid(unsafe_code)]

pub mod boxed;
pub mod cache;
pub mod embed;
pub mod error;
pub mod index;
pub mod kv;
pub mod math;
pub mod policy;
pub mod types;

/// Test-support seam doubles (fault injection, edge-case policies). Compiled only under `test` or the
/// `test-support` feature — never in the default/air-gap build.
#[cfg(any(test, feature = "test-support"))]
pub mod testing;

pub use boxed::DynCache;
pub use cache::SemanticCache;
pub use embed::{Embedder, HashEmbedder};
pub use error::RecallError;
pub use index::{AnnIndex, BruteForceIndex};
pub use kv::{MemKv, Store};
pub use policy::{StaticThreshold, ThresholdPolicy};
pub use types::{Entry, Key, Lookup, ModelId, Namespace, Outcome, Scored, Vector, Verdict};

#[cfg(any(test, feature = "test-support"))]
pub use testing::{AlwaysHit, FailingIndex};
