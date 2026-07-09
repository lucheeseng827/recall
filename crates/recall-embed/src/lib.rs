//! Real embedder backends for recall, behind feature flags.
//!
//! - **default build**: nothing heavy — this crate just re-exports [`HashEmbedder`] from
//!   `recall-core`, so depending on `recall-embed` never silently pulls ML deps.
//! - **`static` feature**: [`Model2VecEmbedder`], a model2vec/potion static embedder — a token-vector
//!   lookup with mean pooling, sub-millisecond per short query, no transformer forward pass. This is
//!   recall's recommended default embedder (PLAN.md §1 embeddings). The model is loaded from a local
//!   directory only; the `hf-hub` feature is disabled, so there is no runtime network.
//!
//! See PLAN.md §1 for why static embeddings are the right primitive for a *cache key* (determinism +
//! latency matter more than SOTA semantics).

pub use recall_core::HashEmbedder;

#[cfg(feature = "static")]
mod static_model;
#[cfg(feature = "static")]
pub use static_model::Model2VecEmbedder;
