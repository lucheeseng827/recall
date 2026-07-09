//! # recall-calibrate
//!
//! recall's adaptive threshold engine (PLAN.md §5). A single global cosine cutoff is
//! structurally wrong across a real embedding space: dense regions sit at high baseline similarity
//! (a fixed τ admits false hits) while sparse regions sit lower (the same τ rejects true hits), the
//! cost of a false hit is workload-dependent, and a static cutoff never learns. [`AdaptiveThreshold`]
//! replaces the magic number with a per-namespace bound that targets an operator-chosen **false-hit
//! rate**, learned from feedback when it is available and safe (≈ a per-embedder prior) when it is
//! not.
//!
//! It is a pure-Rust [`ThresholdPolicy`](recall_core::ThresholdPolicy) implementation — math plus
//! counters, no I/O, no license surface — that drops in behind the existing seam with zero facade
//! changes, exactly where `recall-core`'s `StaticThreshold` sits.
#![forbid(unsafe_code)]

pub mod adaptive;
pub mod moments;
pub mod stats;

pub use adaptive::{AdaptiveConfig, AdaptiveThreshold};
pub use moments::EwMoments;
