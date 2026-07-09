//! # recall-index
//!
//! A pure-Rust, zero-dependency **HNSW** (Hierarchical Navigable Small World) implementation of
//! recall-core's [`AnnIndex`](recall_core::AnnIndex) seam — the scale escalation from the exact
//! `BruteForceIndex` oracle (PLAN.md §3-OSS, T3). Search is approximate but sublinear; the gate is
//! `recall@1 >= 0.98` against the brute-force oracle, which is exercised as a test.
//!
//! Why hand-rolled rather than `hnsw_rs`/`instant-distance`: recall's whole pitch is the
//! single-static-binary / air-gap / single-supply-chain property. An external ANN crate drags in a
//! large transitive tree; keeping the graph in-house (like `recall-calibrate` keeps the adaptive
//! engine dependency-free) preserves that envelope even for the at-scale index.
//!
//! Vectors are expected L2-normalized by the caller (the `SemanticCache` facade does this), so a dot
//! product is cosine similarity and `1 - dot` is a valid distance — identical to `BruteForceIndex`,
//! which is what makes the recall comparison meaningful.

#![forbid(unsafe_code)]

mod hnsw;

pub use hnsw::{HnswConfig, HnswIndex};
