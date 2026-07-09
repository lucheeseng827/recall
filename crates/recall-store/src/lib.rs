//! # recall-store
//!
//! Durable [`Store`](recall_core::Store) backends for recall. The seam itself — and the in-memory
//! `MemKv` the MVP uses — live in `recall-core`; this crate adds restart-surviving implementations
//! behind feature flags so the dependency-light, air-gap-clean default build is never perturbed
//! (PLAN.md §3-OSS).
//!
//! Today there is one backend: [`RedbStore`] (feature `redb`) — `redb` is pure-Rust ACID
//! (copy-on-write B-tree with checksums, no WAL, no C dependency), the recommended durable default.
//! `store-fjall` (LSM, high-churn/large-blob) is the future escalation behind its own feature.
#![forbid(unsafe_code)]

#[cfg(feature = "redb")]
pub mod redb_store;

#[cfg(feature = "redb")]
pub use redb_store::RedbStore;
