//! The single error type the seams return. Deliberately tiny: `recall-core` has no I/O and no
//! network, so the only failures are programmer/contract violations (bad namespace, dim mismatch)
//! and whatever a real backend wraps into `Backend`.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum RecallError {
    /// A namespace was empty or contained the `U+001F` join delimiter, which would make the
    /// `namespace ⊕ embedder-id` key derivation collision-prone. Rejected up front (PLAN.md §2.3).
    #[error("invalid namespace: must be non-empty and must not contain U+001F")]
    InvalidNamespace,

    /// An `Embedder` returned no vectors for a single-text `embed_one`.
    #[error("embedder returned an empty embedding")]
    EmptyEmbedding,

    /// Query/stored vector dimensions disagree — almost always a backend wired with the wrong
    /// embedder. Surfaced rather than silently truncated so cross-model bugs are loud.
    #[error("vector dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    /// Escape hatch for a real (durable / FFI) backend to wrap its own error without `recall-core`
    /// taking a dependency on it.
    #[error("backend error: {0}")]
    Backend(String),
}
