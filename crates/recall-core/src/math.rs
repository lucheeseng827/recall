//! Tiny vector helpers shared by the embedder normalization and the brute-force index. Kept in one
//! place so the index and the facade agree on exactly what "cosine" means: a dot product over
//! L2-normalized vectors.

use crate::types::Vector;

/// Dot product. With both operands L2-normalized this equals cosine similarity. Callers guarantee
/// equal length (the index checks; see `index.rs`).
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// L2-normalize in place and return. A zero vector (e.g. an empty prompt under `HashEmbedder`) is
/// returned unchanged rather than producing NaNs — it will simply never match anything.
pub fn normalize(mut v: Vector) -> Vector {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
    v
}
