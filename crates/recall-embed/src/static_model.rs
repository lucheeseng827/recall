//! The `static` embedder backend: model2vec / potion static embeddings via `model2vec-rs`.

use std::path::Path;

use model2vec_rs::model::StaticModel;
use recall_core::{Embedder, ModelId, RecallError, Vector};

/// A static (model2vec/potion) embedder. Encoding is a per-token embedding lookup + mean pool — no
/// attention, no matmul — so a short query embeds in sub-millisecond time, leaving the hit budget to
/// ANN search (PLAN.md §1).
pub struct Model2VecEmbedder {
    model: StaticModel,
    dim: usize,
    id: ModelId,
}

impl Model2VecEmbedder {
    /// Load a local model2vec/potion model directory (expects `config.json`, `tokenizer.json`, and
    /// `model.safetensors`). **Local only** — the `hf-hub` feature is intentionally off, so a path
    /// that does not exist returns an error rather than reaching the network.
    ///
    /// `id` becomes the [`Embedder::id`], which the cache folds into every namespace; use a stable
    /// `"name@version"` (e.g. `"potion-base-8M@1"`) so a model swap never returns stale hits.
    pub fn load_local(path: impl AsRef<Path>, id: impl Into<String>) -> Result<Self, RecallError> {
        let path = path.as_ref();
        let model = StaticModel::from_pretrained(path, None, None, None).map_err(|e| {
            RecallError::Backend(format!("model2vec load failed for {path:?}: {e}"))
        })?;
        // model2vec-rs exposes no dimension accessor, so probe once with a non-empty string.
        let dim = model.encode_single("dimension probe").len();
        if dim == 0 {
            return Err(RecallError::Backend(
                "model2vec produced a zero-dimension embedding".into(),
            ));
        }
        Ok(Self {
            model,
            dim,
            id: ModelId::new(id),
        })
    }
}

impl Embedder for Model2VecEmbedder {
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vector>, RecallError> {
        // model2vec-rs takes owned Strings; short cache queries make this allocation negligible.
        let owned: Vec<String> = texts.iter().map(|s| s.to_string()).collect();
        let out = self.model.encode(&owned);
        // Defend the index's dimension invariant: a ragged result would be a model bug.
        if let Some(bad) = out.iter().find(|v| v.len() != self.dim) {
            return Err(RecallError::DimensionMismatch {
                expected: self.dim,
                got: bad.len(),
            });
        }
        Ok(out)
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn id(&self) -> ModelId {
        self.id.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use recall_core::Embedder;

    /// Live model test — only runs when `RECALL_MODEL2VEC_PATH` points at a local model directory,
    /// so CI without a model still passes. Run with e.g.:
    ///   RECALL_MODEL2VEC_PATH=/path/to/potion-base-8M cargo test -p recall-embed --features static
    #[test]
    fn loads_local_model_and_embeds() {
        let Ok(path) = std::env::var("RECALL_MODEL2VEC_PATH") else {
            eprintln!("skipping: set RECALL_MODEL2VEC_PATH to a local model2vec dir to run");
            return;
        };
        let emb = Model2VecEmbedder::load_local(&path, "test-model@1").expect("load");
        assert!(emb.dim() > 0);
        let v = emb.embed(&["hello world", "hello world"]).expect("embed");
        assert_eq!(v.len(), 2);
        assert_eq!(v[0].len(), emb.dim());
        // Identical inputs embed identically.
        assert_eq!(v[0], v[1]);
    }

    /// Local-only guarantee: a non-existent path must error, never reach the network.
    #[test]
    fn missing_path_errors_offline() {
        let err = Model2VecEmbedder::load_local("/no/such/recall/model", "x@1");
        assert!(err.is_err(), "a missing model dir must error, not download");
    }
}
