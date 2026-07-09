//! The blob-store seam and its MVP implementation. `MemKv` is a plain in-memory map; the durable
//! backends (`store-redb` pure-Rust ACID, `store-fjall` LSM) live in `recall-store` behind feature
//! flags and implement this same trait (PLAN.md §3-OSS).

use std::collections::HashMap;
use std::sync::RwLock;

use crate::error::RecallError;
use crate::types::{Entry, Key};

/// Object-safe `Key → Entry` store. Kept synchronous in core; a durable async backend can wrap its
/// own runtime behind this signature (a server bridges sync↔async via `spawn_blocking`).
pub trait Store: Send + Sync {
    fn get(&self, key: Key) -> Result<Option<Entry>, RecallError>;
    fn put(&self, key: Key, entry: &Entry) -> Result<(), RecallError>;
    fn remove(&self, key: Key) -> Result<(), RecallError>;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Enumerate every `(Key, Entry)` so `SemanticCache::rehydrate` can rebuild the in-memory ANN
    /// index + exact-map from a durable store after a restart. Default: unsupported — a backend that
    /// cannot (or need not) enumerate returns `Backend`, and rehydration is then a no-op. `MemKv`
    /// and the durable `recall-store` backends override it. Returning the full set is fine because
    /// this runs once at startup, not on the hot path.
    fn scan(&self) -> Result<Vec<(Key, Entry)>, RecallError> {
        Err(RecallError::Backend(
            "this store does not support scan() / rehydration".into(),
        ))
    }
}

/// In-memory store: `RwLock<HashMap<Key, Entry>>`. Durable, restart-surviving persistence is the
/// `recall-store` job; this is the MVP/test backend.
#[derive(Default)]
pub struct MemKv {
    map: RwLock<HashMap<Key, Entry>>,
}

impl MemKv {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Store for MemKv {
    fn get(&self, key: Key) -> Result<Option<Entry>, RecallError> {
        Ok(self.map.read().unwrap().get(&key).cloned())
    }

    fn put(&self, key: Key, entry: &Entry) -> Result<(), RecallError> {
        self.map.write().unwrap().insert(key, entry.clone());
        Ok(())
    }

    fn remove(&self, key: Key) -> Result<(), RecallError> {
        self.map.write().unwrap().remove(&key);
        Ok(())
    }

    fn len(&self) -> usize {
        self.map.read().unwrap().len()
    }

    fn scan(&self) -> Result<Vec<(Key, Entry)>, RecallError> {
        Ok(self
            .map
            .read()
            .unwrap()
            .iter()
            .map(|(k, v)| (*k, v.clone()))
            .collect())
    }
}
