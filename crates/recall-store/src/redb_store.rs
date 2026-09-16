//! A durable, restart-surviving [`Store`] backed by [`redb`] — pure-Rust ACID (copy-on-write B-tree
//! with XXH3 checksums, no WAL, no C dependency), the recommended durable default (PLAN.md §3-OSS).
//!
//! Layout: one table `recall.entries` mapping the 32-byte content [`Key`] to a `serde_json`-encoded
//! [`Entry`]. Each `put`/`remove` is its own committed transaction, so a crash leaves a consistent
//! database; `get` reads a lock-free snapshot. `Database` is `Send + Sync`, so one `RedbStore` is
//! shared across all proxy workers.

use std::path::Path;
use std::sync::Arc;

use recall_core::{Entry, Key, RecallError, Store};
use redb::{Database, ReadableTable, ReadableTableMetadata, TableDefinition};

/// `Key (32 bytes) → serde_json(Entry)`.
const ENTRIES: TableDefinition<&[u8], &[u8]> = TableDefinition::new("recall.entries");

/// A single-row table holding the savings-ledger snapshot as an opaque blob (the codec belongs to
/// the caller — this store persists bytes, it does not interpret the ledger).
const SAVINGS: TableDefinition<&str, &[u8]> = TableDefinition::new("recall.savings");
const SAVINGS_ROW: &str = "snapshot";

/// A `Store` whose entries survive a process restart. Open once and share — `Clone` is a cheap
/// `Arc` bump, so one handle can live inside the cache (as `Box<dyn Store>`) while another feeds
/// the savings-ledger flusher.
#[derive(Clone)]
pub struct RedbStore {
    db: Arc<Database>,
}

/// Wrap any redb / serde error into the seam's opaque backend error — `recall-core` never takes a
/// dependency on redb's error types.
fn backend<E: std::fmt::Display>(e: E) -> RecallError {
    RecallError::Backend(e.to_string())
}

impl RedbStore {
    /// Open the database at `path`, creating it (and both tables) if absent. Creating the tables
    /// up front means the first read on a brand-new database returns `None` rather than a
    /// "no such table" error.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, RecallError> {
        let db = Database::create(path).map_err(backend)?;
        let wtxn = db.begin_write().map_err(backend)?;
        wtxn.open_table(ENTRIES).map_err(backend)?;
        wtxn.open_table(SAVINGS).map_err(backend)?;
        wtxn.commit().map_err(backend)?;
        Ok(Self { db: Arc::new(db) })
    }

    /// Persist the savings-ledger snapshot blob (one committed transaction; the previous snapshot
    /// is atomically replaced — a crash leaves either the old or the new blob, never a torn one).
    pub fn save_savings_blob(&self, bytes: &[u8]) -> Result<(), RecallError> {
        let wtxn = self.db.begin_write().map_err(backend)?;
        {
            let mut table = wtxn.open_table(SAVINGS).map_err(backend)?;
            table.insert(SAVINGS_ROW, bytes).map_err(backend)?;
        }
        wtxn.commit().map_err(backend)?;
        Ok(())
    }

    /// Read back the persisted savings-ledger snapshot blob, if one was ever saved. A database
    /// created before the savings table existed reports `None` (open() creates the empty table),
    /// so an upgraded binary starts a fresh ledger rather than erroring.
    pub fn load_savings_blob(&self) -> Result<Option<Vec<u8>>, RecallError> {
        let rtxn = self.db.begin_read().map_err(backend)?;
        let table = rtxn.open_table(SAVINGS).map_err(backend)?;
        Ok(table
            .get(SAVINGS_ROW)
            .map_err(backend)?
            .map(|guard| guard.value().to_vec()))
    }
}

impl Store for RedbStore {
    fn get(&self, key: Key) -> Result<Option<Entry>, RecallError> {
        let rtxn = self.db.begin_read().map_err(backend)?;
        let table = rtxn.open_table(ENTRIES).map_err(backend)?;
        match table.get(key.as_bytes().as_slice()).map_err(backend)? {
            Some(guard) => Ok(Some(
                serde_json::from_slice(guard.value()).map_err(backend)?,
            )),
            None => Ok(None),
        }
    }

    fn put(&self, key: Key, entry: &Entry) -> Result<(), RecallError> {
        let bytes = serde_json::to_vec(entry).map_err(backend)?;
        let wtxn = self.db.begin_write().map_err(backend)?;
        {
            let mut table = wtxn.open_table(ENTRIES).map_err(backend)?;
            table
                .insert(key.as_bytes().as_slice(), bytes.as_slice())
                .map_err(backend)?;
        }
        wtxn.commit().map_err(backend)?;
        Ok(())
    }

    fn remove(&self, key: Key) -> Result<(), RecallError> {
        let wtxn = self.db.begin_write().map_err(backend)?;
        {
            let mut table = wtxn.open_table(ENTRIES).map_err(backend)?;
            table.remove(key.as_bytes().as_slice()).map_err(backend)?;
        }
        wtxn.commit().map_err(backend)?;
        Ok(())
    }

    fn len(&self) -> usize {
        // The seam's `len` is infallible; on a read error report 0 — the next get/put surfaces it.
        (|| -> Result<usize, RecallError> {
            let rtxn = self.db.begin_read().map_err(backend)?;
            let table = rtxn.open_table(ENTRIES).map_err(backend)?;
            Ok(table.len().map_err(backend)? as usize)
        })()
        .unwrap_or(0)
    }

    /// Enumerate the whole table so `SemanticCache::rehydrate` can rebuild the in-memory index +
    /// exact-map on startup — this is what turns `--store redb` from KV-only into a cache whose
    /// *lookups* survive a restart. A blob written before `Entry::namespace` existed still decodes
    /// (the field is `#[serde(default)]`); rehydrate then skips it (empty namespace won't parse).
    fn scan(&self) -> Result<Vec<(Key, Entry)>, RecallError> {
        let rtxn = self.db.begin_read().map_err(backend)?;
        let table = rtxn.open_table(ENTRIES).map_err(backend)?;
        let mut out = Vec::with_capacity(table.len().map_err(backend)? as usize);
        for row in table.iter().map_err(backend)? {
            let (k, v) = row.map_err(backend)?;
            let kb: [u8; 32] = k
                .value()
                .try_into()
                .map_err(|_| RecallError::Backend("stored key is not 32 bytes".into()))?;
            let entry: Entry = serde_json::from_slice(v.value()).map_err(backend)?;
            out.push((Key::from_bytes(kb), entry));
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use recall_core::{Entry, Key, ModelId, Store};

    fn entry(completion: &str) -> Entry {
        Entry {
            prompt: "the prompt".into(),
            completion: completion.into(),
            model_id: ModelId::new("test@1"),
            created_at_unix: 0,
            namespace: "ns".into(),
        }
    }

    // A unique temp path per test (process id + name) so parallel tests don't collide.
    fn temp_db(tag: &str) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!("recall-redb-{}-{tag}.redb", std::process::id()));
        let _ = std::fs::remove_file(&p);
        p
    }

    #[test]
    fn put_get_remove_and_len() {
        let path = temp_db("crud");
        let store = RedbStore::open(&path).unwrap();
        let k = Key::derive("ns", "the prompt");

        assert_eq!(store.len(), 0);
        assert!(store.get(k).unwrap().is_none());

        store.put(k, &entry("answer")).unwrap();
        assert_eq!(store.len(), 1);
        assert_eq!(store.get(k).unwrap().unwrap().completion, "answer");

        store.remove(k).unwrap();
        assert!(store.get(k).unwrap().is_none());
        assert_eq!(store.len(), 0);

        std::fs::remove_file(&path).ok();
    }

    // The whole point of the durable backend: entries outlive the process that wrote them.
    #[test]
    fn entries_survive_reopen() {
        let path = temp_db("reopen");
        let k = Key::derive("ns", "durable");
        {
            let store = RedbStore::open(&path).unwrap();
            store.put(k, &entry("persisted")).unwrap();
        } // drop closes the database (simulating a restart)
        {
            let store = RedbStore::open(&path).unwrap(); // reopen
            assert_eq!(store.get(k).unwrap().unwrap().completion, "persisted");
        }
        std::fs::remove_file(&path).ok();
    }

    // `scan` is what `SemanticCache::rehydrate` consumes on startup: it must enumerate every stored
    // pair, and the pairs must outlive the process that wrote them (so a reopened DB yields them).
    #[test]
    fn scan_enumerates_all_entries_across_reopen() {
        let path = temp_db("scan");
        let k1 = Key::derive("ns", "one");
        let k2 = Key::derive("ns", "two");
        {
            let store = RedbStore::open(&path).unwrap();
            store.put(k1, &entry("first")).unwrap();
            store.put(k2, &entry("second")).unwrap();
        } // "restart"

        let store = RedbStore::open(&path).unwrap();
        let mut scanned = store.scan().unwrap();
        assert_eq!(scanned.len(), 2, "scan returns every persisted entry");
        // Order is btree key order, not insertion order — sort by completion to assert deterministically.
        scanned.sort_by(|a, b| a.1.completion.cmp(&b.1.completion));
        let keys: Vec<Key> = scanned.iter().map(|(k, _)| *k).collect();
        assert!(keys.contains(&k1) && keys.contains(&k2), "keys round-trip");
        assert_eq!(scanned[0].1.completion, "first");
        assert_eq!(scanned[1].1.completion, "second");

        std::fs::remove_file(&path).ok();
    }

    // The savings ledger blob: saved, atomically replaced, and read back across a reopen — the
    // persistence primitive behind "the $ saved figure survives a deploy". The store deliberately
    // knows nothing about the ledger's shape: it persists opaque bytes, and the codec belongs to
    // whoever owns the counters (recall-proxy), which is why this test asserts on bytes.
    #[test]
    fn savings_blob_replaces_and_survives_reopen() {
        let path = temp_db("savings-blob");
        {
            let store = RedbStore::open(&path).unwrap();
            assert!(
                store.load_savings_blob().unwrap().is_none(),
                "a fresh database has no snapshot"
            );
            store
                .save_savings_blob(br#"{"version":1,"hits":1}"#)
                .unwrap();
            // A second save REPLACES the first (single-row semantics) — the ledger is a current
            // value, not an append-only log, so a reopen must never see a stale generation.
            store
                .save_savings_blob(br#"{"version":1,"hits":2}"#)
                .unwrap();
        } // drop closes the database (simulating a restart)

        let store = RedbStore::open(&path).unwrap(); // reopen
        let blob = store
            .load_savings_blob()
            .unwrap()
            .expect("snapshot survives the reopen");
        assert_eq!(
            blob,
            br#"{"version":1,"hits":2}"#.to_vec(),
            "the LATEST snapshot is the one that persisted"
        );

        // Arbitrary bytes round-trip: the store must not assume UTF-8 or any structure.
        store.save_savings_blob(&[0xffu8, 0x00, 0x7f]).unwrap();
        assert_eq!(
            store.load_savings_blob().unwrap().unwrap(),
            vec![0xffu8, 0x00, 0x7f]
        );

        std::fs::remove_file(&path).ok();
    }

    // Clone shares the same underlying database: a blob saved through one handle is visible
    // through the other — the shape serve uses (one handle in the cache, one in the flusher).
    #[test]
    fn cloned_handle_shares_the_database() {
        let path = temp_db("clone-shared");
        let a = RedbStore::open(&path).unwrap();
        let b = a.clone();
        b.save_savings_blob(b"ledger-bytes").unwrap();
        assert_eq!(
            a.load_savings_blob().unwrap().as_deref(),
            Some(b"ledger-bytes".as_ref()),
            "both handles see the same table"
        );
        a.put(Key::derive("ns", "p"), &entry("x")).unwrap();
        assert_eq!(b.len(), 1);
        std::fs::remove_file(&path).ok();
    }

    // The full cross-restart contract, end-to-end on the real durable backend — the loop that
    // `recall serve --store redb` runs. Store a completion through a cache, drop it (closing the
    // db), reopen into a *fresh* cache with a cold index, and confirm the lookup is invisible until
    // `rehydrate` rebuilds the in-memory structures from the persisted blobs.
    #[test]
    fn cache_lookups_survive_restart_via_rehydrate() {
        use recall_core::{
            BruteForceIndex, HashEmbedder, Lookup, Namespace, SemanticCache, StaticThreshold,
        };

        let path = temp_db("rehydrate-e2e");
        let ns = Namespace::new("tenant-a/chat").unwrap();
        let prompt = "what is the capital of france";

        // Session 1: store through a cache backed by the durable store, then drop (closes the db).
        {
            let cache = SemanticCache::new(
                HashEmbedder::default(),
                BruteForceIndex::new(),
                RedbStore::open(&path).unwrap(),
                StaticThreshold::new(0.5),
            );
            cache.put_embedding(&ns, prompt, "Paris.").unwrap();
        }

        // Session 2 ("restart"): fresh cache + cold index over the reopened durable store.
        let cache = SemanticCache::new(
            HashEmbedder::default(),
            BruteForceIndex::new(),
            RedbStore::open(&path).unwrap(),
            StaticThreshold::new(0.5),
        );
        assert!(
            matches!(cache.get(&ns, prompt).unwrap(), Lookup::Miss { .. }),
            "the persisted blob is invisible to a cold index until rehydrated"
        );
        assert_eq!(cache.rehydrate().unwrap(), 1, "the one entry rehydrates");
        match cache.get(&ns, prompt).unwrap() {
            Lookup::Hit { entry, .. } => assert_eq!(entry.completion, "Paris."),
            Lookup::Miss { .. } => panic!("a rehydrated lookup must hit after restart"),
        }

        std::fs::remove_file(&path).ok();
    }
}
