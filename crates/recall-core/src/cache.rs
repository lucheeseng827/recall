//! `SemanticCache` — the facade that ties the four seams together with two methods, `get()` and
//! `put()`. The cache **never calls the LLM**: on a miss it returns the already-computed query
//! vector, and the caller hands the fresh completion back via `put()`. That keeps `recall-core`
//! network-free and air-gap clean (PLAN.md §2.2, §2.3).

use std::collections::HashMap;
use std::sync::RwLock;

use crate::embed::Embedder;
use crate::error::RecallError;
use crate::index::AnnIndex;
use crate::kv::Store;
use crate::math::normalize;
use crate::policy::ThresholdPolicy;
use crate::types::{Entry, Key, Lookup, Namespace, Outcome, Scored, Verdict};

/// Generic over the four seams so the MVP wires concrete in-memory impls while the binary can hold
/// `Box<dyn _>` and pick backends at runtime. No `async` here — a server bridges to its runtime.
pub struct SemanticCache<E, I, S, P>
where
    E: Embedder,
    I: AnnIndex,
    S: Store,
    P: ThresholdPolicy,
{
    embedder: E,
    index: I,
    kv: S,
    policy: P,
    /// `(ns_key, prompt-hash) → Key`: an O(1) *certain* hit with zero ANN/threshold false-hit risk,
    /// and it skips the embed entirely — the cheapest correctness win (PLAN.md §2.3, step 1).
    exact: RwLock<HashMap<(String, [u8; 32]), Key>>,
    top_k: usize,
    /// When true, a neighbor above τ is only served if its stored prompt equals the query exactly —
    /// an optional guard against obvious false hits. Off by default in the MVP; the seam exists.
    verify_on_hit: bool,
}

impl<E, I, S, P> SemanticCache<E, I, S, P>
where
    E: Embedder,
    I: AnnIndex,
    S: Store,
    P: ThresholdPolicy,
{
    pub fn new(embedder: E, index: I, kv: S, policy: P) -> Self {
        Self {
            embedder,
            index,
            kv,
            policy,
            exact: RwLock::new(HashMap::new()),
            top_k: 5,
            verify_on_hit: false,
        }
    }

    /// Builder knob: how many neighbors the ANN seam returns (only the top one decides a hit today;
    /// the extra neighbors feed the adaptive policy's 2nd-neighbor ambiguity gate later — §5.3).
    pub fn with_top_k(mut self, k: usize) -> Self {
        self.top_k = k.max(1);
        self
    }

    /// Builder knob: enable the exact-equality verify-on-hit guard.
    pub fn with_verify_on_hit(mut self, on: bool) -> Self {
        self.verify_on_hit = on;
        self
    }

    /// `namespace ⊕ embedder.id()` — the model identity is folded in so a backend swap never returns
    /// a stale cross-model hit (PLAN.md §2.3, step 3). `\u{1f}` is collision-safe because
    /// `Namespace::new` rejects it.
    fn ns_key(&self, ns: &Namespace) -> String {
        format!("{}\u{1f}{}", ns.as_str(), self.embedder.id())
    }

    /// THE read path: exact shortcut → embed → search → decide → {verify+serve | miss}.
    pub fn get(&self, ns: &Namespace, prompt: &str) -> Result<Lookup, RecallError> {
        let nsk = self.ns_key(ns);

        // 1. Exact-match shortcut: a certain hit, no embed, no false-hit risk.
        let phash = *blake3::hash(prompt.as_bytes()).as_bytes();
        if let Some(&key) = self.exact.read().unwrap().get(&(nsk.clone(), phash)) {
            if let Some(entry) = self.kv.get(key)? {
                if entry.prompt == prompt {
                    return Ok(Lookup::Hit {
                        key,
                        score: 1.0,
                        entry,
                    });
                }
            }
        }

        // 2. Semantic path — exactly one embed per request.
        let vector = normalize(self.embedder.embed_one(prompt)?);
        let hits = self.index.search(&nsk, &vector, self.top_k)?;
        let top = hits.first().copied();

        // Pair the verdict with the actual top neighbor. A pluggable `ThresholdPolicy` could return
        // `Verdict::Hit` even when `top` is `None`; treat that as a miss rather than unwrapping, so a
        // buggy policy can never panic the hot path.
        match (self.policy.decide(&nsk, top.map(|s| s.score)), top) {
            (Verdict::Hit, Some(Scored { key, score })) => {
                match self.kv.get(key)? {
                    // Serve unless verify-on-hit is on and the stored prompt differs.
                    Some(entry) if !self.verify_on_hit || entry.prompt == prompt => {
                        Ok(Lookup::Hit { key, score, entry })
                    }
                    // index/kv drift or verify rejected → miss; the query vector is reused by put().
                    _ => Ok(Lookup::Miss { vector }),
                }
            }
            (Verdict::Hit, None) | (Verdict::Miss, _) => Ok(Lookup::Miss { vector }),
        }
    }

    /// THE write path: store the fresh completion, then index it (store-BEFORE-index so a concurrent
    /// `get()` never resolves a hit to a missing entry — PLAN.md §2.3, step 8→9). `vector` is the
    /// one returned by the preceding `Miss`, so there is no second embed.
    pub fn put(
        &self,
        ns: &Namespace,
        prompt: &str,
        completion: &str,
        vector: &[f32],
    ) -> Result<Key, RecallError> {
        let nsk = self.ns_key(ns);
        let key = Key::derive(&nsk, prompt);
        let entry = Entry {
            prompt: prompt.to_string(),
            completion: completion.to_string(),
            model_id: self.embedder.id(),
            created_at_unix: now_unix(),
            namespace: ns.as_str().to_string(),
        };

        self.kv.put(key, &entry)?; // 8. store first
                                   // 9. then index — if a durable index backend fails the insert, roll back the just-written KV
                                   // blob so a failed `put` leaves no unreachable orphan (store and index stay consistent).
        if let Err(e) = self.index.insert(&nsk, key, vector) {
            let _ = self.kv.remove(key);
            return Err(e);
        }
        self.exact
            .write()
            .unwrap()
            .insert((nsk, *blake3::hash(prompt.as_bytes()).as_bytes()), key);
        Ok(key)
    }

    /// Convenience: embed `prompt` and store it (used when there is no prior `Miss` vector to reuse,
    /// e.g. warming the cache). Costs one embed.
    pub fn put_embedding(
        &self,
        ns: &Namespace,
        prompt: &str,
        completion: &str,
    ) -> Result<Key, RecallError> {
        let vector = normalize(self.embedder.embed_one(prompt)?);
        self.put(ns, prompt, completion, &vector)
    }

    /// Rebuild the in-memory ANN index + exact-map from a durable [`Store`] so cached *lookups* —
    /// not just the persisted KV blobs — survive a restart. Without this, reopening a durable store
    /// leaves `get()` blind to every persisted entry (the index/exact-map start empty), which is why
    /// `--store redb` was KV-only until now.
    ///
    /// Scans the store and, for each entry written under the **current** embedder, re-embeds its
    /// prompt — deterministic, so it reproduces the originally-indexed vector — and re-inserts it
    /// into the index + exact-map under `ns_key = namespace ⊕ embedder.id()`. Re-embedding (rather
    /// than persisting vectors) keeps the durable record small at the cost of O(entries) embeds at
    /// startup; for the in-process static embedder that is sub-millisecond each.
    ///
    /// Entries stored under a *different* embedder are skipped: `ns_key` folds in `embedder.id()`,
    /// so resurrecting them into the current embedder's space would be exactly the stale cross-model
    /// hit that fold is there to prevent. A record whose namespace no longer parses is skipped too.
    ///
    /// **Best-effort**: a per-entry re-embed or index failure skips just that entry and continues,
    /// so one bad record can never strand the rest in a half-warm state. The only hard error is a
    /// store-`scan` failure (nothing could be read at all); the binary treats that, and only that, as
    /// a genuinely cold index. Returns the number of entries rehydrated.
    pub fn rehydrate(&self) -> Result<usize, RecallError> {
        let my_id = self.embedder.id();
        let mut rehydrated = 0usize;
        for (key, entry) in self.kv.scan()? {
            if entry.model_id != my_id {
                continue; // stored under a different embedder — skip (would be a cross-model hit)
            }
            let Ok(ns) = Namespace::new(entry.namespace.clone()) else {
                continue; // defensively skip a record with an unparseable/empty namespace
            };
            let nsk = self.ns_key(&ns);
            // Skip (don't abort) an entry that can't be re-embedded or indexed — keeps rehydration
            // best-effort so a single failure doesn't leave a partially-warm cache behind an `Err`.
            let Ok(vector) = self.embedder.embed_one(&entry.prompt) else {
                continue;
            };
            let vector = normalize(vector);
            if self.index.insert(&nsk, key, &vector).is_err() {
                continue;
            }
            self.exact.write().unwrap().insert(
                (nsk, *blake3::hash(entry.prompt.as_bytes()).as_bytes()),
                key,
            );
            rehydrated += 1;
        }
        Ok(rehydrated)
    }

    /// Feed an adaptive [`ThresholdPolicy`] the outcome of a previously served hit so it can retune
    /// its per-namespace cutoff toward the operator's false-hit target (PLAN.md §5). This is the
    /// signal that turns the adaptive engine from inert (resting at cold-start τ) into learning. A
    /// no-op for `StaticThreshold` (its `observe` is the default no-op).
    ///
    /// Off the hot path. `score` is the similarity of the hit being judged — the value `get()`
    /// returned on the `Hit` — and `ns` must be the namespace that hit was served under, so the
    /// feedback lands on the right per-namespace state (it is folded into the same `ns_key` the
    /// `decide`/`observe` seam is keyed by).
    pub fn observe(&self, ns: &Namespace, score: f32, outcome: Outcome) {
        self.policy.observe(&self.ns_key(ns), score, outcome);
    }

    /// Read-only accessors so a binary/bench can report state without owning the seams.
    pub fn entries(&self) -> usize {
        self.kv.len()
    }
    pub fn policy_id(&self) -> &str {
        self.policy.id()
    }
}

/// Unix-epoch SECONDS. Stored on `Entry` so TTL math survives a restart (an `Instant` would not).
fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::HashEmbedder;
    use crate::index::BruteForceIndex;
    use crate::kv::MemKv;
    use crate::policy::StaticThreshold;

    fn cache(
        tau: f32,
        verify: bool,
    ) -> SemanticCache<HashEmbedder, BruteForceIndex, MemKv, StaticThreshold> {
        SemanticCache::new(
            HashEmbedder::default(),
            BruteForceIndex::new(),
            MemKv::new(),
            StaticThreshold::new(tau),
        )
        .with_verify_on_hit(verify)
    }

    fn ns() -> Namespace {
        Namespace::new("tenant-a/chat").unwrap()
    }

    // The MVP exit criterion: miss → put → exact-hit.
    #[test]
    fn miss_then_put_then_exact_hit() {
        let c = cache(0.9, false);
        let prompt = "How do I reset my password?";

        let v = match c.get(&ns(), prompt).unwrap() {
            Lookup::Miss { vector } => vector,
            Lookup::Hit { .. } => panic!("empty cache must miss"),
        };
        c.put(&ns(), prompt, "Click 'Forgot password'.", &v)
            .unwrap();

        match c.get(&ns(), prompt).unwrap() {
            Lookup::Hit { score, entry, .. } => {
                assert_eq!(score, 1.0, "identical prompt is an exact hit");
                assert_eq!(entry.completion, "Click 'Forgot password'.");
            }
            Lookup::Miss { .. } => panic!("identical prompt must hit"),
        }
    }

    // A near-paraphrase hits at a permissive τ and misses at a strict τ — proving the threshold is
    // the knob, and that the HashEmbedder gives paraphrases non-trivial (but <1.0) similarity.
    #[test]
    fn paraphrase_hits_below_tau_misses_above() {
        let stored = "what is the capital of france";
        let para = "what is the capital city of france";

        // Store the canonical prompt, then query a close paraphrase.
        let loose = cache(0.5, false);
        loose.put_embedding(&ns(), stored, "Paris.").unwrap();
        assert!(
            matches!(loose.get(&ns(), para).unwrap(), Lookup::Hit { .. }),
            "a close paraphrase should hit at τ=0.5"
        );

        // The same paraphrase is not identical, so a near-1.0 cutoff must reject it.
        let strict = cache(0.999, false);
        strict.put_embedding(&ns(), stored, "Paris.").unwrap();
        assert!(
            !matches!(strict.get(&ns(), para).unwrap(), Lookup::Hit { .. }),
            "the same paraphrase should miss at τ=0.999"
        );
    }

    // verify-on-hit rejects a forced near-collision: two different prompts that the index ranks as
    // neighbors must not serve each other's answer when the guard is on.
    #[test]
    fn verify_on_hit_rejects_non_identical() {
        // τ=0.0 forces every search to "hit" on the top neighbor, isolating the verify guard.
        let c = cache(0.0, true);
        c.put_embedding(&ns(), "alpha beta gamma", "ANSWER-A")
            .unwrap();

        // A different prompt: with verify-on-hit, even a forced top-neighbor hit is rejected to a
        // miss because the stored prompt != the query.
        match c.get(&ns(), "delta epsilon zeta").unwrap() {
            Lookup::Miss { .. } => {}
            Lookup::Hit { entry, .. } => {
                panic!(
                    "verify-on-hit must reject a non-identical prompt, got {:?}",
                    entry
                )
            }
        }
    }

    // Namespace isolation: the same prompt in two namespaces never cross-resolves.
    #[test]
    fn namespaces_are_isolated() {
        let c = cache(0.0, false);
        let a = Namespace::new("tenant-a").unwrap();
        let b = Namespace::new("tenant-b").unwrap();
        c.put_embedding(&a, "shared prompt", "A-secret").unwrap();

        match c.get(&b, "shared prompt").unwrap() {
            Lookup::Miss { .. } => {}
            Lookup::Hit { .. } => panic!("tenant-b must not see tenant-a's entry"),
        }
    }

    // A failing ANN insert must roll back the KV write so `put` leaves no unreachable orphan
    // (store/index atomicity — exercised via the test-support `FailingIndex`).
    #[test]
    fn put_rolls_back_kv_when_index_insert_fails() {
        use crate::testing::FailingIndex;
        let cache = SemanticCache::new(
            HashEmbedder::default(),
            FailingIndex::new(),
            MemKv::new(),
            StaticThreshold::new(0.9),
        );
        let vector = match cache.get(&ns(), "q").unwrap() {
            Lookup::Miss { vector } => vector,
            Lookup::Hit { .. } => unreachable!("empty cache must miss"),
        };
        let err = cache.put(&ns(), "q", "answer", &vector).unwrap_err();
        assert!(
            matches!(err, RecallError::Backend(_)),
            "put surfaces the index failure"
        );
        assert_eq!(
            cache.entries(),
            0,
            "the KV write is rolled back — no orphaned entry remains"
        );
    }

    // A custom policy returning `Hit` with no neighbor must resolve to a miss, never a panic
    // (the defensive `(Hit, None)` arm — exercised via the test-support `AlwaysHit` policy).
    #[test]
    fn hit_verdict_without_neighbor_is_a_miss_not_a_panic() {
        use crate::testing::AlwaysHit;
        let cache = SemanticCache::new(
            HashEmbedder::default(),
            BruteForceIndex::new(), // empty → search yields no neighbors → top == None
            MemKv::new(),
            AlwaysHit,
        );
        assert!(
            matches!(cache.get(&ns(), "anything").unwrap(), Lookup::Miss { .. }),
            "a Hit verdict with no neighbor must degrade to a miss"
        );
    }

    #[test]
    fn rejects_invalid_namespace() {
        assert!(Namespace::new("").is_err());
        assert!(Namespace::new("a\u{1f}b").is_err());
        assert!(Namespace::new("ok/ns").is_ok());
    }

    // Rehydration is the cross-restart contract for a durable store: an entry that lives only in the
    // KV blob (a fresh index/exact-map, as after a reopen) is invisible to `get()` until `rehydrate`
    // rebuilds the in-memory structures. This simulates a restart by populating a `MemKv` directly —
    // mirroring `ns_key` (the documented `namespace ⊕ embedder.id()` join) — then handing it to a
    // cold cache. It checks both rebuilt paths (exact shortcut + semantic index) and the
    // cross-embedder skip.
    #[test]
    fn rehydrate_rebuilds_index_and_exact_and_skips_other_embedders() {
        use crate::types::ModelId;

        let embedder = HashEmbedder::default();
        let store = MemKv::new();
        let ns = Namespace::new("tenant-a/chat").unwrap();
        // Mirror SemanticCache::ns_key — the U+001F join of namespace and embedder id.
        let nsk = format!("{}\u{1f}{}", ns.as_str(), embedder.id());

        let prompt = "what is the capital of france";
        let key = Key::derive(&nsk, prompt);
        store
            .put(
                key,
                &Entry {
                    prompt: prompt.into(),
                    completion: "Paris.".into(),
                    model_id: embedder.id(),
                    created_at_unix: 0,
                    namespace: ns.as_str().into(),
                },
            )
            .unwrap();

        // A second entry written under a *different* embedder must be skipped on rehydrate.
        let other = "totally unrelated prompt";
        store
            .put(
                Key::derive(&nsk, other),
                &Entry {
                    prompt: other.into(),
                    completion: "STALE".into(),
                    model_id: ModelId::new("some-other-embedder@1"),
                    created_at_unix: 0,
                    namespace: ns.as_str().into(),
                },
            )
            .unwrap();

        // "Restart": same store + embedder, but a cold index and empty exact-map.
        let cache = SemanticCache::new(
            embedder,
            BruteForceIndex::new(),
            store,
            StaticThreshold::new(0.5),
        );

        // Before rehydrate the cold index can't find the persisted entry.
        assert!(
            matches!(cache.get(&ns, prompt).unwrap(), Lookup::Miss { .. }),
            "a cold index must miss until rehydrated"
        );

        let n = cache.rehydrate().unwrap();
        assert_eq!(n, 1, "only the current-embedder entry rehydrates");

        // Exact shortcut rebuilt: identical prompt is a certain hit.
        match cache.get(&ns, prompt).unwrap() {
            Lookup::Hit { score, entry, .. } => {
                assert_eq!(score, 1.0, "identical prompt is an exact hit");
                assert_eq!(entry.completion, "Paris.");
            }
            Lookup::Miss { .. } => panic!("rehydrated entry must hit"),
        }

        // Semantic index rebuilt: a close paraphrase hits via the index (not the exact shortcut).
        assert!(
            matches!(
                cache
                    .get(&ns, "what is the capital city of france")
                    .unwrap(),
                Lookup::Hit { .. }
            ),
            "a paraphrase must hit through the rehydrated index at τ=0.5"
        );

        // The other-embedder entry was skipped, so it stays a miss.
        assert!(
            matches!(cache.get(&ns, other).unwrap(), Lookup::Miss { .. }),
            "an entry from a different embedder must not be rehydrated"
        );
    }

    // `observe` is the feedback hook that activates the adaptive engine. The cache must forward it
    // to the policy keyed by the SAME `ns_key` (namespace ⊕ embedder id) that `decide` is keyed by —
    // otherwise learning lands on the wrong per-namespace state. A spy policy sharing an `Arc` log
    // captures the forwarded call after it's moved into the cache.
    #[test]
    fn observe_forwards_to_policy_under_the_ns_key() {
        use crate::types::Outcome;
        use std::sync::{Arc, Mutex};

        struct SpyPolicy {
            seen: Arc<Mutex<Vec<(String, f32, Outcome)>>>,
        }
        impl ThresholdPolicy for SpyPolicy {
            fn id(&self) -> &str {
                "spy"
            }
            fn decide(&self, _ns: &str, _top: Option<f32>) -> Verdict {
                Verdict::Miss
            }
            fn observe(&self, ns: &str, score: f32, outcome: Outcome) {
                self.seen
                    .lock()
                    .unwrap()
                    .push((ns.to_string(), score, outcome));
            }
        }

        let log = Arc::new(Mutex::new(Vec::new()));
        let embedder = HashEmbedder::default();
        // The expected key mirrors SemanticCache::ns_key — computed before `embedder` is moved.
        let expected_nsk = format!("{}\u{1f}{}", ns().as_str(), embedder.id());
        let cache = SemanticCache::new(
            embedder,
            BruteForceIndex::new(),
            MemKv::new(),
            SpyPolicy { seen: log.clone() },
        );

        cache.observe(&ns(), 0.8125, Outcome::Wrong);

        let seen = log.lock().unwrap();
        assert_eq!(seen.len(), 1, "observe is forwarded exactly once");
        assert_eq!(
            seen[0],
            (expected_nsk, 0.8125, Outcome::Wrong),
            "forwarded under the ns_key, with the score and outcome intact"
        );
    }
}
