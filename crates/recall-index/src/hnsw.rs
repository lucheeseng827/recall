//! The HNSW graph behind the [`AnnIndex`] seam. One independent graph per namespace string (the same
//! partitioning `BruteForceIndex` uses). Single-threaded, interior-mutable behind an `RwLock`; insert
//! is incremental (one vector at a time, matching the cache's miss-path), and delete is a tombstone
//! (HNSW node removal is hard; the cache only needs search to never return a removed key).

use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::RwLock;

use recall_core::math::dot;
use recall_core::{AnnIndex, Key, RecallError, Scored};

/// Cosine distance for L2-normalized vectors: `1 - dot`. Smaller is closer. The cache normalizes
/// before insert/search, so this matches `BruteForceIndex`'s `dot` similarity exactly (`score = 1 -
/// dist`).
fn dist(a: &[f32], b: &[f32]) -> f32 {
    1.0 - dot(a, b)
}

/// Tunables for the graph. Defaults follow the HNSW paper's common settings (M=16) with a search
/// breadth chosen so `recall@1 >= 0.98` holds at the cache's small-N regime.
#[derive(Clone, Copy, Debug)]
pub struct HnswConfig {
    /// Max neighbors per node on layers above 0.
    pub m: usize,
    /// Max neighbors per node on layer 0 (conventionally `2*m`).
    pub m0: usize,
    /// Candidate-list breadth while building (higher = better graph, slower insert).
    pub ef_construction: usize,
    /// Candidate-list breadth while searching (higher = better recall, slower search).
    pub ef_search: usize,
    /// Level-generation normalizer `mL` (conventionally `1/ln(M)`).
    pub ml: f64,
    /// Seed for the (deterministic) level-assignment PRNG, so a graph is reproducible from its
    /// insert order.
    pub seed: u64,
}

impl Default for HnswConfig {
    fn default() -> Self {
        let m = 16;
        Self {
            m,
            m0: m * 2,
            ef_construction: 200,
            ef_search: 64,
            ml: 1.0 / (m as f64).ln(),
            seed: 0x243F_6A88_85A3_08D3,
        }
    }
}

/// A scored candidate during traversal. Ordered by distance with a total order (NaN sorts as the
/// farthest), so it can live in a `BinaryHeap` (a max-heap on distance).
#[derive(Clone, Copy)]
struct Cand {
    dist: f32,
    id: usize,
}
impl PartialEq for Cand {
    fn eq(&self, o: &Self) -> bool {
        self.cmp(o) == Ordering::Equal
    }
}
impl Eq for Cand {}
impl PartialOrd for Cand {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Cand {
    fn cmp(&self, o: &Self) -> Ordering {
        // Total order on distance; NaN is treated as the largest (farthest) so a degenerate vector
        // can never win a nearest-neighbor slot.
        match self.dist.partial_cmp(&o.dist) {
            Some(ord) => ord,
            None => match (self.dist.is_nan(), o.dist.is_nan()) {
                (true, true) => Ordering::Equal,
                (true, false) => Ordering::Greater,
                (false, true) => Ordering::Less,
                (false, false) => Ordering::Equal,
            },
        }
    }
}

struct Node {
    key: Key,
    vec: Vec<f32>,
    /// `links[layer]` = neighbor node ids at that layer; `links.len()` = this node's top layer + 1.
    links: Vec<Vec<usize>>,
    tombstoned: bool,
}

/// One namespace's HNSW graph.
#[derive(Default)]
struct Graph {
    nodes: Vec<Node>,
    key_to_id: HashMap<Key, usize>,
    entry: Option<usize>,
    top_layer: usize,
    /// Count of non-tombstoned nodes (what `len` reports).
    live: usize,
    /// splitmix64 state for level assignment.
    rng: u64,
}

/// Advance a splitmix64 state and return the next value. No external RNG dependency.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl Graph {
    /// A uniform draw in `(0, 1]`.
    fn next_unit(&mut self) -> f64 {
        // Top 53 bits → [0,1); map to (0,1] so `ln` is always finite.
        let u = (splitmix64(&mut self.rng) >> 11) as f64 / (1u64 << 53) as f64;
        1.0 - u
    }

    /// Draw a node's top layer: `floor(-ln(U) * mL)`, capped to keep the tower shallow.
    fn random_level(&mut self, ml: f64) -> usize {
        let r = self.next_unit();
        ((-r.ln()) * ml).floor() as usize
    }

    /// The HNSW `SEARCH-LAYER` primitive: the `ef` nearest nodes to `query` reachable on `layer`
    /// starting from `entry_points`, returned sorted nearest-first. Includes tombstoned nodes (they
    /// still route); callers filter them out of the final answer.
    fn search_layer(
        &self,
        query: &[f32],
        entry_points: &[usize],
        ef: usize,
        layer: usize,
    ) -> Vec<Cand> {
        let mut visited: HashSet<usize> = HashSet::new();
        // candidates: min-heap by distance (closest popped first), via Reverse.
        let mut candidates: BinaryHeap<Reverse<Cand>> = BinaryHeap::new();
        // results: max-heap by distance (farthest on top), capped at ef.
        let mut results: BinaryHeap<Cand> = BinaryHeap::new();

        for &ep in entry_points {
            let c = Cand {
                dist: dist(query, &self.nodes[ep].vec),
                id: ep,
            };
            visited.insert(ep);
            candidates.push(Reverse(c));
            results.push(c);
        }

        while let Some(Reverse(c)) = candidates.pop() {
            // If the closest remaining candidate is farther than the current farthest result and we
            // already have ef results, no closer node can be reached — stop.
            if let Some(farthest) = results.peek() {
                if c.dist > farthest.dist && results.len() >= ef {
                    break;
                }
            }
            for &e in &self.nodes[c.id].links[layer] {
                if visited.insert(e) {
                    let d = dist(query, &self.nodes[e].vec);
                    let admit = results.len() < ef || results.peek().is_some_and(|f| d < f.dist);
                    if admit {
                        let cand = Cand { dist: d, id: e };
                        candidates.push(Reverse(cand));
                        results.push(cand);
                        if results.len() > ef {
                            results.pop(); // drop the farthest
                        }
                    }
                }
            }
        }

        let mut out: Vec<Cand> = results.into_vec();
        out.sort_unstable(); // ascending by distance (nearest first)
        out
    }

    /// Re-select `id`'s neighbor list at `layer` to the `cap` closest (the simple HNSW heuristic).
    /// Clones `id`'s vector to sidestep the simultaneous borrow of `nodes`.
    fn prune(&mut self, id: usize, layer: usize, cap: usize) {
        if self.nodes[id].links[layer].len() <= cap {
            return;
        }
        let base = self.nodes[id].vec.clone();
        let mut cands: Vec<Cand> = self.nodes[id].links[layer]
            .iter()
            .map(|&nb| Cand {
                dist: dist(&base, &self.nodes[nb].vec),
                id: nb,
            })
            .collect();
        cands.sort_unstable();
        cands.truncate(cap);
        self.nodes[id].links[layer] = cands.into_iter().map(|c| c.id).collect();
    }

    fn insert(&mut self, key: Key, vec: Vec<f32>, cfg: &HnswConfig) {
        // Re-inserting an existing key: refresh the stored vector in place. In the cache a re-insert
        // is the same (ns, prompt) → the same deterministic vector, so the graph structure stays
        // valid; we do not rebuild links.
        if let Some(&id) = self.key_to_id.get(&key) {
            self.nodes[id].vec = vec;
            return;
        }

        let level = self.random_level(cfg.ml);
        let id = self.nodes.len();
        self.nodes.push(Node {
            key,
            vec,
            links: vec![Vec::new(); level + 1],
            tombstoned: false,
        });
        self.key_to_id.insert(key, id);
        self.live += 1;

        let Some(entry_id) = self.entry else {
            // First node: it is the entry point.
            self.entry = Some(id);
            self.top_layer = level;
            return;
        };

        let query = self.nodes[id].vec.clone();
        let mut ep_ids = vec![entry_id];

        // Phase 1: greedy descent from the top layer down to just above the new node's level (ef=1).
        if self.top_layer > level {
            for lc in ((level + 1)..=self.top_layer).rev() {
                let w = self.search_layer(&query, &ep_ids, 1, lc);
                if let Some(best) = w.first() {
                    ep_ids = vec![best.id];
                }
            }
        }

        // Phase 2: from min(top, level) down to 0, connect bidirectionally.
        let start = self.top_layer.min(level);
        for lc in (0..=start).rev() {
            let w = self.search_layer(&query, &ep_ids, cfg.ef_construction, lc);
            let cap = if lc == 0 { cfg.m0 } else { cfg.m };
            let selected: Vec<usize> = w.iter().take(cap).map(|c| c.id).collect();
            self.nodes[id].links[lc] = selected.clone();
            for &nb in &selected {
                self.nodes[nb].links[lc].push(id);
                self.prune(nb, lc, cap);
            }
            ep_ids = w.iter().map(|c| c.id).collect();
        }

        if level > self.top_layer {
            self.entry = Some(id);
            self.top_layer = level;
        }
    }

    fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<Scored> {
        let Some(entry_id) = self.entry else {
            return Vec::new();
        };
        let mut ep_ids = vec![entry_id];
        // Greedy descent to layer 0.
        for lc in (1..=self.top_layer).rev() {
            let w = self.search_layer(query, &ep_ids, 1, lc);
            if let Some(best) = w.first() {
                ep_ids = vec![best.id];
            }
        }
        // Filtering tombstones AFTER a fixed-width layer-0 search can return fewer than `k` live
        // hits once a graph has churn — enough deleted near-neighbors crowd the live ones out of the
        // `ef`-wide window, which the cache read path would see as a false miss. Widen the search
        // until we have `k` live hits or we've effectively scanned the whole graph.
        let mut width = ef.max(k);
        loop {
            let found = self.search_layer(query, &ep_ids, width, 0);
            let live: Vec<Scored> = found
                .into_iter()
                .filter(|c| !self.nodes[c.id].tombstoned)
                .take(k)
                .map(|c| Scored {
                    key: self.nodes[c.id].key,
                    score: 1.0 - c.dist, // back to cosine similarity for the policy/threshold
                })
                .collect();
            if live.len() == k || width >= self.nodes.len() {
                return live;
            }
            width = (width * 2).min(self.nodes.len());
        }
    }
}

/// An HNSW [`AnnIndex`]. One graph per namespace, guarded by a single `RwLock` (search takes the read
/// lock, insert/remove the write lock — matching `BruteForceIndex`'s concurrency shape).
pub struct HnswIndex {
    cfg: HnswConfig,
    ns: RwLock<HashMap<String, Graph>>,
}

impl HnswIndex {
    pub fn new() -> Self {
        Self::with_config(HnswConfig::default())
    }

    pub fn with_config(cfg: HnswConfig) -> Self {
        Self {
            cfg,
            ns: RwLock::new(HashMap::new()),
        }
    }
}

impl Default for HnswIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl AnnIndex for HnswIndex {
    fn insert(&self, ns: &str, key: Key, vector: &[f32]) -> Result<(), RecallError> {
        let mut map = self.ns.write().unwrap();
        let graph = map.entry(ns.to_string()).or_insert_with(|| Graph {
            rng: self.cfg.seed,
            ..Graph::default()
        });
        // Same-embedder namespaces never mismatch; if a wiring bug feeds a wrong-dim vector, surface
        // it loudly (as BruteForceIndex does) rather than silently zip-truncating in `dot`.
        if let Some(existing) = graph.nodes.first() {
            if existing.vec.len() != vector.len() {
                return Err(RecallError::DimensionMismatch {
                    expected: existing.vec.len(),
                    got: vector.len(),
                });
            }
        }
        graph.insert(key, vector.to_vec(), &self.cfg);
        Ok(())
    }

    fn search(&self, ns: &str, query: &[f32], k: usize) -> Result<Vec<Scored>, RecallError> {
        let map = self.ns.read().unwrap();
        let Some(graph) = map.get(ns) else {
            return Ok(Vec::new());
        };
        if let Some(existing) = graph.nodes.first() {
            if existing.vec.len() != query.len() {
                return Err(RecallError::DimensionMismatch {
                    expected: existing.vec.len(),
                    got: query.len(),
                });
            }
        }
        Ok(graph.search(query, k, self.cfg.ef_search))
    }

    fn remove(&self, ns: &str, key: Key) -> Result<(), RecallError> {
        let mut map = self.ns.write().unwrap();
        if let Some(graph) = map.get_mut(ns) {
            if let Some(id) = graph.key_to_id.remove(&key) {
                if !graph.nodes[id].tombstoned {
                    graph.nodes[id].tombstoned = true;
                    graph.live -= 1;
                }
            }
        }
        Ok(())
    }

    fn len(&self, ns: &str) -> usize {
        self.ns.read().unwrap().get(ns).map_or(0, |g| g.live)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use recall_core::BruteForceIndex;

    // A deterministic, dependency-free vector generator (splitmix64 → unit cube → L2-normalized,
    // matching how the cache feeds the index).
    fn gen_vectors(n: usize, dim: usize, mut seed: u64) -> Vec<Vec<f32>> {
        (0..n)
            .map(|_| {
                let mut v: Vec<f32> = (0..dim)
                    .map(|_| {
                        let u = (splitmix64(&mut seed) >> 11) as f64 / (1u64 << 53) as f64;
                        (u * 2.0 - 1.0) as f32
                    })
                    .collect();
                let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 {
                    for x in &mut v {
                        *x /= norm;
                    }
                }
                v
            })
            .collect()
    }

    fn key(i: usize) -> Key {
        Key::derive("ns", &format!("k{i}"))
    }

    #[test]
    fn insert_search_and_exact_self_match() {
        let idx = HnswIndex::new();
        let vecs = gen_vectors(50, 16, 1);
        for (i, v) in vecs.iter().enumerate() {
            idx.insert("ns", key(i), v).unwrap();
        }
        assert_eq!(idx.len("ns"), 50);
        // Querying a stored vector returns itself as the top-1 (score ~= 1.0).
        for (i, v) in vecs.iter().enumerate() {
            let hits = idx.search("ns", v, 1).unwrap();
            assert_eq!(hits[0].key, key(i), "self-query must return self");
            assert!(hits[0].score > 0.999, "self-similarity is ~1.0");
        }
    }

    #[test]
    fn namespaces_are_isolated() {
        let idx = HnswIndex::new();
        let v = gen_vectors(1, 8, 7).pop().unwrap();
        idx.insert("a", key(0), &v).unwrap();
        assert_eq!(idx.search("b", &v, 1).unwrap().len(), 0);
        assert_eq!(idx.len("b"), 0);
        assert_eq!(idx.len("a"), 1);
    }

    #[test]
    fn remove_tombstones_so_search_skips_it() {
        let idx = HnswIndex::new();
        let vecs = gen_vectors(20, 8, 3);
        for (i, v) in vecs.iter().enumerate() {
            idx.insert("ns", key(i), v).unwrap();
        }
        // Remove the exact match for query #5, then its self-query must not return key(5).
        idx.remove("ns", key(5)).unwrap();
        assert_eq!(idx.len("ns"), 19);
        let hits = idx.search("ns", &vecs[5], 5).unwrap();
        assert!(
            hits.iter().all(|h| h.key != key(5)),
            "a tombstoned key is never returned"
        );
    }

    // Regression: tombstones must not crowd live hits out of the result. A deliberately narrow
    // `ef_search` makes the nearest (now-tombstoned) node fill the entire layer-0 window — without
    // the widening loop, `search` would return empty even though live matches remain.
    #[test]
    fn search_widens_past_tombstones_to_find_live_hit() {
        let cfg = HnswConfig {
            ef_search: 1,
            ..HnswConfig::default()
        };
        let idx = HnswIndex::with_config(cfg);
        let vecs = gen_vectors(30, 16, 11);
        for (i, v) in vecs.iter().enumerate() {
            idx.insert("ns", key(i), v).unwrap();
        }
        // The exact nearest to vecs[0] is key(0) itself; tombstone it, then query it back.
        idx.remove("ns", key(0)).unwrap();
        let hits = idx.search("ns", &vecs[0], 1).unwrap();
        assert_eq!(
            hits.len(),
            1,
            "widening must surface a live hit past the tombstoned nearest"
        );
        assert_ne!(hits[0].key, key(0), "the tombstoned key is never returned");
    }

    #[test]
    fn dimension_mismatch_is_surfaced() {
        let idx = HnswIndex::new();
        idx.insert("ns", key(0), &[1.0, 0.0, 0.0]).unwrap();
        assert!(matches!(
            idx.insert("ns", key(1), &[1.0, 0.0]),
            Err(RecallError::DimensionMismatch {
                expected: 3,
                got: 2
            })
        ));
        assert!(matches!(
            idx.search("ns", &[1.0, 0.0], 1),
            Err(RecallError::DimensionMismatch {
                expected: 3,
                got: 2
            })
        ));
    }

    // THE gate (PLAN.md §3-OSS, T3): HNSW top-1 must agree with the exact brute-force oracle on at
    // least 98% of queries.
    #[test]
    fn recall_at_1_matches_brute_force_oracle() {
        let dim = 32;
        let corpus = gen_vectors(800, dim, 0xABCD);
        let queries = gen_vectors(300, dim, 0x1234);

        let oracle = BruteForceIndex::new();
        let hnsw = HnswIndex::new();
        for (i, v) in corpus.iter().enumerate() {
            oracle.insert("ns", key(i), v).unwrap();
            hnsw.insert("ns", key(i), v).unwrap();
        }

        let mut agree = 0usize;
        for q in &queries {
            let truth = oracle.search("ns", q, 1).unwrap();
            let approx = hnsw.search("ns", q, 1).unwrap();
            if let (Some(t), Some(a)) = (truth.first(), approx.first()) {
                if t.key == a.key {
                    agree += 1;
                }
            }
        }
        let recall = agree as f64 / queries.len() as f64;
        assert!(
            recall >= 0.98,
            "recall@1 vs brute-force oracle = {recall:.4} (< 0.98)"
        );
    }
}
