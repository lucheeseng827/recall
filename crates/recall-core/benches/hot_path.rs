//! Criterion micro-benchmarks for the cache hot path, no I/O (PLAN.md §6.4): the embed, ANN-search,
//! threshold-decide, and end-to-end lookup stages in isolation, so a regression in any one stage is
//! visible on its own rather than hidden in an aggregate. Run with `cargo bench -p recall-core`.
//!
//! The M0 budgets these track (on the named target CPU): **embed < 1 ms** and **decide < 50 µs**
//! (PLAN.md §6.2). The default `HashEmbedder` is tens of microseconds; the heavier static
//! (model2vec) embedder is the one to watch against the 1 ms budget — swap it in here when measuring
//! a production build. `decide < 50 µs` also has a hard CI assertion in `recall-eval`.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use recall_core::{
    AnnIndex, BruteForceIndex, Embedder, HashEmbedder, Key, MemKv, Namespace, SemanticCache,
    StaticThreshold, ThresholdPolicy,
};

/// L2-normalize, matching what the `SemanticCache` facade does before insert/search (both indexes
/// treat a dot product over unit vectors as cosine).
fn normalize(mut v: Vec<f32>) -> Vec<f32> {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        for x in &mut v {
            *x /= n;
        }
    }
    v
}

const PROMPT: &str = "how do i reset my password";
const CORPUS: usize = 1000;

/// Embed one short query → vector. The dominant, embedder-dependent term on the hot path.
fn bench_embed(c: &mut Criterion) {
    let e = HashEmbedder::default();
    c.bench_function("embed/hash_one", |b| {
        b.iter(|| e.embed_one(black_box(PROMPT)).unwrap())
    });
}

/// The threshold decision in isolation — must stay lock-light and well under 50 µs.
fn bench_decide(c: &mut Criterion) {
    let p = StaticThreshold::new(0.9);
    c.bench_function("decide/static", |b| {
        b.iter(|| p.decide(black_box("ns"), black_box(Some(0.95))))
    });
}

/// Exact brute-force cosine search over a warmed 1k-vector namespace — the small-N default and the
/// recall oracle. (HNSW vs flat at scale is benched in `recall-index`.)
fn bench_search(c: &mut Criterion) {
    let e = HashEmbedder::default();
    let idx = BruteForceIndex::new();
    let ns = "bench";
    for i in 0..CORPUS {
        let v = normalize(e.embed_one(&format!("faq question number {i}")).unwrap());
        idx.insert(ns, Key::derive(ns, &i.to_string()), &v).unwrap();
    }
    let q = normalize(e.embed_one(PROMPT).unwrap());
    c.bench_function("search/brute_1k", |b| {
        b.iter(|| idx.search(black_box(ns), black_box(&q), 1).unwrap())
    });
}

/// The full read path through `SemanticCache::get` (exact-shortcut → embed → search → decide), with
/// the in-memory MVP backends and no I/O. The query is *not* an exact stored prompt, so the embed +
/// search + decide path runs rather than the O(1) exact-hash shortcut.
fn bench_lookup(c: &mut Criterion) {
    let cache = SemanticCache::new(
        HashEmbedder::default(),
        BruteForceIndex::new(),
        MemKv::new(),
        StaticThreshold::new(0.9),
    );
    let ns = Namespace::new("bench").unwrap();
    for i in 0..CORPUS {
        cache
            .put_embedding(&ns, &format!("faq question number {i}"), "answer")
            .unwrap();
    }
    c.bench_function("lookup/end_to_end_1k", |b| {
        b.iter(|| cache.get(black_box(&ns), black_box(PROMPT)).unwrap())
    });
}

criterion_group!(
    benches,
    bench_embed,
    bench_decide,
    bench_search,
    bench_lookup
);
criterion_main!(benches);
