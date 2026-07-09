//! Criterion bench: exact flat search (the `recall-core` brute-force oracle) vs the pure-Rust HNSW
//! index as the corpus grows (PLAN.md §6.4 "nearest (flat vs HNSW)"). It isolates *search* latency
//! per query; the `recall ann-bench` subcommand reports the companion recall@1 quality number that
//! makes the latency win meaningful (a fast index that misses neighbours is not a win).
//!
//! Run with `cargo bench -p recall-index --bench ann`.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use recall_core::{AnnIndex, BruteForceIndex, Key};
use recall_index::HnswIndex;

/// Deterministic L2-normalized random vectors (xorshift, no `rand` dep) — both indexes treat a dot
/// product over unit vectors as cosine, so the corpus is normalized like the real hot path.
fn gen_normalized(n: usize, dims: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut state: u64 = seed | 1;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    };
    (0..n)
        .map(|_| {
            let mut v: Vec<f32> = (0..dims).map(|_| next() as f32).collect();
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

fn bench_search(c: &mut Criterion) {
    const DIMS: usize = 256;
    let ns = "bench";
    let query = &gen_normalized(1, DIMS, 0x5EED)[0];

    let mut group = c.benchmark_group("ann_search_top1");
    for &n in &[1_000usize, 10_000] {
        let corpus = gen_normalized(n, DIMS, 0xC0DE_F00D);
        let brute = BruteForceIndex::new();
        let hnsw = HnswIndex::new();
        for (i, v) in corpus.iter().enumerate() {
            let key = Key::derive(ns, &i.to_string());
            brute.insert(ns, key, v).unwrap();
            hnsw.insert(ns, key, v).unwrap();
        }
        group.bench_with_input(BenchmarkId::new("brute", n), &n, |b, _| {
            b.iter(|| brute.search(ns, black_box(query), 1).unwrap())
        });
        group.bench_with_input(BenchmarkId::new("hnsw", n), &n, |b, _| {
            b.iter(|| hnsw.search(ns, black_box(query), 1).unwrap())
        });
    }
    group.finish();
}

criterion_group!(benches, bench_search);
criterion_main!(benches);
