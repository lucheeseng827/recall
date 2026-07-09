//! `recall` — the MVP CLI that demonstrates and measures the cache loop.
//!
//!   recall ask --ns <namespace> "<prompt>"   # HIT prints the cached answer; MISS reads a
//!                                            # completion from stdin and stores it
//!   recall bench [--tau <f32>] [--iters <n>] [--model <dir>]
//!                                            # synthetic FAQ/paraphrase workload; reports hit-rate +
//!                                            # embed vs lookup p50/p99 (--model uses a real embedder)
//!   recall replay --file <reqs.jsonl>        # drive a running `serve` with a request log; reports
//!                                            # hit-rate + tokens saved (input/output split) + $ est.
//!   recall ann-bench [--dims d] [--sizes ..] # brute vs HNSW: recall@1 + search latency as N grows
//!   recall serve  [--config recall.toml]     # the OpenAI/Anthropic-compatible caching proxy
//!
//! `ask`/`bench`/`ann-bench` run entirely in-process against the in-memory MVP backends; `serve`/`replay`
//! speak HTTP. Arg parsing is hand-rolled to keep clap out of the MVP build.

use std::io::Read;
use std::sync::Arc;
use std::time::Instant;

use recall_core::{
    AnnIndex, BruteForceIndex, Embedder, HashEmbedder, Key, Lookup, MemKv, Namespace,
    StaticThreshold,
};
use recall_proxy::{boxed_cache_full, Config, ProxyState};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let code = match args.first().map(String::as_str) {
        Some("ask") => cmd_ask(&args[1..]),
        Some("bench") => cmd_bench(&args[1..]),
        Some("replay") => cmd_replay(&args[1..]),
        Some("ann-bench") => cmd_ann_bench(&args[1..]),
        Some("calibrate") => cmd_calibrate(&args[1..]),
        Some("serve") => cmd_serve(&args[1..]),
        Some("-h") | Some("--help") | None => {
            usage();
            0
        }
        Some(other) => {
            eprintln!("recall: unknown subcommand '{other}'\n");
            usage();
            2
        }
    };
    std::process::exit(code);
}

fn usage() {
    eprintln!(
        "recall — self-hosted semantic cache\n\n\
         USAGE:\n  \
         recall serve [--config recall.toml] [--listen addr] [--upstream url]\n               \
         [--base-namespace ns] [--max-temperature f] [--tau f] [--model dir]\n               \
         [--index brute|hnsw] [--store memory|redb] [--db-path file] [--policy static|adaptive]\n               \
         [--watch-config]  (hot-reload the threshold policy from --config; other fields need a restart)\n  \
         recall ask --ns <namespace> \"<prompt>\"\n  \
         recall bench [--tau <f32>] [--iters <n>] [--model <dir>]\n  \
         recall replay --file <reqs.jsonl> [--target url] [--path p] [--verify-sample r]\n               \
         [--upstream url] [--price-input <usd/Mtok>] [--price-output <usd/Mtok>]\n  \
         recall ann-bench [--dims <d>] [--queries <q>] [--sizes 100,1000,10000] [--top-k <k>]\n  \
         recall calibrate --file <pairs.jsonl> [--model <dir>] [--target-fhr <r>]\n\n\
         `serve` runs the OpenAI-compatible proxy (point your client's base URL at it).\n\
         `calibrate` picks the similarity threshold for the configured embedder from a labeled\n\
         JSONL pair set ({{\"a\":..,\"b\":..,\"should_hit\":true|false}} per line).\n\
         `replay` drives a running `serve` with a JSONL request log and reports hit-rate + tokens saved.\n\
         `ann-bench` compares the brute-force and HNSW indexes (recall@1 + search latency) as N grows.\n\
         Config precedence: flags > env (RECALL_*) > --config file > defaults.\n\
         On an `ask` MISS, the completion is read from stdin and stored for next time."
    );
}

/// `--flag value` lookup over a raw arg slice. Returns the value following `flag`, if present.
fn flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
}

fn cmd_ask(args: &[String]) -> i32 {
    let Some(ns_raw) = flag(args, "--ns") else {
        eprintln!("recall ask: --ns <namespace> is required");
        return 2;
    };
    // The prompt is the first non-flag, non-flag-value positional argument.
    let prompt = args
        .iter()
        .enumerate()
        .find(|(i, a)| {
            !a.starts_with("--") && args.get(i.wrapping_sub(1)).map(String::as_str) != Some("--ns")
        })
        .map(|(_, a)| a.clone());
    let Some(prompt) = prompt else {
        eprintln!("recall ask: a \"<prompt>\" positional argument is required");
        return 2;
    };
    let ns = match Namespace::new(ns_raw) {
        Ok(ns) => ns,
        Err(e) => {
            eprintln!("recall ask: {e}");
            return 2;
        }
    };

    // A fresh in-memory cache per invocation: this MVP CLI proves the loop, it does not persist
    // across runs (durable backends are the OSS `recall-store` job, PLAN.md §3-OSS).
    let cache = boxed_cache_full(
        Box::new(HashEmbedder::default()),
        Box::new(MemKv::new()),
        Box::new(StaticThreshold::new(0.9)),
    );
    match cache.get(&ns, &prompt).expect("lookup") {
        Lookup::Hit { score, entry, .. } => {
            println!("HIT (score {score:.4})\n{}", entry.completion);
            0
        }
        Lookup::Miss { vector } => {
            eprintln!("MISS — reading completion from stdin to store…");
            let mut completion = String::new();
            if std::io::stdin().read_to_string(&mut completion).is_err() {
                eprintln!("recall ask: failed to read completion from stdin");
                return 1;
            }
            let completion = completion.trim_end().to_string();
            cache
                .put(&ns, &prompt, &completion, &vector)
                .expect("store");
            println!("MISS (stored {} bytes)", completion.len());
            0
        }
    }
}

fn cmd_bench(args: &[String]) -> i32 {
    let tau: f32 = flag(args, "--tau")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.85);
    let iters: usize = flag(args, "--iters")
        .and_then(|s| s.parse().ok())
        .unwrap_or(2000);
    if iters == 0 {
        // Zero iterations would leave `latencies_us` empty and panic the percentile/max indexing.
        eprintln!("recall bench: --iters must be greater than 0");
        return 2;
    }

    // A tiny synthetic FAQ workload: each canonical question plus a paraphrase that *should* hit.
    let faq: &[(&str, &str)] = &[
        (
            "how do i reset my password",
            "Use the 'Forgot password' link.",
        ),
        ("what are your support hours", "We are open 9am-5pm ET."),
        ("how do i cancel my subscription", "Open Billing → Cancel."),
        (
            "where can i download an invoice",
            "Billing → Invoices → Download.",
        ),
        (
            "how do i change my email address",
            "Settings → Account → Email.",
        ),
    ];
    let paraphrases: &[&str] = &[
        "how can i reset the password",
        "what hours is support available",
        "how do i cancel the subscription",
        "where do i download my invoice",
        "how can i change my email",
    ];

    // Optional real embedder: `--model <dir>` (or RECALL_MODEL2VEC_PATH) runs the bench with the
    // SAME static model2vec/potion embedder a production deploy uses, so the reported numbers are a
    // defensible baseline — not just the near-free hash stub. Needs a `--features static` build;
    // otherwise it falls back to the hash embedder (build_embedder prints a note).
    let model_path = flag(args, "--model").map(str::to_string).or_else(|| {
        let p = std::env::var("RECALL_MODEL2VEC_PATH").ok();
        if let Some(ref path) = p {
            eprintln!("recall bench: using model from RECALL_MODEL2VEC_PATH={path}");
        }
        p
    });

    // The embedder that lives in the cache, plus a probe used ONLY to time the embed step in
    // isolation (the cache's embedder is moved in, so it can't be timed separately). When no model
    // path is given the hash embedder is a trivial default — no model load, so we skip the second
    // build_embedder call and construct one directly.
    let embedder = build_embedder(model_path.as_deref());
    let embedder_id = embedder.id().to_string();
    let probe: Box<dyn Embedder> = match &model_path {
        Some(p) => build_embedder(Some(p.as_str())),
        None => Box::new(HashEmbedder::default()),
    };

    let cache = boxed_cache_full(
        embedder,
        Box::new(MemKv::new()),
        Box::new(StaticThreshold::new(tau)),
    );
    let ns = Namespace::new("bench").unwrap();
    // Warm the canonical answers.
    for (q, a) in faq {
        cache.put_embedding(&ns, q, a).unwrap();
    }

    let mut hits = 0usize;
    // Nanosecond resolution: a hash embed is tens of ns and would floor to 0 µs otherwise.
    let mut embed_ns: Vec<u128> = Vec::with_capacity(iters);
    let mut lookup_ns: Vec<u128> = Vec::with_capacity(iters);
    for i in 0..iters {
        // Alternate exact questions and paraphrases to get a realistic mix.
        let query = if i % 2 == 0 {
            faq[i % faq.len()].0
        } else {
            paraphrases[i % paraphrases.len()]
        };
        // Embed cost in isolation — the dominant, embedder-dependent term on the hot path.
        let te = Instant::now();
        let _ = probe.embed_one(query).expect("embed");
        embed_ns.push(te.elapsed().as_nanos());
        // Full lookup: exact-shortcut → embed → ANN search → threshold decide.
        let tl = Instant::now();
        let res = cache.get(&ns, query).expect("lookup");
        lookup_ns.push(tl.elapsed().as_nanos());
        if matches!(res, Lookup::Hit { .. }) {
            hits += 1;
        }
    }

    embed_ns.sort_unstable();
    lookup_ns.sort_unstable();
    let pct = |v: &[u128], q: f64| -> u128 {
        let idx = ((v.len() as f64 - 1.0) * q).round() as usize;
        v[idx]
    };
    let us = |ns: u128| ns as f64 / 1000.0;
    let hit_rate = 100.0 * hits as f64 / iters as f64;
    let embed_p50 = pct(&embed_ns, 0.50);
    let lookup_p50 = pct(&lookup_ns, 0.50);
    // In-process ANN + threshold ≈ the full lookup minus the embed it contains. Median-vs-median, and
    // the exact-match shortcut (which skips embed+search) makes this a slight over-estimate of pure
    // search — hence "~".
    let index_p50 = lookup_p50.saturating_sub(embed_p50);
    if lookup_p50 < embed_p50 {
        eprintln!(
            "recall bench: note: lookup_p50 ({:.1} µs) < embed_p50 ({:.1} µs) — \
             the exact-match shortcut dominates the median; index_p50 is clamped to 0",
            us(lookup_p50),
            us(embed_p50)
        );
    }

    println!(
        "recall bench  (embedder: {embedder_id}, policy: {})",
        cache.policy_id()
    );
    println!("  entries warmed  : {}", cache.entries());
    println!("  iterations      : {iters}");
    println!("  hit-rate        : {hit_rate:.1}%  ({hits}/{iters})");
    println!(
        "  embed  p50/p99  : {:.1} / {:.1} µs   (query → vector; embedder-bound)",
        us(embed_p50),
        us(pct(&embed_ns, 0.99))
    );
    println!(
        "  lookup p50/p99  : {:.1} / {:.1} µs   (full get: shortcut → embed → ANN → decide)",
        us(lookup_p50),
        us(pct(&lookup_ns, 0.99))
    );
    println!(
        "  index  p50      : ~{:.1} µs   (lookup_p50 − embed_p50: in-process ANN + threshold)",
        us(index_p50)
    );
    println!(
        "  lookup max      : {:.1} µs",
        us(lookup_ns[lookup_ns.len() - 1])
    );
    0
}

/// Optional `recall.toml`. Every field is optional; env and flags override anything set here.
#[derive(serde::Deserialize, Default, Clone)]
struct FileConfig {
    listen: Option<String>,
    upstream: Option<String>,
    base_namespace: Option<String>,
    max_temperature: Option<f64>,
    tau: Option<f32>,
    model: Option<String>,
    index: Option<String>,
    store: Option<String>,
    db_path: Option<String>,
    policy: Option<String>,
    target_fhr: Option<f64>,
}

/// Deterministic L2-normalized random vectors from a seed (no `rand` dependency — a small xorshift
/// keeps the benchmark reproducible and the binary deps-free). Normalized because both indexes treat
/// a dot product as cosine similarity and expect unit vectors (the `SemanticCache` facade normalizes).
fn gen_normalized(n: usize, dims: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut state: u64 = seed | 1; // never 0 (xorshift fixed point)
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        // Map to (-1, 1).
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

/// `recall ann-bench`: compare the exact `BruteForceIndex` oracle against the pure-Rust HNSW index as
/// the corpus grows — the data behind the "recall@1 ≥ 0.98, sublinear at scale" claim. For each size
/// it warms both indexes with the same vectors and reports per-query search latency (p50/p99) plus
/// HNSW's top-k recall against the brute-force top-1.
fn cmd_ann_bench(args: &[String]) -> i32 {
    let dims: usize = flag(args, "--dims")
        .and_then(|s| s.parse().ok())
        .unwrap_or(256);
    let queries: usize = flag(args, "--queries")
        .and_then(|s| s.parse().ok())
        .unwrap_or(500);
    let top_k: usize = flag(args, "--top-k")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let sizes: Vec<usize> = flag(args, "--sizes")
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![100, 1_000, 10_000, 50_000]);
    if queries < 100 || sizes.is_empty() || top_k == 0 {
        eprintln!(
            "recall ann-bench: --queries must be ≥ 100 for meaningful percentiles; --top-k and at least one --sizes value must be > 0"
        );
        return 2;
    }

    let qs = gen_normalized(queries, dims, 0x51A1_5EED);
    let ns = "annbench";
    let pct = |v: &[u128], q: f64| -> u128 {
        let idx = ((v.len() as f64 - 1.0) * q).round() as usize;
        v[idx]
    };
    let us = |ns: u128| ns as f64 / 1000.0;

    // Time `queries` searches and collect results in one pass — avoids a redundant recall-only
    // search round and ensures timing and recall measurements come from the same calls.
    let time_and_collect =
        |idx: &dyn AnnIndex, k: usize| -> (Vec<u128>, Vec<Vec<recall_core::Scored>>) {
            let mut times: Vec<u128> = Vec::with_capacity(qs.len());
            let mut results: Vec<Vec<recall_core::Scored>> = Vec::with_capacity(qs.len());
            for q in &qs {
                let start = Instant::now();
                let r = idx.search(ns, q, k).expect("search");
                times.push(start.elapsed().as_nanos());
                results.push(r);
            }
            times.sort_unstable();
            (times, results)
        };

    println!("recall ann-bench  (dims: {dims}, queries: {queries}, top-k: {top_k})");
    println!(
        "  {:>8} | {:>16} | {:>16} | {:>8} | {:>9}",
        "corpus", "brute p50/p99 µs", "hnsw p50/p99 µs", "speedup", "recall@1"
    );
    for &n in &sizes {
        let corpus = gen_normalized(n, dims, 0xC0DE_F00D);
        let brute = BruteForceIndex::new();
        let hnsw = recall_index::HnswIndex::new();
        for (i, v) in corpus.iter().enumerate() {
            let key = Key::derive(ns, &i.to_string());
            brute.insert(ns, key, v).expect("brute insert");
            hnsw.insert(ns, key, v).expect("hnsw insert");
        }

        let (b, brute_results) = time_and_collect(&brute, top_k);
        let (h, hnsw_results) = time_and_collect(&hnsw, top_k);

        // recall@1: how often HNSW's top-k contains the brute-force oracle's single best neighbor.
        let mut agree = 0usize;
        for (br, hr) in brute_results.iter().zip(hnsw_results.iter()) {
            if let Some(oracle) = br.first() {
                if hr.iter().any(|s| s.key == oracle.key) {
                    agree += 1;
                }
            }
        }
        let recall = agree as f64 / qs.len() as f64;
        let bp50 = pct(&b, 0.50);
        let hp50 = pct(&h, 0.50);
        let speedup = if hp50 > 0 {
            bp50 as f64 / hp50 as f64
        } else {
            f64::INFINITY
        };

        println!(
            "  {:>8} | {:>7.1} / {:>6.1} | {:>7.1} / {:>6.1} | {:>7.2}x | {:>8.4}",
            n,
            us(bp50),
            us(pct(&b, 0.99)),
            us(hp50),
            us(pct(&h, 0.99)),
            speedup,
            recall
        );
    }
    println!("\n  (brute is the exact oracle; HNSW is approximate but sublinear — recall@1 should hold ≥ 0.98)");
    0
}

/// One labeled pair for `recall calibrate`: two prompts and whether they *should* be treated as the
/// same cached answer (a paraphrase = should_hit; a look-alike-but-different = should not).
#[derive(serde::Deserialize)]
struct CalibPair {
    a: String,
    b: String,
    should_hit: bool,
}

/// Cosine similarity of two embedding vectors (normalize then dot) — both indexes treat a dot product
/// over unit vectors as cosine, exactly as the `SemanticCache` facade does on the hot path.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    let (na, nb) = (norm(a), norm(b));
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>() / (na * nb)
}

/// `recall calibrate --file <pairs.jsonl>`: there is no portable magic threshold across embedders
/// (PLAN.md §5.1), so this measures the similarity distributions of labeled paraphrase (should-hit)
/// and look-alike (should-not-hit) pairs *under the configured embedder* and prints the cutoff τ that
/// best separates them — the value to put in `recall.toml`'s `tau` (or the cold-start prior for the
/// adaptive policy). With `--target-fhr <r>` it instead reports the most permissive τ that still holds
/// the false-hit rate at or below `r` (maximize cache hit-rate subject to the correctness budget).
fn cmd_calibrate(args: &[String]) -> i32 {
    let Some(file) = flag(args, "--file") else {
        eprintln!("recall calibrate: --file <pairs.jsonl> is required");
        return 2;
    };
    let target_fhr: Option<f64> = match flag(args, "--target-fhr") {
        Some(s) => match s.parse::<f64>() {
            Ok(r) if (0.0..=1.0).contains(&r) => Some(r),
            _ => {
                eprintln!("recall calibrate: --target-fhr must be a number in [0.0, 1.0]");
                return 2;
            }
        },
        None => None,
    };
    let model_path = flag(args, "--model")
        .map(str::to_string)
        .or_else(|| std::env::var("RECALL_MODEL2VEC_PATH").ok());
    let embedder = build_embedder(model_path.as_deref());

    let content = match std::fs::read_to_string(file) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("recall calibrate: cannot read {file}: {e}");
            return 1;
        }
    };

    // Cosine similarity of each labeled pair, split into the should-hit (pos) and should-not (neg)
    // distributions. The optimal τ lives in the gap between them.
    let (mut pos, mut neg): (Vec<f32>, Vec<f32>) = (Vec::new(), Vec::new());
    for (i, line) in content.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let pair: CalibPair = match serde_json::from_str(line) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("recall calibrate: {file}:{}: invalid pair JSON: {e}", i + 1);
                return 1;
            }
        };
        let (va, vb) = match (embedder.embed_one(&pair.a), embedder.embed_one(&pair.b)) {
            (Ok(va), Ok(vb)) => (va, vb),
            (Err(e), _) | (_, Err(e)) => {
                eprintln!("recall calibrate: {file}:{}: embed failed: {e}", i + 1);
                return 1;
            }
        };
        let sim = cosine(&va, &vb);
        if pair.should_hit {
            pos.push(sim);
        } else {
            neg.push(sim);
        }
    }
    if pos.is_empty() || neg.is_empty() {
        eprintln!(
            "recall calibrate: need at least one should_hit AND one should_not_hit pair (got {} / {})",
            pos.len(),
            neg.len()
        );
        return 2;
    }

    // Metrics at a candidate cutoff τ ("serve as a hit" ⇔ sim ≥ τ).
    let total_pos = pos.len() as f64;
    let total_neg = neg.len() as f64;
    let metrics = |tau: f32| -> (f64, f64) {
        let true_hits = pos.iter().filter(|&&s| s >= tau).count() as f64;
        let false_hits = neg.iter().filter(|&&s| s >= tau).count() as f64;
        let hit_rate = true_hits / total_pos;
        let served = true_hits + false_hits;
        // False-hit rate as the operator means it: wrong answers among the hits actually served.
        let fhr = if served > 0.0 {
            false_hits / served
        } else {
            0.0
        };
        (hit_rate, fhr)
    };

    // Candidate cutoffs: every observed similarity (each is a real decision boundary).
    let mut candidates: Vec<f32> = pos.iter().chain(neg.iter()).copied().collect();
    candidates.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    candidates.dedup();

    let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;
    let pos_min = pos.iter().cloned().fold(f32::INFINITY, f32::min);
    let neg_max = neg.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    println!(
        "recall calibrate  (embedder: {}, pairs: {}  [{} hit / {} no-hit])",
        embedder.id(),
        pos.len() + neg.len(),
        pos.len(),
        neg.len()
    );
    println!(
        "  should-hit cosine : mean {:.4}  (min {:.4})",
        mean(&pos),
        pos_min
    );
    println!(
        "  should-not cosine : mean {:.4}  (max {:.4})",
        mean(&neg),
        neg_max
    );
    println!(
        "  separation gap    : {:.4}  ({})",
        pos_min - neg_max,
        if pos_min > neg_max {
            "cleanly separable"
        } else {
            "distributions overlap — no τ separates them perfectly"
        }
    );

    // Recommended τ: the cutoff maximizing Youden's J (hit-rate − false-positive-rate), the standard
    // separation point. Tie-break toward the higher τ (the more conservative, fewer-false-hits one).
    let fpr = |tau: f32| neg.iter().filter(|&&s| s >= tau).count() as f64 / total_neg;
    let mut best = (f32::NAN, f64::NEG_INFINITY);
    for &tau in &candidates {
        let (hit_rate, _) = metrics(tau);
        let j = hit_rate - fpr(tau);
        if j > best.1 || (j == best.1 && tau > best.0) {
            best = (tau, j);
        }
    }
    let (rec_hit, rec_fhr) = metrics(best.0);
    println!(
        "  recommended τ     : {:.4}   (Youden J {:.3}; hit-rate {:.1}%, false-hit {:.1}%)",
        best.0,
        best.1,
        rec_hit * 100.0,
        rec_fhr * 100.0
    );

    // Optional: the τ with the highest hit-rate that still holds the false-hit rate ≤ target (max
    // value subject to the correctness budget). FHR = false_hits / served is NOT monotonic in τ, so
    // the first cutoff under the cap need not be the best — evaluate every qualifying τ and pick the
    // max hit-rate, breaking ties toward the higher (more conservative) τ.
    if let Some(target) = target_fhr {
        let pick = candidates
            .iter()
            .copied()
            .filter_map(|tau| {
                let (hit_rate, fhr) = metrics(tau);
                (fhr <= target).then_some((tau, hit_rate, fhr))
            })
            .max_by(|a, b| {
                a.1.partial_cmp(&b.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal))
            });
        match pick {
            Some((tau, hr, fhr)) => {
                println!(
                    "  τ @ FHR ≤ {:.1}%   : {:.4}   (hit-rate {:.1}%, false-hit {:.1}%)",
                    target * 100.0,
                    tau,
                    hr * 100.0,
                    fhr * 100.0
                );
            }
            None => println!(
                "  τ @ FHR ≤ {:.1}%   : unreachable on this set (even τ=max admits a false hit)",
                target * 100.0
            ),
        }
    }
    0
}

/// `recall replay --file <reqs.jsonl>`: drive a running `serve` with a request log and report the
/// hit-rate and token savings it produced (the validation step for the "we saved $X" claim).
fn cmd_replay(args: &[String]) -> i32 {
    let Some(file) = flag(args, "--file") else {
        eprintln!("recall replay: --file <reqs.jsonl> is required");
        return 2;
    };
    let target = flag(args, "--target")
        .unwrap_or("http://127.0.0.1:8080")
        .to_string();
    let default_path = flag(args, "--path")
        .unwrap_or("/v1/chat/completions")
        .to_string();
    let verify_rate: f64 = flag(args, "--verify-sample")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);
    let upstream = flag(args, "--upstream")
        .map(str::to_string)
        .or_else(|| std::env::var("RECALL_UPSTREAM").ok());
    let price_input: Option<f64> = flag(args, "--price-input").and_then(|s| s.parse().ok());
    let price_output: Option<f64> = flag(args, "--price-output").and_then(|s| s.parse().ok());
    // Verification re-calls the upstream, so it needs the upstream URL and (almost always) a key. The
    // key is env-only, never a flag — same rule as `serve`.
    let upstream_key = std::env::var("RECALL_UPSTREAM_API_KEY")
        .ok()
        .or_else(|| std::env::var("RECALL_ANTHROPIC_API_KEY").ok());
    if !(0.0..=1.0).contains(&verify_rate) {
        eprintln!("recall replay: --verify-sample must be in [0.0, 1.0]");
        return 2;
    }
    if verify_rate > 0.0 && upstream.is_none() {
        eprintln!(
            "recall replay: --verify-sample needs --upstream (or RECALL_UPSTREAM) to compare against"
        );
        return 2;
    }

    let content = match std::fs::read_to_string(file) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("recall replay: cannot read {file}: {e}");
            return 1;
        }
    };
    let lines: Vec<String> = content.lines().map(str::to_string).collect();
    let total_lines = lines.len();

    let opts = recall_proxy::replay::ReplayOpts {
        target: target.clone(),
        default_path,
        verify_rate,
        upstream,
        upstream_key,
        anthropic_version: "2023-06-01".to_string(),
    };

    let rt = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("recall replay: failed to start runtime: {e}");
            return 1;
        }
    };
    let rep = match rt.block_on(recall_proxy::replay::run_replay(lines, opts)) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("recall replay: {e}");
            return 1;
        }
    };

    println!("recall replay  (target: {target}, {total_lines} lines)");
    println!("  requests        : {}", rep.requests);
    println!("  hits            : {}", rep.hits);
    println!("  misses          : {}", rep.misses);
    println!("  bypass          : {}", rep.bypass);
    println!("  errors          : {}", rep.errors);
    println!(
        "  hit-rate        : {:.1}%  (over cacheable)",
        rep.hit_ratio * 100.0
    );
    println!("  tokens saved    : {} total", rep.tokens_saved);
    println!("    input         : {}", rep.input_tokens_saved);
    println!("    output        : {}", rep.output_tokens_saved);
    match (price_input, price_output) {
        (Some(pi), Some(po)) => {
            let usd = rep.input_tokens_saved as f64 / 1e6 * pi
                + rep.output_tokens_saved as f64 / 1e6 * po;
            println!("  est. saved      : ${usd:.4}  (in ${pi}/Mtok, out ${po}/Mtok)");
        }
        _ => println!(
            "  est. saved      : (pass --price-input/--price-output <usd per 1M tokens> to price)"
        ),
    }
    if rep.verified > 0 {
        let fhr = rep.verify_mismatch as f64 / rep.verified as f64 * 100.0;
        println!(
            "  verify          : {} sampled, {} mismatch ({fhr:.1}% candidate false-hit), {} unchecked",
            rep.verified, rep.verify_mismatch, rep.verify_unchecked
        );
    }
    // A run where every request errored exercised nothing (proxy unreachable, upstream down, malformed
    // log) — surface that as a non-zero exit so a CI/SSM gate sees red instead of a misleading 0% hit.
    if rep.requests > 0 && rep.errors == rep.requests {
        eprintln!(
            "recall replay: all {} requests errored — nothing measured",
            rep.requests
        );
        return 1;
    }
    0
}

fn cmd_serve(args: &[String]) -> i32 {
    // Structured logging for the long-running proxy: honor RUST_LOG (e.g. `RUST_LOG=recall=debug`
    // for per-request hit/miss detail), defaulting to `info`. `try_init` so a double-install (tests)
    // is a no-op rather than a panic. Arg/parse errors below still go to stderr directly.
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_target(false)
        .try_init();

    // Lowest precedence: an optional config file.
    let file = match flag(args, "--config") {
        Some(path) => match std::fs::read_to_string(path).map(|s| toml::from_str::<FileConfig>(&s))
        {
            Ok(Ok(fc)) => fc,
            Ok(Err(e)) => {
                eprintln!("recall serve: invalid {path}: {e}");
                return 2;
            }
            Err(e) => {
                eprintln!("recall serve: cannot read {path}: {e}");
                return 2;
            }
        },
        None => FileConfig::default(),
    };
    // Captured before `resolve` consumes `file`'s fields: the hot-reload baseline (for detecting
    // non-policy config changes) and the path the watcher re-reads.
    let startup_file = file.clone();
    let config_path = flag(args, "--config").map(str::to_string);

    // Resolve each setting as flag > env (RECALL_*) > file > default.
    let resolve = |flag_name: &str, env: &str, file: Option<String>, default: &str| -> String {
        flag(args, flag_name)
            .map(str::to_string)
            .or_else(|| std::env::var(env).ok())
            .or(file)
            .unwrap_or_else(|| default.to_string())
    };

    let listen = resolve("--listen", "RECALL_LISTEN", file.listen, "127.0.0.1:8080");
    let upstream = resolve(
        "--upstream",
        "RECALL_UPSTREAM",
        file.upstream,
        "https://api.openai.com",
    );
    let base_namespace = resolve(
        "--base-namespace",
        "RECALL_BASE_NAMESPACE",
        file.base_namespace,
        "default",
    );
    let max_temperature: f64 = resolve(
        "--max-temperature",
        "RECALL_MAX_TEMPERATURE",
        file.max_temperature.map(|v| v.to_string()),
        "1.0",
    )
    .parse()
    .unwrap_or(1.0);
    let tau: f32 = resolve(
        "--tau",
        "RECALL_TAU",
        file.tau.map(|v| v.to_string()),
        "0.9",
    )
    .parse()
    .unwrap_or(0.9);
    let model = flag(args, "--model")
        .map(str::to_string)
        .or_else(|| std::env::var("RECALL_MODEL2VEC_PATH").ok())
        .or(file.model);
    // The upstream key is read from the environment only — never a flag (it would leak into `ps`).
    let upstream_api_key = std::env::var("RECALL_UPSTREAM_API_KEY").ok();
    // Anthropic (`/v1/messages`) upstream — separate base/version/key from the OpenAI upstream.
    let anthropic_upstream = resolve(
        "--anthropic-upstream",
        "RECALL_ANTHROPIC_UPSTREAM",
        None,
        "https://api.anthropic.com",
    );
    let anthropic_version =
        std::env::var("RECALL_ANTHROPIC_VERSION").unwrap_or_else(|_| "2023-06-01".to_string());
    // Like the OpenAI key, the Anthropic key is env-only — never a flag.
    let anthropic_api_key = std::env::var("RECALL_ANTHROPIC_API_KEY").ok();
    // Store backend: `memory` (default, ephemeral) or `redb` (durable; needs a `--features
    // store-redb` build). The db path is where redb persists.
    let store_kind = resolve("--store", "RECALL_STORE", file.store, "memory");
    let db_path = resolve("--db-path", "RECALL_DB_PATH", file.db_path, "recall.redb");
    // Threshold policy: `static` (fixed τ, default) or `adaptive` (learns a per-namespace τ toward a
    // target false-hit rate; rests at the cold-start τ until feedback arrives — PLAN.md §5).
    let policy_kind = resolve("--policy", "RECALL_POLICY", file.policy, "static");
    let target_fhr: f64 = resolve(
        "--target-fhr",
        "RECALL_TARGET_FHR",
        file.target_fhr.map(|v| v.to_string()),
        "0.02",
    )
    .parse()
    .unwrap_or(0.02);
    let initial_policy = match build_policy(&policy_kind, tau, target_fhr) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("recall serve: {e}");
            return 2;
        }
    };
    if policy_kind == "adaptive" {
        tracing::info!(
            target_fhr,
            cold_start_tau = tau,
            "adaptive threshold active; resting at cold-start τ until feedback arrives"
        );
    }

    // Optional hot-reload (`--watch-config`): wrap the policy so the config watcher can swap it in
    // place. ONLY the threshold policy is hot-reloadable (PLAN.md §3-OSS); the listener, index,
    // store, and embedder own bound sockets / loaded weights and need a restart.
    let watch_config = args.iter().any(|a| a == "--watch-config");
    let (policy, reload_handle): (Box<dyn recall_core::ThresholdPolicy>, _) = if watch_config {
        if config_path.is_none() {
            eprintln!("recall serve: --watch-config requires --config <file>");
            return 2;
        }
        let handle = recall_proxy::PolicyHandle::new(initial_policy);
        let swappable = recall_proxy::SwappablePolicy::new(handle.clone());
        (Box::new(swappable), Some(handle))
    } else {
        (initial_policy, None)
    };
    // ANN index: `brute` (exact, default) or `hnsw` (approximate, sublinear at scale — recall@1 ≥
    // 0.98 vs the brute-force oracle, PLAN.md §3-OSS T3). Both are pure-Rust, no extra deps.
    let index_kind = resolve("--index", "RECALL_INDEX", file.index, "brute");
    let index: Box<dyn recall_core::AnnIndex> = match index_kind.as_str() {
        "brute" => Box::new(BruteForceIndex::new()),
        "hnsw" => {
            eprintln!(
                "recall serve: HNSW index (approximate ANN, recall@1 ≥ 0.98 vs the brute-force oracle)"
            );
            Box::new(recall_index::HnswIndex::new())
        }
        other => {
            eprintln!("recall serve: unknown --index '{other}' (expected: brute | hnsw)");
            return 2;
        }
    };

    let embedder = build_embedder(model.as_deref());
    // Fail fast if an explicitly requested backend can't be honored, rather than silently
    // downgrading the durability/behavior the operator asked for.
    let cache = match build_cache(embedder, index, &store_kind, &db_path, policy) {
        Ok(cache) => Arc::new(cache),
        Err(e) => {
            eprintln!("recall serve: {e}");
            return 2;
        }
    };
    let config = Config {
        listen: listen.clone(),
        upstream_base: upstream.clone(),
        upstream_api_key,
        base_namespace,
        max_temperature,
        anthropic_upstream_base: anthropic_upstream,
        anthropic_version,
        anthropic_api_key,
        ..Config::default()
    };
    let state = ProxyState::new(cache, config);

    // Start the config watcher (if `--watch-config`): it swaps the threshold policy in place when the
    // file changes. `config_path` is guaranteed Some here (validated above).
    if let Some(handle) = reload_handle {
        let pins = ReloadPins {
            tau_pinned: flag(args, "--tau").is_some() || std::env::var("RECALL_TAU").is_ok(),
            policy_pinned: flag(args, "--policy").is_some()
                || std::env::var("RECALL_POLICY").is_ok(),
            fhr_pinned: flag(args, "--target-fhr").is_some()
                || std::env::var("RECALL_TARGET_FHR").is_ok(),
            tau,
            policy_kind: policy_kind.clone(),
            target_fhr,
        };
        spawn_config_watcher(
            config_path.expect("--watch-config validated --config present"),
            handle,
            pins,
            startup_file,
        );
    }

    let rt = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("recall serve: failed to start runtime: {e}");
            return 1;
        }
    };
    tracing::info!(
        listen = %listen,
        openai_upstream = %upstream,
        anthropic_upstream = %state.config.anthropic_upstream_base,
        "recall serve: listening (cache-miss forwarding)"
    );
    // Propagate a bind/serve failure as a non-zero exit so a supervisor/script sees a dead proxy.
    match rt.block_on(async move { recall_proxy::serve(state).await }) {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("recall serve: server error: {e}");
            1
        }
    }
}

/// Pick the embedder: the static model2vec backend if `--model`/`RECALL_MODEL2VEC_PATH` is set AND
/// the binary was built with `--features static`; otherwise the deterministic hash embedder.
fn build_embedder(model_path: Option<&str>) -> Box<dyn Embedder> {
    if let Some(path) = model_path {
        #[cfg(feature = "static")]
        {
            match recall_embed::Model2VecEmbedder::load_local(path, format!("model2vec@{path}")) {
                Ok(m) => {
                    eprintln!("recall serve: loaded static embedder from {path}");
                    return Box::new(m);
                }
                Err(e) => eprintln!("recall serve: model load failed ({e}); using hash embedder"),
            }
        }
        #[cfg(not(feature = "static"))]
        {
            eprintln!(
                "recall serve: --model needs a `--features static` build; ignoring {path}, \
                 using hash embedder"
            );
        }
    }
    Box::new(HashEmbedder::default())
}

/// Build the cache from the chosen ANN index, store backend, and threshold policy. Returns an error
/// (so `serve` exits non-zero) when an explicitly requested backend can't be honored — an unknown
/// `--store`, or `redb` on a binary built without the `store-redb` feature, or a redb open failure —
/// rather than silently downgrading to the in-memory store.
fn build_cache(
    embedder: Box<dyn Embedder>,
    index: Box<dyn recall_core::AnnIndex>,
    store_kind: &str,
    db_path: &str,
    policy: Box<dyn recall_core::ThresholdPolicy>,
) -> Result<recall_core::DynCache, String> {
    match store_kind {
        "memory" => Ok(recall_proxy::boxed_cache_full_with_index(
            embedder,
            index,
            Box::new(MemKv::new()),
            policy,
        )),
        "redb" => {
            #[cfg(feature = "store-redb")]
            {
                let store = recall_store::RedbStore::open(db_path)
                    .map_err(|e| format!("--store redb: open {db_path} failed: {e}"))?;
                let cache = recall_proxy::boxed_cache_full_with_index(
                    embedder,
                    index,
                    Box::new(store),
                    policy,
                );
                let persisted = cache.entries(); // KV blob count (kv.len) through the boxed store
                                                 // Rebuild the in-memory index + exact-map from the persisted blobs so cached
                                                 // *lookups* — not just the KV entries — survive the restart. On failure we keep
                                                 // serving with a cold index (the blobs are intact; entries re-learn on first miss)
                                                 // rather than refusing to start.
                match cache.rehydrate() {
                    Ok(n) => tracing::info!(
                        db_path,
                        rehydrated = n,
                        persisted,
                        "durable redb store — rehydrated cached lookups across restart"
                    ),
                    Err(e) => tracing::warn!(
                        db_path,
                        error = %e,
                        "durable redb store — rehydration failed; serving with a cold index (KV blobs intact)"
                    ),
                }
                Ok(cache)
            }
            #[cfg(not(feature = "store-redb"))]
            {
                let _ = (embedder, index, policy, db_path);
                Err("--store redb requires a `--features store-redb` build".to_string())
            }
        }
        other => Err(format!(
            "unknown --store '{other}' (expected: memory | redb)"
        )),
    }
}

/// Build a threshold policy from its config fields. Factored out so both startup and the hot-reload
/// watcher construct the policy the same way.
fn build_policy(
    kind: &str,
    tau: f32,
    target_fhr: f64,
) -> Result<Box<dyn recall_core::ThresholdPolicy>, String> {
    match kind {
        "static" => Ok(Box::new(StaticThreshold::new(tau))),
        "adaptive" => Ok(Box::new(recall_calibrate::AdaptiveThreshold::new(
            recall_calibrate::AdaptiveConfig::new(tau, target_fhr),
        ))),
        other => Err(format!(
            "unknown --policy '{other}' (expected: static | adaptive)"
        )),
    }
}

/// What the startup command line pinned, plus the startup-resolved policy values. On reload, a field
/// pinned by a flag or env keeps its startup value (flag > env > file precedence is preserved); an
/// unpinned field tracks the config file.
struct ReloadPins {
    tau_pinned: bool,
    policy_pinned: bool,
    fhr_pinned: bool,
    tau: f32,
    policy_kind: String,
    target_fhr: f64,
}

/// The effective `(policy, tau, target_fhr)` after a reload: the file's value for each unpinned field,
/// the startup value for a pinned one. Pure — unit-tested.
fn effective_policy(new: &FileConfig, pins: &ReloadPins) -> (String, f32, f64) {
    let kind = if pins.policy_pinned {
        pins.policy_kind.clone()
    } else {
        new.policy.clone().unwrap_or_else(|| "static".to_string())
    };
    let tau = if pins.tau_pinned {
        pins.tau
    } else {
        new.tau.unwrap_or(0.9)
    };
    let fhr = if pins.fhr_pinned {
        pins.target_fhr
    } else {
        new.target_fhr.unwrap_or(0.02)
    };
    (kind, tau, fhr)
}

/// Non-policy config fields that changed vs the startup baseline — these are NOT hot-reloadable (they
/// own bound sockets / loaded weights / file handles), so the watcher logs them and leaves them at
/// their startup values. Pure — unit-tested.
fn ignored_nonpolicy_changes(new: &FileConfig, startup: &FileConfig) -> Vec<&'static str> {
    let mut changed = Vec::new();
    if new.listen != startup.listen {
        changed.push("listen");
    }
    if new.upstream != startup.upstream {
        changed.push("upstream");
    }
    if new.base_namespace != startup.base_namespace {
        changed.push("base_namespace");
    }
    if new.model != startup.model {
        changed.push("model");
    }
    if new.index != startup.index {
        changed.push("index");
    }
    if new.store != startup.store {
        changed.push("store");
    }
    if new.db_path != startup.db_path {
        changed.push("db_path");
    }
    if new.max_temperature != startup.max_temperature {
        changed.push("max_temperature");
    }
    changed
}

/// Re-read the config file and swap the threshold policy in place. On a read or parse error, keep the
/// previous policy (a typo can never take the cache down — PLAN.md §3-OSS); non-policy changes are
/// logged and ignored. Used by the watcher and exercised directly by tests.
fn apply_reload(
    path: &str,
    handle: &recall_proxy::PolicyHandle,
    pins: &ReloadPins,
    startup: &FileConfig,
) {
    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => {
            tracing::warn!(error = %e, "config reload skipped: cannot read file; keeping previous policy");
            return;
        }
    };
    let new: FileConfig = match toml::from_str(&content) {
        Ok(c) => c,
        Err(e) => {
            tracing::warn!(error = %e, "config reload skipped: parse error; keeping previous policy");
            return;
        }
    };
    for field in ignored_nonpolicy_changes(&new, startup) {
        tracing::warn!(
            field,
            "config field changed but is not hot-reloadable; restart to apply"
        );
    }
    let (kind, tau, fhr) = effective_policy(&new, pins);
    match build_policy(&kind, tau, fhr) {
        Ok(p) => {
            let from = handle.current_id();
            handle.store(p);
            tracing::info!(from = %from, to = %handle.current_id(), "hot-reloaded threshold policy");
        }
        Err(e) => {
            tracing::warn!(error = %e, "config reload skipped: invalid policy; keeping previous")
        }
    }
}

/// Watch the config file's directory and apply a debounced reload on change. Runs on a dedicated
/// thread (notify uses its own anyway); a watcher-init failure disables hot-reload with a warning
/// rather than taking the server down.
fn spawn_config_watcher(
    path: String,
    handle: std::sync::Arc<recall_proxy::PolicyHandle>,
    pins: ReloadPins,
    startup: FileConfig,
) {
    use notify::{RecursiveMode, Watcher};
    std::thread::spawn(move || {
        let (tx, rx) = std::sync::mpsc::channel();
        let mut watcher = match notify::recommended_watcher(move |res| {
            let _ = tx.send(res);
        }) {
            Ok(w) => w,
            Err(e) => {
                tracing::warn!(error = %e, "hot-reload disabled: watcher init failed");
                return;
            }
        };
        // Watch the parent directory, not the file inode: editors replace the file on save (write to
        // a temp + rename), which a file-inode watch would miss after the first write.
        let pathbuf = std::path::PathBuf::from(&path);
        let dir = pathbuf
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .map(std::path::Path::to_path_buf)
            .unwrap_or_else(|| std::path::PathBuf::from("."));
        if let Err(e) = watcher.watch(&dir, RecursiveMode::NonRecursive) {
            tracing::warn!(error = %e, dir = %dir.display(), "hot-reload disabled: cannot watch directory");
            return;
        }
        tracing::info!(path = %path, "watching config for hot-reload (threshold policy only)");
        loop {
            // Block for the first event, then drain a ~200 ms burst (one save fires several events).
            if rx.recv().is_err() {
                break; // the watcher was dropped
            }
            while rx
                .recv_timeout(std::time::Duration::from_millis(200))
                .is_ok()
            {}
            apply_reload(&path, &handle, &pins, &startup);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::{
        apply_reload, build_policy, cosine, effective_policy, ignored_nonpolicy_changes,
        FileConfig, ReloadPins,
    };
    use recall_proxy::PolicyHandle;

    #[test]
    fn cosine_is_one_for_identical_and_zero_for_orthogonal() {
        let a = [1.0_f32, 2.0, 3.0];
        assert!((cosine(&a, &a) - 1.0).abs() < 1e-6, "identical → 1.0");
        assert!(
            cosine(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6,
            "orthogonal → 0.0"
        );
        // A zero vector has no direction — defined as 0 similarity, never NaN.
        assert_eq!(cosine(&[0.0, 0.0], &[1.0, 1.0]), 0.0);
        // Magnitude-invariant: scaling a vector does not change the angle.
        assert!((cosine(&[2.0, 0.0], &[5.0, 0.0]) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn build_policy_known_and_unknown() {
        assert_eq!(
            build_policy("static", 0.9, 0.02).unwrap().id(),
            "static@0.900"
        );
        assert!(build_policy("adaptive", 0.85, 0.02)
            .unwrap()
            .id()
            .starts_with("adaptive@"));
        assert!(build_policy("bogus", 0.9, 0.02).is_err());
    }

    fn unpinned() -> ReloadPins {
        ReloadPins {
            tau_pinned: false,
            policy_pinned: false,
            fhr_pinned: false,
            tau: 0.9,
            policy_kind: "static".into(),
            target_fhr: 0.02,
        }
    }

    #[test]
    fn effective_policy_tracks_file_when_unpinned_and_startup_when_pinned() {
        let file = FileConfig {
            policy: Some("adaptive".into()),
            tau: Some(0.7),
            target_fhr: Some(0.05),
            ..Default::default()
        };

        // Unpinned: the file wins.
        let (kind, tau, fhr) = effective_policy(&file, &unpinned());
        assert_eq!((kind.as_str(), tau, fhr), ("adaptive", 0.7, 0.05));

        // Pinned by flag/env: the startup value wins, the file is ignored.
        let pins = ReloadPins {
            tau_pinned: true,
            policy_pinned: true,
            fhr_pinned: true,
            tau: 0.95,
            policy_kind: "static".into(),
            target_fhr: 0.01,
        };
        let (kind, tau, fhr) = effective_policy(&file, &pins);
        assert_eq!((kind.as_str(), tau, fhr), ("static", 0.95, 0.01));
    }

    #[test]
    fn ignored_changes_flags_nonpolicy_fields_only() {
        let startup = FileConfig::default();
        // A POLICY field change must NOT be flagged as a non-reloadable change.
        let policy_only = FileConfig {
            tau: Some(0.5),
            ..Default::default()
        };
        assert!(ignored_nonpolicy_changes(&policy_only, &startup).is_empty());

        // Non-policy fields are flagged (they need a restart).
        let nonpolicy = FileConfig {
            tau: Some(0.5),
            listen: Some("0.0.0.0:9".into()),
            model: Some("/m".into()),
            ..Default::default()
        };
        let ignored = ignored_nonpolicy_changes(&nonpolicy, &startup);
        assert!(ignored.contains(&"listen") && ignored.contains(&"model"));
        assert!(!ignored.contains(&"tau"));
    }

    #[test]
    fn apply_reload_swaps_on_valid_and_keeps_previous_on_parse_error() {
        let dir = std::env::temp_dir();
        let path = dir.join("recall_reload_test.toml");
        let path_str = path.to_str().unwrap();
        let handle = PolicyHandle::new(build_policy("static", 0.9, 0.02).unwrap());
        let startup = FileConfig::default();

        // Valid reload lowering τ → the policy swaps live.
        std::fs::write(&path, "tau = 0.5\n").unwrap();
        apply_reload(path_str, &handle, &unpinned(), &startup);
        assert_eq!(
            handle.current_id(),
            "static@0.500",
            "valid reload swapped τ"
        );

        // Switch policy kind too.
        std::fs::write(&path, "policy = \"adaptive\"\ntau = 0.8\n").unwrap();
        apply_reload(path_str, &handle, &unpinned(), &startup);
        assert!(
            handle.current_id().starts_with("adaptive@"),
            "reload switched policy kind"
        );

        // A malformed file must NOT take the cache down — keep the previous good policy.
        std::fs::write(&path, "tau = = broken\n").unwrap();
        apply_reload(path_str, &handle, &unpinned(), &startup);
        assert!(
            handle.current_id().starts_with("adaptive@"),
            "parse error keeps the previous policy"
        );

        let _ = std::fs::remove_file(&path);
    }
}
