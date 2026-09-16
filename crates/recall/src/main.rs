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
        Some("status") => cmd_status(&args[1..]),
        Some("export") => cmd_export(&args[1..]),
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
         [--index brute|hnsw] [--hnsw-m n] [--hnsw-ef-construction n] [--hnsw-ef-search n]\n               \
         [--ambiguity-eps f] [--store memory|redb] [--db-path file] [--policy static|adaptive]\n               \
         [--savings-flush-secs n]  (persist the savings counters; needs --store redb)\n               \
         [--watch-config]  (hot-reload the threshold policy from --config; other fields need a restart)\n  \
         recall ask --ns <namespace> \"<prompt>\"\n  \
         recall bench [--tau <f32>] [--iters <n>] [--model <dir>]\n  \
         recall replay --file <reqs.jsonl> [--target url] [--path p] [--verify-sample r]\n               \
         [--upstream url] [--price-input <usd/Mtok>] [--price-output <usd/Mtok>]\n  \
         recall status [--target url] [--json]\n  \
         recall export [--target url]\n  \
         recall ann-bench [--dims <d>] [--queries <q>] [--sizes 100,1000,10000] [--top-k <k>]\n               \
         [--m <n>] [--ef-construction <n>] [--ef-search <n>]\n  \
         recall calibrate --file <pairs.jsonl> [--model <dir>] [--target-fhr <r>]\n\n\
         `serve` runs the OpenAI-compatible proxy (point your client's base URL at it).\n\
         `calibrate` picks the similarity threshold for the configured embedder from a labeled\n\
         JSONL pair set ({{\"a\":..,\"b\":..,\"should_hit\":true|false}} per line).\n\
         `replay` drives a running `serve` with a JSONL request log and reports the dollars, tokens,\n\
         and hit-rate that log would have saved — run it BEFORE putting recall in your request path.\n\
         `status` summarizes a running `serve` in one screen — what it has saved, and what that\n\
         figure does and does not account for.\n\
         `export` emits the same facts as a versioned JSON document (recall.export/v1) — the\n\
         stable shape to pipe into your own roll-up, chargeback, or reporting tooling.\n\
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
    /// HNSW graph tunables, used only with `index = "hnsw"`. Defaults are validated at the bundled
    /// embedder's 256-dim width (see BENCHMARKS.md); `ef_search` is the recall-vs-latency dial to
    /// raise first when `ann-bench` shows recall@1 short of the gate for your dims/N.
    hnsw_m: Option<usize>,
    hnsw_ef_construction: Option<usize>,
    hnsw_ef_search: Option<usize>,
    /// Second-neighbor ambiguity gate: a would-be hit whose top-two neighbours sit within this
    /// similarity of each other AND store different answers is served as a miss. 0 (default)
    /// disarms it.
    ambiguity_eps: Option<f32>,
    /// How often (seconds) the savings counters are snapshotted into the durable store, so the
    /// "$ saved" figure survives a restart. Only with `store = "redb"`. 0 (the default) keeps the
    /// documented behaviour: counters reset with the process.
    savings_flush_secs: Option<u64>,
    /// `[pricing.<model>]` — the operator's own USD-per-million-token rates, used to turn saved
    /// tokens into `recall_cost_saved_usd_total`. File-only (there is no flag or env form: a rate
    /// table is not a scalar), optional, and empty by default — recall ships no price list.
    #[serde(default)]
    pricing: std::collections::HashMap<String, recall_proxy::pricing::RateSpec>,
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
    // HNSW tunables, defaulting to the shipped `HnswConfig::default()` — the same graph `serve
    // --index hnsw` builds — so a plain `ann-bench` run validates the real deployed config, and
    // the flags let an operator sweep for THEIR dims/N before opting in at scale.
    let hnsw_defaults = recall_index::HnswConfig::default();
    let hnsw_m: usize = flag(args, "--m")
        .and_then(|s| s.parse().ok())
        .unwrap_or(hnsw_defaults.m);
    let hnsw_efc: usize = flag(args, "--ef-construction")
        .and_then(|s| s.parse().ok())
        .unwrap_or(hnsw_defaults.ef_construction);
    let hnsw_efs: usize = flag(args, "--ef-search")
        .and_then(|s| s.parse().ok())
        .unwrap_or(hnsw_defaults.ef_search);
    let cfg = recall_index::HnswConfig::with_params(hnsw_m, hnsw_efc, hnsw_efs);
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

    println!(
        "recall ann-bench  (dims: {dims}, queries: {queries}, top-k: {top_k}, \
         hnsw: m={hnsw_m} ef_construction={hnsw_efc} ef_search={hnsw_efs})"
    );
    println!(
        "  {:>8} | {:>16} | {:>16} | {:>8} | {:>9}",
        "corpus", "brute p50/p99 µs", "hnsw p50/p99 µs", "speedup", "recall@1"
    );
    for &n in &sizes {
        let corpus = gen_normalized(n, dims, 0xC0DE_F00D);
        let brute = BruteForceIndex::new();
        let hnsw = recall_index::HnswIndex::with_config(cfg);
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

/// Print the savings block — the headline of a replay run.
///
/// The dollar figure comes from the proxy's own per-model pricing (`recall_cost_saved_usd_total`)
/// whenever the target has a `[pricing.<model>]` table; that is the only way a log spanning several
/// models is priced correctly. `--price-input`/`--price-output` remain as a fallback for a target
/// with no price book, and are labeled as what they are: one blended rate across every model in the
/// log. When neither is available the line says how to get a number rather than printing `$0.00`,
/// which would read as "the cache saved nothing".
///
/// Where a false-hit rate was actually measured (`--verify-sample`), the discounted figure is
/// printed **next to** the gross one. A wrong hit is not a saving — it bought a wrong answer with
/// the money it did not spend — and an ROI number quoted gross when a measured error rate exists is
/// the kind that gets a cache ripped back out a quarter later. With no sampling, the un-discounted
/// figure is labeled as such instead of being silently passed off as verified.
fn print_savings(
    rep: &recall_proxy::replay::ReplayReport,
    price_input: Option<f64>,
    price_output: Option<f64>,
) {
    if rep.cost_saved_micros > 0 {
        println!(
            "  $ saved         : ${:.4}   (priced per model by the target's [pricing] table)",
            rep.cost_saved_usd()
        );
        match rep.cost_saved_usd_discounted() {
            Some(net) => println!(
                "                    ${net:.4} less the {:.1}% sampled false-hit rate",
                rep.sampled_false_hit_rate().unwrap_or(0.0) * 100.0
            ),
            None => println!(
                "                    not discounted for wrong hits — add --verify-sample to measure them"
            ),
        }
        if rep.unpriced_tokens_saved > 0 {
            println!(
                "                    understated: {} saved tokens were on models with no price rule",
                rep.unpriced_tokens_saved
            );
        }
        return;
    }
    // No per-model figure. Fall back to the flat rates, or explain how to get a real one.
    match (price_input, price_output) {
        (Some(pi), Some(po)) => {
            let usd = rep.input_tokens_saved as f64 / 1e6 * pi
                + rep.output_tokens_saved as f64 / 1e6 * po;
            println!(
                "  $ saved (est.)  : ${usd:.4}   (one blended rate: in ${pi}/Mtok, out ${po}/Mtok)"
            );
        }
        // $0 has several causes and the snapshot can tell most of them apart (same
        // disambiguation as `recall status`): blaming a missing price book when one exists
        // sends the user to fix something that is not broken.
        _ if rep.unpriced_tokens_saved > 0 => println!(
            "  $ saved         : not priced — the hits were on models with no matching [pricing.<model>]\n                    rule on the target (its table exists but does not cover this log's models)"
        ),
        _ if rep.hits_usage_unknown > 0 => println!(
            "  $ saved         : not priced — the replayed completions reported no token usage to price"
        ),
        _ => println!(
            "  $ saved         : not priced — give the target a [pricing.<model>] table (see recall.toml),\n                    or pass --price-input/--price-output <usd per 1M tokens> for a flat estimate"
        ),
    }
}

/// `recall replay --file <reqs.jsonl>`: drive a running `serve` with a request log and report the
/// hit-rate and dollar savings it produced (the validation step for the "we saved $X" claim).
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
    // The money first: it is the line an engineer forwards to whoever owns the budget, and every
    // other number here exists to let that person check it.
    print_savings(&rep, price_input, price_output);
    println!(
        "  hit-rate        : {:.1}%  ({} of {} cacheable)",
        rep.hit_ratio * 100.0,
        rep.hits,
        rep.hits + rep.misses
    );
    println!(
        "  tokens saved    : {} total  (input {}, output {})",
        rep.tokens_saved, rep.input_tokens_saved, rep.output_tokens_saved
    );
    println!(
        "  requests        : {}  (hits {}, misses {}, bypass {}, errors {})",
        rep.requests, rep.hits, rep.misses, rep.bypass, rep.errors
    );
    if rep.verified > 0 {
        let fhr = rep.sampled_false_hit_rate().unwrap_or(0.0) * 100.0;
        println!(
            "  false-hit       : {fhr:.1}% sampled  ({} of {} checked disagreed with a fresh upstream answer, {} unchecked)",
            rep.verify_mismatch, rep.verified, rep.verify_unchecked
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

/// `recall status [--target url]`: one screen summarizing a running `serve`.
///
/// This is the thing an engineer forwards to whoever owns the budget, so it leads with the money
/// and then says, in the same breath, what that number does and does not account for. Every caveat
/// printed here is measured, not assumed:
///
/// - the savings are discounted by the **observed** false-hit rate when feedback has been reported,
///   and explicitly labelled un-discounted when none has — never silently treated as error-free;
/// - saved tokens on models with no price rule, and hits whose stored completion carried no usage
///   at all, are called out as *under-counts* rather than folded away, because a figure that is
///   quietly too low erodes trust exactly as fast as one that is too high.
///
/// It reads `/metrics` over HTTP and computes nothing the proxy did not already measure.
/// What window the node's cumulative counters actually cover, in words.
///
/// "since this process started" is right — and was the only possibility — until a node could
/// persist its counters. With `--savings-flush-secs` on they span the ledger instead, and saying
/// otherwise would understate a figure by exactly the history the operator turned persistence on
/// to keep. A node that predates the ledger reports `0` and reads as before.
fn counter_window(snap: &recall_proxy::scrape::Snapshot) -> String {
    if snap.ledger_since_unix == 0 {
        return "since this process started".to_string();
    }
    match snap
        .process_start_unix
        .checked_sub(snap.ledger_since_unix)
        .filter(|gap| *gap > 0)
    {
        // The ledger predates this process: report how much history it carries, since that is the
        // whole reason the number is worth quoting after a deploy.
        Some(gap) => format!("over the {} the ledger has run", human_duration(gap)),
        // Same instant (or a clock that moved backwards): this ledger began with this process.
        None => "since this ledger began, with this process".to_string(),
    }
}

/// A coarse, human duration — days/hours/minutes. Deliberately approximate: this labels a figure,
/// it is not the figure.
fn human_duration(secs: u64) -> String {
    match secs {
        s if s >= 172_800 => format!("{} days", s / 86_400),
        s if s >= 7_200 => format!("{} hours", s / 3_600),
        s if s >= 120 => format!("{} minutes", s / 60),
        s => format!("{s} seconds"),
    }
}

fn cmd_status(args: &[String]) -> i32 {
    let target = flag(args, "--target")
        .map(str::to_string)
        .or_else(|| std::env::var("RECALL_TARGET").ok())
        .unwrap_or_else(|| "http://127.0.0.1:8080".to_string());
    let as_json = args.iter().any(|a| a == "--json");

    let rt = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("recall status: failed to start runtime: {e}");
            return 1;
        }
    };
    let snap = match rt.block_on(recall_proxy::scrape::fetch_url(&target)) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("recall status: {e}");
            return 1;
        }
    };

    if as_json {
        match serde_json::to_string_pretty(&snap) {
            Ok(j) => println!("{j}"),
            Err(e) => {
                eprintln!("recall status: cannot render json: {e}");
                return 1;
            }
        }
        return 0;
    }

    println!("recall status  (target: {target})");
    match snap.cost_saved_micros {
        // $0 has several causes and the snapshot can tell most of them apart. Naming the wrong one
        // sends an operator to fix a price book that is already correct.
        0 => {
            let why = if snap.hits == 0 {
                "no hits yet"
            } else if snap.unpriced_tokens_saved > 0 {
                "hits so far were on models with no matching [pricing.<model>] rule"
            } else if snap.hits_usage_unknown > 0 {
                "the completions replayed so far reported no token usage to price"
            } else {
                "add a [pricing.<model>] table to the proxy's config (see recall.toml)"
            };
            println!("  $ saved         : not priced — {why}");
        }
        _ => {
            // The window is part of the figure: "$18,400" means nothing without "over what". It is
            // only spelled out once the ledger outlives the process, since until then every reader
            // already knew the answer was "this process".
            if snap.ledger_since_unix > 0 {
                println!(
                    "  $ saved         : ${:.4}   ({})",
                    snap.cost_saved_usd(),
                    counter_window(&snap)
                );
            } else {
                println!("  $ saved         : ${:.4}", snap.cost_saved_usd());
            }
            match snap.cost_saved_usd_discounted() {
                Some(net) => println!(
                    "                    ${net:.4} less the {:.1}% observed false-hit rate",
                    snap.observed_false_hit_rate().unwrap_or(0.0) * 100.0
                ),
                None => println!(
                    "                    not discounted for wrong hits — none have been reported to /v1/cache/feedback"
                ),
            }
        }
    }
    match snap.hit_rate() {
        Some(r) => println!(
            "  hit-rate        : {:.1}%  ({} of {} cacheable)",
            r * 100.0,
            snap.hits,
            snap.hits + snap.misses
        ),
        None => println!("  hit-rate        : no cacheable traffic yet"),
    }
    println!(
        "  tokens saved    : {} total  (input {}, output {})",
        snap.tokens_saved, snap.input_tokens_saved, snap.output_tokens_saved
    );
    // The fill side: what this process paid to build the cache it now holds. Reported next to the
    // savings because it is the denominator of the same question, and deliberately not framed as
    // anything more: every cache pays to be filled once, on one node or on fifty, so this is the
    // price of having a cache — not evidence of any particular topology. What running N of them
    // costs is arithmetic over N nodes' series, which belongs to whatever scrapes them all.
    if snap.fill_entries > 0 {
        let cost = if snap.fill_cost_micros > 0 {
            format!("  (${:.4} at the configured rates)", snap.fill_cost_usd())
        } else {
            String::new()
        };
        println!(
            "  filled          : {} entries bought and stored {window}{cost}",
            snap.fill_entries,
            window = counter_window(&snap)
        );
    }
    println!(
        "  entries         : {} held by the store",
        snap.cache_entries
    );
    if snap.rehydrated_entries > 0 || snap.persisted_entries_at_start > 0 {
        println!(
            "  recovered       : {} of {} persisted entries rebuilt at start-up",
            snap.rehydrated_entries, snap.persisted_entries_at_start
        );
    }
    // Not a partition of `requests`, and formatted so it does not read as one: a miss whose
    // upstream forward then failed is counted in both `misses` and `upstream errors`, and a request
    // rejected before it reached the cache is in none of them.
    println!(
        "  requests        : {}   hits {} · misses {} · bypass {} · upstream errors {}",
        snap.requests, snap.hits, snap.misses, snap.bypass, snap.upstream_errors
    );
    let judged = snap.feedback_agree + snap.feedback_wrong;
    match snap.observed_false_hit_rate() {
        Some(fhr) => println!(
            "  false-hit       : {:.1}% observed  ({} of {} judged hits reported wrong)",
            fhr * 100.0,
            snap.feedback_wrong,
            judged
        ),
        None => println!(
            "  false-hit       : unmeasured — no outcomes reported to POST /v1/cache/feedback"
        ),
    }

    // Everything below is a caveat on the figures above. Printed only when it applies, so a clean
    // deployment gets a clean screen and a caveat always means something.
    let mut caveats = Vec::new();
    if snap.unpriced_tokens_saved > 0 {
        caveats.push(format!(
            "$ saved understates: {} saved tokens were on models with no price rule",
            snap.unpriced_tokens_saved
        ));
    }
    if snap.hits_usage_unknown > 0 {
        caveats.push(format!(
            "$ saved and tokens saved understate: {} hits replayed a completion that reported no token usage",
            snap.hits_usage_unknown
        ));
    }
    if snap.unpriced_fill_tokens > 0 {
        caveats.push(format!(
            "fill cost understates: {} purchased tokens were on models with no price rule",
            snap.unpriced_fill_tokens
        ));
    }
    if snap.fill_usage_unknown > 0 {
        caveats.push(format!(
            "fill cost understates: {} stored entries came from a response that reported no token usage",
            snap.fill_usage_unknown
        ));
    }
    if snap.stream_not_stored > 0 {
        caveats.push(format!(
            "{} streamed misses were not cached (stream ended without its terminator, or truncated)",
            snap.stream_not_stored
        ));
    }
    if !caveats.is_empty() {
        println!("  caveats         :");
        for c in caveats {
            println!("                    - {c}");
        }
    }
    // Where the savings came from. Only the top rows: this is a status screen, and the full list
    // is one `curl /v1/cache/namespaces` away. Absent entirely on a node that predates the
    // endpoint, which is why nothing above depends on it.
    if !snap.by_namespace.is_empty() {
        const SHOWN: usize = 5;
        println!("  top namespaces  :");
        for row in snap.by_namespace.iter().take(SHOWN) {
            let dollars = if row.unpriced_tokens > 0 && row.cost_micros == 0 {
                "unpriced".to_string()
            } else {
                format!("${:.4}", row.cost_micros as f64 / 1e6)
            };
            println!(
                "                    {:>10}  {:<44} ({} hits)",
                dollars, row.namespace, row.hits
            );
        }
        if snap.by_namespace.len() > SHOWN {
            println!(
                "                    … {} more (GET /v1/cache/namespaces for all)",
                snap.by_namespace.len() - SHOWN
            );
        }
    }
    0
}

/// `recall export [--target url]`: the node's savings and usage as one versioned JSON document.
///
/// Where `status` is a screen for a human, `export` is a shape for a machine: `recall.export/v1`,
/// documented in API.md, evolved additively only. Money is integer micro-USD, the under-count
/// flags ride alongside every figure, and `export_id` is content-derived so an ingest ledger can
/// accept the same document twice without double-counting. Pipe it wherever the roll-up lives.
fn cmd_export(args: &[String]) -> i32 {
    let target = flag(args, "--target")
        .map(str::to_string)
        .or_else(|| std::env::var("RECALL_TARGET").ok())
        .unwrap_or_else(|| "http://127.0.0.1:8080".to_string());

    let rt = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("recall export: failed to start runtime: {e}");
            return 1;
        }
    };
    let snap = match rt.block_on(recall_proxy::scrape::fetch_url(&target)) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("recall export: {e}");
            return 1;
        }
    };
    // A 0 stamp would be a valid-LOOKING document that the paid side silently drops as stale
    // (every baseline is at or after it), so a broken clock must fail the command, not ship.
    let exported_at_unix_ms = match std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
    {
        Ok(d) => d.as_millis() as u64,
        Err(e) => {
            eprintln!("recall export: system clock is before the Unix epoch ({e}) — refusing to emit a document that would be discarded as stale");
            return 1;
        }
    };
    let doc = recall_proxy::export::build(&snap, &target, exported_at_unix_ms);
    match serde_json::to_string_pretty(&doc) {
        Ok(j) => {
            println!("{j}");
            0
        }
        Err(e) => {
            eprintln!("recall export: cannot render json: {e}");
            1
        }
    }
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
    // The operator's price book (`[pricing.<model>]`). File-only, and deliberately non-fatal: a bad
    // rate is warned about and skipped rather than refusing to start, because a typo in an optional
    // reporting table must not take a cache proxy down. Both states are logged, so an operator never
    // has to guess why `recall_cost_saved_usd_total` is zero.
    let (pricing, rejected_rates) = recall_proxy::pricing::PriceBook::load(&file.pricing);
    for key in &rejected_rates {
        tracing::warn!(
            model = key.as_str(),
            "ignoring [pricing] entry: rates must be finite, non-negative USD per 1M tokens"
        );
    }
    if pricing.is_empty() {
        tracing::info!(
            "no [pricing.<model>] rates configured; recall_cost_saved_usd_total stays 0 and saved \
             tokens are reported as unpriced"
        );
    } else {
        tracing::info!(
            models = pricing.len(),
            "price book loaded; cache savings will also be reported in USD"
        );
    }

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
    // ANN index: `brute` (exact, default) or `hnsw` (approximate, sublinear at scale — PLAN.md
    // §3-OSS T3). Both are pure-Rust, no extra deps.
    let index_kind = resolve("--index", "RECALL_INDEX", file.index, "brute");
    // HNSW graph tunables (used only with --index hnsw): flag > env > file > validated default.
    // These are THE recall-vs-latency dials at scale — `recall ann-bench --m/--ef-search` measures
    // the trade for your dims/N before you commit a fleet to them.
    let hnsw_defaults = recall_index::HnswConfig::default();
    let hnsw_num = |flag_name: &str, env: &str, file_v: Option<usize>, default: usize| -> usize {
        let raw = resolve(
            flag_name,
            env,
            file_v.map(|v| v.to_string()),
            &default.to_string(),
        );
        parse_or_warn(flag_name, &raw, default)
    };
    let hnsw_m = hnsw_num("--hnsw-m", "RECALL_HNSW_M", file.hnsw_m, hnsw_defaults.m);
    let hnsw_efc = hnsw_num(
        "--hnsw-ef-construction",
        "RECALL_HNSW_EF_CONSTRUCTION",
        file.hnsw_ef_construction,
        hnsw_defaults.ef_construction,
    );
    let hnsw_efs = hnsw_num(
        "--hnsw-ef-search",
        "RECALL_HNSW_EF_SEARCH",
        file.hnsw_ef_search,
        hnsw_defaults.ef_search,
    );
    let index: Box<dyn recall_core::AnnIndex> = match index_kind.as_str() {
        "brute" => Box::new(BruteForceIndex::new()),
        "hnsw" => {
            eprintln!(
                "recall serve: HNSW index (approximate ANN; m={hnsw_m} ef_construction={hnsw_efc} \
                 ef_search={hnsw_efs} — validate recall@1 for your dims/N with `recall ann-bench`)"
            );
            Box::new(recall_index::HnswIndex::with_config(
                recall_index::HnswConfig::with_params(hnsw_m, hnsw_efc, hnsw_efs),
            ))
        }
        other => {
            eprintln!("recall serve: unknown --index '{other}' (expected: brute | hnsw)");
            return 2;
        }
    };
    // §5.3 second-neighbour ambiguity gate (0 = off, the default): near-tied top-2 neighbours with
    // different answers are served as misses instead of a coin-flip hit.
    let ambiguity_eps: f32 = parse_or_warn(
        "--ambiguity-eps",
        &resolve(
            "--ambiguity-eps",
            "RECALL_AMBIGUITY_EPS",
            file.ambiguity_eps.map(|v| v.to_string()),
            "0",
        ),
        0.0,
    );
    if ambiguity_eps > 0.0 {
        tracing::info!(
            eps = ambiguity_eps,
            "ambiguity gate armed — near-tied lookalikes with different answers will miss"
        );
    }
    // Savings-ledger persistence. Off by default: `/metrics` and `recall.export/v1` both document
    // the process as the boundary its cumulative counters reset at, so counters that outlive a
    // restart are a change of meaning for every consumer and are never switched on unasked.
    let savings_flush_secs: u64 = parse_or_warn(
        "--savings-flush-secs",
        &resolve(
            "--savings-flush-secs",
            "RECALL_SAVINGS_FLUSH_SECS",
            file.savings_flush_secs.map(|v| v.to_string()),
            "0",
        ),
        0,
    );

    let embedder = build_embedder(model.as_deref());
    // Fail fast if an explicitly requested backend can't be honored, rather than silently
    // downgrading the durability/behavior the operator asked for.
    let built = match build_cache(embedder, index, &store_kind, &db_path, policy) {
        Ok(built) => built,
        Err(e) => {
            eprintln!("recall serve: {e}");
            return 2;
        }
    };
    // The durable handle (if any) is taken before the cache is boxed, so the savings ledger can be
    // persisted through the same database the entries live in.
    #[cfg(feature = "store-redb")]
    let ledger_store = built.ledger;
    let store_facts = built.facts;
    let cache = Arc::new(built.cache.with_ambiguity_eps(ambiguity_eps));
    let config = Config {
        listen: listen.clone(),
        upstream_base: upstream.clone(),
        upstream_api_key,
        base_namespace,
        max_temperature,
        anthropic_upstream_base: anthropic_upstream,
        anthropic_version,
        anthropic_api_key,
        pricing,
        store: store_facts,
        ..Config::default()
    };
    let state = ProxyState::new(cache, config);

    // Savings-ledger persistence (durable store only, opt-in). Restore first — so the counters are
    // already whole before the listener accepts anything — then keep flushing them. The handle and
    // epoch are kept for the shutdown flush after `serve` returns.
    #[cfg(feature = "store-redb")]
    let mut ledger_at_shutdown = None;
    #[cfg(feature = "store-redb")]
    if savings_flush_secs > 0 {
        match ledger_store {
            Some(store) => {
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0);
                // A restored epoch travels with the ledger; a fresh one starts now.
                let since = restore_ledger(&store, &state).unwrap_or(now);
                state
                    .ledger_since_unix
                    .store(since, std::sync::atomic::Ordering::Relaxed);
                // Write immediately, so a node that restarts before the first tick still records
                // its epoch (and an operator can see the ledger is live without waiting).
                if let Err(e) = flush_ledger(&store, &state, since) {
                    tracing::warn!(error = %e, "first savings ledger flush failed; will retry");
                }
                let flusher =
                    spawn_ledger_flusher(state.clone(), store.clone(), since, savings_flush_secs);
                ledger_at_shutdown = Some((store, since, flusher));
                tracing::info!(
                    every_secs = savings_flush_secs,
                    since_unix = since,
                    "savings ledger persistence active — counters now outlive this process"
                );
            }
            None => tracing::warn!(
                "--savings-flush-secs needs a durable store; ignoring it (counters reset with this process)"
            ),
        }
    }

    // Same knob on a build without the durable store: say so rather than accepting it silently,
    // since an operator who asked for a persistent figure would otherwise get a resetting one and
    // no indication why.
    #[cfg(not(feature = "store-redb"))]
    if savings_flush_secs > 0 {
        tracing::warn!(
            "--savings-flush-secs needs a `--features store-redb` build; ignoring it \
             (counters reset with this process)"
        );
    }

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
    let code = match rt.block_on(async {
        let state = state.clone();
        recall_proxy::serve(state).await
    }) {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("recall serve: server error: {e}");
            1
        }
    };
    // Shutdown flush: `serve` returning is the one moment we know no more traffic is coming, so
    // the last interval's counters are written rather than dropped. Without this a planned deploy
    // loses up to `--savings-flush-secs` of savings — indistinguishable from a crash, for a figure
    // whose whole purpose is to be quotable across deploys.
    #[cfg(feature = "store-redb")]
    if let Some((store, since, flusher)) = ledger_at_shutdown {
        // Order matters: the periodic thread must be finished before this write, or its in-flight
        // (older) snapshot could land after it and become the durable one.
        flusher.stop_and_join();
        match flush_ledger(&store, &state, since) {
            Ok(hits) => tracing::info!(hits, "savings ledger flushed on shutdown"),
            Err(e) => tracing::warn!(error = %e, "savings ledger shutdown flush failed"),
        }
    }
    code
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
/// A store opened at start-up, with what it recovered.
///
/// `build_cache` used to discard the rehydration numbers into a log line, which meant "did my cache
/// survive the restart" could only be answered by whoever happened to be watching stderr at the
/// time. They come back with the cache now, so `/metrics` can report them.
/// Restore a persisted savings ledger into a freshly built node, returning the ledger's epoch (the
/// value `recall_ledger_since_time_seconds` reports) — or `None` when there is nothing usable to
/// restore, in which case the node starts a fresh ledger at `now`.
///
/// Every failure mode lands on "start fresh, with a warning": a first run on this volume, a
/// snapshot written by a newer build, a corrupt blob, an unreadable database. Savings accounting
/// must never be the reason a cache refuses to serve.
#[cfg(feature = "store-redb")]
fn restore_ledger(
    store: &recall_store::RedbStore,
    state: &std::sync::Arc<ProxyState>,
) -> Option<u64> {
    let blob = match store.load_savings_blob() {
        Ok(Some(b)) => b,
        Ok(None) => return None, // first run on this volume
        Err(e) => {
            tracing::warn!(error = %e, "savings ledger unreadable; starting a fresh ledger");
            return None;
        }
    };
    let snap: recall_proxy::ledger::LedgerSnapshot = match serde_json::from_slice(&blob) {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!(error = %e, "savings ledger snapshot is corrupt; starting a fresh ledger");
            return None;
        }
    };
    if !recall_proxy::ledger::restore(&snap, &state.metrics, &state.namespaces) {
        tracing::warn!(
            version = snap.version,
            supported = recall_proxy::ledger::VERSION,
            "savings ledger snapshot is a version this build does not know; starting a fresh ledger"
        );
        return None;
    }
    tracing::info!(
        hits = snap.hits,
        cost_saved_micros = snap.cost_saved_micros,
        since_unix = snap.since_unix,
        "savings ledger restored — the $ saved figure survives the restart"
    );
    Some(snap.since_unix)
}

/// Persist the ledger once. Shared by the periodic flusher and the flush-on-shutdown path, so both
/// write exactly the same bytes.
#[cfg(feature = "store-redb")]
fn flush_ledger(
    store: &recall_store::RedbStore,
    state: &std::sync::Arc<ProxyState>,
    since_unix: u64,
) -> Result<u64, String> {
    let snap = recall_proxy::ledger::snapshot(&state.metrics, &state.namespaces, since_unix);
    let bytes = serde_json::to_vec(&snap).map_err(|e| e.to_string())?;
    store.save_savings_blob(&bytes).map_err(|e| e.to_string())?;
    Ok(snap.hits)
}

/// A running ledger flusher: drop or signal the sender to stop it, then join the handle to know it
/// has finished writing.
#[cfg(feature = "store-redb")]
struct LedgerFlusher {
    stop: std::sync::mpsc::Sender<()>,
    handle: std::thread::JoinHandle<()>,
}

#[cfg(feature = "store-redb")]
impl LedgerFlusher {
    /// Stop the flusher and wait for it to finish. **Must** be called before the shutdown flush:
    /// otherwise the thread can snapshot older counters, the shutdown flush can commit newer ones,
    /// and the thread's late write then makes the *older* snapshot the durable one — silently
    /// undoing the very accounting the shutdown flush exists to save.
    fn stop_and_join(self) {
        // Either arm ends the wait: an explicit signal, or the sender dropping. Both mean "stop".
        let _ = self.stop.send(());
        if self.handle.join().is_err() {
            tracing::warn!("savings ledger flusher panicked; its last write may not have landed");
        }
    }
}

/// Persist the ledger every `every_secs`, until told to stop.
///
/// A dedicated OS thread rather than a Tokio task: persistence is the one thing that must still
/// happen when the reactor is saturated, and redb's write transaction is blocking anyway. A failed
/// flush warns and retries on the next tick — the in-memory counters are still correct, so a
/// transient write error costs at most one interval of durability, never the running figure.
///
/// The wait is a channel timeout rather than a sleep so shutdown is prompt: a `--savings-flush-secs
/// 300` node still exits immediately instead of blocking five minutes on a sleeping thread.
#[cfg(feature = "store-redb")]
fn spawn_ledger_flusher(
    state: std::sync::Arc<ProxyState>,
    store: recall_store::RedbStore,
    since_unix: u64,
    every_secs: u64,
) -> LedgerFlusher {
    let (stop, rx) = std::sync::mpsc::channel::<()>();
    let handle = std::thread::spawn(move || {
        let interval = std::time::Duration::from_secs(every_secs);
        // Timing out means "another interval elapsed, flush". Anything else — a signal, or the
        // sender dropped — means shutdown, and the caller's flush supersedes ours.
        while matches!(
            rx.recv_timeout(interval),
            Err(std::sync::mpsc::RecvTimeoutError::Timeout)
        ) {
            match flush_ledger(&store, &state, since_unix) {
                Ok(hits) => tracing::debug!(hits, "savings ledger flushed"),
                Err(e) => tracing::warn!(error = %e, "savings ledger flush failed; will retry"),
            }
        }
    });
    LedgerFlusher { stop, handle }
}

struct BuiltCache {
    cache: recall_core::DynCache,
    facts: recall_proxy::config::StoreFacts,
    /// A second handle on the durable store, kept OUTSIDE the boxed cache so the binary can also
    /// persist the savings ledger through it. `None` for the in-memory store — and always `None`
    /// on a build without the feature, which is what makes the ledger a redb-only capability.
    #[cfg(feature = "store-redb")]
    ledger: Option<recall_store::RedbStore>,
}

fn build_cache(
    embedder: Box<dyn Embedder>,
    index: Box<dyn recall_core::AnnIndex>,
    store_kind: &str,
    db_path: &str,
    policy: Box<dyn recall_core::ThresholdPolicy>,
) -> Result<BuiltCache, String> {
    match store_kind {
        "memory" => {
            let cache = recall_proxy::boxed_cache_full_with_index(
                embedder,
                index,
                Box::new(MemKv::new()),
                policy,
            );
            // An ephemeral store starts empty by design, not by loss — say so once, so a zero
            // hit-rate in the first minutes reads as expected rather than as a fault, and so the
            // cost of refilling after each restart is a decision the operator makes knowingly.
            tracing::info!(
                "in-memory store — the cache starts empty and is not kept across a restart \
                 (use --store redb --db-path … to keep it)"
            );
            Ok(BuiltCache {
                cache,
                facts: recall_proxy::config::StoreFacts::default(),
                #[cfg(feature = "store-redb")]
                ledger: None,
            })
        }
        "redb" => {
            #[cfg(feature = "store-redb")]
            {
                let store = recall_store::RedbStore::open(db_path)
                    .map_err(|e| format!("--store redb: open {db_path} failed: {e}"))?;
                // A cheap Arc clone of the same database: one handle lives inside the cache, the
                // other carries the savings ledger.
                let ledger = store.clone();
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
                let rehydrated = match cache.rehydrate() {
                    Ok(n) => {
                        tracing::info!(
                            db_path,
                            rehydrated = n,
                            persisted,
                            "durable redb store — rehydrated cached lookups across restart"
                        );
                        n
                    }
                    Err(e) => {
                        tracing::warn!(
                            db_path,
                            error = %e,
                            "durable redb store — rehydration failed; serving with a cold index (KV blobs intact)"
                        );
                        0
                    }
                };
                Ok(BuiltCache {
                    cache,
                    facts: recall_proxy::config::StoreFacts {
                        backend: "redb",
                        durable: true,
                        persisted_at_start: persisted as u64,
                        rehydrated: rehydrated as u64,
                    },
                    ledger: Some(ledger),
                })
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

/// Parse a resolved config value, saying so when it cannot rather than falling back in silence.
///
/// A bare `.parse().unwrap_or(default)` reads to the operator as "you never asked for it": a
/// typo'd `--ambiguity-eps 0.O5` leaves a false-hit guard disarmed and `--savings-flush-secs 60s`
/// leaves the savings figure resetting every deploy, both without a word anywhere. The default
/// still applies — a malformed knob must never stop the proxy serving — but it stops being
/// invisible.
fn parse_or_warn<T: std::str::FromStr>(setting: &str, raw: &str, default: T) -> T {
    raw.parse().unwrap_or_else(|_| {
        tracing::warn!(setting, value = raw, "unparseable value; using the default");
        default
    })
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
    // The graph is built once and the gate is baked into the cache at construction, so these are
    // startup-only too. Without them here, `--watch-config` would accept an edit in silence and
    // leave the operator believing a live change took effect.
    if new.hnsw_m != startup.hnsw_m
        || new.hnsw_ef_construction != startup.hnsw_ef_construction
        || new.hnsw_ef_search != startup.hnsw_ef_search
    {
        changed.push("hnsw_*");
    }
    if new.ambiguity_eps != startup.ambiguity_eps {
        changed.push("ambiguity_eps");
    }
    if new.savings_flush_secs != startup.savings_flush_secs {
        changed.push("savings_flush_secs");
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

    /// The `[pricing.<model>]` TOML shape, end to end: parsed off a real config snippet (not a
    /// hand-built map), so a rename or a shape change in `FileConfig` fails here rather than
    /// silently pricing nothing in production.
    #[test]
    fn pricing_table_parses_from_toml_and_loads() {
        let toml = r#"
tau = 0.9

[pricing."gpt-4o"]
input = 2.50
output = 10.00

[pricing."gpt-4o-mini*"]
input = 0.15
output = 0.60
"#;
        let fc: FileConfig = toml::from_str(toml).expect("config parses");
        assert_eq!(fc.pricing.len(), 2);
        let (book, rejected) = recall_proxy::pricing::PriceBook::load(&fc.pricing);
        assert!(rejected.is_empty(), "{rejected:?}");
        // 1M input @ $2.50 + 1M output @ $10.00 = $12.50 → 12_500_000 micro-USD.
        assert_eq!(
            book.cost_micros("gpt-4o", 1_000_000, 1_000_000),
            Some(12_500_000)
        );
        // The prefix rule reaches a dated snapshot the exact key would not.
        assert_eq!(
            book.cost_micros("gpt-4o-mini-2024-07-18", 1_000_000, 0),
            Some(150_000)
        );
        // An unrelated model stays unpriced rather than being guessed at.
        assert_eq!(book.cost_micros("claude-sonnet-4", 1_000, 1_000), None);
    }

    /// A config with no `[pricing]` section at all must still parse — the table is optional and an
    /// empty price book is a supported state, not a startup failure.
    #[test]
    fn config_without_pricing_still_parses_to_an_empty_book() {
        let fc: FileConfig = toml::from_str("tau = 0.9\n").expect("config parses");
        assert!(recall_proxy::pricing::PriceBook::load(&fc.pricing)
            .0
            .is_empty());
    }

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
    fn startup_only_keys_are_flagged_as_needing_a_restart() {
        // The graph knobs, the gate and the flush interval are all read once at startup. If they
        // are missing from this list, `--watch-config` accepts an edit to them in silence and the
        // operator is left believing a live change took effect.
        let startup = FileConfig::default();
        let edited = FileConfig {
            hnsw_ef_search: Some(512),
            ambiguity_eps: Some(0.02),
            savings_flush_secs: Some(30),
            ..Default::default()
        };
        let ignored = ignored_nonpolicy_changes(&edited, &startup);
        for key in ["hnsw_*", "ambiguity_eps", "savings_flush_secs"] {
            assert!(ignored.contains(&key), "{key} must warn; got {ignored:?}");
        }
    }

    #[test]
    fn an_unparseable_value_falls_back_without_taking_the_proxy_down() {
        // The default still applies — a typo must never stop the proxy serving — and the operator
        // gets a warning rather than silence (asserted by the caller's log, not here).
        assert_eq!(
            super::parse_or_warn("--ambiguity-eps", "0.O5", 0.0_f32),
            0.0
        );
        assert_eq!(
            super::parse_or_warn("--savings-flush-secs", "60s", 0_u64),
            0
        );
        // A good value still parses.
        assert_eq!(super::parse_or_warn("--hnsw-m", "64", 32_usize), 64);
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
