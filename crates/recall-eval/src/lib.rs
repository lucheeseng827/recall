//! # recall-eval — the adaptive threshold's evidence harness (PLAN.md §5.6, §6.4)
//!
//! The adaptive-threshold claim is only credible with a **reproducible hit-rate vs false-hit-rate
//! curve**, so this crate runs the *same labeled stream* through the real shipped policies — a fixed
//! `StaticThreshold` at the commonly documented `0.8` default, a `static@best` baseline (the single global τ
//! tuned in hindsight to hold a 2% false-hit rate — a deliberately hard bar), and the
//! `AdaptiveThreshold` (feedback off, then on) — and reports, at a fixed false-hit budget, how much
//! more traffic each serves from cache.
//!
//! **The thesis it demonstrates (PLAN.md §5.1).** Similarity is not calibrated across an embedding
//! space: some regions sit at high baseline cosine (a fixed τ admits false hits there), others sit
//! lower (the same τ rejects true hits there). No single global τ is right in both. The shipped
//! `AdaptiveThreshold` learns one band *per namespace*, so this harness models the two regions as two
//! namespaces (a *dense* boilerplate region and a *sparse* technical region) — exactly the structure a
//! single cutoff cannot serve, and the per-namespace adaptive τ can.
//!
//! **Honesty of the setup.** The traffic is **synthetic with controllable density** (PLAN.md §5.6) —
//! the offset between regions is the whole point, and a hand-rolled deterministic PRNG makes every
//! number byte-reproducible. The harness drives the *real* `decide`/`observe` seam, so it measures the
//! shipped engine, not a model of it. Feedback is supplied by an offline oracle over the labeled
//! stream (optionally corrupted by a [`NoisyVerifier`] to prove graceful degradation under wrong
//! feedback). Train and test use disjoint streams (different seeds) so nothing is measured on the data
//! it was tuned on. Reported deltas (X% more) are *computed*, never asserted.

use recall_core::{Outcome, ThresholdPolicy, Verdict};

/// One region of the embedding space: a should-hit (paraphrase) cosine distribution and a should-not
/// (look-alike-but-different) distribution, each approximately Gaussian, plus how many of each to
/// draw. The gap between regions' means is what makes a single global τ structurally wrong.
#[derive(Clone, Debug)]
pub struct Region {
    /// Namespace this region's traffic lands in (the adaptive policy learns one band per namespace).
    pub ns: &'static str,
    pub pos_mean: f32,
    pub pos_std: f32,
    pub neg_mean: f32,
    pub neg_std: f32,
    pub n_pos: usize,
    pub n_neg: usize,
}

/// A single lookup event: the namespace, the top-neighbour cosine the index returned, and the ground
/// truth — whether that neighbour is actually the right cached answer (`should_hit`) or a look-alike
/// (`!should_hit`). The policy only ever sees `sim`; `should_hit` is the oracle the harness scores
/// against (and the source of feedback in the feedback-on runs).
#[derive(Clone, Copy, Debug)]
pub struct Event {
    pub ns: &'static str,
    pub sim: f32,
    pub should_hit: bool,
}

/// Outcome of running one policy over a labeled stream.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Stats {
    /// should-hit events the policy served (`decide == Hit` on a true paraphrase).
    pub true_hits: u64,
    /// should-not events the policy wrongly served (`decide == Hit` on a look-alike) — the harm.
    pub false_hits: u64,
    /// total should-hit events in the stream (the denominator for hit-rate).
    pub total_pos: u64,
    /// total should-not events in the stream.
    pub total_neg: u64,
}

impl Stats {
    /// Fraction of true paraphrases served from cache — the cache's *value*.
    pub fn hit_rate(&self) -> f64 {
        if self.total_pos == 0 {
            0.0
        } else {
            self.true_hits as f64 / self.total_pos as f64
        }
    }

    /// Fraction of *served* hits that were wrong — the correctness budget the operator sets (PLAN.md
    /// §5: a target false-hit rate, not a magic cosine number). Zero when nothing was served.
    pub fn false_hit_rate(&self) -> f64 {
        let served = self.true_hits + self.false_hits;
        if served == 0 {
            0.0
        } else {
            self.false_hits as f64 / served as f64
        }
    }
}

/// A small, fast, fully-deterministic PRNG (xorshift64*) — no `rand` dependency, so the eval is
/// byte-reproducible from a seed (PLAN.md §5.6 "byte-reproducible").
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        // Never 0 (xorshift's fixed point); fold the seed so small seeds still mix well.
        Self(seed ^ 0x9E37_79B9_7F4A_7C15 | 1)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in [0, 1).
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Approximately N(0, 1) via the Irwin–Hall sum of 12 uniforms minus 6 — deps-free and plenty
    /// accurate for generating region distributions.
    fn normal(&mut self) -> f64 {
        let sum: f64 = (0..12).map(|_| self.unit()).sum();
        sum - 6.0
    }

    /// A cosine sample from N(mean, std), clamped to the valid similarity range [-1, 1].
    fn cosine_sample(&mut self, mean: f32, std: f32) -> f32 {
        let v = mean as f64 + std as f64 * self.normal();
        v.clamp(-1.0, 1.0) as f32
    }
}

/// Generate a labeled stream from the regions, deterministically from `seed`. Events are interleaved
/// across regions and shuffled so an online policy sees a realistic mix rather than one region then
/// the next.
pub fn gen_stream(regions: &[Region], seed: u64) -> Vec<Event> {
    let mut rng = Rng::new(seed);
    let mut events = Vec::new();
    for r in regions {
        for _ in 0..r.n_pos {
            events.push(Event {
                ns: r.ns,
                sim: rng.cosine_sample(r.pos_mean, r.pos_std),
                should_hit: true,
            });
        }
        for _ in 0..r.n_neg {
            events.push(Event {
                ns: r.ns,
                sim: rng.cosine_sample(r.neg_mean, r.neg_std),
                should_hit: false,
            });
        }
    }
    // Fisher–Yates shuffle with the same PRNG so the order is reproducible too.
    for i in (1..events.len()).rev() {
        let j = (rng.next_u64() % (i as u64 + 1)) as usize;
        events.swap(i, j);
    }
    events
}

/// The reference two-region workload: a *dense* region where both the should-hit and should-not
/// cosine clusters sit high (a low τ admits false hits), and a *sparse* region where both sit lower
/// (a high τ rejects true hits). Within each region the clusters are separable; across regions they
/// are offset — the structure a single global τ cannot serve but a per-namespace τ can. Equal counts
/// per region so a "single τ can only win one region" result reads as a clean ~50% ceiling.
pub fn demo_regions() -> Vec<Region> {
    vec![
        Region {
            ns: "dense",
            pos_mean: 0.93,
            pos_std: 0.015,
            neg_mean: 0.84,
            neg_std: 0.015,
            n_pos: 1500,
            n_neg: 1500,
        },
        Region {
            ns: "sparse",
            pos_mean: 0.76,
            pos_std: 0.02,
            neg_mean: 0.64,
            neg_std: 0.02,
            n_pos: 1500,
            n_neg: 1500,
        },
    ]
}

/// Score a policy over a labeled stream using only `decide` (no learning) — used for every policy at
/// measurement time, and for the static baselines (which never learn).
pub fn eval_policy(policy: &dyn ThresholdPolicy, events: &[Event]) -> Stats {
    let mut s = Stats::default();
    for e in events {
        if e.should_hit {
            s.total_pos += 1;
        } else {
            s.total_neg += 1;
        }
        if matches!(policy.decide(e.ns, Some(e.sim)), Verdict::Hit) {
            if e.should_hit {
                s.true_hits += 1;
            } else {
                s.false_hits += 1;
            }
        }
    }
    s
}

/// A feedback source that returns the *correct* outcome with probability `p_correct` and flips it
/// otherwise — models the absence of a perfect production oracle (PLAN.md §5.4/§5.6 `NoisyVerifier`).
/// `p_correct = 1.0` is a perfect offline oracle.
pub struct NoisyVerifier {
    pub p_correct: f64,
    rng: Rng,
}

impl NoisyVerifier {
    pub fn new(p_correct: f64, seed: u64) -> Self {
        Self {
            p_correct: p_correct.clamp(0.0, 1.0),
            rng: Rng::new(seed),
        }
    }

    /// The (possibly corrupted) verdict for an event: should_hit → Agree, else Wrong, flipped with
    /// probability `1 - p_correct`.
    fn outcome_for(&mut self, should_hit: bool) -> Outcome {
        let truth = if should_hit {
            Outcome::Agree
        } else {
            Outcome::Wrong
        };
        if self.rng.unit() < self.p_correct {
            truth
        } else {
            match truth {
                Outcome::Agree => Outcome::Wrong,
                Outcome::Wrong => Outcome::Agree,
            }
        }
    }
}

/// Train an adaptive policy by replaying the labeled stream through `observe`, with feedback supplied
/// by `verifier`. This is the offline-oracle / replay feedback mode (PLAN.md §5.6): the harness has a
/// label for every pair, so it can train the per-namespace bands from the full stream. Static
/// policies ignore `observe`, so calling this on one is a harmless no-op.
pub fn train_policy(policy: &dyn ThresholdPolicy, events: &[Event], verifier: &mut NoisyVerifier) {
    for e in events {
        let outcome = verifier.outcome_for(e.should_hit);
        policy.observe(e.ns, e.sim, outcome);
    }
}

/// Find the single global τ that maximizes hit-rate while holding the **global** false-hit rate at or
/// below `target_fhr` on `events` — the `static@best` baseline (the best a single cutoff can do, tuned
/// in hindsight on this exact stream, which is a deliberately strong bar to beat). Sweeps τ over a
/// fine grid in [0.50, 0.99]; returns the chosen τ.
pub fn static_best_tau(events: &[Event], target_fhr: f64) -> f32 {
    let mut best_tau = 0.99_f32;
    let mut best_hits = -1.0_f64;
    // 0.50..=0.99 in 0.0025 steps — fine enough to find the knee where FHR crosses the budget.
    for step in 0..=196 {
        let tau = 0.50 + step as f32 * 0.0025;
        let policy = recall_core::StaticThreshold::new(tau);
        let s = eval_policy(&policy, events);
        if s.false_hit_rate() <= target_fhr && s.hit_rate() > best_hits {
            best_hits = s.hit_rate();
            best_tau = tau;
        }
    }
    best_tau
}

#[cfg(test)]
mod tests {
    use super::*;
    use recall_calibrate::{AdaptiveConfig, AdaptiveThreshold};
    use recall_core::StaticThreshold;
    use std::time::Instant;

    use super::demo_regions as regions;

    const TARGET_FHR: f64 = 0.02;

    fn adaptive() -> AdaptiveThreshold {
        // Cold-start τ₀ = 0.80 so feedback-off behaves like the common static default; target 2% FHR.
        AdaptiveThreshold::new(AdaptiveConfig::new(0.80, TARGET_FHR))
    }

    #[test]
    fn adaptive_feedback_on_beats_static_best_at_the_same_fhr() {
        let train = gen_stream(&regions(), 1);
        let test = gen_stream(&regions(), 2);

        // static@best: the single global τ tuned on train to hold 2% FHR — measured on test.
        let best_tau = static_best_tau(&train, TARGET_FHR);
        let static_best = eval_policy(&StaticThreshold::new(best_tau), &test);

        // adaptive, trained with a perfect offline oracle on train — measured on test.
        let policy = adaptive();
        train_policy(&policy, &train, &mut NoisyVerifier::new(1.0, 7));
        let adaptive_on = eval_policy(&policy, &test);

        // The headline: at (about) the same correctness budget, the per-namespace policy serves
        // materially more from cache because it holds each region's FHR with its own τ.
        assert!(
            adaptive_on.false_hit_rate() <= 0.04,
            "adaptive holds ~the FHR budget (got {:.3})",
            adaptive_on.false_hit_rate()
        );
        assert!(
            adaptive_on.hit_rate() > static_best.hit_rate() * 1.3,
            "adaptive feedback-on hit-rate {:.3} should clear static@best {:.3} by a wide margin \
             (per-namespace τ captures the sparse region a single τ must reject)",
            adaptive_on.hit_rate(),
            static_best.hit_rate()
        );
        // And it learned *different* bands per region — the actual mechanism.
        assert!(
            policy.tau("dense") > policy.tau("sparse") + 0.05,
            "dense band {:.3} should sit well above the sparse band {:.3}",
            policy.tau("dense"),
            policy.tau("sparse")
        );
    }

    #[test]
    fn static_at_0_8_blows_the_false_hit_budget() {
        // The commonly documented 0.8 default is wrong here: it sits below the dense negatives (~0.84), so
        // it serves them as hits — a false-hit rate far over any sane budget.
        let test = gen_stream(&regions(), 2);
        let s = eval_policy(&StaticThreshold::new(0.8), &test);
        assert!(
            s.false_hit_rate() > 0.2,
            "static@0.8 over-serves the dense region (FHR {:.3})",
            s.false_hit_rate()
        );
    }

    #[test]
    fn feedback_off_matches_the_cold_start_static_cutoff() {
        // With no feedback the adaptive policy rests at its cold-start τ₀, so it must behave exactly
        // like the equivalent static cutoff — the "safe to enable before a feedback source exists"
        // property (PLAN.md §5.5).
        let test = gen_stream(&regions(), 2);
        let off = eval_policy(&adaptive(), &test); // never trained
        let stat = eval_policy(&StaticThreshold::new(0.80), &test);
        assert_eq!(off.true_hits, stat.true_hits, "feedback-off == static@τ₀");
        assert_eq!(off.false_hits, stat.false_hits);
    }

    #[test]
    fn adaptive_degrades_gracefully_under_noisy_feedback() {
        // Even with 15% of feedback labels flipped, per-namespace learning should still beat the best
        // single global τ — the engine must not require a perfect oracle (PLAN.md §5.6).
        let train = gen_stream(&regions(), 1);
        let test = gen_stream(&regions(), 2);
        let static_best = eval_policy(
            &StaticThreshold::new(static_best_tau(&train, TARGET_FHR)),
            &test,
        );

        let policy = adaptive();
        train_policy(&policy, &train, &mut NoisyVerifier::new(0.85, 11));
        let noisy = eval_policy(&policy, &test);
        assert!(
            noisy.hit_rate() > static_best.hit_rate(),
            "noisy-feedback adaptive {:.3} still beats static@best {:.3}",
            noisy.hit_rate(),
            static_best.hit_rate()
        );
    }

    #[test]
    fn decide_is_well_under_50us() {
        // The adaptive policy must cost no latency: `decide` is read-only on a per-namespace snapshot (PLAN.md
        // §5.3 target < 50 µs). Time a long run of decisions and assert the per-call mean clears it.
        let policy = adaptive();
        train_policy(
            &policy,
            &gen_stream(&regions(), 1),
            &mut NoisyVerifier::new(1.0, 3),
        );
        let n = 200_000;
        let start = Instant::now();
        let mut sink = 0u64;
        for i in 0..n {
            let ns = if i % 2 == 0 { "dense" } else { "sparse" };
            if matches!(policy.decide(ns, Some(0.9)), Verdict::Hit) {
                sink += 1;
            }
        }
        let per_call = start.elapsed().as_nanos() as f64 / n as f64;
        assert!(sink <= n, "guard against the loop being optimized away");
        assert!(
            per_call < 50_000.0,
            "decide averaged {per_call:.0} ns/call, must be < 50 µs"
        );
    }
}
