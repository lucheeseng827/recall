//! `recall-eval` — run the adaptive-threshold policy comparison and print the hit-rate vs
//! false-hit-rate evidence (PLAN.md §5.6). Drives the real shipped policies over a reproducible,
//! controllable-density synthetic stream; the regression-gate assertions live in the library tests.
//!
//!   cargo run -p recall-eval                 # default: target FHR 2%, two-region workload
//!   cargo run -p recall-eval -- --target-fhr 0.05 --noise 0.85
//!
//! Output: a per-policy table (hit-rate, false-hit-rate, learned bands) plus the one-line headline —
//! at a fixed false-hit budget, how much more traffic the adaptive engine serves than a single τ.

use recall_calibrate::{AdaptiveConfig, AdaptiveThreshold};
use recall_core::StaticThreshold;
use recall_eval::{
    demo_regions, eval_policy, gen_stream, static_best_tau, train_policy, NoisyVerifier, Stats,
};

fn flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let target_fhr: f64 = flag(&args, "--target-fhr")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.02);
    // Cold-start prior τ₀ for the adaptive policy and the equivalent feedback-off static cutoff.
    let tau0: f32 = flag(&args, "--tau0")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.80);
    // Fraction of feedback labels the oracle gets right (1.0 = perfect; lower = NoisyVerifier).
    let p_correct: f64 = flag(&args, "--noise")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1.0);

    // Validate the rate inputs before any reporting math depends on them: a non-positive --target-fhr
    // divides to ±∞ in the "over budget by N×" line, and a --noise outside [0,1] is not a probability.
    if !(target_fhr > 0.0 && target_fhr <= 1.0) {
        eprintln!("recall-eval: --target-fhr must be in (0.0, 1.0] (got {target_fhr})");
        std::process::exit(2);
    }
    if !(0.0..=1.0).contains(&p_correct) {
        eprintln!("recall-eval: --noise must be in [0.0, 1.0] (got {p_correct})");
        std::process::exit(2);
    }

    let regions = demo_regions();
    // Disjoint train/test streams (different seeds) — nothing is measured on what it was tuned on.
    let train = gen_stream(&regions, 1);
    let test = gen_stream(&regions, 2);

    // The policies under comparison, all scored on the same test stream.
    let s_080 = eval_policy(&StaticThreshold::new(0.80), &test);
    let best_tau = static_best_tau(&train, target_fhr);
    let s_best = eval_policy(&StaticThreshold::new(best_tau), &test);
    let s_off = eval_policy(
        &AdaptiveThreshold::new(AdaptiveConfig::new(tau0, target_fhr)),
        &test,
    );

    let adaptive = AdaptiveThreshold::new(AdaptiveConfig::new(tau0, target_fhr));
    train_policy(&adaptive, &train, &mut NoisyVerifier::new(p_correct, 7));
    let s_on = eval_policy(&adaptive, &test);

    let pct = |x: f64| x * 100.0;
    let row = |name: &str, detail: &str, s: &Stats| {
        println!(
            "  {name:<26} {detail:<22} {:>7.1}% {:>9.1}%",
            pct(s.hit_rate()),
            pct(s.false_hit_rate()),
        );
    };

    println!("recall-eval — adaptive threshold vs a single static cutoff");
    println!(
        "  workload: {} regions, {} events (synthetic, controllable density; seed-reproducible)",
        regions.len(),
        test.len()
    );
    println!(
        "  correctness budget: target false-hit rate {:.1}%\n",
        pct(target_fhr)
    );
    println!(
        "  {:<26} {:<22} {:>8} {:>10}",
        "policy", "config", "hit-rate", "false-hit"
    );
    row("static@0.8 (fixed default)", "τ=0.800", &s_080);
    row("static@best", &format!("τ={best_tau:.3} (tuned)"), &s_best);
    row("adaptive (feedback off)", &format!("τ₀={tau0:.3}"), &s_off);
    row(
        "adaptive (feedback on)",
        &format!(
            "dense {:.3}/sparse {:.3}",
            adaptive.tau("dense"),
            adaptive.tau("sparse")
        ),
        &s_on,
    );

    // Headline deltas — computed, never asserted (PLAN.md §5.6). Reported only against baselines that
    // actually held the budget, so "X% more at the same FHR" is apples-to-apples.
    println!();
    if p_correct < 1.0 {
        println!(
            "  feedback oracle: {:.0}% correct (NoisyVerifier)",
            pct(p_correct)
        );
    }
    let more = |a: &Stats, b: &Stats| {
        if b.hit_rate() > 0.0 {
            (a.hit_rate() / b.hit_rate() - 1.0) * 100.0
        } else {
            f64::INFINITY
        }
    };
    // Only claim "X% more at the same budget" when static@best actually held the budget on the *test*
    // stream — it is tuned on `train`, so it can run over budget out-of-sample, which would make the
    // comparison apples-to-oranges and overstate the adaptive win.
    if s_best.false_hit_rate() <= target_fhr {
        println!(
            "  → at ≤{:.1}% false-hit, adaptive (feedback on) serves {:.0}% more from cache than \
             static@best ({:.1}% vs {:.1}% hit-rate),",
            pct(target_fhr),
            more(&s_on, &s_best),
            pct(s_on.hit_rate()),
            pct(s_best.hit_rate()),
        );
    } else {
        println!(
            "  → adaptive (feedback on) holds {:.1}% hit-rate at {:.1}% false-hit, while static@best — \
             tuned on the train stream — ran {:.1}% false-hit out-of-sample, OVER the ≤{:.1}% budget, \
             so there is no same-budget single-τ baseline to compare against,",
            pct(s_on.hit_rate()),
            pct(s_on.false_hit_rate()),
            pct(s_best.false_hit_rate()),
            pct(target_fhr),
        );
    }
    println!(
        "    while static@0.8 sits at a {:.1}% false-hit rate — over budget by {:.0}× — because no \
         single τ fits both regions.",
        pct(s_080.false_hit_rate()),
        s_080.false_hit_rate() / target_fhr,
    );
}
