//! `AdaptiveThreshold` — a feedback-driven [`ThresholdPolicy`] that targets an operator-chosen
//! **false-hit rate** instead of a fixed cosine cutoff (PLAN.md §5). A drop-in behind
//! the existing seam: the cache calls `decide` on the hot path and `observe` off it, exactly as for
//! `StaticThreshold`.
//!
//! The core idea (PLAN.md §5.4, "FHR is authoritative"): track the distribution of similarity scores
//! that were *confirmed wrong* (a served hit later judged a false hit). The threshold sits a
//! `z(target_fhr)`-sigma margin above that distribution's mean, so only a fraction ≈ `target_fhr` of
//! future false-hit-like scores can clear it. With no feedback yet, τ rests at a per-embedder
//! cold-start prior and the policy behaves like a static cutoff — so it is safe to enable before a
//! ground-truth feedback source exists (PLAN.md §5.5).
//!
//! Scope of this increment: a single adaptive band per namespace (per-region bands and the richer
//! per-query gates of §5.3 are a later seam extension), in-memory state, learning exercised by
//! direct `observe` calls. Wiring a production feedback source (a verifier) is the documented next
//! step (§5.4).

use std::collections::HashMap;
use std::sync::RwLock;

use recall_core::{Outcome, ThresholdPolicy, Verdict};

use crate::moments::EwMoments;
use crate::stats::{ema, inv_upper_normal};

/// Tunables for the adaptive threshold. Construct with [`AdaptiveConfig::new`] so the safety clamps
/// are derived from the cold-start prior.
#[derive(Clone, Debug)]
pub struct AdaptiveConfig {
    /// The operator's SLO: the fraction of served hits allowed to be wrong. An auditable target
    /// ("≤2% wrong cache hits") rather than a magic cosine number. Clamped to `[1e-4, 0.5]`.
    pub target_false_hit_rate: f64,
    /// τ before any feedback — a per-embedder prior (PLAN.md §5.5). Clamped to `[0, 1]`.
    pub cold_start_tau: f32,
    /// Hard floor / ceiling on the learned τ so even adversarial feedback can't drive it to nonsense.
    pub tau_min: f32,
    pub tau_max: f32,
    /// Confirmed-wrong samples required before the Gaussian retune engages; below it, τ holds near
    /// the cold-start prior.
    pub min_samples: u64,
    /// EMA rate at which τ moves toward the freshly-computed FHR target.
    pub learn_rate: f64,
    /// Decay rate of the per-distribution moments.
    pub decay: f64,
}

impl AdaptiveConfig {
    /// Build a config from the two knobs an operator actually sets — the cold-start prior and the
    /// target false-hit rate — deriving the safety clamps (`tau_min = τ₀ − 0.1`, `tau_max = 0.995`,
    /// PLAN.md §5.5) and sensible learning defaults.
    pub fn new(cold_start_tau: f32, target_false_hit_rate: f64) -> Self {
        let tau0 = cold_start_tau.clamp(0.0, 1.0);
        Self {
            target_false_hit_rate: target_false_hit_rate.clamp(1e-4, 0.5),
            cold_start_tau: tau0,
            tau_min: (tau0 - 0.1).max(0.0),
            tau_max: 0.995,
            min_samples: 30,
            learn_rate: 0.1,
            decay: 0.05,
        }
    }
}

impl Default for AdaptiveConfig {
    /// A reasonable static-embedder prior: τ₀ = 0.85, target FHR = 2%.
    fn default() -> Self {
        Self::new(0.85, 0.02)
    }
}

/// One namespace's learned state.
struct NsState {
    tau: f32,
    /// Similarities that proved WRONG (false hits) — drives the FHR-targeting τ.
    neg: EwMoments,
    /// Similarities that proved RIGHT — used only as a recall guard so τ isn't raised above the
    /// proven-good cluster.
    pos: EwMoments,
}

/// A `ThresholdPolicy` whose per-namespace τ adapts toward a target false-hit rate as feedback
/// arrives. Interior-mutable (`decide` is `&self` on the hot path; `observe` is `&self` off it), so
/// it drops into the cache exactly where `StaticThreshold` did.
pub struct AdaptiveThreshold {
    cfg: AdaptiveConfig,
    id: String,
    ns: RwLock<HashMap<String, NsState>>,
}

impl AdaptiveThreshold {
    pub fn new(cfg: AdaptiveConfig) -> Self {
        let id = format!(
            "adaptive@fhr={:.3}:tau0={:.3}",
            cfg.target_false_hit_rate, cfg.cold_start_tau
        );
        Self {
            cfg,
            id,
            ns: RwLock::new(HashMap::new()),
        }
    }

    /// The current learned τ for a namespace, or the cold-start prior if it has no feedback yet.
    /// Exposed for tests and operator introspection.
    pub fn tau(&self, ns: &str) -> f32 {
        self.ns
            .read()
            .unwrap()
            .get(ns)
            .map_or(self.cfg.cold_start_tau, |s| s.tau)
    }
}

impl ThresholdPolicy for AdaptiveThreshold {
    fn id(&self) -> &str {
        &self.id
    }

    fn decide(&self, ns: &str, top: Option<f32>) -> Verdict {
        match top {
            Some(score) if score >= self.tau(ns) => Verdict::Hit,
            _ => Verdict::Miss,
        }
    }

    fn observe(&self, ns: &str, score: f32, outcome: Outcome) {
        let mut map = self.ns.write().unwrap();
        let state = map.entry(ns.to_string()).or_insert_with(|| NsState {
            tau: self.cfg.cold_start_tau,
            neg: EwMoments::new(self.cfg.decay),
            pos: EwMoments::new(self.cfg.decay),
        });
        match outcome {
            Outcome::Wrong => state.neg.update(score as f64),
            Outcome::Agree => state.pos.update(score as f64),
        }
        retune(state, &self.cfg);
    }
}

/// Recompute τ from the confirmed-wrong distribution. FHR is authoritative (PLAN.md §5.4): place τ a
/// `z(target_fhr)`-sigma margin above the mean of similarities that proved wrong, so ≈ `target_fhr`
/// of that distribution still clears τ and the rest is rejected. Until enough wrong samples exist,
/// hold τ near the cold-start prior. A recall guard keeps τ from rising above the proven-good
/// cluster (don't start rejecting answers feedback has confirmed correct).
fn retune(state: &mut NsState, cfg: &AdaptiveConfig) {
    if state.neg.count() < cfg.min_samples {
        state.tau = ema(state.tau as f64, cfg.cold_start_tau as f64, 0.05) as f32;
        return;
    }
    let z = inv_upper_normal(cfg.target_false_hit_rate);
    let mut target = state.neg.mean() + z * state.neg.std();
    if state.pos.count() >= cfg.min_samples {
        // Recall guard: don't raise τ *into* the proven-correct cluster — capping at its mean would
        // reject roughly half the correct hits (those below the mean). Cap at a lower-bound
        // statistic (mean − 2σ) so the bulk of the positive cluster still clears τ.
        let pos_floor = state.pos.mean() - 2.0 * state.pos.std();
        target = target.min(pos_floor);
    }
    let next = ema(state.tau as f64, target, cfg.learn_rate) as f32;
    state.tau = next.clamp(cfg.tau_min, cfg.tau_max);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> AdaptiveThreshold {
        AdaptiveThreshold::new(AdaptiveConfig::new(0.85, 0.02))
    }

    #[test]
    fn cold_start_behaves_like_a_static_cutoff() {
        let p = policy();
        assert_eq!(p.tau("ns"), 0.85);
        assert!(matches!(p.decide("ns", Some(0.90)), Verdict::Hit));
        assert!(matches!(p.decide("ns", Some(0.80)), Verdict::Miss));
        assert!(matches!(p.decide("ns", None), Verdict::Miss));
    }

    #[test]
    fn false_hit_feedback_raises_tau_and_makes_the_cache_more_precise() {
        let p = policy();
        // A spread of confirmed-WRONG hits centered ~0.85 — the cache was over-serving here.
        for i in 0..120 {
            let score = 0.80 + (i % 11) as f32 * 0.01; // 0.80..=0.90, mean ≈ 0.85
            p.observe("ns", score, Outcome::Wrong);
        }
        // τ climbs a margin above the false-hit mean (precision up): a 0.85 candidate that hit at
        // cold-start now misses, while a clearly-similar 0.97 candidate still hits.
        assert!(
            p.tau("ns") > 0.88,
            "tau rose above the false-hit band, got {}",
            p.tau("ns")
        );
        assert!(p.tau("ns") <= 0.995, "respects the ceiling");
        assert!(matches!(p.decide("ns", Some(0.85)), Verdict::Miss));
        assert!(matches!(p.decide("ns", Some(0.97)), Verdict::Hit));
    }

    #[test]
    fn learning_is_per_namespace() {
        let p = policy();
        for i in 0..120 {
            p.observe("hot", 0.80 + (i % 11) as f32 * 0.01, Outcome::Wrong);
        }
        // The untouched namespace still decides at the cold-start prior.
        assert_eq!(p.tau("other"), 0.85);
        assert!(matches!(p.decide("other", Some(0.86)), Verdict::Hit));
        // While the namespace that saw false hits is now stricter.
        assert!(matches!(p.decide("hot", Some(0.86)), Verdict::Miss));
    }

    #[test]
    fn agree_feedback_does_not_raise_tau() {
        let p = policy();
        for _ in 0..120 {
            p.observe("ns", 0.95, Outcome::Agree);
        }
        // Confirmed-correct hits are not false hits — τ must not climb on them.
        assert!(
            p.tau("ns") <= 0.85 + 1e-6,
            "agree feedback keeps tau at the prior"
        );
    }

    #[test]
    fn recall_guard_does_not_cut_into_the_correct_cluster() {
        let p = policy();
        // Confirmed-WRONG hits sit high (~0.95) — the FHR target alone would push τ up near there.
        // Confirmed-CORRECT hits form a spread cluster around ~0.90 (0.86..=0.94). The recall guard
        // must keep τ below the *lower edge* of that cluster, not at its mean.
        for i in 0..80 {
            p.observe("ns", 0.93 + (i % 5) as f32 * 0.01, Outcome::Wrong); // 0.93..=0.97
            p.observe("ns", 0.86 + (i % 9) as f32 * 0.01, Outcome::Agree); // 0.86..=0.94, mean ≈ 0.90
        }
        // τ is pulled below the positive mean — the old `min(pos.mean())` guard would have parked it
        // at ≈0.90 and rejected the lower half of the correct cluster.
        assert!(
            p.tau("ns") < 0.90,
            "guard keeps tau below the positive mean, got {}",
            p.tau("ns")
        );
        // A correct hit below the positive mean still serves.
        assert!(matches!(p.decide("ns", Some(0.88)), Verdict::Hit));
    }
}
