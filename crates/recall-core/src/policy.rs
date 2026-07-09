//! The threshold seam and its MVP implementation. `StaticThreshold` is a single clamped cosine
//! cutoff — the common fixed-threshold primitive whose limitations the adaptive engine (PLAN.md
//! §5) exists to fix. `observe()` defaults to a no-op so the adaptive per-region policy is a drop-in
//! later with zero facade changes.

use crate::types::{Outcome, Verdict};

/// Object-safe hit-vs-miss seam. `decide` is on the hot path (target <50µs, lock-light); `observe`
/// is off it (PLAN.md §2.2, §5.3).
pub trait ThresholdPolicy: Send + Sync {
    /// Includes the algorithm + config hash + the embedder id, so a calibration snapshot learned
    /// under one embedder is never reused under another.
    fn id(&self) -> &str;
    fn decide(&self, ns: &str, top: Option<f32>) -> Verdict;
    fn observe(&self, _ns: &str, _score: f32, _outcome: Outcome) {}
}

/// A single global cosine cutoff. Correct primitive for the MVP; structurally wrong across a real
/// embedding space, which is the §5 thesis.
pub struct StaticThreshold {
    tau: f32,
    id: String,
}

impl StaticThreshold {
    pub fn new(tau: f32) -> Self {
        // `f32::clamp` passes NaN through unchanged, and `score >= NaN` is always false — a NaN τ
        // would silently force every lookup to miss. Fail it closed to 1.0 (exact-only) so a
        // misconfigured threshold stays conservative instead of silently disabling the cache.
        let tau = if tau.is_nan() {
            1.0
        } else {
            tau.clamp(0.0, 1.0)
        };
        Self {
            id: format!("static@{tau:.3}"),
            tau,
        }
    }

    pub fn tau(&self) -> f32 {
        self.tau
    }
}

impl ThresholdPolicy for StaticThreshold {
    fn id(&self) -> &str {
        &self.id
    }

    fn decide(&self, _ns: &str, top: Option<f32>) -> Verdict {
        match top {
            Some(score) if score >= self.tau => Verdict::Hit,
            _ => Verdict::Miss,
        }
    }
    // observe: intentionally the default no-op — a static threshold does not learn.
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nan_tau_fails_closed_not_open() {
        // A NaN τ must not silently turn every lookup into a miss; it clamps to exact-only (1.0).
        let p = StaticThreshold::new(f32::NAN);
        assert_eq!(p.tau(), 1.0);
        assert!(matches!(p.decide("ns", Some(0.99)), Verdict::Miss));
        assert!(matches!(p.decide("ns", Some(1.0)), Verdict::Hit));
    }

    #[test]
    fn tau_is_clamped_into_unit_range() {
        assert_eq!(StaticThreshold::new(-1.0).tau(), 0.0);
        assert_eq!(StaticThreshold::new(2.0).tau(), 1.0);
    }
}
