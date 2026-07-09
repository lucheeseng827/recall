//! Exponentially-weighted running mean and variance — the per-distribution state the adaptive
//! threshold keeps for the confirmed-wrong (and confirmed-right) similarity scores. Decayed rather
//! than cumulative so the threshold tracks a drifting workload instead of being anchored by ancient
//! samples (PLAN.md §5.2).

/// Decayed (exponentially-weighted) mean and variance, updated one sample at a time in O(1) space.
/// `count` is the raw number of samples seen — used only to gate the cold-start period, not the
/// statistics themselves.
#[derive(Clone, Debug)]
pub struct EwMoments {
    alpha: f64,
    mean: f64,
    var: f64,
    count: u64,
    initialized: bool,
}

impl EwMoments {
    /// `alpha` is the decay rate in `(0, 1]` — larger adapts faster (weights recent samples more),
    /// smaller is smoother. Clamped to a sane range.
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha: alpha.clamp(1e-3, 1.0),
            mean: 0.0,
            var: 0.0,
            count: 0,
            initialized: false,
        }
    }

    /// Fold in one observation. Uses West's incremental exponentially-weighted variance, seeded from
    /// the first sample so the mean isn't dragged from an arbitrary zero.
    pub fn update(&mut self, x: f64) {
        self.count += 1;
        if !self.initialized {
            self.mean = x;
            self.var = 0.0;
            self.initialized = true;
            return;
        }
        let delta = x - self.mean;
        self.mean += self.alpha * delta;
        self.var = (1.0 - self.alpha) * (self.var + self.alpha * delta * delta);
    }

    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// Standard deviation (`sqrt` of the decayed variance), floored at 0 against tiny negatives.
    pub fn std(&self) -> f64 {
        self.var.max(0.0).sqrt()
    }

    pub fn count(&self) -> u64 {
        self.count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_stream_converges_to_the_constant_with_zero_spread() {
        let mut m = EwMoments::new(0.1);
        for _ in 0..200 {
            m.update(0.75);
        }
        assert!((m.mean() - 0.75).abs() < 1e-9);
        assert!(m.std() < 1e-9);
        assert_eq!(m.count(), 200);
    }

    #[test]
    fn tracks_mean_and_nonzero_spread_of_a_varying_stream() {
        let mut m = EwMoments::new(0.2);
        // Alternate around 0.5; the decayed mean should sit near 0.5 with positive spread.
        for i in 0..500 {
            m.update(if i % 2 == 0 { 0.4 } else { 0.6 });
        }
        assert!(
            (m.mean() - 0.5).abs() < 0.05,
            "mean near 0.5, got {}",
            m.mean()
        );
        assert!(m.std() > 0.0, "alternating stream has spread");
    }

    #[test]
    fn first_sample_seeds_the_mean() {
        let mut m = EwMoments::new(0.05);
        m.update(0.9);
        assert!((m.mean() - 0.9).abs() < 1e-9, "not dragged from zero");
        assert_eq!(m.std(), 0.0);
    }
}
