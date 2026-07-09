//! Small, dependency-free statistics the adaptive threshold needs: the inverse normal CDF (to turn
//! a target false-hit rate into a sigma multiplier) and an EMA step.

/// Inverse standard-normal CDF (probit): the `z` such that `P(Z ≤ z) = p`, for `0 < p < 1`.
/// Acklam's rational approximation — absolute error < 1.15e-9 over the open interval, which is far
/// tighter than the threshold needs. Returns ±∞ at the boundaries.
// The coefficients are Acklam's published constants, quoted verbatim at full precision — that some
// round-trip to a shorter f64 literal is expected, so the precision lint is silenced here.
#[allow(clippy::excessive_precision)]
pub fn probit(p: f64) -> f64 {
    // Coefficients for Acklam's algorithm.
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_690e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239e0,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838e0,
        -2.549_732_539_343_734e0,
        4.374_664_141_464_968e0,
        2.938_163_982_698_783e0,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996e0,
        3.754_408_661_907_416e0,
    ];
    const P_LOW: f64 = 0.024_25;
    const P_HIGH: f64 = 1.0 - P_LOW;

    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= P_HIGH {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

/// The upper-tail quantile: the `z` such that `P(Z > z) = p`. This is the sigma multiplier that
/// admits a fraction `p` of a Gaussian above the bound — exactly the false-hit budget. `z(0.02) ≈
/// 2.05`, `z(0.05) ≈ 1.64`, `z(0.5) = 0`.
pub fn inv_upper_normal(p: f64) -> f64 {
    probit(1.0 - p)
}

/// One exponential-moving step from `current` toward `target` at rate `alpha ∈ [0, 1]`.
pub fn ema(current: f64, target: f64, alpha: f64) -> f64 {
    current + alpha * (target - current)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probit_known_quantiles() {
        assert!(probit(0.5).abs() < 1e-6, "median is 0");
        assert!((probit(0.975) - 1.959_964).abs() < 1e-3, "97.5% ≈ 1.96");
        assert!((probit(0.95) - 1.644_854).abs() < 1e-3, "95% ≈ 1.645");
        // Symmetry.
        assert!((probit(0.1) + probit(0.9)).abs() < 1e-6);
    }

    #[test]
    fn inv_upper_normal_is_the_false_hit_sigma() {
        assert!(
            (inv_upper_normal(0.02) - 2.053_749).abs() < 1e-3,
            "FHR 2% ≈ 2.05σ"
        );
        assert!(
            (inv_upper_normal(0.05) - 1.644_854).abs() < 1e-3,
            "FHR 5% ≈ 1.64σ"
        );
        assert!(inv_upper_normal(0.5).abs() < 1e-6, "FHR 50% = 0σ");
        // A stricter FHR demands a larger margin.
        assert!(inv_upper_normal(0.01) > inv_upper_normal(0.05));
    }

    #[test]
    fn ema_steps_toward_target() {
        assert!((ema(0.0, 1.0, 0.5) - 0.5).abs() < 1e-9);
        assert!(
            (ema(0.8, 0.8, 0.1) - 0.8).abs() < 1e-9,
            "no move when already at target"
        );
    }
}
