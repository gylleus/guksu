//! Seeded PRNG shared by tests, benches, and the recall harness.
//!
//! Hand-rolled so that seeded output is bit-stable across guksu releases,
//! platforms, and toolchains — `rand`'s `StdRng` explicitly does not promise
//! that, and reproducible corpora are a harness requirement (`--seed 42` must
//! mean the same vectors years from now). Hidden from docs: not a public API
//! contract beyond determinism.

/// SplitMix64: 64-bit state, full period 2^64, passes BigCrush. The standard
/// seeding/streaming PRNG (Steele, Lea, Flood 2014).
#[derive(Clone, Debug)]
pub struct SplitMix64 {
    state: u64,
    /// Second output of the last Box-Muller pair, served before drawing anew.
    cached_gaussian: Option<f64>,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self { state: seed, cached_gaussian: None }
    }

    /// Derive an independent stream for a substream index (e.g. one per row),
    /// so generation is deterministic regardless of iteration or thread order.
    pub fn substream(seed: u64, index: u64) -> Self {
        // One scramble round separates (seed, index) pairs; SplitMix64's
        // output function decorrelates the sequential states that follow.
        let mut root = Self::new(seed ^ index.wrapping_mul(0x9E3779B97F4A7C15));
        let derived = root.next_u64();
        Self::new(derived)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1) with 53-bit precision.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform in [0, 1) as f32.
    pub fn next_f32(&mut self) -> f32 {
        self.next_f64() as f32
    }

    /// Uniform in [0, n) without modulo bias (multiply-shift).
    pub fn next_range(&mut self, n: u64) -> u64 {
        ((self.next_u64() as u128 * n as u128) >> 64) as u64
    }

    /// Standard normal via Box-Muller; pairs are cached so consecutive calls
    /// consume one uniform pair per two gaussians.
    pub fn next_gaussian(&mut self) -> f64 {
        if let Some(g) = self.cached_gaussian.take() {
            return g;
        }
        // u1 in (0, 1] so ln(u1) is finite.
        let u1 = 1.0 - self.next_f64();
        let u2 = self.next_f64();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = std::f64::consts::TAU * u2;
        self.cached_gaussian = Some(r * theta.sin());
        r * theta.cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_vectors_seed_zero() {
        // Published SplitMix64 test vectors for seed 0 — pins the constants.
        let mut r = SplitMix64::new(0);
        assert_eq!(r.next_u64(), 0xE220A8397B1DCDAF);
        assert_eq!(r.next_u64(), 0x6E789E6AA1B965F4);
    }

    #[test]
    fn deterministic_per_seed() {
        let a: Vec<u64> = {
            let mut r = SplitMix64::new(42);
            (0..64).map(|_| r.next_u64()).collect()
        };
        let b: Vec<u64> = {
            let mut r = SplitMix64::new(42);
            (0..64).map(|_| r.next_u64()).collect()
        };
        assert_eq!(a, b);
        let c = {
            let mut r = SplitMix64::new(43);
            r.next_u64()
        };
        assert_ne!(a[0], c);
    }

    #[test]
    fn substreams_differ_and_are_deterministic() {
        let first = |i: u64| SplitMix64::substream(7, i).next_u64();
        assert_eq!(first(0), first(0));
        assert_ne!(first(0), first(1));
        assert_ne!(first(1), first(2));
    }

    #[test]
    fn uniform_range_and_moments() {
        let mut r = SplitMix64::new(1);
        let n = 100_000;
        let mut sum = 0.0;
        for _ in 0..n {
            let x = r.next_f64();
            assert!((0.0..1.0).contains(&x));
            sum += x;
        }
        let mean = sum / n as f64;
        assert!((mean - 0.5).abs() < 0.01, "mean {mean}");
        for bound in [1u64, 2, 3, 1000] {
            for _ in 0..1000 {
                assert!(r.next_range(bound) < bound);
            }
        }
    }

    #[test]
    fn gaussian_moments() {
        let mut r = SplitMix64::new(2);
        let n = 100_000;
        let (mut sum, mut sum_sq) = (0.0, 0.0);
        for _ in 0..n {
            let g = r.next_gaussian();
            assert!(g.is_finite());
            sum += g;
            sum_sq += g * g;
        }
        let mean = sum / n as f64;
        let var = sum_sq / n as f64 - mean * mean;
        assert!(mean.abs() < 0.02, "mean {mean}");
        assert!((var - 1.0).abs() < 0.03, "var {var}");
    }
}
