//! Distance/similarity kernels over borrowed rows.
//!
//! Score convention: **higher is better**. All `dot_*` functions are
//! similarities (the corpus is expected L2-normalized, so dot = cosine).
//! [`hamming`] is the one raw distance (lower is better, exact);
//! [`hamming_score`] adapts it to the score convention as `-(h as f32)`.
//!
//! # Binary code layout
//!
//! Packed sign-bit codes are `binary_code_len(dim)` bytes, **MSB-first within
//! each byte**: dimension `i` lives in byte `i / 8`, bit `7 - (i % 8)`. This is
//! numpy `packbits(bitorder='big')` and Voyage `ubinary`, so such exports can
//! be used directly. **Padding bits in the final byte must be zero** — the
//! symmetric [`hamming`] kernel XORs whole bytes and relies on equal padding
//! cancelling; the asymmetric kernels decode only the `dim` real bits and
//! never read padding.
//!
//! # Determinism
//!
//! For a fixed backend, every kernel is bit-deterministic (fixed unroll,
//! reduction tree, and tail path). Integer kernels (`dot_i8`, `hamming`,
//! `dot_i8_bin`) are exact and therefore bit-identical across backends; f32
//! kernels may differ across backends by summation order within a small
//! tolerance. NaN/Inf pass through untouched (NaN in ⇒ NaN out); callers that
//! sort scores use `f32::total_cmp`.
//!
//! # Dispatch
//!
//! Free functions dispatch through [`Kernels::detected`] on every call (one
//! atomic load + an indirect call). Inside a scan loop, hoist the table:
//!
//! ```
//! let k = guksu::kernels::Kernels::detected();
//! let score = (k.dot_f32)(&[1.0, 0.0], &[0.5, 0.5]);
//! assert_eq!(score, 0.5);
//! ```
//!
//! Every table entry validates its input lengths itself, so calling through a
//! table is exactly as safe as the free functions.

use std::sync::OnceLock;

#[cfg(target_arch = "aarch64")]
mod aarch64;
mod scalar;
#[cfg(target_arch = "x86_64")]
mod x86_64;

/// Number of bytes in a packed sign-bit code for `dim` dimensions (`ceil(dim/8)`).
pub const fn binary_code_len(dim: usize) -> usize {
    dim.div_ceil(8)
}

#[inline]
pub(crate) fn check_equal_len(a: usize, b: usize) {
    assert!(a == b, "kernel input length mismatch: {a} vs {b}");
}

#[inline]
pub(crate) fn check_code_len(q_len: usize, code_len: usize) {
    assert!(
        code_len == binary_code_len(q_len),
        "binary code is {code_len} bytes but binary_code_len({q_len}) = {}",
        binary_code_len(q_len)
    );
}

/// A resolved set of kernel implementations (plain safe fn pointers).
///
/// Obtain via [`Kernels::detected`] (best for this CPU), [`Kernels::scalar`]
/// (portable reference, also the test oracle), or [`Kernels::by_name`] to pin
/// a specific backend. The `backend` field names the selected implementation
/// for logs and bench labels.
#[derive(Clone, Copy)]
pub struct Kernels {
    /// Backend name: `"scalar"`, `"neon"`, `"neon_dotprod"`, or `"avx2"`.
    pub backend: &'static str,
    pub dot_f32: fn(&[f32], &[f32]) -> f32,
    pub dot_i8: fn(&[i8], &[i8]) -> i32,
    pub hamming: fn(&[u8], &[u8]) -> u32,
    pub dot_f32_bin: fn(&[f32], &[u8]) -> f32,
    pub dot_i8_bin: fn(&[i8], &[u8]) -> i32,
}

static SCALAR: Kernels = Kernels {
    backend: "scalar",
    dot_f32: scalar::dot_f32,
    dot_i8: scalar::dot_i8,
    hamming: scalar::hamming,
    dot_f32_bin: scalar::dot_f32_bin,
    dot_i8_bin: scalar::dot_i8_bin,
};

#[cfg(target_arch = "aarch64")]
const VALID_BACKENDS: &[&str] = &["scalar", "neon", "neon_dotprod"];
#[cfg(target_arch = "x86_64")]
const VALID_BACKENDS: &[&str] = &["scalar", "avx2"];
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
const VALID_BACKENDS: &[&str] = &["scalar"];

impl Kernels {
    /// Backend names valid for this compilation target, weakest first. Names
    /// the running CPU cannot execute are still listed — [`Kernels::by_name`]
    /// is the arbiter of what is actually available.
    pub const NAMES: &'static [&'static str] = VALID_BACKENDS;

    /// Portable reference backend — the oracle SIMD backends are tested against.
    pub const fn scalar() -> &'static Kernels {
        &SCALAR
    }

    /// Best backend for this CPU, detected once per process. To pin a
    /// specific backend instead, use [`Kernels::by_name`].
    pub fn detected() -> &'static Kernels {
        static DETECTED: OnceLock<&'static Kernels> = OnceLock::new();
        DETECTED.get_or_init(Self::best)
    }

    /// The backend named `name`, or `None` when the name is not in
    /// [`Kernels::NAMES`] or the running CPU lacks the features it needs — a
    /// returned table is always sound to call.
    pub fn by_name(name: &str) -> Option<&'static Kernels> {
        match name {
            "scalar" => Some(&SCALAR),
            #[cfg(target_arch = "aarch64")]
            "neon" => Some(&aarch64::TABLE_NEON),
            #[cfg(target_arch = "aarch64")]
            "neon_dotprod" => std::arch::is_aarch64_feature_detected!("dotprod")
                .then_some(&aarch64::TABLE_NEON_DOTPROD),
            #[cfg(target_arch = "x86_64")]
            "avx2" => (std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
                && std::arch::is_x86_feature_detected!("popcnt"))
            .then_some(&x86_64::TABLE_AVX2),
            _ => None,
        }
    }

    fn best() -> &'static Kernels {
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("dotprod") {
                return &aarch64::TABLE_NEON_DOTPROD;
            }
            #[allow(unreachable_code)]
            return &aarch64::TABLE_NEON;
        }
        #[cfg(target_arch = "x86_64")]
        {
            // POPCNT and FMA are not implied by AVX2 as target features —
            // detect all three (every real AVX2 CPU has them).
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
                && std::arch::is_x86_feature_detected!("popcnt")
            {
                return &x86_64::TABLE_AVX2;
            }
        }
        #[allow(unreachable_code)]
        &SCALAR
    }
}

/// f32 dot product (cosine similarity on L2-normalized inputs). Panics if lengths differ.
#[inline]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    (Kernels::detected().dot_f32)(a, b)
}

/// Exact int8 dot product with i32 accumulation, over the full `[-128, 127]`
/// domain. Rank-equivalent score under a corpus-wide fixed scale.
#[inline]
pub fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
    (Kernels::detected().dot_i8)(a, b)
}

/// Scaled int8 dot: `dot_i8(a, b) as f32 * (sa * sb)` — the ranking score
/// under per-vector scales.
#[inline]
pub fn dot_i8_scaled(a: &[i8], sa: f32, b: &[i8], sb: f32) -> f32 {
    dot_i8(a, b) as f32 * (sa * sb)
}

/// Exact Hamming distance over packed sign-bit codes (lower is better). Both
/// codes must honor the zero-padding contract (see module docs).
#[inline]
pub fn hamming(a: &[u8], b: &[u8]) -> u32 {
    (Kernels::detected().hamming)(a, b)
}

/// Hamming adapted to the higher-is-better score convention: `-(hamming as f32)`.
#[inline]
pub fn hamming_score(a: &[u8], b: &[u8]) -> f32 {
    -(hamming(a, b) as f32)
}

/// Asymmetric f32 query × binary code: the ±1 dot `Σᵢ qᵢ·(2bᵢ−1)`, computed by
/// sign-flipping `qᵢ` where `bᵢ = 0` (exact — no extra rounding vs multiplying
/// by ±1.0).
#[inline]
pub fn dot_f32_bin(q: &[f32], code: &[u8]) -> f32 {
    (Kernels::detected().dot_f32_bin)(q, code)
}

/// Asymmetric int8 query × binary code: the exact i32 ±1 dot `Σᵢ qᵢ·(2bᵢ−1)`.
/// The query-side scale is rank-invariant per query; apply it outside if
/// absolute values are needed.
#[inline]
pub fn dot_i8_bin(q: &[i8], code: &[u8]) -> i32 {
    (Kernels::detected().dot_i8_bin)(q, code)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::{pack_sign_bits_vec, quantize_i8_vec, max_abs_scale, unpack_sign_bits};
    use crate::rng::SplitMix64;

    fn bit(code: &[u8], i: usize) -> bool {
        (code[i / 8] >> (7 - (i % 8))) & 1 == 1
    }

    fn normalized_gaussian(rng: &mut SplitMix64, d: usize) -> Vec<f32> {
        let mut v: Vec<f32> = (0..d).map(|_| rng.next_gaussian() as f32).collect();
        let norm = v.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt() as f32;
        if norm > 0.0 {
            for x in &mut v {
                *x /= norm;
            }
        }
        v
    }

    fn random_code(rng: &mut SplitMix64, d: usize) -> Vec<u8> {
        let n = binary_code_len(d);
        let mut c: Vec<u8> = (0..n).map(|_| rng.next_range(256) as u8).collect();
        let pad = n * 8 - d;
        if pad > 0 {
            c[n - 1] &= 0xFFu8 << pad;
        }
        c
    }

    #[test]
    fn empty_inputs_are_zero() {
        assert_eq!(dot_f32(&[], &[]), 0.0);
        assert_eq!(dot_i8(&[], &[]), 0);
        assert_eq!(hamming(&[], &[]), 0);
        assert_eq!(dot_f32_bin(&[], &[]), 0.0);
        assert_eq!(dot_i8_bin(&[], &[]), 0);
    }

    #[test]
    fn i8_extremes_exact() {
        // Alternating ±127 self-dot: every term is +16129. This is the
        // permanent tripwire for saturating-i16 implementations (maddubs).
        let alt: Vec<i8> = (0..1024).map(|i| if i % 2 == 0 { 127 } else { -127 }).collect();
        assert_eq!(dot_i8(&alt, &alt), 1024 * 16129);

        // Full-domain −128 must be exact (Voyage int8 rows can contain it).
        let neg = vec![-128i8; 1024];
        assert_eq!(dot_i8(&neg, &neg), 1024 * 16384);
        let pos = vec![127i8; 1024];
        assert_eq!(dot_i8(&neg, &pos), 1024 * -128 * 127);
    }

    #[test]
    fn f32_specials_pass_through() {
        assert_eq!(dot_f32(&[0.0, -0.0], &[1.0, 1.0]), 0.0);
        assert!(dot_f32(&[1e30], &[1e30]).is_infinite());
        assert!(dot_f32(&[f32::NAN], &[1.0]).is_nan());
        assert!(dot_f32_bin(&[f32::NAN], &[0x80]).is_nan());
        // Denormal inputs stay finite, no panic.
        assert!(dot_f32(&[1e-40], &[1e-40]).is_finite());
    }

    #[test]
    fn hamming_matches_per_bit_naive() {
        let mut rng = SplitMix64::new(11);
        for d in [1usize, 2, 7, 8, 9, 15, 16, 17, 63, 64, 65, 127, 128, 1000, 1024, 1027] {
            for _ in 0..20 {
                let a = random_code(&mut rng, d);
                let b = random_code(&mut rng, d);
                let naive: u32 = (0..d).map(|i| (bit(&a, i) != bit(&b, i)) as u32).sum();
                assert_eq!(hamming(&a, &b), naive, "d={d}");
                assert_eq!(hamming(&a, &b), hamming(&b, &a), "d={d}");
                assert_eq!(hamming(&a, &a), 0, "d={d}");
            }
        }
    }

    #[test]
    fn dirty_padding_corrupts_hamming() {
        // Documents that the zero-padding contract is load-bearing: equal
        // codes with unequal padding bits do NOT compare equal.
        // d = 13: 2 bytes, 3 padding bits.
        let clean = pack_sign_bits_vec(&[1.0f32; 13]);
        let mut dirty = clean.clone();
        dirty[1] |= 0b0000_0101;
        assert_eq!(hamming(&clean, &clean), 0);
        assert_eq!(hamming(&clean, &dirty), 2);
    }

    #[test]
    fn asym_f32_bin_equals_unpacked_dot_exactly_on_scalar() {
        // Sign-flipping is exact, so the scalar asymmetric kernel must equal
        // the scalar f32 dot against the ±1 reconstruction bit-for-bit.
        let s = Kernels::scalar();
        let mut rng = SplitMix64::new(12);
        for d in [1usize, 7, 8, 64, 65, 1000, 1024, 1027] {
            let q = normalized_gaussian(&mut rng, d);
            let x = normalized_gaussian(&mut rng, d);
            let code = pack_sign_bits_vec(&x);
            let mut unpacked = vec![0.0f32; d];
            unpack_sign_bits(&code, &mut unpacked);
            assert_eq!((s.dot_f32_bin)(&q, &code), (s.dot_f32)(&q, &unpacked), "d={d}");
        }
    }

    #[test]
    fn asym_i8_bin_matches_naive() {
        let mut rng = SplitMix64::new(13);
        for d in [1usize, 7, 8, 64, 65, 1000, 1024, 1027] {
            let q: Vec<i8> = (0..d).map(|_| (rng.next_range(256) as i16 - 128) as i8).collect();
            let code = random_code(&mut rng, d);
            let naive: i32 = (0..d)
                .map(|i| if bit(&code, i) { q[i] as i32 } else { -(q[i] as i32) })
                .sum();
            assert_eq!(dot_i8_bin(&q, &code), naive, "d={d}");
        }
    }

    #[test]
    fn i8_scaled_tracks_f32_dot_statistically() {
        // Regression guard: symmetric per-vector int8 on normalized 1024-dim
        // vectors stays within 0.02 of the f32 dot.
        let mut rng = SplitMix64::new(14);
        let d = 1024;
        for _ in 0..1000 {
            let a = normalized_gaussian(&mut rng, d);
            let b = normalized_gaussian(&mut rng, d);
            let (sa, sb) = (max_abs_scale(&a), max_abs_scale(&b));
            let qa = quantize_i8_vec(&a, sa);
            let qb = quantize_i8_vec(&b, sb);
            let approx = dot_i8_scaled(&qa, sa, &qb, sb);
            let exact = dot_f32(&a, &b);
            assert!((approx - exact).abs() < 0.02, "{approx} vs {exact}");
        }
    }

    #[test]
    #[should_panic(expected = "length mismatch")]
    fn length_mismatch_panics() {
        dot_f32(&[1.0], &[1.0, 2.0]);
    }

    #[test]
    #[should_panic(expected = "binary code")]
    fn code_length_mismatch_panics() {
        dot_f32_bin(&[1.0; 9], &[0u8; 1]);
    }
}
