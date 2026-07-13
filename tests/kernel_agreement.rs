//! SIMD-vs-scalar agreement: every backend this CPU supports must match the
//! portable reference on every kernel — exactly for integer kernels, within a
//! summation-order bound (checked against an f64 reference) for f32 kernels.
//!
//! Runs green trivially on a CPU with no SIMD backend (scalar is only checked
//! against the f64 reference); each supported arch backend is the real gate.

use guksu::kernels::{Kernels, binary_code_len};
use guksu::rng::SplitMix64;

/// Every CPU-supported backend except the scalar oracle itself.
fn simd_tables() -> Vec<&'static Kernels> {
    Kernels::NAMES
        .iter()
        .filter_map(|&name| Kernels::by_name(name))
        .filter(|k| k.backend != "scalar")
        .collect()
}

/// Crosses every lane width (4/8/16/32), u64 boundary, byte boundary, and
/// unroll boundary, plus the corpus dim.
const DIMS: &[usize] = &[
    1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 1000, 1024,
    1027,
];
const PAIRS: usize = 100;

fn uniform(rng: &mut SplitMix64, d: usize) -> Vec<f32> {
    (0..d).map(|_| rng.next_f32() * 2.0 - 1.0).collect()
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

fn full_domain_i8(rng: &mut SplitMix64, d: usize) -> Vec<i8> {
    (0..d).map(|_| (rng.next_range(256) as i16 - 128) as i8).collect()
}

/// Random code honoring the zero-padding contract.
fn random_code(rng: &mut SplitMix64, d: usize) -> Vec<u8> {
    let n = binary_code_len(d);
    let mut c: Vec<u8> = (0..n).map(|_| rng.next_range(256) as u8).collect();
    let pad = n * 8 - d;
    if pad > 0 {
        c[n - 1] &= 0xFFu8 << pad;
    }
    c
}

/// (f64 dot, f64 sum of |terms|) — the reference and the scale of its error bound.
fn dot_ref_f64(a: &[f32], b: &[f32]) -> (f64, f64) {
    let mut s = 0.0f64;
    let mut sum_abs = 0.0f64;
    for (&x, &y) in a.iter().zip(b) {
        let t = x as f64 * y as f64;
        s += t;
        sum_abs += t.abs();
    }
    (s, sum_abs)
}

fn f32_bin_ref_f64(q: &[f32], code: &[u8]) -> (f64, f64) {
    let mut s = 0.0f64;
    let mut sum_abs = 0.0f64;
    for (i, &qi) in q.iter().enumerate() {
        let signed = if (code[i / 8] >> (7 - (i % 8))) & 1 == 1 { qi as f64 } else { -(qi as f64) };
        s += signed;
        sum_abs += (qi as f64).abs();
    }
    (s, sum_abs)
}

/// |x − ref| ≤ 32·ε_f32·Σ|terms|: covers summation-order and FMA-vs-mul+add
/// differences at these dims while catching any real kernel bug (those are off
/// by whole terms, orders of magnitude beyond the bound).
#[track_caller]
fn assert_close(x: f32, ref_f64: f64, sum_abs: f64, ctx: &str) {
    let tol = 32.0 * f32::EPSILON as f64 * sum_abs.max(f32::MIN_POSITIVE as f64);
    assert!(
        (x as f64 - ref_f64).abs() <= tol,
        "{ctx}: {x} vs f64 ref {ref_f64} (tol {tol:.3e})"
    );
}

#[test]
fn every_backend_agrees_with_scalar() {
    let s = Kernels::scalar();
    let tabs = simd_tables();
    let mut rng = SplitMix64::new(0xC0FFEE);

    for &d in DIMS {
        for pair in 0..PAIRS {
            // f32 dot, both input shapes.
            for (a, b, shape) in [
                (uniform(&mut rng, d), uniform(&mut rng, d), "uniform"),
                (normalized_gaussian(&mut rng, d), normalized_gaussian(&mut rng, d), "normalized"),
            ] {
                let (rf, sa) = dot_ref_f64(&a, &b);
                let ctx = format!("dot_f32[scalar] d={d} pair={pair} {shape}");
                assert_close((s.dot_f32)(&a, &b), rf, sa, &ctx);
                for t in &tabs {
                    let ctx = format!("dot_f32[{}] d={d} pair={pair} {shape}", t.backend);
                    assert_close((t.dot_f32)(&a, &b), rf, sa, &ctx);
                }
            }

            // int8 dot: exact across backends, full domain.
            let ia = full_domain_i8(&mut rng, d);
            let ib = full_domain_i8(&mut rng, d);
            for t in &tabs {
                assert_eq!(
                    (s.dot_i8)(&ia, &ib),
                    (t.dot_i8)(&ia, &ib),
                    "dot_i8[{}] d={d} pair={pair}",
                    t.backend
                );
            }

            // Hamming: exact.
            let ca = random_code(&mut rng, d);
            let cb = random_code(&mut rng, d);
            for t in &tabs {
                assert_eq!(
                    (s.hamming)(&ca, &cb),
                    (t.hamming)(&ca, &cb),
                    "hamming[{}] d={d} pair={pair}",
                    t.backend
                );
            }

            // f32 × binary: tolerance.
            let q = normalized_gaussian(&mut rng, d);
            let (rfb, sab) = f32_bin_ref_f64(&q, &ca);
            let ctx = format!("dot_f32_bin[scalar] d={d} pair={pair}");
            assert_close((s.dot_f32_bin)(&q, &ca), rfb, sab, &ctx);
            for t in &tabs {
                let ctx = format!("dot_f32_bin[{}] d={d} pair={pair}", t.backend);
                assert_close((t.dot_f32_bin)(&q, &ca), rfb, sab, &ctx);
            }

            // int8 × binary: exact.
            let iq = full_domain_i8(&mut rng, d);
            for t in &tabs {
                assert_eq!(
                    (s.dot_i8_bin)(&iq, &cb),
                    (t.dot_i8_bin)(&iq, &cb),
                    "dot_i8_bin[{}] d={d} pair={pair}",
                    t.backend
                );
            }
        }
    }
}

/// Anti-silent-skip guard: Apple Silicon must detect the dotprod backend, so
/// the agreement suite above is known to have exercised real SIMD here.
#[test]
#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
fn apple_silicon_detects_neon_dotprod() {
    assert_eq!(Kernels::detected().backend, "neon_dotprod");
}

/// Opt-in CI guard: `GUKSU_REQUIRE=<backend>` makes the run fail if detection
/// picked anything else (e.g. an x86_64 runner that should have AVX2).
#[test]
fn required_backend_is_detected() {
    if let Ok(required) = std::env::var("GUKSU_REQUIRE") {
        assert_eq!(Kernels::detected().backend, required);
    }
}

/// Adversarial constants must agree exactly across backends (integer paths).
#[test]
fn every_backend_agrees_on_extremes() {
    let s = Kernels::scalar();
    let alt: Vec<i8> = (0..1024).map(|i| if i % 2 == 0 { 127 } else { -127 }).collect();
    let neg = vec![-128i8; 1024];
    let pos = vec![127i8; 1024];
    let ones = vec![0xFFu8; 128];
    let zeros = vec![0u8; 128];
    for t in simd_tables() {
        for (a, b) in [(&alt, &alt), (&neg, &neg), (&neg, &pos), (&alt, &neg)] {
            assert_eq!((s.dot_i8)(a, b), (t.dot_i8)(a, b), "dot_i8[{}]", t.backend);
        }
        assert_eq!((t.hamming)(&ones, &zeros), 1024, "hamming[{}]", t.backend);
        assert_eq!((s.dot_i8_bin)(&neg, &ones), (t.dot_i8_bin)(&neg, &ones), "[{}]", t.backend);
        assert_eq!((s.dot_i8_bin)(&pos, &zeros), (t.dot_i8_bin)(&pos, &zeros), "[{}]", t.backend);
    }
}
