//! Symmetric linear int8 quantization (no zero-point), per-vector or fixed scale.

/// Symmetric per-vector scale: `max|x| / 127` (`0.0` for the all-zero row).
/// NaN elements are ignored by the max.
pub fn max_abs_scale(src: &[f32]) -> f32 {
    src.iter().fold(0.0f32, |m, &x| m.max(x.abs())) / 127.0
}

/// Corpus-wide fixed scale: global `max|x| / 127` over a flat sample block of
/// `dim`-length rows. Panics if the block is not whole rows.
pub fn fixed_scale(block: &[f32], dim: usize) -> f32 {
    assert!(dim > 0 && block.len() % dim == 0, "block must be whole rows of dim {dim}");
    max_abs_scale(block)
}

/// `out[i] = round(src[i] / scale)` (half away from zero), clamped to
/// `[-127, 127]` — never emits `-128`. `scale <= 0` (all-zero rows) emits
/// all-zero codes.
pub fn quantize_i8(src: &[f32], scale: f32, out: &mut [i8]) {
    assert_eq!(src.len(), out.len(), "src and out lengths differ");
    let usable = scale > 0.0; // false for zero, negative, and NaN scales
    if !usable {
        out.fill(0);
        return;
    }
    let inv = 1.0 / scale;
    for (o, &x) in out.iter_mut().zip(src) {
        *o = (x * inv).round().clamp(-127.0, 127.0) as i8;
    }
}

/// Allocating convenience for [`quantize_i8`].
pub fn quantize_i8_vec(src: &[f32], scale: f32) -> Vec<i8> {
    let mut out = vec![0i8; src.len()];
    quantize_i8(src, scale, &mut out);
    out
}

/// `out[i] = code[i] as f32 * scale` — the reconstruction the quantizer's
/// error bound is stated against.
pub fn dequantize_i8(code: &[i8], scale: f32, out: &mut [f32]) {
    assert_eq!(code.len(), out.len(), "code and out lengths differ");
    for (o, &c) in out.iter_mut().zip(code) {
        *o = c as f32 * scale;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::SplitMix64;

    #[test]
    fn max_abs_scale_known_values() {
        assert_eq!(max_abs_scale(&[3.0, -4.0]), 4.0 / 127.0);
        assert_eq!(max_abs_scale(&[]), 0.0);
        assert_eq!(max_abs_scale(&[0.0, -0.0]), 0.0);
        assert_eq!(max_abs_scale(&[f32::NAN, 2.0]), 2.0 / 127.0);
    }

    #[test]
    fn round_trip_error_within_half_step() {
        let mut rng = SplitMix64::new(21);
        for _ in 0..100 {
            let src: Vec<f32> = (0..257).map(|_| (rng.next_f32() - 0.5) * 4.0).collect();
            let scale = max_abs_scale(&src);
            let q = quantize_i8_vec(&src, scale);
            let mut back = vec![0.0f32; src.len()];
            dequantize_i8(&q, scale, &mut back);
            for (&x, &y) in src.iter().zip(&back) {
                assert!((x - y).abs() <= scale * 0.5 + 1e-6, "{x} -> {y} (scale {scale})");
            }
        }
    }

    #[test]
    fn never_emits_neg_128_and_clamps() {
        // Out-of-range values (fixed scale smaller than the data) clamp.
        let q = quantize_i8_vec(&[-1e9, 1e9, -1.0, 1.0], 1.0 / 127.0);
        assert_eq!(q, vec![-127, 127, -127, 127]);
        assert!(q.iter().all(|&c| c != -128));
    }

    #[test]
    fn zero_scale_emits_zero_codes() {
        assert_eq!(quantize_i8_vec(&[0.0, -0.0, 0.0], 0.0), vec![0, 0, 0]);
        // Negative scale is treated as degenerate, same as zero.
        assert_eq!(quantize_i8_vec(&[1.0], -1.0), vec![0]);
    }

    #[test]
    fn rounds_half_away_from_zero() {
        let q = quantize_i8_vec(&[0.5, -0.5, 1.5, -1.5, 0.49, -0.49], 1.0);
        assert_eq!(q, vec![1, -1, 2, -2, 0, 0]);
    }

    #[test]
    fn fixed_scale_covers_whole_block() {
        let block = [0.5f32, -1.0, 0.25, 0.75];
        assert_eq!(fixed_scale(&block, 2), 1.0 / 127.0);
    }

    #[test]
    #[should_panic(expected = "whole rows")]
    fn fixed_scale_rejects_ragged_block() {
        fixed_scale(&[1.0, 2.0, 3.0], 2);
    }
}
