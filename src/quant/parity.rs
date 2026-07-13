//! Voyage `output_dtype` parity helpers — pure functions over pre-fetched
//! arrays. Fetching (which needs an API key) stays outside the library; see
//! the README for an export snippet and the dtype conversion table.

use crate::kernels::{binary_code_len, hamming};

/// Convert Voyage `binary` output into guksu codes.
///
/// Voyage's `binary` dtype is *offset binary* stored as i8 (`i8 = u8 - 128`),
/// so each byte converts via `XOR 0x80`. Voyage's `ubinary` dtype is already
/// plain packed u8 and needs no conversion. Ingesting `binary` bytes as-is
/// would silently invert the first dimension of every 8-dim group — the
/// parity signature of that mistake is `bit_mismatch_rate ≈ 0.125`.
pub fn voyage_offset_binary_to_codes(voyage_i8: &[i8], out: &mut [u8]) {
    assert_eq!(voyage_i8.len(), out.len(), "input and output lengths differ");
    for (o, &v) in out.iter_mut().zip(voyage_i8) {
        *o = (v as u8) ^ 0x80;
    }
}

/// Result of comparing client-side sign-bit codes against Voyage codes.
#[derive(Clone, Copy, Debug)]
pub struct BinaryParity {
    pub vectors: usize,
    /// Fraction of all `dim · vectors` bits that differ.
    pub bit_mismatch_rate: f64,
    /// Fraction of rows that match exactly (Hamming 0).
    pub identical_vector_rate: f64,
    /// Differing-bit count of the worst row.
    pub max_row_mismatch: u32,
}

/// Compare client codes (from [`super::pack_sign_bits`] over the f32 export)
/// against Voyage `ubinary`/converted-`binary` codes. Both arguments are flat
/// blocks of `binary_code_len(dim)`-byte rows in the same order.
pub fn binary_parity(client: &[u8], voyage: &[u8], dim: usize) -> BinaryParity {
    assert!(dim > 0, "dim must be > 0");
    let row = binary_code_len(dim);
    assert_eq!(client.len(), voyage.len(), "blocks differ in size");
    assert!(client.len() % row == 0, "blocks are not whole {row}-byte rows");
    let vectors = client.len() / row;

    let mut diff_bits = 0u64;
    let mut identical = 0usize;
    let mut worst = 0u32;
    for (c, v) in client.chunks_exact(row).zip(voyage.chunks_exact(row)) {
        let h = hamming(c, v);
        diff_bits += h as u64;
        identical += (h == 0) as usize;
        worst = worst.max(h);
    }
    BinaryParity {
        vectors,
        bit_mismatch_rate: if vectors == 0 {
            0.0
        } else {
            diff_bits as f64 / (vectors as f64 * dim as f64)
        },
        identical_vector_rate: if vectors == 0 { 1.0 } else { identical as f64 / vectors as f64 },
        max_row_mismatch: worst,
    }
}

/// Result of comparing client-side int8 codes against Voyage int8 codes.
#[derive(Clone, Copy, Debug)]
pub struct Int8Parity {
    pub vectors: usize,
    /// Fraction of elements exactly equal (in code units).
    pub elem_exact_rate: f64,
    /// Mean `|client − voyage|` in code units.
    pub mean_abs_diff: f64,
    pub max_abs_diff: i32,
    /// Mean per-row cosine similarity between the code vectors. Cosine is
    /// invariant to per-row positive scales, so this equals the cosine of the
    /// dequantized rows regardless of scale conventions; all-zero rows count
    /// as similarity 0.
    pub mean_cos_sim: f64,
}

/// Compare client int8 rows (with their per-row scales) against Voyage int8
/// rows. Voyage publishes no scale, so agreement is reported in code units
/// plus scale-invariant cosine; `client_scales` is validated against the row
/// count as a shape check.
pub fn int8_parity(client: &[i8], client_scales: &[f32], voyage: &[i8], dim: usize) -> Int8Parity {
    assert!(dim > 0, "dim must be > 0");
    assert_eq!(client.len(), voyage.len(), "blocks differ in size");
    assert!(client.len() % dim == 0, "blocks are not whole rows of dim {dim}");
    let vectors = client.len() / dim;
    assert_eq!(client_scales.len(), vectors, "expected one scale per row");

    let mut exact = 0u64;
    let mut abs_sum = 0u64;
    let mut max_abs = 0i32;
    let mut cos_sum = 0.0f64;
    for (c, v) in client.chunks_exact(dim).zip(voyage.chunks_exact(dim)) {
        let (mut dot, mut nc, mut nv) = (0f64, 0f64, 0f64);
        for (&x, &y) in c.iter().zip(v) {
            let d = (x as i32 - y as i32).abs();
            exact += (d == 0) as u64;
            abs_sum += d as u64;
            max_abs = max_abs.max(d);
            dot += x as f64 * y as f64;
            nc += (x as f64) * (x as f64);
            nv += (y as f64) * (y as f64);
        }
        if nc > 0.0 && nv > 0.0 {
            cos_sum += dot / (nc.sqrt() * nv.sqrt());
        }
    }
    let elems = (vectors * dim) as f64;
    Int8Parity {
        vectors,
        elem_exact_rate: if vectors == 0 { 1.0 } else { exact as f64 / elems },
        mean_abs_diff: if vectors == 0 { 0.0 } else { abs_sum as f64 / elems },
        max_abs_diff: max_abs,
        mean_cos_sim: if vectors == 0 { 1.0 } else { cos_sum / vectors as f64 },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offset_binary_conversion_is_add_128_and_involutive() {
        for v in i8::MIN..=i8::MAX {
            let mut out = [0u8];
            voyage_offset_binary_to_codes(&[v], &mut out);
            assert_eq!(out[0] as i16, v as i16 + 128, "value {v}");
        }
        // XOR 0x80 twice is the identity.
        let src: Vec<i8> = (i8::MIN..=i8::MAX).collect();
        let mut once = vec![0u8; src.len()];
        voyage_offset_binary_to_codes(&src, &mut once);
        let back: Vec<i8> = once.iter().map(|&b| (b ^ 0x80) as i8).collect();
        assert_eq!(back, src);
    }

    #[test]
    fn binary_parity_counts_planted_diffs() {
        let dim = 16; // 2-byte rows
        let client = vec![0b1010_1010u8, 0b1111_0000, 0b1010_1010, 0b1111_0000];
        let mut voyage = client.clone();
        voyage[2] ^= 0b0000_0011; // 2 differing bits in row 1
        let p = binary_parity(&client, &voyage, dim);
        assert_eq!(p.vectors, 2);
        assert_eq!(p.max_row_mismatch, 2);
        assert_eq!(p.identical_vector_rate, 0.5);
        assert_eq!(p.bit_mismatch_rate, 2.0 / 32.0);
        let clean = binary_parity(&client, &client, dim);
        assert_eq!(clean.bit_mismatch_rate, 0.0);
        assert_eq!(clean.identical_vector_rate, 1.0);
    }

    #[test]
    fn int8_parity_stats_and_scale_invariant_cosine() {
        let dim = 4;
        let client: Vec<i8> = vec![10, -20, 30, -40, 1, 2, 3, 4];
        // Row 0 proportional to client row 0 (cosine 1); row 1 has one off-by-2.
        let voyage: Vec<i8> = vec![20, -40, 60, -80, 1, 2, 3, 6];
        let p = int8_parity(&client, &[0.5, 0.25], &voyage, dim);
        assert_eq!(p.vectors, 2);
        assert_eq!(p.max_abs_diff, 40);
        assert_eq!(p.elem_exact_rate, 3.0 / 8.0);
        // Row 0 cosine is exactly 1; row 1 cosine = 38 / (√30·√50).
        let cos1 = 38.0 / (30f64.sqrt() * 50f64.sqrt());
        assert!((p.mean_cos_sim - (1.0 + cos1) / 2.0).abs() < 1e-12);
    }

    #[test]
    fn empty_blocks_are_degenerate_but_defined() {
        let p = binary_parity(&[], &[], 8);
        assert_eq!(p.vectors, 0);
        assert_eq!(p.bit_mismatch_rate, 0.0);
        let q = int8_parity(&[], &[], &[], 4);
        assert_eq!(q.vectors, 0);
        assert_eq!(q.elem_exact_rate, 1.0);
    }
}
