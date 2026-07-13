//! Sign-bit binary packing (MSB-first; see crate::kernels module docs).

use crate::kernels::binary_code_len;

/// Pack sign bits MSB-first: bit `i` is set iff `src[i] > 0.0`. Padding bits
/// in the final byte are zeroed. `out.len()` must be `binary_code_len(src.len())`.
pub fn pack_sign_bits(src: &[f32], out: &mut [u8]) {
    assert_eq!(
        out.len(),
        binary_code_len(src.len()),
        "output must be binary_code_len(dim) bytes"
    );
    out.fill(0);
    for (i, &x) in src.iter().enumerate() {
        if x > 0.0 {
            out[i / 8] |= 1 << (7 - (i % 8));
        }
    }
}

/// Allocating convenience for [`pack_sign_bits`].
pub fn pack_sign_bits_vec(src: &[f32]) -> Vec<u8> {
    let mut out = vec![0u8; binary_code_len(src.len())];
    pack_sign_bits(src, &mut out);
    out
}

/// Reconstruct the ±1.0 vector a code represents (set bit → `+1.0`, clear →
/// `-1.0`). `code.len()` must be `binary_code_len(out.len())`.
pub fn unpack_sign_bits(code: &[u8], out: &mut [f32]) {
    assert_eq!(
        code.len(),
        binary_code_len(out.len()),
        "code must be binary_code_len(dim) bytes"
    );
    for (i, o) in out.iter_mut().enumerate() {
        let bit = (code[i / 8] >> (7 - (i % 8))) & 1;
        *o = if bit == 1 { 1.0 } else { -1.0 };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packs_msb_first() {
        // dims 0..8 land in byte 0, bit 7 down to bit 0.
        let src = [1.0f32, -1.0, 0.5, 2.0, -0.1, -3.0, -0.5, 1.0];
        assert_eq!(pack_sign_bits_vec(&src), vec![0b1011_0001]);
    }

    #[test]
    fn zero_and_nan_pack_as_clear() {
        assert_eq!(pack_sign_bits_vec(&[0.0, -0.0, f32::NAN, 1.0]), vec![0b0001_0000]);
    }

    #[test]
    fn padding_bits_are_zero() {
        // d=13 → 2 bytes, low 3 bits of the last byte are padding.
        let code = pack_sign_bits_vec(&[1.0f32; 13]);
        assert_eq!(code, vec![0b1111_1111, 0b1111_1000]);
    }

    #[test]
    fn unpack_recovers_signs() {
        let src = [0.3f32, -0.7, 0.0, 5.0, -0.0, -1e-30, 1e-30, 2.0, -2.0];
        let code = pack_sign_bits_vec(&src);
        let mut out = vec![0.0f32; src.len()];
        unpack_sign_bits(&code, &mut out);
        for (i, (&x, &u)) in src.iter().zip(&out).enumerate() {
            let expected = if x > 0.0 { 1.0 } else { -1.0 };
            assert_eq!(u, expected, "dim {i}");
        }
    }

    #[test]
    #[should_panic(expected = "binary_code_len")]
    fn wrong_output_len_panics() {
        pack_sign_bits(&[1.0; 9], &mut [0u8; 1]);
    }
}
