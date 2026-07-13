//! Portable reference implementations — the oracle every SIMD backend is
//! tested against. Deterministic: sequential accumulation, no FMA.

use super::{check_code_len, check_equal_len};

pub(super) fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    check_equal_len(a.len(), b.len());
    let mut acc = 0.0f32;
    for (&x, &y) in a.iter().zip(b) {
        acc += x * y;
    }
    acc
}

pub(super) fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
    check_equal_len(a.len(), b.len());
    let mut acc = 0i32;
    for (&x, &y) in a.iter().zip(b) {
        acc += x as i32 * y as i32;
    }
    acc
}

pub(super) fn hamming(a: &[u8], b: &[u8]) -> u32 {
    check_equal_len(a.len(), b.len());
    let mut acc = 0u32;
    let mut wa = a.chunks_exact(8);
    let mut wb = b.chunks_exact(8);
    for (ca, cb) in (&mut wa).zip(&mut wb) {
        let x = u64::from_ne_bytes(ca.try_into().unwrap());
        let y = u64::from_ne_bytes(cb.try_into().unwrap());
        acc += (x ^ y).count_ones();
    }
    for (&x, &y) in wa.remainder().iter().zip(wb.remainder()) {
        acc += (x ^ y).count_ones();
    }
    acc
}

pub(super) fn dot_f32_bin(q: &[f32], code: &[u8]) -> f32 {
    check_code_len(q.len(), code.len());
    let mut acc = 0.0f32;
    for (i, &qi) in q.iter().enumerate() {
        if (code[i >> 3] >> (7 - (i & 7))) & 1 == 1 {
            acc += qi;
        } else {
            acc -= qi;
        }
    }
    acc
}

pub(super) fn dot_i8_bin(q: &[i8], code: &[u8]) -> i32 {
    check_code_len(q.len(), code.len());
    let mut acc = 0i32;
    for (i, &qi) in q.iter().enumerate() {
        if (code[i >> 3] >> (7 - (i & 7))) & 1 == 1 {
            acc += qi as i32;
        } else {
            acc -= qi as i32;
        }
    }
    acc
}
