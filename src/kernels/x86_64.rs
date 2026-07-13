//! AVX2 backend (x86_64). Requires AVX2 + FMA + POPCNT — none implies the
//! others as target features, so detection checks all three (every real AVX2
//! CPU has them).
//!
//! int8 kernels deliberately use sign-extend (`vpmovsxbw`) + `vpmaddwd`
//! instead of `maddubs`/`sign_epi8`: those saturate in i16 and mishandle
//! −128 (`abs_epi8(-128) == -128`), and rows ingested from Voyage int8 can
//! contain −128. The widened path is exact over the full domain.
//!
//! # Safety model
//!
//! Table entries are safe fns: they validate lengths, then call an `unsafe`
//! `#[target_feature]` inner fn. Preconditions: (a) the validated lengths
//! (every read is bounded by the loop conditions), and (b) AVX2/FMA/POPCNT
//! presence — guaranteed because `TABLE_AVX2` is only handed out after
//! `is_x86_feature_detected!` succeeds for all three.

use std::arch::x86_64::*;

use super::{Kernels, check_code_len, check_equal_len};

pub(super) static TABLE_AVX2: Kernels = Kernels {
    backend: "avx2",
    dot_f32,
    dot_i8,
    hamming,
    dot_f32_bin,
    dot_i8_bin,
};

fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    check_equal_len(a.len(), b.len());
    // SAFETY: lengths validated; features guaranteed by post-detection install.
    unsafe { dot_f32_avx2(a, b) }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn dot_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    // SAFETY: every read is at an index < n, guarded by the loop conditions.
    unsafe {
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();
        let mut i = 0;
        while i + 32 <= n {
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)), acc0);
            acc1 = _mm256_fmadd_ps(
                _mm256_loadu_ps(pa.add(i + 8)),
                _mm256_loadu_ps(pb.add(i + 8)),
                acc1,
            );
            acc2 = _mm256_fmadd_ps(
                _mm256_loadu_ps(pa.add(i + 16)),
                _mm256_loadu_ps(pb.add(i + 16)),
                acc2,
            );
            acc3 = _mm256_fmadd_ps(
                _mm256_loadu_ps(pa.add(i + 24)),
                _mm256_loadu_ps(pb.add(i + 24)),
                acc3,
            );
            i += 32;
        }
        while i + 8 <= n {
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)), acc0);
            i += 8;
        }
        let sum256 = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
        let s4 = _mm_add_ps(_mm256_castps256_ps128(sum256), _mm256_extractf128_ps(sum256, 1));
        let s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
        let s1 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 0b01));
        let mut sum = _mm_cvtss_f32(s1);
        while i < n {
            sum += *pa.add(i) * *pb.add(i);
            i += 1;
        }
        sum
    }
}

fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
    check_equal_len(a.len(), b.len());
    // SAFETY: lengths validated; features guaranteed by post-detection install.
    unsafe { dot_i8_avx2(a, b) }
}

/// Reduce an i32x8 accumulator to a scalar (fixed tree; integers are exact).
#[target_feature(enable = "avx2")]
unsafe fn reduce_i32(acc: __m256i) -> i32 {
    // SAFETY: register-only ops.
    unsafe {
        let s = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
        let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b00_00_11_10));
        let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b00_00_00_01));
        _mm_cvtsi128_si32(s)
    }
}

#[target_feature(enable = "avx2")]
unsafe fn dot_i8_avx2(a: &[i8], b: &[i8]) -> i32 {
    let n = a.len();
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    // SAFETY: reads bounded by the loop conditions. Exactness: madd of
    // sign-extended i8 values cannot saturate (|product pair sum| ≤ 2·16384).
    unsafe {
        let mut acc0 = _mm256_setzero_si256();
        let mut acc1 = _mm256_setzero_si256();
        let mut i = 0;
        while i + 32 <= n {
            let wa0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(pa.add(i) as *const __m128i));
            let wb0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(pb.add(i) as *const __m128i));
            acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(wa0, wb0));
            let wa1 = _mm256_cvtepi8_epi16(_mm_loadu_si128(pa.add(i + 16) as *const __m128i));
            let wb1 = _mm256_cvtepi8_epi16(_mm_loadu_si128(pb.add(i + 16) as *const __m128i));
            acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(wa1, wb1));
            i += 32;
        }
        while i + 16 <= n {
            let wa = _mm256_cvtepi8_epi16(_mm_loadu_si128(pa.add(i) as *const __m128i));
            let wb = _mm256_cvtepi8_epi16(_mm_loadu_si128(pb.add(i) as *const __m128i));
            acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(wa, wb));
            i += 16;
        }
        let mut sum = reduce_i32(_mm256_add_epi32(acc0, acc1));
        while i < n {
            sum += *pa.add(i) as i32 * *pb.add(i) as i32;
            i += 1;
        }
        sum
    }
}

fn hamming(a: &[u8], b: &[u8]) -> u32 {
    check_equal_len(a.len(), b.len());
    // SAFETY: lengths validated; POPCNT guaranteed by post-detection install.
    unsafe { hamming_popcnt(a, b) }
}

#[target_feature(enable = "popcnt")]
unsafe fn hamming_popcnt(a: &[u8], b: &[u8]) -> u32 {
    let n = a.len();
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    let words = n / 8;
    // SAFETY: word w reads bytes [8w, 8w+8) with w < words = n/8; unaligned
    // u64 reads are done via read_unaligned. Four independent accumulators
    // dodge the pre-Ice-Lake popcnt destination false dependency.
    unsafe {
        let (mut c0, mut c1, mut c2, mut c3) = (0u64, 0u64, 0u64, 0u64);
        let load = |p: *const u8, w: usize| (p.add(w * 8) as *const u64).read_unaligned();
        let mut w = 0;
        while w + 4 <= words {
            c0 += ((load(pa, w) ^ load(pb, w)).count_ones()) as u64;
            c1 += ((load(pa, w + 1) ^ load(pb, w + 1)).count_ones()) as u64;
            c2 += ((load(pa, w + 2) ^ load(pb, w + 2)).count_ones()) as u64;
            c3 += ((load(pa, w + 3) ^ load(pb, w + 3)).count_ones()) as u64;
            w += 4;
        }
        while w < words {
            c0 += ((load(pa, w) ^ load(pb, w)).count_ones()) as u64;
            w += 1;
        }
        let mut total = (c0 + c1 + c2 + c3) as u32;
        let mut i = words * 8;
        while i < n {
            total += (*pa.add(i) ^ *pb.add(i)).count_ones();
            i += 1;
        }
        total
    }
}

/// Per-lane left shifts that move code bit `7 - lane` (MSB-first) to bit 31.
static SHIFTS: [i32; 8] = [24, 25, 26, 27, 28, 29, 30, 31];

fn dot_f32_bin(q: &[f32], code: &[u8]) -> f32 {
    check_code_len(q.len(), code.len());
    // SAFETY: lengths validated; features guaranteed by post-detection install.
    unsafe { dot_f32_bin_avx2(q, code) }
}

#[target_feature(enable = "avx2")]
unsafe fn dot_f32_bin_avx2(q: &[f32], code: &[u8]) -> f32 {
    let n = q.len();
    let full_bytes = n / 8;
    let pq = q.as_ptr();
    // SAFETY: byte b touches q[8b .. 8b+8], b < full_bytes = n/8; the bit tail
    // indexes q and code in-bounds by construction.
    unsafe {
        let shifts = _mm256_loadu_si256(SHIFTS.as_ptr() as *const __m256i);
        let sign = _mm256_set1_epi32(u32::MAX.wrapping_shl(31) as i32); // 0x8000_0000
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut b = 0;
        while b + 2 <= full_bytes {
            // Bit clear ⇒ flip: broadcast !byte, move each lane's bit to the
            // f32 sign position, XOR onto the query lanes (exact sign flip).
            let nc0 = _mm256_set1_epi32(!(*code.get_unchecked(b) as u32) as i32);
            let nc1 = _mm256_set1_epi32(!(*code.get_unchecked(b + 1) as u32) as i32);
            let m0 = _mm256_and_si256(_mm256_sllv_epi32(nc0, shifts), sign);
            let m1 = _mm256_and_si256(_mm256_sllv_epi32(nc1, shifts), sign);
            let q0 = _mm256_loadu_ps(pq.add(b * 8));
            let q1 = _mm256_loadu_ps(pq.add(b * 8 + 8));
            acc0 = _mm256_add_ps(acc0, _mm256_xor_ps(q0, _mm256_castsi256_ps(m0)));
            acc1 = _mm256_add_ps(acc1, _mm256_xor_ps(q1, _mm256_castsi256_ps(m1)));
            b += 2;
        }
        if b < full_bytes {
            let nc = _mm256_set1_epi32(!(*code.get_unchecked(b) as u32) as i32);
            let m = _mm256_and_si256(_mm256_sllv_epi32(nc, shifts), sign);
            let qv = _mm256_loadu_ps(pq.add(b * 8));
            acc0 = _mm256_add_ps(acc0, _mm256_xor_ps(qv, _mm256_castsi256_ps(m)));
            b += 1;
        }
        let sum256 = _mm256_add_ps(acc0, acc1);
        let s4 = _mm_add_ps(_mm256_castps256_ps128(sum256), _mm256_extractf128_ps(sum256, 1));
        let s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
        let s1 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 0b01));
        let mut sum = _mm_cvtss_f32(s1);
        for t in full_bytes * 8..n {
            let bit = (code[t >> 3] >> (7 - (t & 7))) & 1;
            if bit == 1 {
                sum += q[t];
            } else {
                sum -= q[t];
            }
        }
        sum
    }
}

/// Shuffle control replicating each of 4 code bytes across 8 lanes
/// (`_mm256_shuffle_epi8` indexes within each 128-bit half, and
/// `_mm256_set1_epi32` puts the same 4 bytes in both halves).
static SPREAD: [u8; 32] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
];
/// MSB-first per-lane bit masks, one byte's worth per 8 lanes.
static BIT_MASKS: [u8; 32] = [
    128, 64, 32, 16, 8, 4, 2, 1, 128, 64, 32, 16, 8, 4, 2, 1, 128, 64, 32, 16, 8, 4, 2, 1, 128,
    64, 32, 16, 8, 4, 2, 1,
];

fn dot_i8_bin(q: &[i8], code: &[u8]) -> i32 {
    check_code_len(q.len(), code.len());
    // SAFETY: lengths validated; features guaranteed by post-detection install.
    unsafe { dot_i8_bin_avx2(q, code) }
}

#[target_feature(enable = "avx2")]
unsafe fn dot_i8_bin_avx2(q: &[i8], code: &[u8]) -> i32 {
    let n = q.len();
    let quads = n / 32;
    let pq = q.as_ptr();
    // SAFETY: quad p touches q[32p .. 32p+32] and code[4p .. 4p+4];
    // p < quads = n/32 keeps both in-bounds (code has ceil(n/8) bytes).
    unsafe {
        let spread = _mm256_loadu_si256(SPREAD.as_ptr() as *const __m256i);
        let bits = _mm256_loadu_si256(BIT_MASKS.as_ptr() as *const __m256i);
        let two = _mm256_set1_epi8(2);
        let neg_one = _mm256_set1_epi8(-1);
        let mut acc0 = _mm256_setzero_si256();
        let mut acc1 = _mm256_setzero_si256();
        for p in 0..quads {
            let c = code.as_ptr().add(4 * p);
            let quad = u32::from_le_bytes([*c, *c.add(1), *c.add(2), *c.add(3)]);
            let bytes = _mm256_shuffle_epi8(_mm256_set1_epi32(quad as i32), spread);
            let set = _mm256_cmpeq_epi8(_mm256_and_si256(bytes, bits), bits); // 0xFF where set
            let signs = _mm256_add_epi8(_mm256_and_si256(set, two), neg_one); // set → +1, clear → -1
            let q0 = _mm_loadu_si128(pq.add(32 * p) as *const __m128i);
            let q1 = _mm_loadu_si128(pq.add(32 * p + 16) as *const __m128i);
            let s0 = _mm256_castsi256_si128(signs);
            let s1 = _mm256_extracti128_si256(signs, 1);
            acc0 = _mm256_add_epi32(
                acc0,
                _mm256_madd_epi16(_mm256_cvtepi8_epi16(q0), _mm256_cvtepi8_epi16(s0)),
            );
            acc1 = _mm256_add_epi32(
                acc1,
                _mm256_madd_epi16(_mm256_cvtepi8_epi16(q1), _mm256_cvtepi8_epi16(s1)),
            );
        }
        let mut sum = reduce_i32(_mm256_add_epi32(acc0, acc1));
        for t in quads * 32..n {
            let bit = (code[t >> 3] >> (7 - (t & 7))) & 1;
            if bit == 1 {
                sum += q[t] as i32;
            } else {
                sum -= q[t] as i32;
            }
        }
        sum
    }
}
