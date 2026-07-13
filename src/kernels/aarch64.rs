//! NEON backend (aarch64). NEON is baseline on every aarch64 std target; the
//! `dotprod` extension (ARMv8.2, present on all Apple Silicon) upgrades the
//! two int8 kernels — the tables differ only in those two entries.
//!
//! # Safety model
//!
//! Table entries are safe fns: they validate lengths, then enter one unsafe
//! region whose preconditions are (a) the validated lengths (every pointer
//! read is bounded by the loop conditions) and (b), for the `*_dotprod` fns
//! only, presence of the `dotprod` target feature — guaranteed because
//! `TABLE_NEON_DOTPROD` is only handed out after
//! `is_aarch64_feature_detected!("dotprod")` succeeds.
//!
//! # NEON cheat sheet
//!
//! Intrinsic names decode as `v<op><mods>[q]_<lane type>`: `q` = 128-bit
//! register, `l` = long (result lanes widen), `p` = pairwise, trailing `v` =
//! reduce across the vector, `_n` = scalar operand, `_high` = upper half.
//! Everything used here:
//!
//! - `vld1q_{f32,s8,u8,s32}` — unaligned 16-byte load: 4×f32/i32, 16×i8/u8
//! - `vdupq_n_*`, `vdup_n_u8` — broadcast one scalar to every lane
//! - `vaddq_*`, `vandq_*`, `veorq_u8` — lane-wise add / AND / XOR
//! - `vfmaq_f32(acc, a, b)` — fused `acc[i] + a[i]·b[i]` per f32 lane
//! - `vmull_s8`, `vmull_high_s8` — widening multiply: 8×(i8·i8) → 8×i16
//! - `vpadalq_s16(acc, x)` — pairwise fold: `acc[i] += x[2i] + x[2i+1]`,
//!   i16 pairs into i32 lanes (`vpadalq_u8` likewise folds u8 into u16)
//! - `vaddvq_f32`, `vaddvq_s32` — sum all lanes to one scalar
//! - `vaddlvq_u16` — widening sum of all lanes → u32
//! - `vcntq_u8` — per-byte popcount
//! - `vtstq_u8(a, b)` — per lane: `(a & b) != 0` → 0xFF, else 0x00
//! - `vshlq_u32(x, s)` — shift each lane left by its own amount `s[i]`
//! - `vget_low_s8`, `vcombine_u8` — split / join 64-bit register halves
//! - `vreinterpretq_*` — bit-cast between lane views (free)

use std::arch::aarch64::*;
use std::arch::asm;

use super::{Kernels, check_code_len, check_equal_len};

/// `sdot acc.4s, a.16b, b.16b` via inline asm — the dotprod intrinsic
/// (`vdotq_s32`) is still unstable (`stdarch_neon_dotprod`); the instruction
/// itself is not. Exact over the full i8 domain including −128.
///
/// # Safety
/// The CPU must support the `dotprod` feature.
#[inline(always)]
unsafe fn sdot(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
    let mut out = acc;
    // SAFETY: register-only (nomem), deterministic (pure); caller guarantees
    // the dotprod feature is present.
    unsafe {
        asm!(
            "sdot {acc:v}.4s, {a:v}.16b, {b:v}.16b",
            acc = inout(vreg) out,
            a = in(vreg) a,
            b = in(vreg) b,
            options(pure, nomem, nostack)
        );
    }
    out
}

pub(super) static TABLE_NEON: Kernels = Kernels {
    backend: "neon",
    dot_f32,
    dot_i8: dot_i8_mull,
    hamming,
    dot_f32_bin,
    dot_i8_bin: dot_i8_bin_mull,
};

pub(super) static TABLE_NEON_DOTPROD: Kernels = Kernels {
    backend: "neon_dotprod",
    dot_f32,
    dot_i8: dot_i8_dotprod,
    hamming,
    dot_f32_bin,
    dot_i8_bin: dot_i8_bin_dotprod,
};

fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    check_equal_len(a.len(), b.len());
    let n = a.len();
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    // SAFETY: every read is at an index < n, guarded by the loop conditions.
    unsafe {
        // 16 floats/iter over four independent FMA chains — a single
        // accumulator would stall on FMA latency between iterations.
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);
        let mut i = 0;
        while i + 16 <= n {
            acc0 = vfmaq_f32(acc0, vld1q_f32(pa.add(i)), vld1q_f32(pb.add(i)));
            acc1 = vfmaq_f32(acc1, vld1q_f32(pa.add(i + 4)), vld1q_f32(pb.add(i + 4)));
            acc2 = vfmaq_f32(acc2, vld1q_f32(pa.add(i + 8)), vld1q_f32(pb.add(i + 8)));
            acc3 = vfmaq_f32(acc3, vld1q_f32(pa.add(i + 12)), vld1q_f32(pb.add(i + 12)));
            i += 16;
        }
        while i + 4 <= n {
            acc0 = vfmaq_f32(acc0, vld1q_f32(pa.add(i)), vld1q_f32(pb.add(i)));
            i += 4;
        }
        let mut sum = vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
        while i < n {
            sum += *pa.add(i) * *pb.add(i);
            i += 1;
        }
        sum
    }
}

fn dot_i8_mull(a: &[i8], b: &[i8]) -> i32 {
    check_equal_len(a.len(), b.len());
    let n = a.len();
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    // SAFETY: reads bounded by the loop conditions. Exactness: i8×i8 products
    // fit i16 (|x| ≤ 16384); vpadalq_s16 widens each pair sum to i32.
    unsafe {
        // 16 products/iter: vmull widens each 8-lane half to i16 exactly,
        // vpadalq folds the i16 pairs into the i32 accumulators.
        let mut acc0 = vdupq_n_s32(0);
        let mut acc1 = vdupq_n_s32(0);
        let mut i = 0;
        while i + 16 <= n {
            let va = vld1q_s8(pa.add(i));
            let vb = vld1q_s8(pb.add(i));
            acc0 = vpadalq_s16(acc0, vmull_s8(vget_low_s8(va), vget_low_s8(vb)));
            acc1 = vpadalq_s16(acc1, vmull_high_s8(va, vb));
            i += 16;
        }
        let mut sum = vaddvq_s32(vaddq_s32(acc0, acc1));
        while i < n {
            sum += *pa.add(i) as i32 * *pb.add(i) as i32;
            i += 1;
        }
        sum
    }
}

fn dot_i8_dotprod(a: &[i8], b: &[i8]) -> i32 {
    check_equal_len(a.len(), b.len());
    let n = a.len();
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    // SAFETY: reads bounded by the loop conditions; sdot's dotprod requirement
    // is guaranteed because this table is only installed post-detection.
    unsafe {
        // 64 products/iter: each sdot fuses 16 i8·i8 products straight into
        // 4 i32 lanes, so there is no widening step to interleave.
        let mut acc0 = vdupq_n_s32(0);
        let mut acc1 = vdupq_n_s32(0);
        let mut acc2 = vdupq_n_s32(0);
        let mut acc3 = vdupq_n_s32(0);
        let mut i = 0;
        while i + 64 <= n {
            acc0 = sdot(acc0, vld1q_s8(pa.add(i)), vld1q_s8(pb.add(i)));
            acc1 = sdot(acc1, vld1q_s8(pa.add(i + 16)), vld1q_s8(pb.add(i + 16)));
            acc2 = sdot(acc2, vld1q_s8(pa.add(i + 32)), vld1q_s8(pb.add(i + 32)));
            acc3 = sdot(acc3, vld1q_s8(pa.add(i + 48)), vld1q_s8(pb.add(i + 48)));
            i += 64;
        }
        while i + 16 <= n {
            acc0 = sdot(acc0, vld1q_s8(pa.add(i)), vld1q_s8(pb.add(i)));
            i += 16;
        }
        let mut sum = vaddvq_s32(vaddq_s32(vaddq_s32(acc0, acc1), vaddq_s32(acc2, acc3)));
        while i < n {
            sum += *pa.add(i) as i32 * *pb.add(i) as i32;
            i += 1;
        }
        sum
    }
}

fn hamming(a: &[u8], b: &[u8]) -> u32 {
    check_equal_len(a.len(), b.len());
    let n = a.len();
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    // SAFETY: reads bounded by the loop conditions. The u16 lane accumulator
    // is drained every ≤512 iterations (each adds ≤16 per lane), far below
    // u16 saturation, so rows of any length count exactly.
    unsafe {
        // Per 16 bytes: XOR, per-byte popcount, fold byte pairs into u16
        // lanes; each block's lane sums then drain into the u32 total.
        let mut total = 0u32;
        let mut i = 0;
        while i + 16 <= n {
            let iters = ((n - i) / 16).min(512);
            let mut acc16 = vdupq_n_u16(0);
            for _ in 0..iters {
                let x = veorq_u8(vld1q_u8(pa.add(i)), vld1q_u8(pb.add(i)));
                acc16 = vpadalq_u8(acc16, vcntq_u8(x));
                i += 16;
            }
            total += vaddlvq_u16(acc16);
        }
        while i < n {
            total += (*pa.add(i) ^ *pb.add(i)).count_ones();
            i += 1;
        }
        total
    }
}

/// Per-lane left shifts that move code bit `7 - lane` (MSB-first) to bit 31.
static SHIFTS_LO: [i32; 4] = [24, 25, 26, 27];
static SHIFTS_HI: [i32; 4] = [28, 29, 30, 31];

fn dot_f32_bin(q: &[f32], code: &[u8]) -> f32 {
    check_code_len(q.len(), code.len());
    let n = q.len();
    let full_bytes = n / 8;
    let pq = q.as_ptr();
    // SAFETY: byte b touches q[8b .. 8b+8], b < full_bytes = n/8; the bit tail
    // indexes q and code in-bounds by construction.
    unsafe {
        // One code byte drives 8 query floats: turn the byte into per-lane
        // sign-bit masks and XOR-flip the lanes — adds only, no multiplies.
        let shifts_lo = vld1q_s32(SHIFTS_LO.as_ptr());
        let shifts_hi = vld1q_s32(SHIFTS_HI.as_ptr());
        let sign = vdupq_n_u32(0x8000_0000);
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        for b in 0..full_bytes {
            // Bit clear ⇒ flip: broadcast !byte, move each lane's bit to the
            // f32 sign position, XOR onto the query lanes (exact sign flip).
            let nc = vdupq_n_u32(!(*code.get_unchecked(b) as u32));
            let m0 = vandq_u32(vshlq_u32(nc, shifts_lo), sign);
            let m1 = vandq_u32(vshlq_u32(nc, shifts_hi), sign);
            let q0 = vld1q_f32(pq.add(b * 8));
            let q1 = vld1q_f32(pq.add(b * 8 + 4));
            acc0 = vaddq_f32(acc0, vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(q0), m0)));
            acc1 = vaddq_f32(acc1, vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(q1), m1)));
        }
        let mut sum = vaddvq_f32(vaddq_f32(acc0, acc1));
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

/// MSB-first per-lane bit masks for two broadcast code bytes.
static BIT_MASKS: [u8; 16] = [128, 64, 32, 16, 8, 4, 2, 1, 128, 64, 32, 16, 8, 4, 2, 1];

/// Expand two code bytes into 16 i8 lanes of ±1.
#[inline(always)]
unsafe fn expand_signs(c0: u8, c1: u8) -> int8x16_t {
    // SAFETY: caller is in a NEON context; BIT_MASKS is 16 bytes.
    unsafe {
        let bytes = vcombine_u8(vdup_n_u8(c0), vdup_n_u8(c1));
        let set = vtstq_u8(bytes, vld1q_u8(BIT_MASKS.as_ptr())); // 0xFF where bit set
        let m = vreinterpretq_s8_u8(set); // -1 where set, 0 where clear
        vaddq_s8(vandq_s8(m, vdupq_n_s8(2)), vdupq_n_s8(-1)) // set → +1, clear → -1
    }
}

#[inline(always)]
unsafe fn dot_i8_bin_tail(q: &[i8], code: &[u8], from: usize) -> i32 {
    let mut sum = 0i32;
    for t in from..q.len() {
        let bit = (code[t >> 3] >> (7 - (t & 7))) & 1;
        if bit == 1 {
            sum += q[t] as i32;
        } else {
            sum -= q[t] as i32;
        }
    }
    sum
}

fn dot_i8_bin_mull(q: &[i8], code: &[u8]) -> i32 {
    check_code_len(q.len(), code.len());
    let n = q.len();
    let pairs = n / 16;
    let pq = q.as_ptr();
    // SAFETY: pair p touches q[16p .. 16p+16] and code[2p], code[2p+1];
    // p < pairs = n/16 keeps both in-bounds (code has ceil(n/8) bytes).
    unsafe {
        // 16 dims (2 code bytes)/iter: expand bits to ±1 lanes, then the
        // same widen-and-fold dot as dot_i8_mull.
        let mut acc = vdupq_n_s32(0);
        for p in 0..pairs {
            let signs = expand_signs(*code.get_unchecked(2 * p), *code.get_unchecked(2 * p + 1));
            let vq = vld1q_s8(pq.add(16 * p));
            acc = vpadalq_s16(acc, vmull_s8(vget_low_s8(vq), vget_low_s8(signs)));
            acc = vpadalq_s16(acc, vmull_high_s8(vq, signs));
        }
        vaddvq_s32(acc) + dot_i8_bin_tail(q, code, pairs * 16)
    }
}

fn dot_i8_bin_dotprod(q: &[i8], code: &[u8]) -> i32 {
    check_code_len(q.len(), code.len());
    let n = q.len();
    let pairs = n / 16;
    let pq = q.as_ptr();
    // SAFETY: same bounds argument as dot_i8_bin_mull; sdot's dotprod
    // requirement is guaranteed by post-detection table install.
    unsafe {
        // Same ±1 expansion, with sdot folding each 16-lane batch directly.
        let mut acc = vdupq_n_s32(0);
        for p in 0..pairs {
            let signs = expand_signs(*code.get_unchecked(2 * p), *code.get_unchecked(2 * p + 1));
            acc = sdot(acc, vld1q_s8(pq.add(16 * p)), signs);
        }
        vaddvq_s32(acc) + dot_i8_bin_tail(q, code, pairs * 16)
    }
}
