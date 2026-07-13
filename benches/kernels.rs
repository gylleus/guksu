//! Kernel throughput: every kernel × every CPU-supported backend × {hot, stream}.
//!
//! `hot` cycles 256 cache-resident rows (compute ceiling); `stream` walks a
//! 128 MB block (bandwidth ceiling). Throughput is doc-row bytes per op, so
//! criterion's GB/s column is directly comparable across representations —
//! the spec's sanity bar is stream rows/s scaling near the byte ratio
//! binary:int8:f32 ≈ 32:4:1 at dim 1024 (128 B / 1 KB / 4 KB rows).

use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use guksu::kernels::{Kernels, binary_code_len};
use guksu::quant::{max_abs_scale, pack_sign_bits, quantize_i8};
use guksu::rng::SplitMix64;

const DIM: usize = 1024;
const HOT_ROWS: usize = 256;
const STREAM_BYTES: usize = 128 << 20;

fn tables() -> Vec<(&'static str, &'static Kernels)> {
    Kernels::NAMES
        .iter()
        .filter_map(|&name| Kernels::by_name(name).map(|k| (name, k)))
        .collect()
}

fn f32_rows(rng: &mut SplitMix64, n: usize) -> Vec<f32> {
    (0..n * DIM).map(|_| rng.next_f32() * 2.0 - 1.0).collect()
}

fn i8_rows(rng: &mut SplitMix64, n: usize) -> Vec<i8> {
    (0..n * DIM).map(|_| (rng.next_range(255) as i16 - 127) as i8).collect()
}

fn bin_rows(rng: &mut SplitMix64, n: usize) -> Vec<u8> {
    // DIM = 1024 has no padding bits, so raw random bytes are valid codes.
    (0..n * binary_code_len(DIM)).map(|_| rng.next_range(256) as u8).collect()
}

/// Bench one kernel over hot and stream working sets for every backend.
#[allow(clippy::too_many_arguments)]
fn run_regimes<Q: ?Sized, R: ?Sized>(
    c: &mut Criterion,
    group: &str,
    row_bytes: usize,
    query: &Q,
    hot: &R,
    stream: &R,
    row: impl Fn(&R, usize) -> &R,
    rows_in: impl Fn(&R) -> usize,
    call: impl Fn(&'static Kernels, &Q, &R) -> f64 + Copy,
) {
    let mut g = c.benchmark_group(group);
    g.throughput(Throughput::Bytes(row_bytes as u64));
    for (name, k) in tables() {
        for (regime, data) in [("hot", hot), ("stream", stream)] {
            let rows = rows_in(data);
            g.bench_function(BenchmarkId::new(name, regime), |b| {
                let mut i = 0usize;
                b.iter(|| {
                    let r = call(k, black_box(query), black_box(row(data, i)));
                    i = (i + 1) % rows;
                    black_box(r)
                });
            });
        }
    }
    g.finish();
}

fn bench_dot_f32(c: &mut Criterion) {
    let mut rng = SplitMix64::new(0xBE7C_0001);
    let q = f32_rows(&mut rng, 1);
    let hot = f32_rows(&mut rng, HOT_ROWS);
    let stream = f32_rows(&mut rng, STREAM_BYTES / (DIM * 4));
    run_regimes(
        c,
        "dot_f32",
        DIM * 4,
        q.as_slice(),
        hot.as_slice(),
        stream.as_slice(),
        |d, i| &d[i * DIM..(i + 1) * DIM],
        |d| d.len() / DIM,
        |k, q, r| (k.dot_f32)(q, r) as f64,
    );
}

fn bench_dot_i8(c: &mut Criterion) {
    let mut rng = SplitMix64::new(0xBE7C_0002);
    let q = i8_rows(&mut rng, 1);
    let hot = i8_rows(&mut rng, HOT_ROWS);
    let stream = i8_rows(&mut rng, STREAM_BYTES / DIM);
    run_regimes(
        c,
        "dot_i8",
        DIM,
        q.as_slice(),
        hot.as_slice(),
        stream.as_slice(),
        |d, i| &d[i * DIM..(i + 1) * DIM],
        |d| d.len() / DIM,
        |k, q, r| (k.dot_i8)(q, r) as f64,
    );
}

fn bench_hamming(c: &mut Criterion) {
    let mut rng = SplitMix64::new(0xBE7C_0003);
    let code_len = binary_code_len(DIM);
    let q = bin_rows(&mut rng, 1);
    let hot = bin_rows(&mut rng, HOT_ROWS);
    let stream = bin_rows(&mut rng, STREAM_BYTES / code_len);
    run_regimes(
        c,
        "hamming",
        code_len,
        q.as_slice(),
        hot.as_slice(),
        stream.as_slice(),
        |d, i| &d[i * code_len..(i + 1) * code_len],
        |d| d.len() / code_len,
        |k, q, r| (k.hamming)(q, r) as f64,
    );
}

fn bench_dot_f32_bin(c: &mut Criterion) {
    let mut rng = SplitMix64::new(0xBE7C_0004);
    let code_len = binary_code_len(DIM);
    let q = f32_rows(&mut rng, 1);
    let hot = bin_rows(&mut rng, HOT_ROWS);
    let stream = bin_rows(&mut rng, STREAM_BYTES / code_len);
    run_regimes(
        c,
        "dot_f32_bin",
        code_len,
        q.as_slice(),
        hot.as_slice(),
        stream.as_slice(),
        |d, i| &d[i * code_len..(i + 1) * code_len],
        |d| d.len() / code_len,
        |k, q, r| (k.dot_f32_bin)(q, r) as f64,
    );
}

fn bench_dot_i8_bin(c: &mut Criterion) {
    let mut rng = SplitMix64::new(0xBE7C_0005);
    let code_len = binary_code_len(DIM);
    let q = i8_rows(&mut rng, 1);
    let hot = bin_rows(&mut rng, HOT_ROWS);
    let stream = bin_rows(&mut rng, STREAM_BYTES / code_len);
    run_regimes(
        c,
        "dot_i8_bin",
        code_len,
        q.as_slice(),
        hot.as_slice(),
        stream.as_slice(),
        |d, i| &d[i * code_len..(i + 1) * code_len],
        |d| d.len() / code_len,
        |k, q, r| (k.dot_i8_bin)(q, r) as f64,
    );
}

/// Documents the "hoist `Kernels::detected()` out of scan loops" guidance
/// with a number (per-call OnceLock load + indirect call vs hoisted table).
fn bench_dispatch_overhead(c: &mut Criterion) {
    let mut rng = SplitMix64::new(0xBE7C_0006);
    let d = 16;
    let a: Vec<f32> = (0..d).map(|_| rng.next_f32()).collect();
    let b_: Vec<f32> = (0..d).map(|_| rng.next_f32()).collect();
    let mut g = c.benchmark_group("dispatch_overhead");
    g.bench_function("hoisted_table", |b| {
        let k = Kernels::detected();
        b.iter(|| black_box((k.dot_f32)(black_box(&a), black_box(&b_))));
    });
    g.bench_function("free_fn_per_call", |b| {
        b.iter(|| black_box(guksu::kernels::dot_f32(black_box(&a), black_box(&b_))));
    });
    g.finish();
}

fn bench_quantize(c: &mut Criterion) {
    let mut rng = SplitMix64::new(0xBE7C_0007);
    let src = f32_rows(&mut rng, 1);
    let mut g = c.benchmark_group("quantize");
    g.throughput(Throughput::Bytes((DIM * 4) as u64));
    g.bench_function("pack_sign_bits", |b| {
        let mut out = vec![0u8; binary_code_len(DIM)];
        b.iter(|| {
            pack_sign_bits(black_box(&src), &mut out);
            black_box(out[0])
        });
    });
    g.bench_function("quantize_i8", |b| {
        let scale = max_abs_scale(&src);
        let mut out = vec![0i8; DIM];
        b.iter(|| {
            quantize_i8(black_box(&src), black_box(scale), &mut out);
            black_box(out[0])
        });
    });
    g.finish();
}

fn config() -> Criterion {
    Criterion::default()
        .sample_size(20)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(2))
}

criterion_group! {
    name = benches;
    config = config();
    targets = bench_dot_f32, bench_dot_i8, bench_hamming, bench_dot_f32_bin, bench_dot_i8_bin,
        bench_dispatch_overhead, bench_quantize
}
criterion_main!(benches);
