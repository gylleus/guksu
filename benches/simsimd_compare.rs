//! Performance cross-check against simsimd (C kernels) — never a runtime
//! dependency; enabled only via `cargo bench --features bench-simsimd`.
//! Informal bar: guksu within ~1.5× of simsimd on each kernel; a bigger gap
//! means a kernel is mis-written for the target.

use std::time::Duration;

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use guksu::kernels::Kernels;
use guksu::rng::SplitMix64;
use simsimd::{BinarySimilarity, SpatialSimilarity};

const DIM: usize = 1024;

fn bench_crosscheck(c: &mut Criterion) {
    let mut rng = SplitMix64::new(0x51D5);
    let af: Vec<f32> = (0..DIM).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
    let bf: Vec<f32> = (0..DIM).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
    let ai: Vec<i8> = (0..DIM).map(|_| (rng.next_range(255) as i16 - 127) as i8).collect();
    let bi: Vec<i8> = (0..DIM).map(|_| (rng.next_range(255) as i16 - 127) as i8).collect();
    let ab: Vec<u8> = (0..DIM / 8).map(|_| rng.next_range(256) as u8).collect();
    let bb: Vec<u8> = (0..DIM / 8).map(|_| rng.next_range(256) as u8).collect();

    let k = Kernels::detected();

    // Value sanity before timing anything.
    let g_dot = (k.dot_f32)(&af, &bf) as f64;
    let s_dot = f32::dot(&af, &bf).expect("simsimd f32 dot");
    assert!((g_dot - s_dot).abs() <= 1e-3 * g_dot.abs().max(1.0), "f32 dot: {g_dot} vs {s_dot}");
    let g_i8 = (k.dot_i8)(&ai, &bi) as f64;
    let s_i8 = i8::dot(&ai, &bi).expect("simsimd i8 dot");
    assert!((g_i8 - s_i8).abs() < 1.0, "i8 dot: {g_i8} vs {s_i8}");
    let g_h = (k.hamming)(&ab, &bb) as f64;
    let s_h = u8::hamming(&ab, &bb).expect("simsimd hamming");
    assert_eq!(g_h, s_h, "hamming: {g_h} vs {s_h}");

    let mut g = c.benchmark_group("crosscheck_simsimd");
    g.bench_function("dot_f32/guksu", |b| {
        b.iter(|| black_box((k.dot_f32)(black_box(&af), black_box(&bf))))
    });
    g.bench_function("dot_f32/simsimd", |b| {
        b.iter(|| black_box(f32::dot(black_box(&af), black_box(&bf))))
    });
    g.bench_function("dot_i8/guksu", |b| {
        b.iter(|| black_box((k.dot_i8)(black_box(&ai), black_box(&bi))))
    });
    g.bench_function("dot_i8/simsimd", |b| {
        b.iter(|| black_box(i8::dot(black_box(&ai), black_box(&bi))))
    });
    g.bench_function("hamming/guksu", |b| {
        b.iter(|| black_box((k.hamming)(black_box(&ab), black_box(&bb))))
    });
    g.bench_function("hamming/simsimd", |b| {
        b.iter(|| black_box(u8::hamming(black_box(&ab), black_box(&bb))))
    });
    g.finish();
}

fn config() -> Criterion {
    Criterion::default()
        .sample_size(50)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(2))
}

criterion_group! {
    name = benches;
    config = config();
    targets = bench_crosscheck
}
criterion_main!(benches);
