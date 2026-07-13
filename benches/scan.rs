//! Flat-scan latency: representation × filter selectivity at k = 48 (the
//! production k), plus n-scaling and rerank micro-benches. Queries rotate per
//! iteration so the query row does not become unrealistically L1-resident.

use std::time::Duration;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use guksu::rng::SplitMix64;
use guksu::{BinaryBlock, Bitset, F32Block, I8Block, I8Query, Scorer, ViewQuery};

const DIM: usize = 1024;
const K: usize = 48;
const N: usize = 100_000;
const QUERIES: usize = 64;

struct Data {
    f32s: F32Block,
    i8s: I8Block,
    bins: BinaryBlock,
    q_f32: Vec<Vec<f32>>,
    q_i8: Vec<(Vec<i8>, f32)>,
    q_bits: Vec<Vec<u8>>,
}

fn setup(n: usize) -> Data {
    let mut rng = SplitMix64::new(0x5CA7);
    let flat: Vec<f32> = (0..n * DIM).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
    let f32s = F32Block::from_flat(&flat, DIM);
    let i8s = I8Block::from_f32_per_vector(&f32s);
    let bins = BinaryBlock::from_f32(&f32s);
    let q_f32: Vec<Vec<f32>> = (0..QUERIES)
        .map(|_| (0..DIM).map(|_| rng.next_f32() * 2.0 - 1.0).collect())
        .collect();
    let q_i8 = q_f32
        .iter()
        .map(|q| {
            let s = guksu::quant::max_abs_scale(q);
            (guksu::quant::quantize_i8_vec(q, s), s)
        })
        .collect();
    let q_bits = q_f32
        .iter()
        .map(|q| guksu::quant::pack_sign_bits_vec(q))
        .collect();
    Data {
        f32s,
        i8s,
        bins,
        q_f32,
        q_i8,
        q_bits,
    }
}

fn bench_scan_representations(c: &mut Criterion) {
    let data = setup(N);
    let filters: Vec<(String, Option<Bitset>)> = vec![
        ("s1.00".into(), None),
        ("s0.10".into(), Some(Bitset::random(N, 0.10, 1))),
        ("s0.02".into(), Some(Bitset::random(N, 0.02, 2))),
    ];
    let mut g = c.benchmark_group(format!("scan_n{N}_k{K}"));
    for (fname, filter) in &filters {
        let f = filter.as_ref();
        let mut q = 0usize;
        g.bench_function(BenchmarkId::new("f32", fname), |b| {
            b.iter(|| {
                let r = Scorer::F32 {
                    query: ViewQuery {
                        view: data.f32s.view(),
                        query: black_box(&data.q_f32[q]),
                    },
                }
                .top_k(K, f)
                .unwrap();
                q = (q + 1) % QUERIES;
                black_box(r)
            })
        });
        g.bench_function(BenchmarkId::new("i8", fname), |b| {
            b.iter(|| {
                let (qi, qs) = &data.q_i8[q];
                let r = Scorer::I8 {
                    query: ViewQuery {
                        view: data.i8s.view(),
                        query: I8Query {
                            codes: black_box(qi),
                            scale: *qs,
                        },
                    },
                }
                .top_k(K, f)
                .unwrap();
                q = (q + 1) % QUERIES;
                black_box(r)
            })
        });
        g.bench_function(BenchmarkId::new("bin", fname), |b| {
            b.iter(|| {
                let r = Scorer::Bin {
                    query: ViewQuery {
                        view: data.bins.view(),
                        query: black_box(&data.q_bits[q]),
                    },
                }
                .top_k(K, f)
                .unwrap();
                q = (q + 1) % QUERIES;
                black_box(r)
            })
        });
        g.bench_function(BenchmarkId::new("f32xbin", fname), |b| {
            b.iter(|| {
                let r = Scorer::F32Bin {
                    query: ViewQuery {
                        view: data.bins.view(),
                        query: black_box(&data.q_f32[q]),
                    },
                }
                .top_k(K, f)
                .unwrap();
                q = (q + 1) % QUERIES;
                black_box(r)
            })
        });
        g.bench_function(BenchmarkId::new("i8xbin", fname), |b| {
            b.iter(|| {
                let (qi, qs) = &data.q_i8[q];
                let r = Scorer::I8Bin {
                    query: ViewQuery {
                        view: data.bins.view(),
                        query: I8Query {
                            codes: black_box(qi),
                            scale: *qs,
                        },
                    },
                }
                .top_k(K, f)
                .unwrap();
                q = (q + 1) % QUERIES;
                black_box(r)
            })
        });
    }
    g.finish();
}

fn bench_scan_scaling(c: &mut Criterion) {
    let mut g = c.benchmark_group("scan_scaling_f32xbin");
    for n in [10_000usize, 100_000] {
        let data = setup(n);
        let mut q = 0usize;
        g.bench_function(BenchmarkId::from_parameter(n), |b| {
            b.iter(|| {
                let r = Scorer::F32Bin {
                    query: ViewQuery {
                        view: data.bins.view(),
                        query: black_box(&data.q_f32[q]),
                    },
                }
                .top_k(K, None)
                .unwrap();
                q = (q + 1) % QUERIES;
                black_box(r)
            })
        });
    }
    g.finish();
}

fn bench_rerank(c: &mut Criterion) {
    let data = setup(N);
    // A realistic 8×k coarse pool per query.
    let pools: Vec<Vec<u32>> = data
        .q_bits
        .iter()
        .map(|qb| {
            let pool = Scorer::Bin {
                query: ViewQuery {
                    view: data.bins.view(),
                    query: qb,
                },
            }
            .top_k(8 * K, None)
            .unwrap();
            pool.iter().map(|h| h.id).collect()
        })
        .collect();
    let mut g = c.benchmark_group("rerank_r384_k48");
    let mut q = 0usize;
    g.bench_function("f32", |b| {
        b.iter(|| {
            let r = Scorer::F32 {
                query: ViewQuery {
                    view: data.f32s.view(),
                    query: black_box(&data.q_f32[q]),
                },
            }
            .rerank(&pools[q], K)
            .unwrap();
            q = (q + 1) % QUERIES;
            black_box(r)
        })
    });
    g.bench_function("i8", |b| {
        b.iter(|| {
            let (qi, qs) = &data.q_i8[q];
            let r = Scorer::I8 {
                query: ViewQuery {
                    view: data.i8s.view(),
                    query: I8Query {
                        codes: black_box(qi),
                        scale: *qs,
                    },
                },
            }
            .rerank(&pools[q], K)
            .unwrap();
            q = (q + 1) % QUERIES;
            black_box(r)
        })
    });
    g.finish();
}

fn config() -> Criterion {
    Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(2))
}

criterion_group! {
    name = benches;
    config = config();
    targets = bench_scan_representations, bench_scan_scaling, bench_rerank
}
criterion_main!(benches);
