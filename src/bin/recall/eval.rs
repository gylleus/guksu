//! Config-matrix evaluation: quantized data prep, ground truth (+ cache),
//! recall math, and the cell grid itself.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use guksu::{
    BinaryBlock, Bitset, Block, F32Block, Hit, I8Block, I8Query, Scorer, StorageError, ViewQuery,
};

// ---------------------------------------------------------------- data prep

/// Everything quantized once and shared across the whole matrix.
pub struct Data {
    pub f32s: F32Block,
    pub i8s: I8Block,
    pub bins: BinaryBlock,
    pub q_f32: Vec<Vec<f32>>,
    pub q_i8: Vec<Vec<i8>>,
    pub q_scales: Vec<f32>,
    pub q_bits: Vec<Vec<u8>>,
}

impl Data {
    /// Fails if `corpus_flat` is not whole rows of `dim`.
    pub fn prepare(
        corpus_flat: &[f32],
        queries_flat: &[f32],
        dim: usize,
    ) -> Result<Data, StorageError> {
        let f32s = F32Block::from_flat(corpus_flat, dim)?;
        let i8s = I8Block::from_f32_per_vector(&f32s);
        let bins = BinaryBlock::from_f32(&f32s);
        let q_f32: Vec<Vec<f32>> = queries_flat
            .chunks_exact(dim)
            .map(<[f32]>::to_vec)
            .collect();
        let q_scales: Vec<f32> = q_f32
            .iter()
            .map(|q| guksu::quant::max_abs_scale(q))
            .collect();
        let q_i8 = q_f32
            .iter()
            .zip(&q_scales)
            .map(|(q, &s)| guksu::quant::quantize_i8_vec(q, s))
            .collect();
        let q_bits = q_f32
            .iter()
            .map(|q| guksu::quant::pack_sign_bits_vec(q))
            .collect();
        Ok(Data {
            f32s,
            i8s,
            bins,
            q_f32,
            q_i8,
            q_scales,
            q_bits,
        })
    }

    pub fn queries(&self) -> usize {
        self.q_f32.len()
    }
}

/// Parallel map over query indices with deterministic slot assignment.
pub fn par_map_queries<T: Send>(q: usize, threads: usize, f: impl Fn(usize) -> T + Sync) -> Vec<T> {
    let mut out: Vec<Option<T>> = (0..q).map(|_| None).collect();
    let chunk = q.div_ceil(threads.max(1)).max(1);
    std::thread::scope(|s| {
        for (t, slots) in out.chunks_mut(chunk).enumerate() {
            let f = &f;
            s.spawn(move || {
                for (j, slot) in slots.iter_mut().enumerate() {
                    *slot = Some(f(t * chunk + j));
                }
            });
        }
    });
    out.into_iter().map(|o| o.expect("slot filled")).collect()
}

// ---------------------------------------------------------------- the matrix

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Coarse {
    F32Control,
    I8Sym,
    BinSym,
    BinAsymF32q,
    BinAsymI8q,
}

impl Coarse {
    pub fn label(self) -> &'static str {
        match self {
            Coarse::F32Control => "f32 (control)",
            Coarse::I8Sym => "int8 sym",
            Coarse::BinSym => "bin sym",
            Coarse::BinAsymF32q => "bin asym f32q",
            Coarse::BinAsymI8q => "bin asym i8q",
        }
    }

    /// The coarse-stage scorer for query `qi`.
    fn scorer<'a>(self, data: &'a Data, qi: usize) -> Scorer<'a> {
        match self {
            Coarse::F32Control => Scorer::F32 {
                query: ViewQuery {
                    view: data.f32s.view(),
                    query: &data.q_f32[qi],
                },
            },
            Coarse::I8Sym => Scorer::I8 {
                query: ViewQuery {
                    view: data.i8s.view(),
                    query: I8Query {
                        codes: &data.q_i8[qi],
                        scale: data.q_scales[qi],
                    },
                },
            },
            Coarse::BinSym => Scorer::Bin {
                query: ViewQuery {
                    view: data.bins.view(),
                    query: &data.q_bits[qi],
                },
            },
            Coarse::BinAsymF32q => Scorer::F32Bin {
                query: ViewQuery {
                    view: data.bins.view(),
                    query: &data.q_f32[qi],
                },
            },
            Coarse::BinAsymI8q => Scorer::I8Bin {
                query: ViewQuery {
                    view: data.bins.view(),
                    query: I8Query {
                        codes: &data.q_i8[qi],
                        scale: data.q_scales[qi],
                    },
                },
            },
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Store {
    I8,
    F32,
}

impl Store {
    pub fn label(self) -> &'static str {
        match self {
            Store::I8 => "i8",
            Store::F32 => "f32",
        }
    }

    /// The rerank scorer against this store for query `qi`.
    fn scorer<'a>(self, data: &'a Data, qi: usize) -> Scorer<'a> {
        match self {
            Store::F32 => Scorer::F32 {
                query: ViewQuery {
                    view: data.f32s.view(),
                    query: &data.q_f32[qi],
                },
            },
            Store::I8 => Scorer::I8 {
                query: ViewQuery {
                    view: data.i8s.view(),
                    query: I8Query {
                        codes: &data.q_i8[qi],
                        scale: data.q_scales[qi],
                    },
                },
            },
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Cell {
    pub coarse: Coarse,
    /// `None` = single-stage; `Some((store, factor))` = coarse top-(factor·k)
    /// pool reranked against `store`.
    pub two_stage: Option<(Store, usize)>,
}

/// The evaluation grid. Deliberately absent: int8-asym single-stage (no
/// f32×i8 kernel in M0 and int8 error is already second-order), f32→anything
/// (rerank of exact is a no-op), and sym/asym on the rerank stage (rerank
/// precision is defined by the store).
pub fn matrix(rerank_factors: &[usize], full: bool) -> Vec<Cell> {
    let mut cells = vec![
        Cell {
            coarse: Coarse::F32Control,
            two_stage: None,
        },
        Cell {
            coarse: Coarse::I8Sym,
            two_stage: None,
        },
        Cell {
            coarse: Coarse::BinSym,
            two_stage: None,
        },
        Cell {
            coarse: Coarse::BinAsymF32q,
            two_stage: None,
        },
        Cell {
            coarse: Coarse::BinAsymI8q,
            two_stage: None,
        },
    ];
    for coarse in [Coarse::BinSym, Coarse::BinAsymF32q, Coarse::BinAsymI8q] {
        for store in [Store::I8, Store::F32] {
            for &factor in rerank_factors {
                cells.push(Cell {
                    coarse,
                    two_stage: Some((store, factor)),
                });
            }
        }
    }
    if full {
        for &factor in rerank_factors {
            cells.push(Cell {
                coarse: Coarse::I8Sym,
                two_stage: Some((Store::F32, factor)),
            });
        }
    }
    cells
}

// ---------------------------------------------------------------- ground truth

pub fn ground_truth(
    data: &Data,
    kmax: usize,
    filter: Option<&Bitset>,
    threads: usize,
) -> Vec<Vec<Hit>> {
    par_map_queries(data.queries(), threads, |qi| {
        Scorer::F32 {
            query: ViewQuery {
                view: data.f32s.view(),
                query: &data.q_f32[qi],
            },
        }
        .top_k(kmax, filter)
        .unwrap()
    })
}

const GT_MAGIC: &[u8; 6] = b"GKGT1\n";

/// Load GT if the cache file exists and its params string matches; otherwise
/// compute and (best-effort) save. Returns (gt, came_from_cache).
pub fn load_or_compute_gt(
    path: Option<&Path>,
    params: &str,
    compute: impl FnOnce() -> Vec<Vec<Hit>>,
) -> (Vec<Vec<Hit>>, bool) {
    if let Some(p) = path {
        if let Some(gt) = try_load_gt(p, params) {
            return (gt, true);
        }
    }
    let gt = compute();
    if let Some(p) = path {
        if let Err(e) = save_gt(p, params, &gt) {
            eprintln!("warning: could not write gt cache {}: {e}", p.display());
        }
    }
    (gt, false)
}

/// Per-selectivity cache file so multi-filter runs do not clobber each other.
pub fn gt_cache_path(base: &Path, s: f64) -> PathBuf {
    PathBuf::from(format!("{}.s{s:.4}", base.display()))
}

fn try_load_gt(path: &Path, params: &str) -> Option<Vec<Vec<Hit>>> {
    let bytes = std::fs::read(path).ok()?;
    let mut at = 0usize;
    let take = |at: &mut usize, n: usize| -> Option<&[u8]> {
        let s = bytes.get(*at..*at + n)?;
        *at += n;
        Some(s)
    };
    if take(&mut at, 6)? != GT_MAGIC {
        return None;
    }
    let u32_at =
        |at: &mut usize| -> Option<u32> { Some(u32::from_le_bytes(take(at, 4)?.try_into().ok()?)) };
    let plen = u32_at(&mut at)? as usize;
    if std::str::from_utf8(take(&mut at, plen)?).ok()? != params {
        return None; // stale params: recompute and overwrite, never error
    }
    let queries = u32_at(&mut at)? as usize;
    let mut gt = Vec::with_capacity(queries);
    for _ in 0..queries {
        let len = u32_at(&mut at)? as usize;
        let mut hits = Vec::with_capacity(len);
        for _ in 0..len {
            let id = u32_at(&mut at)?;
            let score = f32::from_le_bytes(take(&mut at, 4)?.try_into().ok()?);
            hits.push(Hit { id, score });
        }
        gt.push(hits);
    }
    (at == bytes.len()).then_some(gt)
}

fn save_gt(path: &Path, params: &str, gt: &[Vec<Hit>]) -> std::io::Result<()> {
    let mut w = std::io::BufWriter::new(std::fs::File::create(path)?);
    w.write_all(GT_MAGIC)?;
    w.write_all(&(params.len() as u32).to_le_bytes())?;
    w.write_all(params.as_bytes())?;
    w.write_all(&(gt.len() as u32).to_le_bytes())?;
    for hits in gt {
        w.write_all(&(hits.len() as u32).to_le_bytes())?;
        for h in hits {
            w.write_all(&h.id.to_le_bytes())?;
            w.write_all(&h.score.to_le_bytes())?;
        }
    }
    w.flush()
}

// ---------------------------------------------------------------- recall

/// recall@k = |result@k ∩ GT@k| / min(k, |GT|); vacuous (1.0) when GT is
/// empty. The GT prefix passed in is already min(k, |GT|) long; the result is
/// truncated at k (it can legitimately hold more when scanned at kmax).
pub fn recall_at_k(result: &[Hit], k: usize, gt_prefix_sorted_ids: &[u32]) -> f64 {
    let denom = gt_prefix_sorted_ids.len();
    if denom == 0 {
        return 1.0;
    }
    let hits = result
        .iter()
        .take(k)
        .filter(|h| gt_prefix_sorted_ids.binary_search(&h.id).is_ok())
        .count();
    hits as f64 / denom as f64
}

/// `[query][k_idx]` → sorted GT-prefix ids (prefix length `min(k, |GT|)`).
pub fn gt_prefixes(gt: &[Vec<Hit>], ks: &[usize]) -> Vec<Vec<Vec<u32>>> {
    gt.iter()
        .map(|hits| {
            ks.iter()
                .map(|&k| {
                    let mut ids: Vec<u32> =
                        hits[..k.min(hits.len())].iter().map(|h| h.id).collect();
                    ids.sort_unstable();
                    ids
                })
                .collect()
        })
        .collect()
}

// ---------------------------------------------------------------- run

pub struct Row {
    pub coarse: &'static str,
    pub store: Option<&'static str>,
    pub factor: Option<usize>,
    /// Pool size per k (two-stage only): factor·k.
    pub pool_at: Vec<Option<usize>>,
    pub recalls: Vec<f64>,
    pub us_per_query: f64,
}

impl Row {
    pub fn label(&self) -> String {
        match (self.store, self.factor) {
            (Some(store), Some(factor)) => {
                format!("{:<13} -> {:<3} x{factor}", self.coarse, store)
            }
            _ => self.coarse.to_string(),
        }
    }
}

/// Evaluate every cell. Two-stage cells share ONE coarse scan per coarse kind
/// at Rmax = max(factor)·kmax; pools for smaller (k, factor) combinations are
/// prefixes of it (valid by the selection prefix property). `us/q` for those
/// cells includes the shared Rmax coarse scan, slightly overstating small
/// factors — benches own rigorous per-config timings.
pub fn run_matrix(
    data: &Data,
    cells: &[Cell],
    ks: &[usize],
    gt: &[Vec<Hit>],
    filter: Option<&Bitset>,
    threads: usize,
) -> Vec<Row> {
    let q = data.queries();
    let kmax = *ks.iter().max().expect("ks non-empty");
    let prefixes = gt_prefixes(gt, ks);
    let mean_recalls = |per_query: &dyn Fn(usize, usize) -> f64| -> Vec<f64> {
        (0..ks.len())
            .map(|ki| (0..q).map(|qi| per_query(qi, ki)).sum::<f64>() / q as f64)
            .collect()
    };
    let mut rows = Vec::new();

    // Single-stage cells: one scan at kmax per query; recall@k from prefixes.
    for cell in cells.iter().filter(|c| c.two_stage.is_none()) {
        let t0 = Instant::now();
        let results = par_map_queries(q, threads, |qi| {
            cell.coarse.scorer(data, qi).top_k(kmax, filter).unwrap()
        });
        let us = t0.elapsed().as_secs_f64() * 1e6 / q as f64;
        let recalls = mean_recalls(&|qi, ki| recall_at_k(&results[qi], ks[ki], &prefixes[qi][ki]));
        rows.push(Row {
            coarse: cell.coarse.label(),
            store: None,
            factor: None,
            pool_at: vec![None; ks.len()],
            recalls,
            us_per_query: us,
        });
    }

    // Two-stage cells, grouped by coarse kind.
    for coarse in [
        Coarse::I8Sym,
        Coarse::BinSym,
        Coarse::BinAsymF32q,
        Coarse::BinAsymI8q,
    ] {
        let group: Vec<&Cell> = cells
            .iter()
            .filter(|c| c.coarse == coarse && c.two_stage.is_some())
            .collect();
        if group.is_empty() {
            continue;
        }
        let fmax = group.iter().map(|c| c.two_stage.unwrap().1).max().unwrap();
        let rmax = fmax * kmax;
        let t0 = Instant::now();
        let pools: Vec<Vec<u32>> = par_map_queries(q, threads, |qi| {
            coarse
                .scorer(data, qi)
                .top_k(rmax, filter)
                .unwrap()
                .iter()
                .map(|h| h.id)
                .collect()
        });
        let coarse_us = t0.elapsed().as_secs_f64() * 1e6 / q as f64;

        for cell in group {
            let (store, factor) = cell.two_stage.unwrap();
            let t1 = Instant::now();
            let results: Vec<Vec<Vec<Hit>>> = par_map_queries(q, threads, |qi| {
                ks.iter()
                    .map(|&k| {
                        let pool = &pools[qi][..(factor * k).min(pools[qi].len())];
                        store.scorer(data, qi).rerank(pool, k).unwrap()
                    })
                    .collect()
            });
            let rerank_us = t1.elapsed().as_secs_f64() * 1e6 / q as f64;
            let recalls =
                mean_recalls(&|qi, ki| recall_at_k(&results[qi][ki], ks[ki], &prefixes[qi][ki]));
            rows.push(Row {
                coarse: coarse.label(),
                store: Some(store.label()),
                factor: Some(factor),
                pool_at: ks.iter().map(|&k| Some(factor * k)).collect(),
                recalls,
                us_per_query: coarse_us + rerank_us,
            });
        }
    }
    rows
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::Dist;

    #[test]
    fn matrix_row_counts() {
        assert_eq!(matrix(&[2, 4, 8], false).len(), 23);
        assert_eq!(matrix(&[2, 4, 8], true).len(), 26);
        assert_eq!(matrix(&[4], false).len(), 5 + 6);
    }

    #[test]
    fn recall_math_with_ties_and_denominator() {
        let hit = |id, score| Hit { id, score };
        let gt = vec![vec![hit(1, 3.0), hit(2, 2.0), hit(3, 1.0)]];
        let pre = gt_prefixes(&gt, &[2, 10]);
        // Exact @2.
        assert_eq!(recall_at_k(&[hit(2, 9.0), hit(1, 8.0)], 2, &pre[0][0]), 1.0);
        // One of two @2.
        assert_eq!(recall_at_k(&[hit(2, 9.0), hit(9, 8.0)], 2, &pre[0][0]), 0.5);
        // Result truncates at k: the match at position 3 must not count @2.
        assert_eq!(
            recall_at_k(&[hit(9, 9.0), hit(8, 8.0), hit(1, 7.0)], 2, &pre[0][0]),
            0.0
        );
        // k=10 > |GT|=3: denominator clamps to 3.
        assert_eq!(
            recall_at_k(&[hit(3, 1.0), hit(1, 1.0), hit(7, 1.0)], 10, &pre[0][1]),
            2.0 / 3.0
        );
        // Empty GT is vacuously perfect.
        assert_eq!(recall_at_k(&[hit(1, 0.0)], 5, &[]), 1.0);
    }

    #[test]
    fn control_row_is_exact_and_rerank_is_monotone() {
        let dim = 64;
        let corpus = crate::synth::generate(Dist::Gmm, 400, dim, 8, 5, 0, 2);
        let queries =
            crate::synth::generate(Dist::Gmm, 30, dim, 8, 5, crate::synth::QUERY_STREAM, 2);
        let data = Data::prepare(&corpus, &queries, dim).unwrap();
        let ks = [10usize];
        let gt = ground_truth(&data, 10, None, 2);
        let rows = run_matrix(&data, &matrix(&[2, 8], false), &ks, &gt, None, 2);
        let control = &rows[0];
        assert_eq!(control.coarse, "f32 (control)");
        assert_eq!(control.recalls[0], 1.0, "control must be exact");
        // Rerank recall must not decrease with factor for the same config.
        for coarse in ["bin sym", "bin asym f32q", "bin asym i8q"] {
            for store in ["i8", "f32"] {
                let r: Vec<f64> = rows
                    .iter()
                    .filter(|r| r.coarse == coarse && r.store == Some(store))
                    .map(|r| r.recalls[0])
                    .collect();
                assert_eq!(r.len(), 2);
                assert!(r[1] >= r[0] - 1e-9, "{coarse}->{store}: {r:?} not monotone");
            }
        }
    }

    #[test]
    fn gt_cache_round_trip_and_staleness() {
        let dir = std::env::temp_dir().join("guksu_gt_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("gt.cache");
        let gt = vec![vec![Hit { id: 3, score: 0.5 }], vec![]];
        save_gt(&path, "params v1", &gt).unwrap();
        let loaded = try_load_gt(&path, "params v1").unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0], gt[0]);
        assert!(loaded[1].is_empty());
        assert!(
            try_load_gt(&path, "params v2").is_none(),
            "stale params must miss"
        );
        let (recomputed, from_cache) = load_or_compute_gt(Some(&path), "params v2", || gt.clone());
        assert!(!from_cache);
        assert_eq!(recomputed[0], gt[0]);
        assert!(
            try_load_gt(&path, "params v2").is_some(),
            "cache must be overwritten"
        );
    }
}
