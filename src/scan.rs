//! Flat exact top-k over quantized blocks, with optional filter bitmaps and a
//! rerank primitive for two-stage (coarse → precise) search. Every scan goes
//! through a [`Scorer`]: a query bound to the corpus representation it is
//! scored against, one variant per distance kernel.
//!
//! # Ordering contract
//!
//! A [`Hit`] is *greater* (better) than another if its score is higher under
//! `f32::total_cmp`, or the scores are equal and its id is lower. Every result
//! vector is sorted best-first, i.e. `(score descending, id ascending)`. The
//! rule is shared by ground truth, top-k, and rerank alike — with binary codes
//! score ties are massive (1024-bit codes admit 1025 Hamming values), so the
//! tie order is what makes results and recall numbers reproducible.
//!
//! Selection keeps the k greatest hits of the candidate stream; because the
//! order is total and ids are unique, the result is a deterministic function
//! of the candidate *set* and has the **prefix property**:
//! `top_k(k1) == top_k(k2)[..k1]` for `k1 <= k2`.
//!
//! # Scores are representation-local
//!
//! A Hamming-derived score and a dot score are not comparable. Two-stage
//! search composes a coarse scan with a rerank that *recomputes* scores from
//! the precise store — coarse scores are never blended:
//!
//! ```
//! use guksu::{BinaryBlock, Block, F32Block, Scorer, ViewQuery, quant};
//! let corpus = F32Block::from_flat(&[1.0, 0.0, -0.6, 0.8, 0.6, 0.8], 2).unwrap();
//! let bin = BinaryBlock::from_f32(&corpus);
//! let query = [0.6f32, -0.8];
//! let q_bits = quant::pack_sign_bits_vec(&query);
//!
//! // Coarse scan (R = 2), then precise rerank.
//! let coarse = Scorer::Bin {
//!     query: ViewQuery { view: bin.view(), query: &q_bits },
//! };
//! let pool = coarse.top_k(2, None).unwrap();
//! let ids: Vec<u32> = pool.iter().map(|h| h.id).collect();
//! let precise = Scorer::F32 {
//!     query: ViewQuery { view: corpus.view(), query: &query },
//! };
//! let hits = precise.rerank(&ids, 1).unwrap();
//! assert_eq!(hits[0].id, 0);
//! ```

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::fmt;

use crate::bitset::Bitset;
use crate::kernels::Kernels;
use crate::query::{I8Query, ViewQuery};
use crate::storage::{BinaryView, F32View, I8View};

/// A scored hit. See the module docs for the ordering contract.
#[derive(Clone, Copy, Debug)]
pub struct Hit {
    pub id: u32,
    pub score: f32,
}

impl Ord for Hit {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .total_cmp(&other.score)
            .then_with(|| other.id.cmp(&self.id))
    }
}

impl PartialOrd for Hit {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// Not derived: equality must agree with `Ord` (total_cmp), not with f32 `==`,
// which disagrees on -0.0/NaN and would break the ordering contract.
impl PartialEq for Hit {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Hit {}

/// Bounded selection core: scores every candidate id and keeps the `k`
/// greatest hits, returned best-first. Every [`Scorer`] scan and rerank is a
/// thin wrapper over this. Duplicate candidate ids are the caller's bug.
pub fn select_top_k(
    candidates: impl IntoIterator<Item = u32>,
    mut score: impl FnMut(u32) -> f32,
    k: usize,
) -> Vec<Hit> {
    if k == 0 {
        return Vec::new();
    }
    // Min-heap of the k best so far; a candidate replaces the current worst
    // only if strictly greater, so the kept set is exactly the k greatest.
    let mut heap: BinaryHeap<std::cmp::Reverse<Hit>> = BinaryHeap::with_capacity(k);
    for id in candidates {
        let hit = Hit {
            id,
            score: score(id),
        };
        if heap.len() < k {
            heap.push(std::cmp::Reverse(hit));
        } else {
            let mut worst = heap.peek_mut().expect("heap is non-empty");
            if hit > worst.0 {
                *worst = std::cmp::Reverse(hit);
            }
        }
    }
    let mut out: Vec<Hit> = heap.into_iter().map(|r| r.0).collect();
    out.sort_unstable_by(|a, b| b.cmp(a));
    out
}

/// A scan precondition violation — the query or filter shape does not match
/// the view it is scored against. Returned by [`Scorer::top_k`] and
/// [`Scorer::rerank`]; each variant carries both sides of the mismatch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScanError {
    /// The filter covers a different id universe than the view has rows.
    FilterLen { filter_len: usize, rows: usize },
    /// The query vector's dimension differs from the view's row dimension.
    QueryDim { query_dim: usize, view_dim: usize },
}

impl fmt::Display for ScanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            ScanError::FilterLen { filter_len, rows } => {
                write!(
                    f,
                    "filter covers {filter_len} ids but the view has {rows} rows"
                )
            }
            ScanError::QueryDim {
                query_dim,
                view_dim,
            } => {
                write!(f, "query dim {query_dim} vs view dim {view_dim}")
            }
        }
    }
}

impl std::error::Error for ScanError {}

fn check_filter(filter: Option<&Bitset>, rows: usize) -> Result<(), ScanError> {
    match filter {
        Some(f) if f.len() != rows => Err(ScanError::FilterLen {
            filter_len: f.len(),
            rows,
        }),
        _ => Ok(()),
    }
}

/// A query bound to the corpus representation that scores it — one variant per
/// distance kernel, so only meaningful [`ViewQuery`] pairings are
/// representable and the variant fixes the score formula.
///
/// Construction is free (`Copy`, borrows only). Scan a whole view with
/// [`Scorer::top_k`] or score an explicit candidate list with
/// [`Scorer::rerank`].
#[derive(Clone, Copy)]
pub enum Scorer<'a> {
    /// Exact f32 dot (cosine on L2-normalized data) — the ground-truth scorer.
    /// Score: `dot_f32(query, row)`.
    F32 {
        query: ViewQuery<F32View<'a>>,
    },
    /// Symmetric scaled int8 dot; the query is pre-quantized, its scale
    /// riding along in [`I8Query`].
    /// Score: `dot_i8(query.codes, row) as f32 * (query.scale * row_scale)`.
    I8 {
        query: ViewQuery<I8View<'a>>,
    },
    /// Symmetric binary similarity; the query is packed sign bits like the
    /// rows. Score: `-(hamming(query, row) as f32)`, so higher is better.
    Bin {
        query: ViewQuery<BinaryView<'a>>,
    },
    /// Asymmetric f32-query × binary-doc dot. Score: `dot_f32_bin(query, row)`.
    F32Bin {
        query: ViewQuery<BinaryView<'a>, &'a [f32]>,
    },
    /// Asymmetric int8-query × binary-doc dot (the query scale keeps the
    /// score in dot units; it is rank-invariant per query).
    /// Score: `dot_i8_bin(query.codes, row) as f32 * query.scale`.
    I8Bin {
        query: ViewQuery<BinaryView<'a>, I8Query<'a>>,
    },
}

impl<'a> Scorer<'a> {
    /// Exact top-k over the whole view, or over the filter's set ids only
    /// (cost scales with set bits, not rows). Results are best-first; see the
    /// module docs for the ordering contract. Fails with
    /// [`ScanError::QueryDim`] on a query/view dim mismatch (`Bin` queries are
    /// packed bytes, validated per row by the kernel instead) and
    /// [`ScanError::FilterLen`] on a filter/view length mismatch.
    pub fn top_k(self, k: usize, filter: Option<&Bitset>) -> Result<Vec<Hit>, ScanError> {
        self.check_query_dim()?;
        check_filter(filter, self.rows())?;
        Ok(match filter {
            Some(f) => self.select(f.iter_ones(), k),
            None => self.select(0..self.rows() as u32, k),
        })
    }

    /// Top-k among an explicit candidate id list — the rerank primitive of
    /// two-stage search. Scores are recomputed from this scorer's view and
    /// never blended with the pool's coarse scores; rerank from a *more
    /// precise* representation than the pool's source. Fails with
    /// [`ScanError::QueryDim`] like [`Scorer::top_k`]; duplicate or
    /// out-of-range candidate ids are the caller's bug.
    pub fn rerank(self, candidates: &[u32], k: usize) -> Result<Vec<Hit>, ScanError> {
        self.check_query_dim()?;
        Ok(self.select(candidates.iter().copied(), k))
    }

    fn rows(&self) -> usize {
        match self {
            Scorer::F32 { query } => query.rows(),
            Scorer::I8 { query } => query.rows(),
            Scorer::Bin { query } => query.rows(),
            Scorer::F32Bin { query } => query.rows(),
            Scorer::I8Bin { query } => query.rows(),
        }
    }

    fn check_query_dim(&self) -> Result<(), ScanError> {
        match self {
            Scorer::F32 { query } => query.check_dim(),
            Scorer::I8 { query } => query.check_dim(),
            Scorer::Bin { query } => query.check_dim(),
            Scorer::F32Bin { query } => query.check_dim(),
            Scorer::I8Bin { query } => query.check_dim(),
        }
    }

    /// One match per scan, never per row: the kernel table is fetched once
    /// (the hoisting contract from the `kernels` docs) and each arm hands its
    /// concrete scoring closure to the monomorphized selection loop.
    fn select(self, candidates: impl IntoIterator<Item = u32>, k: usize) -> Vec<Hit> {
        let kern = Kernels::detected();
        match self {
            Scorer::F32 {
                query: ViewQuery { view, query },
            } => select_top_k(candidates, |id| (kern.dot_f32)(query, view.row(id)), k),
            Scorer::I8 {
                query: ViewQuery { view, query },
            } => select_top_k(
                candidates,
                |id| {
                    (kern.dot_i8)(query.codes, view.row(id)) as f32
                        * (query.scale * view.scale(id))
                },
                k,
            ),
            Scorer::Bin {
                query: ViewQuery { view, query },
            } => select_top_k(
                candidates,
                |id| -((kern.hamming)(query, view.row(id)) as f32),
                k,
            ),
            Scorer::F32Bin {
                query: ViewQuery { view, query },
            } => select_top_k(candidates, |id| (kern.dot_f32_bin)(query, view.row(id)), k),
            Scorer::I8Bin {
                query: ViewQuery { view, query },
            } => select_top_k(
                candidates,
                |id| (kern.dot_i8_bin)(query.codes, view.row(id)) as f32 * query.scale,
                k,
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::{max_abs_scale, pack_sign_bits_vec, quantize_i8_vec};
    use crate::rng::SplitMix64;
    use crate::storage::{BinaryBlock, Block, F32Block, I8Block};

    fn naive_top_k(hits: impl IntoIterator<Item = Hit>, k: usize) -> Vec<Hit> {
        let mut all: Vec<Hit> = hits.into_iter().collect();
        all.sort_unstable_by(|a, b| b.cmp(a));
        all.truncate(k);
        all
    }

    fn random_block(rng: &mut SplitMix64, n: usize, dim: usize) -> F32Block {
        let flat: Vec<f32> = (0..n * dim).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
        F32Block::from_flat(&flat, dim).unwrap()
    }

    #[test]
    fn hit_ordering_contract() {
        let better = Hit { id: 3, score: 2.0 };
        let worse = Hit { id: 1, score: 1.0 };
        assert!(better > worse);
        // Equal scores: lower id is greater (better).
        assert!(Hit { id: 1, score: 1.0 } > Hit { id: 2, score: 1.0 });
        // total_cmp: +0.0 beats -0.0; equality follows cmp, not f32 ==.
        assert!(Hit { id: 0, score: 0.0 } > Hit { id: 0, score: -0.0 });
        assert_ne!(Hit { id: 0, score: 0.0 }, Hit { id: 0, score: -0.0 });
        assert_eq!(Hit { id: 7, score: 0.5 }, Hit { id: 7, score: 0.5 });
    }

    #[test]
    fn select_matches_naive_sort() {
        let mut rng = SplitMix64::new(51);
        for n in [0usize, 1, 5, 100, 1000] {
            let scores: Vec<f32> = (0..n).map(|_| rng.next_f32()).collect();
            for k in [0usize, 1, 7, n / 2, n, n + 10] {
                let got = select_top_k(0..n as u32, |id| scores[id as usize], k);
                let want = naive_top_k(
                    scores.iter().enumerate().map(|(i, &s)| Hit {
                        id: i as u32,
                        score: s,
                    }),
                    k,
                );
                assert_eq!(got, want, "n={n} k={k}");
            }
        }
    }

    #[test]
    fn all_ties_take_lowest_ids() {
        let got = select_top_k(0..100u32, |_| 1.0, 5);
        let ids: Vec<u32> = got.iter().map(|h| h.id).collect();
        assert_eq!(ids, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn prefix_property_holds_under_heavy_ties() {
        let mut rng = SplitMix64::new(52);
        // Scores quantized to 4 levels → massive ties, like Hamming plateaus.
        let scores: Vec<f32> = (0..500).map(|_| rng.next_range(4) as f32).collect();
        let big = select_top_k(0..500u32, |id| scores[id as usize], 400);
        for k in [1usize, 10, 48, 100, 399] {
            let small = select_top_k(0..500u32, |id| scores[id as usize], k);
            assert_eq!(small.as_slice(), &big[..k], "k={k}");
        }
    }

    #[test]
    fn f32_top_k_matches_naive() {
        let mut rng = SplitMix64::new(53);
        let block = random_block(&mut rng, 300, 17);
        let query: Vec<f32> = (0..17).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
        let v = block.view();
        let got = Scorer::F32 {
            query: ViewQuery { view: v, query: &query },
        }
        .top_k(10, None)
        .unwrap();
        let want = naive_top_k(
            (0..300u32).map(|id| Hit {
                id,
                score: crate::kernels::dot_f32(&query, v.row(id)),
            }),
            10,
        );
        assert_eq!(got, want);
    }

    #[test]
    fn every_scan_variant_matches_its_kernel() {
        let mut rng = SplitMix64::new(54);
        let (n, dim, k) = (200usize, 64usize, 12usize);
        let block = random_block(&mut rng, n, dim);
        let i8s = I8Block::from_f32_per_vector(&block);
        let bins = BinaryBlock::from_f32(&block);
        let query: Vec<f32> = (0..dim).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
        let q_scale = max_abs_scale(&query);
        let q_i8 = quantize_i8_vec(&query, q_scale);
        let q_bits = pack_sign_bits_vec(&query);
        let kern = Kernels::detected();

        let cases: Vec<(Vec<Hit>, Vec<Hit>)> = vec![
            (
                Scorer::I8 {
                    query: ViewQuery {
                        view: i8s.view(),
                        query: I8Query {
                            codes: &q_i8,
                            scale: q_scale,
                        },
                    },
                }
                .top_k(k, None)
                .unwrap(),
                naive_top_k(
                    (0..n as u32).map(|id| Hit {
                        id,
                        score: (kern.dot_i8)(&q_i8, i8s.view().row(id)) as f32
                            * (q_scale * i8s.view().scale(id)),
                    }),
                    k,
                ),
            ),
            (
                Scorer::Bin {
                    query: ViewQuery {
                        view: bins.view(),
                        query: &q_bits,
                    },
                }
                .top_k(k, None)
                .unwrap(),
                naive_top_k(
                    (0..n as u32).map(|id| Hit {
                        id,
                        score: -((kern.hamming)(&q_bits, bins.view().row(id)) as f32),
                    }),
                    k,
                ),
            ),
            (
                Scorer::F32Bin {
                    query: ViewQuery {
                        view: bins.view(),
                        query: &query,
                    },
                }
                .top_k(k, None)
                .unwrap(),
                naive_top_k(
                    (0..n as u32).map(|id| Hit {
                        id,
                        score: (kern.dot_f32_bin)(&query, bins.view().row(id)),
                    }),
                    k,
                ),
            ),
            (
                Scorer::I8Bin {
                    query: ViewQuery {
                        view: bins.view(),
                        query: I8Query {
                            codes: &q_i8,
                            scale: q_scale,
                        },
                    },
                }
                .top_k(k, None)
                .unwrap(),
                naive_top_k(
                    (0..n as u32).map(|id| Hit {
                        id,
                        score: (kern.dot_i8_bin)(&q_i8, bins.view().row(id)) as f32 * q_scale,
                    }),
                    k,
                ),
            ),
        ];
        for (i, (got, want)) in cases.iter().enumerate() {
            assert_eq!(got, want, "variant {i}");
        }
    }

    #[test]
    fn filtered_scan_matches_naive_over_subset() {
        let mut rng = SplitMix64::new(55);
        let block = random_block(&mut rng, 400, 9);
        let query: Vec<f32> = (0..9).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
        let filter = Bitset::random(400, 0.25, 99);
        let v = block.view();
        let got = Scorer::F32 {
            query: ViewQuery { view: v, query: &query },
        }
        .top_k(20, Some(&filter))
        .unwrap();
        assert!(
            got.iter().all(|h| filter.contains(h.id)),
            "hit outside filter"
        );
        let want = naive_top_k(
            filter.iter_ones().map(|id| Hit {
                id,
                score: crate::kernels::dot_f32(&query, v.row(id)),
            }),
            20,
        );
        assert_eq!(got, want);
    }

    #[test]
    fn empty_filter_and_k_clamps() {
        let mut rng = SplitMix64::new(56);
        let block = random_block(&mut rng, 50, 8);
        let query = vec![1.0f32; 8];
        let v = block.view();
        let scorer = Scorer::F32 {
            query: ViewQuery { view: v, query: &query },
        };
        assert!(scorer.top_k(10, Some(&Bitset::new(50))).unwrap().is_empty());
        assert!(scorer.top_k(0, None).unwrap().is_empty());
        let all = scorer.top_k(500, None).unwrap();
        assert_eq!(all.len(), 50); // k > n clamps to n
        assert!(all.windows(2).all(|w| w[0] > w[1]), "not sorted best-first");
    }

    #[test]
    fn rerank_stays_within_pool_and_reorders() {
        let mut rng = SplitMix64::new(57);
        let (n, dim) = (500usize, 32usize);
        let block = random_block(&mut rng, n, dim);
        let bins = BinaryBlock::from_f32(&block);
        let query: Vec<f32> = (0..dim).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
        let q_bits = pack_sign_bits_vec(&query);
        let filter = Bitset::random(n, 0.3, 7);

        let pool = Scorer::Bin {
            query: ViewQuery {
                view: bins.view(),
                query: &q_bits,
            },
        }
        .top_k(40, Some(&filter))
        .unwrap();
        let ids: Vec<u32> = pool.iter().map(|h| h.id).collect();
        let hits = Scorer::F32 {
            query: ViewQuery {
                view: block.view(),
                query: &query,
            },
        }
        .rerank(&ids, 10)
        .unwrap();

        assert_eq!(hits.len(), 10);
        assert!(
            hits.iter().all(|h| ids.contains(&h.id)),
            "rerank escaped the pool"
        );
        assert!(
            hits.iter().all(|h| filter.contains(h.id)),
            "rerank escaped the filter"
        );
        // Rerank == exact f32 top-k restricted to the pool.
        let v = block.view();
        let want = naive_top_k(
            ids.iter().map(|&id| Hit {
                id,
                score: crate::kernels::dot_f32(&query, v.row(id)),
            }),
            10,
        );
        assert_eq!(hits, want);
    }

    #[test]
    fn i8_rerank_matches_naive() {
        let mut rng = SplitMix64::new(58);
        let (n, dim) = (100usize, 16usize);
        let block = random_block(&mut rng, n, dim);
        let i8s = I8Block::from_f32_per_vector(&block);
        let query: Vec<f32> = (0..dim).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
        let q_scale = max_abs_scale(&query);
        let q_i8 = quantize_i8_vec(&query, q_scale);
        let ids: Vec<u32> = vec![3, 9, 17, 42, 77];
        let got = Scorer::I8 {
            query: ViewQuery {
                view: i8s.view(),
                query: I8Query {
                    codes: &q_i8,
                    scale: q_scale,
                },
            },
        }
        .rerank(&ids, 3)
        .unwrap();
        let v = i8s.view();
        let want = naive_top_k(
            ids.iter().map(|&id| Hit {
                id,
                score: crate::kernels::dot_i8(&q_i8, v.row(id)) as f32 * (q_scale * v.scale(id)),
            }),
            3,
        );
        assert_eq!(got, want);
    }

    #[test]
    fn filter_length_mismatch_errors() {
        let block = F32Block::from_flat(&[1.0, 2.0], 2).unwrap();
        let err = Scorer::F32 {
            query: ViewQuery {
                view: block.view(),
                query: &[1.0, 0.0],
            },
        }
        .top_k(1, Some(&Bitset::new(3)))
        .unwrap_err();
        assert_eq!(
            err,
            ScanError::FilterLen {
                filter_len: 3,
                rows: 1
            }
        );
        assert_eq!(
            err.to_string(),
            "filter covers 3 ids but the view has 1 rows"
        );
    }

    #[test]
    fn query_dim_mismatch_errors() {
        let block = F32Block::from_flat(&[1.0, 2.0], 2).unwrap();
        let scorer = Scorer::F32 {
            query: ViewQuery {
                view: block.view(),
                query: &[1.0],
            },
        };
        let want = ScanError::QueryDim {
            query_dim: 1,
            view_dim: 2,
        };
        assert_eq!(scorer.top_k(1, None).unwrap_err(), want);
        assert_eq!(scorer.rerank(&[0], 1).unwrap_err(), want);
        assert_eq!(want.to_string(), "query dim 1 vs view dim 2");
    }

    #[test]
    fn f32bin_query_dim_mismatch_errors() {
        // BinView::dim() is in bits (the original vector dim), so an
        // asymmetric query must match the unpacked dim, not the byte length.
        let block = F32Block::from_flat(&[1.0, 2.0], 2).unwrap();
        let bins = BinaryBlock::from_f32(&block);
        let err = Scorer::F32Bin {
            query: ViewQuery {
                view: bins.view(),
                query: &[1.0],
            },
        }
        .top_k(1, None)
        .unwrap_err();
        assert_eq!(
            err,
            ScanError::QueryDim {
                query_dim: 1,
                view_dim: 2
            }
        );
    }
}
