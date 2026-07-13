//! Synthetic corpora: uniform-on-sphere and clustered Gaussian mixtures.
//!
//! Every row is generated from `SplitMix64::substream(seed, stream + row)`,
//! so output is bit-identical regardless of thread count or generation order.
//! Uniform-on-sphere at high dim makes all dots ≈ 0 (neighbors are close to
//! arbitrary) — it is a stress case; `gmm` is the default because clustered
//! data is what real embedding corpora look like.

use guksu::rng::SplitMix64;

use crate::cli::Dist;

/// Substream offsets keep corpus rows, query rows, and cluster parameters in
/// disjoint stream index ranges for any n below 2^40.
pub const CORPUS_STREAM: u64 = 0;
pub const QUERY_STREAM: u64 = 1 << 40;
const CLUSTER_STREAM: u64 = 1 << 41;

pub fn normalize(row: &mut [f32]) {
    let norm = row.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt() as f32;
    if norm > 0.0 {
        for x in row {
            *x /= norm;
        }
    }
}

fn fill_gaussian(rng: &mut SplitMix64, row: &mut [f32]) {
    for x in row {
        *x = rng.next_gaussian() as f32;
    }
}

/// One GMM cluster: unit-vector mean + per-coordinate sigma chosen so the
/// TOTAL noise norm (≈ sigma·√dim) lands in [0.15, 0.8] relative to the unit
/// mean — same-cluster cosines of ~0.6–0.98, like real embedding corpora.
/// (A dim-independent per-coordinate sigma would swamp the mean at high dim.)
struct Cluster {
    mean: Vec<f32>,
    sigma: f32,
}

fn clusters(seed: u64, count: usize, dim: usize) -> Vec<Cluster> {
    (0..count)
        .map(|c| {
            let mut rng = SplitMix64::substream(seed, CLUSTER_STREAM + c as u64);
            let mut mean = vec![0.0f32; dim];
            fill_gaussian(&mut rng, &mut mean);
            normalize(&mut mean);
            let spread = 0.15 + 0.65 * rng.next_f32();
            Cluster { mean, sigma: spread / (dim as f32).sqrt() }
        })
        .collect()
}

/// Generate `n` L2-normalized rows into a flat vec, parallelized over
/// contiguous chunks (`stream` picks corpus vs query space).
pub fn generate(
    dist: Dist,
    n: usize,
    dim: usize,
    gmm_clusters: usize,
    seed: u64,
    stream: u64,
    threads: usize,
) -> Vec<f32> {
    let cls = match dist {
        Dist::Gmm => clusters(seed, gmm_clusters, dim),
        Dist::Uniform => Vec::new(),
    };
    let mut flat = vec![0.0f32; n * dim];
    let chunk_rows = n.div_ceil(threads.max(1));
    std::thread::scope(|s| {
        for (t, chunk) in flat.chunks_mut(chunk_rows * dim).enumerate() {
            let cls = &cls;
            s.spawn(move || {
                for (j, row) in chunk.chunks_mut(dim).enumerate() {
                    let i = t * chunk_rows + j;
                    let mut rng = SplitMix64::substream(seed, stream + i as u64);
                    match dist {
                        Dist::Uniform => fill_gaussian(&mut rng, row),
                        Dist::Gmm => {
                            let c = &cls[rng.next_range(cls.len() as u64) as usize];
                            fill_gaussian(&mut rng, row);
                            for (x, &m) in row.iter_mut().zip(&c.mean) {
                                *x = m + c.sigma * *x;
                            }
                        }
                    }
                    normalize(row);
                }
            });
        }
    });
    flat
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_are_normalized_and_deterministic() {
        for dist in [Dist::Uniform, Dist::Gmm] {
            let a = generate(dist, 50, 128, 8, 7, CORPUS_STREAM, 4);
            let b = generate(dist, 50, 128, 8, 7, CORPUS_STREAM, 1); // thread count must not matter
            assert_eq!(a, b, "{dist:?} not thread-order independent");
            for row in a.chunks_exact(128) {
                let norm: f64 = row.iter().map(|&x| (x as f64) * (x as f64)).sum();
                assert!((norm.sqrt() - 1.0).abs() < 1e-3, "{dist:?} row norm {}", norm.sqrt());
            }
        }
    }

    #[test]
    fn corpus_and_queries_differ_but_share_clusters() {
        let corpus = generate(Dist::Gmm, 20, 64, 4, 9, CORPUS_STREAM, 2);
        let queries = generate(Dist::Gmm, 20, 64, 4, 9, QUERY_STREAM, 2);
        assert_ne!(corpus, queries);
        // Same seed, different --gmm-clusters ⇒ different corpus.
        let other = generate(Dist::Gmm, 20, 64, 8, 9, CORPUS_STREAM, 2);
        assert_ne!(corpus, other);
    }

    #[test]
    fn gmm_is_clustered_uniform_is_not() {
        // Mean |dot| between random pairs: near 0 for uniform, clearly
        // positive for a tight mixture.
        let dim = 256;
        let mean_abs_dot = |flat: &[f32]| {
            let rows: Vec<&[f32]> = flat.chunks_exact(dim).collect();
            let mut s = 0.0f64;
            let mut count = 0;
            for i in 0..rows.len() {
                for j in (i + 1)..rows.len() {
                    let d: f32 = rows[i].iter().zip(rows[j]).map(|(&x, &y)| x * y).sum();
                    s += (d as f64).abs();
                    count += 1;
                }
            }
            s / count as f64
        };
        let uni = generate(Dist::Uniform, 60, dim, 1, 3, CORPUS_STREAM, 2);
        let gmm = generate(Dist::Gmm, 60, dim, 4, 3, CORPUS_STREAM, 2);
        let (u, g) = (mean_abs_dot(&uni), mean_abs_dot(&gmm));
        assert!(g > 4.0 * u, "gmm {g} not clustered vs uniform {u}");
    }
}
