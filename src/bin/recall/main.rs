//! recall — the guksu M0 quantization-loss benchmark. See `--help`.

mod cli;
mod eval;
mod npy;
mod synth;
mod table;

use std::process::exit;
use std::time::Instant;

use guksu::Bitset;
use guksu::kernels::Kernels;

fn main() {
    let cfg = cli::parse();
    let backend = Kernels::detected().backend;
    println!(
        "guksu recall v{} | backend: {backend} | threads: {}",
        env!("CARGO_PKG_VERSION"),
        cfg.threads
    );

    let t0 = Instant::now();
    let (corpus, queries, dim, source) = load_data(&cfg);
    let n = corpus.len() / dim;
    let q = queries.len() / dim;
    println!(
        "corpus: {source} n={n} dim={dim} | queries: {q} | loaded in {:.1}s",
        t0.elapsed().as_secs_f64()
    );

    let t1 = Instant::now();
    let data = eval::Data::prepare(&corpus, &queries, dim);
    drop(corpus);
    let code_len = guksu::kernels::binary_code_len(dim);
    println!(
        "bytes/vec: f32 {} | int8 {} (codes+scale) | bin {code_len} | quantized in {:.1}s",
        dim * 4,
        dim + 4,
        t1.elapsed().as_secs_f64()
    );

    let kmax = *cfg.k.iter().max().expect("validated non-empty");
    let cells = eval::matrix(&cfg.rerank, cfg.full);

    for &s in &cfg.filters {
        let filter = (s < 1.0).then(|| Bitset::random(n, s, cfg.seed ^ s.to_bits()));
        let candidates = filter.as_ref().map(|f| f.count_ones()).unwrap_or(n);
        if candidates < 8 * kmax {
            eprintln!(
                "warning: only {candidates} filter survivors at s={s} (< 8·kmax = {}) — cells are degenerate",
                8 * kmax
            );
        }

        let params = format!(
            "guksu-gt v1 | source={source} | n={n} dim={dim} q={q} kmax={kmax} seed={} s={s:.6}",
            cfg.seed
        );
        let cache = cfg.gt_cache.as_ref().map(|b| eval::gt_cache_path(b, s));
        let t2 = Instant::now();
        let (gt, cached) = eval::load_or_compute_gt(cache.as_deref(), &params, || {
            eval::ground_truth(&data, kmax, filter.as_ref(), cfg.threads)
        });
        let rows = eval::run_matrix(&data, &cells, &cfg.k, &gt, filter.as_ref(), cfg.threads);

        let flabel = if s < 1.0 {
            format!("s={s:.2} ({candidates} candidates)")
        } else {
            format!("none (s=1.00, {candidates} candidates)")
        };
        println!();
        println!(
            "== filter: {flabel} | ground truth kmax={kmax}: {:.1}s{} ==",
            t2.elapsed().as_secs_f64(),
            if cached { " (cached)" } else { "" }
        );
        table::print_table(&rows, &cfg.k);

        if let Some(csv) = &cfg.csv {
            let meta = table::CsvMeta { n, dim, queries: q, seed: cfg.seed, backend };
            if let Err(e) = table::append_csv(csv, s, &rows, &cfg.k, &meta) {
                eprintln!("warning: csv write failed: {e}");
            }
        }
    }
}

fn die(msg: &str) -> ! {
    eprintln!("error: {msg}");
    exit(2);
}

/// Returns (corpus flat, queries flat, dim, source label for echo + GT cache key).
fn load_data(cfg: &cli::Config) -> (Vec<f32>, Vec<f32>, usize, String) {
    match (&cfg.vectors_path, &cfg.queries_path) {
        (Some(vp), Some(qp)) => {
            let (mut corpus, dim_c) =
                npy::load_matrix(vp, Some(cfg.dim)).unwrap_or_else(|e| die(&e));
            let (mut queries, dim_q) =
                npy::load_matrix(qp, Some(cfg.dim)).unwrap_or_else(|e| die(&e));
            if dim_c != dim_q {
                die(&format!("corpus dim {dim_c} != query dim {dim_q}"));
            }
            check_or_normalize(&mut corpus, dim_c, cfg.normalize, "corpus");
            check_or_normalize(&mut queries, dim_q, cfg.normalize, "queries");
            let size = std::fs::metadata(vp).map(|m| m.len()).unwrap_or(0);
            (corpus, queries, dim_c, format!("file:{}({size}B)", vp.display()))
        }
        _ => {
            let corpus = synth::generate(
                cfg.dist,
                cfg.n,
                cfg.dim,
                cfg.gmm_clusters,
                cfg.seed,
                synth::CORPUS_STREAM,
                cfg.threads,
            );
            let queries = synth::generate(
                cfg.dist,
                cfg.queries,
                cfg.dim,
                cfg.gmm_clusters,
                cfg.seed,
                synth::QUERY_STREAM,
                cfg.threads,
            );
            let dist = match cfg.dist {
                cli::Dist::Gmm => format!("gmm({})", cfg.gmm_clusters),
                cli::Dist::Uniform => "uniform".to_string(),
            };
            (corpus, queries, cfg.dim, format!("synthetic:{dist}:seed{}", cfg.seed))
        }
    }
}

/// File data must arrive L2-normalized (dot = cosine is the harness's ground
/// truth metric); `--normalize` opts into normalizing it here instead.
fn check_or_normalize(flat: &mut [f32], dim: usize, normalize: bool, what: &str) {
    if normalize {
        for row in flat.chunks_exact_mut(dim) {
            synth::normalize(row);
        }
        return;
    }
    for (i, row) in flat.chunks_exact(dim).take(100).enumerate() {
        let norm = row.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
        if (norm - 1.0).abs() > 1e-3 {
            die(&format!(
                "{what} row {i} has L2 norm {norm:.4}, expected 1.0 — pass --normalize to normalize file data"
            ));
        }
    }
}
