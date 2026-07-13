//! CLI parsing (lexopt) for the recall harness.

use std::path::PathBuf;
use std::process::exit;
use std::str::FromStr;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dist {
    Gmm,
    Uniform,
}

#[derive(Debug)]
pub struct Config {
    pub n: usize,
    pub dim: usize,
    pub queries: usize,
    pub dist: Dist,
    pub gmm_clusters: usize,
    pub seed: u64,
    pub vectors_path: Option<PathBuf>,
    pub queries_path: Option<PathBuf>,
    pub normalize: bool,
    pub k: Vec<usize>,
    pub rerank: Vec<usize>,
    pub filters: Vec<f64>,
    pub full: bool,
    pub gt_cache: Option<PathBuf>,
    pub threads: usize,
    pub csv: Option<PathBuf>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            n: 100_000,
            dim: 1024,
            queries: 1000,
            dist: Dist::Gmm,
            gmm_clusters: 1024,
            seed: 42,
            vectors_path: None,
            queries_path: None,
            normalize: false,
            k: vec![10, 100],
            rerank: vec![2, 4, 8],
            filters: vec![1.0],
            full: false,
            gt_cache: None,
            threads: std::thread::available_parallelism().map(|p| p.get()).unwrap_or(1),
            csv: None,
        }
    }
}

const HELP: &str = "\
recall — quantization-loss benchmark (guksu M0)

Reports recall@k of quantized configs against f32 brute-force ground truth:
{int8, binary, binary→int8-rerank, binary→f32-rerank} × {symmetric, asymmetric}
× rerank depth, optionally under filter bitmaps.

USAGE:
  recall [OPTIONS]                      synthetic corpus (default)
  recall --vectors F --queries-file F   real data (.npy <f4 C-order or raw f32 LE)

DATA (synthetic):
  --n N                 corpus size                        [100000]
  --dim D               dimensionality                     [1024]
  --queries Q           number of queries                  [1000]
  --dist gmm|uniform    synthetic distribution             [gmm]
  --gmm-clusters C      mixture components (~n/100 is a    [1024]
                        realistic density; fewer = harder)
  --seed S              PRNG seed (data, queries, filters) [42]
DATA (files; both flags required together):
  --vectors PATH        corpus file (.npy or raw little-endian f32; raw needs --dim)
  --queries-file PATH   query file, same formats
  --normalize           L2-normalize file data (default: verify and abort if not normalized)
EVALUATION:
  --k LIST              recall depths                      [10,100]
  --rerank LIST         rerank factors (× k)               [2,4,8]
  --filter LIST         filter selectivities (1.0 = none)  [1.0]
  --full                also run int8 → f32 rerank rows
  --gt-cache PATH       reuse ground truth if params match, else compute and save
  --threads N           worker threads                     [available parallelism]
OUTPUT:
  --csv PATH            append machine-readable rows
  -h, --help            this text
";

fn die(msg: &str) -> ! {
    eprintln!("error: {msg}");
    exit(2);
}

fn req_string(parser: &mut lexopt::Parser, flag: &str) -> String {
    match parser.value() {
        Ok(v) => v.into_string().unwrap_or_else(|_| die(&format!("{flag}: non-utf8 value"))),
        Err(e) => die(&format!("{e}")),
    }
}

fn req_parse<T: FromStr>(parser: &mut lexopt::Parser, flag: &str) -> T {
    let s = req_string(parser, flag);
    s.parse().unwrap_or_else(|_| die(&format!("invalid value for {flag}: {s:?}")))
}

fn req_list<T: FromStr>(parser: &mut lexopt::Parser, flag: &str) -> Vec<T> {
    let s = req_string(parser, flag);
    s.split(',')
        .map(|part| {
            part.trim()
                .parse()
                .unwrap_or_else(|_| die(&format!("invalid {flag} element {part:?}")))
        })
        .collect()
}

pub fn parse() -> Config {
    let mut cfg = Config::default();
    let mut parser = lexopt::Parser::from_env();
    loop {
        let arg = match parser.next() {
            Ok(Some(arg)) => arg,
            Ok(None) => break,
            Err(e) => die(&format!("{e}\n\n{HELP}")),
        };
        use lexopt::prelude::*;
        match arg {
            Long("n") => cfg.n = req_parse(&mut parser, "--n"),
            Long("dim") => cfg.dim = req_parse(&mut parser, "--dim"),
            Long("queries") => cfg.queries = req_parse(&mut parser, "--queries"),
            Long("dist") => {
                cfg.dist = match req_string(&mut parser, "--dist").as_str() {
                    "gmm" => Dist::Gmm,
                    "uniform" => Dist::Uniform,
                    other => die(&format!("--dist must be gmm or uniform, got {other:?}")),
                }
            }
            Long("gmm-clusters") => cfg.gmm_clusters = req_parse(&mut parser, "--gmm-clusters"),
            Long("seed") => cfg.seed = req_parse(&mut parser, "--seed"),
            Long("vectors") => cfg.vectors_path = Some(req_string(&mut parser, "--vectors").into()),
            Long("queries-file") => {
                cfg.queries_path = Some(req_string(&mut parser, "--queries-file").into())
            }
            Long("normalize") => cfg.normalize = true,
            Long("k") => cfg.k = req_list(&mut parser, "--k"),
            Long("rerank") => cfg.rerank = req_list(&mut parser, "--rerank"),
            Long("filter") => cfg.filters = req_list(&mut parser, "--filter"),
            Long("full") => cfg.full = true,
            Long("gt-cache") => cfg.gt_cache = Some(req_string(&mut parser, "--gt-cache").into()),
            Long("threads") => cfg.threads = req_parse(&mut parser, "--threads"),
            Long("csv") => cfg.csv = Some(req_string(&mut parser, "--csv").into()),
            Short('h') | Long("help") => {
                print!("{HELP}");
                exit(0);
            }
            other => die(&format!("{}\n\n{HELP}", other.unexpected())),
        }
    }
    validate(&cfg);
    cfg
}

fn validate(cfg: &Config) {
    if cfg.vectors_path.is_some() != cfg.queries_path.is_some() {
        die("--vectors and --queries-file must be given together");
    }
    if cfg.dim == 0 || cfg.n == 0 || cfg.queries == 0 {
        die("--n, --dim and --queries must be > 0");
    }
    if cfg.k.is_empty() || cfg.k.contains(&0) {
        die("--k needs at least one depth > 0");
    }
    if cfg.rerank.is_empty() || cfg.rerank.contains(&0) {
        die("--rerank needs at least one factor > 0");
    }
    if cfg.filters.is_empty() || cfg.filters.iter().any(|&s| !(0.0..=1.0).contains(&s) || s == 0.0)
    {
        die("--filter selectivities must be in (0.0, 1.0]");
    }
    if cfg.threads == 0 {
        die("--threads must be > 0");
    }
    if cfg.dist == Dist::Gmm && cfg.gmm_clusters == 0 {
        die("--gmm-clusters must be > 0");
    }
}
