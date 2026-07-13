//! Aligned stdout table + machine-readable CSV appender.

use std::io::Write;
use std::path::Path;

use crate::eval::Row;

pub fn print_table(rows: &[Row], ks: &[usize]) {
    let label_w =
        rows.iter().map(|r| r.label().len()).chain(["config".len()]).max().unwrap_or(6);
    print!("{:<label_w$}", "config");
    for &k in ks {
        print!("  {:>6}", format!("R@{k}"));
    }
    for &k in ks {
        print!("  {:>11}", format!("recall@{k}"));
    }
    println!("  {:>8}", "us/q");
    for r in rows {
        print!("{:<label_w$}", r.label());
        for p in &r.pool_at {
            match p {
                Some(v) => print!("  {v:>6}"),
                None => print!("  {:>6}", "-"),
            }
        }
        for rec in &r.recalls {
            print!("  {rec:>11.4}");
        }
        println!("  {:>8.0}", r.us_per_query);
    }
}

pub struct CsvMeta<'a> {
    pub n: usize,
    pub dim: usize,
    pub queries: usize,
    pub seed: u64,
    pub backend: &'a str,
}

/// One CSV line per (filter, config, k); header written when creating the file.
pub fn append_csv(
    path: &Path,
    filter_s: f64,
    rows: &[Row],
    ks: &[usize],
    meta: &CsvMeta,
) -> std::io::Result<()> {
    let fresh = !path.exists();
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(path)?;
    if fresh {
        writeln!(
            f,
            "filter_s,config,coarse,rerank,factor,k,r,recall,us_per_query,n,dim,queries,seed,backend"
        )?;
    }
    for r in rows {
        for (ki, &k) in ks.iter().enumerate() {
            writeln!(
                f,
                "{filter_s},{},{},{},{},{k},{},{:.6},{:.1},{},{},{},{},{}",
                r.label(),
                r.coarse,
                r.store.unwrap_or(""),
                r.factor.map(|x| x.to_string()).unwrap_or_default(),
                r.pool_at[ki].map(|x| x.to_string()).unwrap_or_default(),
                r.recalls[ki],
                r.us_per_query,
                meta.n,
                meta.dim,
                meta.queries,
                meta.seed,
                meta.backend,
            )?;
        }
    }
    Ok(())
}
