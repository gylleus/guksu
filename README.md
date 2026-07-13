# guksu

Quantized vector-search building blocks in pure Rust: int8/binary distance
kernels (symmetric **and asymmetric**), sign-bit/int8 quantizers, a flat
filtered top-k scan with rerank primitives, and a recall benchmark harness.

**Status: pre-alpha, API unstable, no graph yet.** This is milestone 0 of a
standalone quantized-HNSW crate — the part that measures *quantization* loss
in isolation, so the graph milestones can be designed around a quantization
config chosen on evidence.

## Scope (M0)

- `kernels` — f32 dot, exact int8 dot (full `[-128, 127]` domain), binary
  Hamming over packed sign bits, and asymmetric `f32×binary` / `int8×binary`
  dots. Portable scalar reference implementations plus AVX2 and NEON(+dotprod)
  paths selected by runtime feature detection; every SIMD path is tested for
  agreement against the scalar reference (exactly for integer kernels, within
  a summation-order bound for f32).
- `quant` — sign-bit binary packing (MSB-first, numpy
  `packbits(bitorder='big')` compatible), symmetric int8 quantization with
  per-vector or fixed scales, and parity helpers for verifying that
  client-side quantization reproduces an embedding provider's server-side
  quantized outputs.
- `scan` — exact top-k over contiguous quantized blocks with an optional
  filter bitmap, plus a rerank primitive for two-stage (coarse → precise)
  search; one `Scorer` enum variant per kernel. This is the production search
  path for small corpora, not just bench scaffolding.
- `recall` harness — computes f32 brute-force ground truth and reports
  recall@k across {int8, binary, binary→int8-rerank, binary→f32-rerank} ×
  {symmetric, asymmetric} × rerank depth, on synthetic or real data.

## Non-goals (for now)

- No HNSW or any graph index (M1+, see roadmap).
- No persistence or mmap format (M1) — but scans already read borrowed views,
  which is the seam the zero-copy format will implement.
- No IVF/PQ/RaBitQ, no training-based quantization.
- Not a server. Zero runtime dependencies by design (the harness binary adds
  exactly one, `lexopt`, behind the `harness` feature).

## Determinism contract

All result orderings are `(score descending, id ascending)` under
`f32::total_cmp` — including ground truth. Binary codes at 1024 bits admit
only 1025 distinct Hamming values, so score ties are the norm; the total
order is what makes recall numbers reproducible across machines. Integer
kernels are bit-identical across backends; f32 kernels may differ across
backends within a documented summation-order tolerance. Selection has the
prefix property: `top_k(k1) == top_k(k2)[..k1]` for `k1 <= k2`.

Backend selection: automatic per CPU (`scalar`, `neon`, `neon_dotprod`,
`avx2`). Pin one explicitly with `Kernels::by_name` — feature-checked, so a
name this CPU cannot execute is `None`, never a crash; `Kernels::NAMES` lists
the candidates. `GUKSU_REQUIRE=<backend>` makes the test suite fail if
detection picked anything else — for CI runners that must exercise a
specific path.

## Quickstart (library)

```rust
use guksu::{BinaryBlock, F32Block, Scorer, ViewQuery};

// Four 4-dim vectors, row-major.
let corpus = F32Block::from_flat(
    &[
        1.0, 0.0, 0.0, 0.0, //
        0.0, 1.0, 0.0, 0.0, //
        0.707, 0.707, 0.0, 0.0, //
        0.0, 0.0, 0.707, 0.707,
    ],
    4,
);
let bins = BinaryBlock::from_f32(&corpus); // 1 bit/dim coarse codes
let query = [0.9f32, 0.1, -0.1, -0.1];

// Two-stage search: asymmetric binary coarse scan → exact f32 rerank.
let coarse = Scorer::F32Bin {
    query: ViewQuery { view: bins.view(), query: &query },
};
let pool = coarse.top_k(3, None).unwrap();
let ids: Vec<u32> = pool.iter().map(|h| h.id).collect();
let precise = Scorer::F32 {
    query: ViewQuery { view: corpus.view(), query: &query },
};
let top = precise.rerank(&ids, 1).unwrap();
assert_eq!(top[0].id, 0);
```

## Recall harness

```sh
# Synthetic corpus (default: 100k × 1024-dim GMM, 1k queries), ~6 s on an M-series laptop
cargo run --release --features harness --bin recall

# The full filtered matrix, machine-readable output, cached ground truth
cargo run --release --features harness --bin recall -- \
    --filter 1.0,0.3,0.1,0.02 --csv results.csv --gt-cache gt.bin

# Real data: .npy (<f4, C-order, shape (n, dim)) or raw little-endian f32 (+ --dim)
cargo run --release --features harness --bin recall -- \
    --vectors corpus.npy --queries-file queries.npy
```

Sample output (synthetic GMM, n=100k, dim=1024, 1k queries, seed 42, Apple
M-series):

```
config                     R@10   R@100    recall@10   recall@100      us/q
f32 (control)                 -       -       1.0000       1.0000      1432
int8 sym                      -       -       0.9494       0.9973       248
bin sym                       -       -       0.2591       0.9314        30
bin asym f32q                 -       -       0.3237       0.9376       510
bin asym i8q                  -       -       0.3234       0.9376       219
bin sym       -> i8  x2      20     200       0.4364       0.9595        43
bin sym       -> i8  x4      40     400       0.6793       0.9664        45
bin sym       -> i8  x8      80     800       0.9137       0.9731        49
bin sym       -> f32 x2      20     200       0.4364       0.9609        48
bin sym       -> f32 x4      40     400       0.6807       0.9678        55
bin sym       -> f32 x8      80     800       0.9491       0.9746        71
bin asym f32q -> i8  x2      20     200       0.5177       0.9722       565
bin asym f32q -> i8  x4      40     400       0.7516       0.9818       567
bin asym f32q -> i8  x8      80     800       0.9336       0.9885       570
bin asym f32q -> f32 x2      20     200       0.5177       0.9736       571
bin asym f32q -> f32 x4      40     400       0.7571       0.9833       578
bin asym f32q -> f32 x8      80     800       0.9741       0.9902       593
bin asym i8q  -> i8  x2      20     200       0.5179       0.9722       233
bin asym i8q  -> i8  x4      40     400       0.7521       0.9818       235
bin asym i8q  -> i8  x8      80     800       0.9338       0.9885       238
bin asym i8q  -> f32 x2      20     200       0.5179       0.9736       238
bin asym i8q  -> f32 x4      40     400       0.7577       0.9833       246
bin asym i8q  -> f32 x8      80     800       0.9747       0.9901       261
```

How to read it: `R@k` is the coarse pool size (rerank factor × k); `us/q` is
wall-clock per query at the printed thread count (two-stage rows include the
shared max-depth coarse scan; use `cargo bench` for rigorous timings).
Patterns that already show on synthetic data: asymmetric beats symmetric
binary at every depth, the int8-query asymmetric variant matches the
f32-query one (int8 query error is second-order), rerank recall is monotone
in depth, and raw binary@10 collapses into Hamming tie plateaus long before
binary@100 does.

At 1M vectors (`--n 1000000 --gmm-clusters 8192`, ~2.5 min wall, ~5.5 GB
resident) the same patterns hold: int8 0.943@10/0.991@100, and
`bin asym → f32 ×8` reaches 0.926@10 / 0.9998@100 — with the int8-query
variant matching the f32-query one at 2.2× less scan cost.

**Synthetic numbers do not decide binary viability.** Recall levels depend
heavily on corpus geometry (`--gmm-clusters`: ~n/100 is a realistic density;
64 makes a brutally hard near-duplicate-heavy corpus — try both). Judgments
belong to a real embedding export via `--vectors`/`--queries-file`.

`--full` adds the int8→f32-rerank rows. `--help` documents everything else.

## Quantization parity with embedding providers

Several embedding APIs can return pre-quantized vectors (int8 or bit-packed
binary) alongside f32. The client-side quantizers here are meant to reproduce
those outputs, so a corpus can be quantized locally from an f32 export
without re-embedding. `quant::binary_parity` / `quant::int8_parity` measure
how close that is on a sample; they take pre-fetched arrays — the library
does no network I/O.

Interop notes:

- The packed-binary layout (MSB-first within each byte) is numpy
  `packbits(bitorder='big')`, which is what providers generally serve —
  unsigned packed binary ingests as-is, bit-identical.
- Some providers serve packed binary as **offset-binary i8** (`u8 − 128`);
  `quant` includes the per-byte conversion (XOR 0x80). A
  `bit_mismatch_rate ≈ 0.125` concentrated on the first bit of every byte is
  the signature of skipping that conversion.
- Provider int8 may use the full `[-128, 127]` range; the kernels accept it
  exactly (guksu's own quantizer emits `[-127, 127]`).

## Benches

```sh
cargo bench                                              # kernels + scan
cargo bench --features bench-simsimd --bench simsimd_compare   # C cross-check
```

Measured on an Apple M-series core (d = 1024, `neon_dotprod` backend):

| kernel | scalar | SIMD | row bytes | stream throughput |
|---|---|---|---|---|
| `dot_f32` | 420 ns | 49–55 ns | 4096 | ~69 GiB/s |
| `dot_i8` | 12.6 ns† | 12.0–13.9 ns | 1024 | ~68 GiB/s |
| `hamming` | 3.3 ns† | 3.0 ns | 128 | ~39 GiB/s |
| `dot_f32_bin` | 609 ns | 75 ns | 128 | compute-bound |
| `dot_i8_bin` | 356 ns | 31 ns | 128 | compute-bound |

† LLVM auto-vectorizes the integer reference loops well on aarch64.

Stream rows/s ratios land at binary:int8:f32 ≈ 18:4:1 against the byte-ratio
ideal of 32:4:1 — Hamming is popcount-compute-bound, not bandwidth-bound,
within the spec's 2× sanity bar. Full 100k×1024 scans: f32 5.4 ms, int8
1.35 ms, binary 0.36 ms, asym f32×bin 7.2 ms — note the asymmetric coarse
scan pays for touching all 1024 query floats per row; its win is the 32×
smaller resident footprint, not flat-scan wall time. Cross-check: guksu
kernels measured 1.5–3.2× *faster* than simsimd's C kernels on this machine.

x86_64/AVX2 is compiled and unit-tested via cross-compilation, but Rosetta 2
does not advertise AVX2 (detection falls back to scalar there — the
`GUKSU_REQUIRE=avx2` guard makes that impossible to miss); the authoritative
AVX2 execution check is an x86_64 CI runner.

## Roadmap

- **M1** — zero-copy graph format + reader: `HnswView::open(&[u8])` at an
  arbitrary offset, flat adjacency arrays, `VectorStore` trait so vectors
  live outside the graph (the `*View` types here are that seam).
- **M2** — parallel builder + serialization, filtered graph search,
  iso-recall benches vs hnswlib-rs/usearch/faiss (build throughput in
  vectors/s/core).
- **M3** — seeded construction (init from an existing view, insert deltas);
  multi-graph iso-recall cost bench.
- **M4** — fuzz the format parser, miri on unsafe code, docs, publish 0.1.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or
  http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or
  http://opensource.org/licenses/MIT)

at your option.

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the Apache-2.0
license, shall be dual licensed as above, without any additional terms or
conditions.
