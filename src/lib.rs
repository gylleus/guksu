//! Quantized vector-search building blocks: distance kernels (f32, int8,
//! binary, and asymmetric float/int8-query × binary-doc), sign-bit/int8
//! quantizers, and a flat filtered top-k scan with rerank primitives.
//!
//! This is milestone 0 of a standalone quantized-HNSW crate: everything needed
//! to measure quantization loss in isolation (no graph yet — traversal loss is
//! a later milestone's concern). The flat scan is production API, not bench
//! scaffolding: it is the intended search path for small corpora.
//!
//! # Conventions
//!
//! - **Scores are f32, higher is better**, in every public API. Raw
//!   [`kernels::hamming`] is the one lower-is-better exception, and
//!   [`kernels::hamming_score`] adapts it.
//! - **Scores are representation-local.** A Hamming-derived score and a dot
//!   score are not comparable; reranking recomputes scores from the
//!   higher-precision store and never blends them with coarse scores.
//! - **Ordering is total and deterministic**: results sort by
//!   `(score descending, id ascending)` using `f32::total_cmp`. Binary codes
//!   at 1024 bits admit only 1025 distinct Hamming values, so massive score
//!   ties are the norm, not the edge case — the tie rule is load-bearing.
//! - **Shape mismatches are errors, not panics, at the public seams**:
//!   [`Scorer::top_k`] and [`Scorer::rerank`] report a query/filter that does
//!   not match the view as a [`ScanError`]; block and view constructors report
//!   inconsistent geometry as a [`StorageError`]. [`Error`] wraps both for
//!   callers composing the seams. Low-level kernels and row accessors keep
//!   their assert-based contracts.
//! - Vector data is always borrowed (`&[f32]`, `&[i8]`, `&[u8]` rows); hot
//!   paths never allocate. Owned blocks exist only as convenience containers.
//!
//! # Quickstart
//!
//! ```
//! use guksu::{BinaryBlock, Block, F32Block, Scorer, ViewQuery};
//!
//! // Four 4-dim vectors, row-major.
//! let corpus = F32Block::from_flat(
//!     &[
//!         1.0, 0.0, 0.0, 0.0, //
//!         0.0, 1.0, 0.0, 0.0, //
//!         0.707, 0.707, 0.0, 0.0, //
//!         0.0, 0.0, 0.707, 0.707,
//!     ],
//!     4,
//! )
//! .unwrap();
//! let bins = BinaryBlock::from_f32(&corpus); // 1 bit/dim coarse codes
//! let query = [0.9f32, 0.1, -0.1, -0.1];
//!
//! // Two-stage search: asymmetric binary coarse scan → exact f32 rerank.
//! let coarse = Scorer::F32Bin {
//!     query: ViewQuery { view: bins.view(), query: &query },
//! };
//! let pool = coarse.top_k(3, None).unwrap();
//! let ids: Vec<u32> = pool.iter().map(|h| h.id).collect();
//! let precise = Scorer::F32 {
//!     query: ViewQuery { view: corpus.view(), query: &query },
//! };
//! let top = precise.rerank(&ids, 1).unwrap();
//! assert_eq!(top[0].id, 0);
//! ```

#![deny(unsafe_op_in_unsafe_fn)]

mod bitset;
mod error;
pub mod kernels;
pub mod quant;
mod query;
#[doc(hidden)]
pub mod rng;
pub mod scan;
mod storage;

pub use bitset::Bitset;
pub use error::Error;
pub use query::{CorpusView, I8Query, QueryData, ViewQuery};
pub use scan::{Hit, ScanError, Scorer, select_top_k};
pub use storage::{
    BinaryBlock, BinaryView, Block, F32Block, F32View, I8Block, I8View, StorageError,
};
