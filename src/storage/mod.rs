//! Owned row-major blocks and the borrowed views scans operate on.
//!
//! Blocks are convenience containers (copy-in constructors + quantizing
//! constructors) sharing the [`Block`] trait; **all reading goes through
//! `Copy` views**, which is the seam a later zero-copy mmap format implements
//! — scan and kernel code never changes when rows come from a mapped file
//! instead of a `Vec`.
//!
//! Constructors that ingest foreign geometry — [`F32Block::from_flat`] and
//! the raw-parts `*View::new` — validate it and return [`StorageError`]; the
//! quantizing constructors are infallible because their source block is
//! already valid. Row accessors keep assert-based contracts (an out-of-range
//! id is the caller's bug), per the errors-at-the-seam rule. `Debug` for
//! blocks and views prints geometry, never row data (corpora are huge).
//!
//! Row starts are padded to a 64-byte stride (at dim 1024 every representation
//! is naturally 64-byte-strided, so padding is free). Views require only
//! `stride >= row_len`; absolute allocation alignment is NOT guaranteed here,
//! so kernels use unaligned loads.
//!
//! Each representation lives in its own submodule (`f32`, `int8`, `binary`);
//! this module owns the shared [`Block`] trait and the geometry helpers.

use crate::scan::CorpusView;

mod binary;
mod error;
mod f32;
mod int8;

pub use binary::{BinaryBlock, BinaryView};
pub use error::StorageError;
pub use f32::{F32Block, F32View};
pub use int8::{I8Block, I8View};

/// Row stride granularity in bytes.
const ROW_ALIGN: usize = 64;

fn round_up(x: usize, to: usize) -> usize {
    x.div_ceil(to) * to
}

fn check_rows(len: usize) -> Result<(), StorageError> {
    if len > u32::MAX as usize {
        return Err(StorageError::TooManyRows { rows: len });
    }
    Ok(())
}

/// What every owned block shares: its geometry and the borrowed view scans
/// read. [`Block::is_empty`] derives from [`Block::len`]; `dim` is in
/// original-vector units (a [`BinaryBlock`] reports bits).
pub trait Block {
    /// The `Copy` view this block lends out — the scannable face of the data.
    type View<'a>: CorpusView
    where
        Self: 'a;

    /// Rows in the block.
    fn len(&self) -> usize;

    /// Row dimension in original-vector units.
    fn dim(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Borrowed view for scanning.
    fn view(&self) -> Self::View<'_>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_blocks_are_fine() {
        let block = F32Block::from_flat(&[], 8).unwrap();
        assert_eq!(block.len(), 0);
        assert!(block.is_empty());
        assert_eq!(I8Block::from_f32_per_vector(&block).len(), 0);
        assert_eq!(BinaryBlock::from_f32(&block).len(), 0);
    }

    #[test]
    fn block_trait_covers_every_representation() {
        fn geometry(b: &impl Block) -> (usize, usize, bool) {
            (b.len(), b.dim(), b.is_empty())
        }
        let f = F32Block::from_flat(&[1.0, -1.0, 0.5, 0.25], 2).unwrap();
        assert_eq!(geometry(&f), (2, 2, false));
        assert_eq!(geometry(&I8Block::from_f32_per_vector(&f)), (2, 2, false));
        assert_eq!(geometry(&BinaryBlock::from_f32(&f)), (2, 2, false));
    }

    #[test]
    fn view_new_rejects_bad_shapes() {
        assert_eq!(
            F32View::new(&[0.0; 8], 8, 4).unwrap_err(),
            StorageError::Stride { stride: 4, min_len: 8 }
        );
        assert_eq!(
            F32View::new(&[0.0; 9], 4, 8).unwrap_err(),
            StorageError::Ragged { data_len: 9, row_len: 8 }
        );
        assert_eq!(F32View::new(&[], 0, 4).unwrap_err(), StorageError::ZeroDim);
        assert_eq!(
            I8View::new(&[0i8; 8], &[1.0], 4, 4).unwrap_err(),
            StorageError::ScaleCount { scales: 1, rows: 2 }
        );
        // Binary strides are bytes: dim 17 packs to 3 bytes, so stride 2 is short.
        assert_eq!(
            BinaryView::new(&[0u8; 2], 17, 2).unwrap_err(),
            StorageError::Stride { stride: 2, min_len: 3 }
        );
    }
}
