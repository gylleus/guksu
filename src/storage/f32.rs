//! Owned row-major f32 vectors and the borrowed view scans read.

use std::fmt;

use super::{Block, ROW_ALIGN, StorageError, check_rows, round_up};

/// Owned row-major f32 vectors, rows padded to a 64-byte stride.
pub struct F32Block {
    data: Box<[f32]>,
    dim: usize,
    stride: usize,
    len: usize,
}

impl fmt::Debug for F32Block {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("F32Block")
            .field("rows", &self.len)
            .field("dim", &self.dim)
            .field("stride", &self.stride)
            .finish_non_exhaustive()
    }
}

impl F32Block {
    /// Copy `data.len() / dim` rows into a padded block. Fails with
    /// [`StorageError::ZeroDim`] or [`StorageError::Ragged`] if `data` is not
    /// whole rows of a positive `dim`.
    pub fn from_flat(data: &[f32], dim: usize) -> Result<Self, StorageError> {
        if dim == 0 {
            return Err(StorageError::ZeroDim);
        }
        if data.len() % dim != 0 {
            return Err(StorageError::Ragged { data_len: data.len(), row_len: dim });
        }
        let len = data.len() / dim;
        check_rows(len)?;
        let stride = round_up(dim * 4, ROW_ALIGN) / 4;
        let mut padded = vec![0.0f32; len * stride];
        for (src, dst) in data.chunks_exact(dim).zip(padded.chunks_exact_mut(stride)) {
            dst[..dim].copy_from_slice(src);
        }
        Ok(Self {
            data: padded.into_boxed_slice(),
            dim,
            stride,
            len,
        })
    }
}

impl Block for F32Block {
    type View<'a>
        = F32View<'a>
    where
        Self: 'a;

    fn len(&self) -> usize {
        self.len
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn view(&self) -> F32View<'_> {
        F32View {
            data: &self.data,
            dim: self.dim,
            stride: self.stride,
        }
    }
}

/// Borrowed row-major f32 data.
#[derive(Clone, Copy)]
pub struct F32View<'a> {
    data: &'a [f32],
    dim: usize,
    stride: usize,
}

impl fmt::Debug for F32View<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("F32View")
            .field("rows", &self.len())
            .field("dim", &self.dim)
            .field("stride", &self.stride)
            .finish_non_exhaustive()
    }
}

impl<'a> F32View<'a> {
    /// Wrap raw parts. Fails with [`StorageError`] unless `stride >= dim > 0`,
    /// `data` is whole strided rows, and the row count fits the u32 id space.
    pub fn new(data: &'a [f32], dim: usize, stride: usize) -> Result<Self, StorageError> {
        if dim == 0 {
            return Err(StorageError::ZeroDim);
        }
        if stride < dim {
            return Err(StorageError::Stride { stride, min_len: dim });
        }
        if data.len() % stride != 0 {
            return Err(StorageError::Ragged { data_len: data.len(), row_len: stride });
        }
        check_rows(data.len() / stride)?;
        Ok(Self { data, dim, stride })
    }

    pub fn len(&self) -> usize {
        self.data.len() / self.stride
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Row `i`, exactly `dim` long (padding excluded). The returned slice
    /// borrows the underlying data (`'a`), not the view.
    pub fn row(&self, i: u32) -> &'a [f32] {
        let start = i as usize * self.stride;
        &self.data[start..start + self.dim]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::SplitMix64;

    fn rows(rng: &mut SplitMix64, n: usize, dim: usize) -> Vec<f32> {
        (0..n * dim).map(|_| rng.next_f32() * 2.0 - 1.0).collect()
    }

    #[test]
    fn f32_round_trip_and_odd_dim_stride() {
        let mut rng = SplitMix64::new(31);
        let (n, dim) = (3, 1000);
        let flat = rows(&mut rng, n, dim);
        let block = F32Block::from_flat(&flat, dim).unwrap();
        assert_eq!(block.len(), n);
        assert_eq!(block.dim(), dim);
        let v = block.view();
        for i in 0..n {
            assert_eq!(v.row(i as u32), &flat[i * dim..(i + 1) * dim]);
        }
        // Rows start on 64-byte boundaries relative to row 0 (dim 1000 → stride 1008 f32).
        let byte_offset = v.row(1).as_ptr() as usize - v.row(0).as_ptr() as usize;
        assert_eq!(byte_offset, 1008 * 4);
        assert_eq!(byte_offset % 64, 0);
    }

    #[test]
    fn view_new_validates() {
        let data = vec![0.0f32; 32];
        let v = F32View::new(&data, 10, 16).unwrap();
        assert_eq!(v.len(), 2);
        assert_eq!(v.row(1).len(), 10);
    }

    #[test]
    fn debug_prints_geometry_not_data() {
        let block = F32Block::from_flat(&[1.0, -1.0, 0.5, 0.25], 2).unwrap();
        assert_eq!(
            format!("{block:?}"),
            "F32Block { rows: 2, dim: 2, stride: 16, .. }"
        );
        assert_eq!(
            format!("{:?}", block.view()),
            "F32View { rows: 2, dim: 2, stride: 16, .. }"
        );
    }

    #[test]
    fn from_flat_rejects_bad_shapes() {
        assert_eq!(
            F32Block::from_flat(&[1.0, 2.0, 3.0], 2).unwrap_err(),
            StorageError::Ragged { data_len: 3, row_len: 2 }
        );
        assert_eq!(F32Block::from_flat(&[], 0).unwrap_err(), StorageError::ZeroDim);
    }
}
