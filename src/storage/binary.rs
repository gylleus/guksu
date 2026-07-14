//! Owned sign-bit-packed codes and the borrowed view scans read.

use std::fmt;

use crate::kernels::binary_code_len;
use crate::quant;
use super::{Block, F32Block, ROW_ALIGN, StorageError, check_rows, round_up};

/// Owned sign-bit-packed vectors (MSB-first; see `kernels` module docs).
pub struct BinaryBlock {
    data: Box<[u8]>,
    dim_bits: usize,
    stride_bytes: usize,
    len: usize,
}

impl fmt::Debug for BinaryBlock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BinaryBlock")
            .field("rows", &self.len)
            .field("dim_bits", &self.dim_bits)
            .field("stride_bytes", &self.stride_bytes)
            .finish_non_exhaustive()
    }
}

impl BinaryBlock {
    /// Pack the sign bits of each row of `src` (`x > 0.0` → bit set).
    pub fn from_f32(src: &F32Block) -> Self {
        let (dim, len) = (src.dim(), src.len());
        let code_len = binary_code_len(dim);
        let stride = round_up(code_len, ROW_ALIGN);
        let mut data = vec![0u8; len * stride];
        let sv = src.view();
        for i in 0..len {
            quant::pack_sign_bits(
                sv.row(i as u32),
                &mut data[i * stride..i * stride + code_len],
            );
        }
        Self {
            data: data.into_boxed_slice(),
            dim_bits: dim,
            stride_bytes: stride,
            len,
        }
    }
}

impl Block for BinaryBlock {
    type View<'a>
        = BinaryView<'a>
    where
        Self: 'a;

    fn len(&self) -> usize {
        self.len
    }

    /// Dimensionality in BITS (the original vector dim).
    fn dim(&self) -> usize {
        self.dim_bits
    }

    fn view(&self) -> BinaryView<'_> {
        BinaryView {
            data: &self.data,
            dim_bits: self.dim_bits,
            stride_bytes: self.stride_bytes,
        }
    }
}

/// Borrowed packed sign-bit codes.
#[derive(Clone, Copy)]
pub struct BinaryView<'a> {
    data: &'a [u8],
    dim_bits: usize,
    stride_bytes: usize,
}

impl fmt::Debug for BinaryView<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BinaryView")
            .field("rows", &self.len())
            .field("dim_bits", &self.dim_bits)
            .field("stride_bytes", &self.stride_bytes)
            .finish_non_exhaustive()
    }
}

impl<'a> BinaryView<'a> {
    /// Wrap raw parts. Fails with [`StorageError`] unless
    /// `stride_bytes >= binary_code_len(dim_bits)` and `data` is whole
    /// strided rows.
    pub fn new(data: &'a [u8], dim_bits: usize, stride_bytes: usize) -> Result<Self, StorageError> {
        if dim_bits == 0 {
            return Err(StorageError::ZeroDim);
        }
        let code_len = binary_code_len(dim_bits);
        if stride_bytes < code_len {
            return Err(StorageError::Stride { stride: stride_bytes, min_len: code_len });
        }
        if data.len() % stride_bytes != 0 {
            return Err(StorageError::Ragged { data_len: data.len(), row_len: stride_bytes });
        }
        check_rows(data.len() / stride_bytes)?;
        Ok(Self {
            data,
            dim_bits,
            stride_bytes,
        })
    }

    pub fn len(&self) -> usize {
        self.data.len() / self.stride_bytes
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Dimensionality in BITS.
    pub fn dim(&self) -> usize {
        self.dim_bits
    }

    /// Row `i`, exactly `binary_code_len(dim)` bytes (stride padding excluded).
    pub fn row(&self, i: u32) -> &'a [u8] {
        let start = i as usize * self.stride_bytes;
        &self.data[start..start + binary_code_len(self.dim_bits)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::pack_sign_bits_vec;
    use crate::rng::SplitMix64;

    fn rows(rng: &mut SplitMix64, n: usize, dim: usize) -> Vec<f32> {
        (0..n * dim).map(|_| rng.next_f32() * 2.0 - 1.0).collect()
    }

    #[test]
    fn bin_rows_match_row_packer() {
        let mut rng = SplitMix64::new(33);
        let (n, dim) = (4, 1027);
        let flat = rows(&mut rng, n, dim);
        let block = F32Block::from_flat(&flat, dim).unwrap();
        let b = BinaryBlock::from_f32(&block);
        assert_eq!(b.dim(), dim);
        let bv = b.view();
        for i in 0..n {
            let expected = pack_sign_bits_vec(&flat[i * dim..(i + 1) * dim]);
            assert_eq!(bv.row(i as u32), expected.as_slice(), "row {i}");
        }
    }
}
