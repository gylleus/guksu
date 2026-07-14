//! Owned int8 codes with per-vector scales, and the borrowed view scans read.

use std::fmt;

use crate::quant;
use super::{Block, F32Block, ROW_ALIGN, StorageError, check_rows, round_up};

/// Owned int8 codes with per-vector scales (`x ≈ code as f32 * scale`).
pub struct I8Block {
    data: Box<[i8]>,
    scales: Box<[f32]>,
    dim: usize,
    stride: usize,
    len: usize,
}

impl fmt::Debug for I8Block {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("I8Block")
            .field("rows", &self.len)
            .field("dim", &self.dim)
            .field("stride", &self.stride)
            .finish_non_exhaustive()
    }
}

impl I8Block {
    /// Quantize each row of `src` with its own `max|x| / 127` scale
    /// (all-zero rows get scale 0 and all-zero codes).
    pub fn from_f32_per_vector(src: &F32Block) -> Self {
        Self::quantize_rows(src, quant::max_abs_scale)
    }

    /// Quantize every row with one caller-supplied scale (e.g.
    /// [`quant::fixed_scale`] over a sample).
    pub fn from_f32_fixed(src: &F32Block, scale: f32) -> Self {
        Self::quantize_rows(src, |_| scale)
    }

    fn quantize_rows(src: &F32Block, scale_of: impl Fn(&[f32]) -> f32) -> Self {
        let (dim, len) = (src.dim(), src.len());
        let stride = round_up(dim, ROW_ALIGN);
        let mut data = vec![0i8; len * stride];
        let mut scales = vec![0.0f32; len];
        let sv = src.view();
        for i in 0..len {
            let row = sv.row(i as u32);
            let scale = scale_of(row);
            scales[i] = scale;
            quant::quantize_i8(row, scale, &mut data[i * stride..i * stride + dim]);
        }
        Self {
            data: data.into_boxed_slice(),
            scales: scales.into_boxed_slice(),
            dim,
            stride,
            len,
        }
    }
}

impl Block for I8Block {
    type View<'a>
        = I8View<'a>
    where
        Self: 'a;

    fn len(&self) -> usize {
        self.len
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn view(&self) -> I8View<'_> {
        I8View {
            data: &self.data,
            scales: &self.scales,
            dim: self.dim,
            stride: self.stride,
        }
    }
}

/// Borrowed int8 codes + parallel per-vector scales.
#[derive(Clone, Copy)]
pub struct I8View<'a> {
    data: &'a [i8],
    scales: &'a [f32],
    dim: usize,
    stride: usize,
}

impl fmt::Debug for I8View<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("I8View")
            .field("rows", &self.len())
            .field("dim", &self.dim)
            .field("stride", &self.stride)
            .finish_non_exhaustive()
    }
}

impl<'a> I8View<'a> {
    /// Wrap raw parts. Fails with [`StorageError`] unless shapes are
    /// consistent (`stride >= dim > 0`, whole rows, one scale per row).
    pub fn new(
        data: &'a [i8],
        scales: &'a [f32],
        dim: usize,
        stride: usize,
    ) -> Result<Self, StorageError> {
        if dim == 0 {
            return Err(StorageError::ZeroDim);
        }
        if stride < dim {
            return Err(StorageError::Stride { stride, min_len: dim });
        }
        if data.len() % stride != 0 {
            return Err(StorageError::Ragged { data_len: data.len(), row_len: stride });
        }
        let len = data.len() / stride;
        check_rows(len)?;
        if scales.len() != len {
            return Err(StorageError::ScaleCount { scales: scales.len(), rows: len });
        }
        Ok(Self {
            data,
            scales,
            dim,
            stride,
        })
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

    pub fn row(&self, i: u32) -> &'a [i8] {
        let start = i as usize * self.stride;
        &self.data[start..start + self.dim]
    }

    pub fn scale(&self, i: u32) -> f32 {
        self.scales[i as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::{max_abs_scale, quantize_i8_vec};
    use crate::rng::SplitMix64;

    fn rows(rng: &mut SplitMix64, n: usize, dim: usize) -> Vec<f32> {
        (0..n * dim).map(|_| rng.next_f32() * 2.0 - 1.0).collect()
    }

    #[test]
    fn i8_per_vector_matches_row_quantizer() {
        let mut rng = SplitMix64::new(32);
        let (n, dim) = (5, 130);
        let flat = rows(&mut rng, n, dim);
        let block = F32Block::from_flat(&flat, dim).unwrap();
        let q = I8Block::from_f32_per_vector(&block);
        let qv = q.view();
        for i in 0..n {
            let row = &flat[i * dim..(i + 1) * dim];
            let scale = max_abs_scale(row);
            assert_eq!(qv.scale(i as u32), scale);
            assert_eq!(qv.row(i as u32), quantize_i8_vec(row, scale).as_slice());
        }
    }

    #[test]
    fn i8_fixed_scale_applies_everywhere() {
        let block = F32Block::from_flat(&[0.5, -0.5, 1.0, 0.25], 2).unwrap();
        let q = I8Block::from_f32_fixed(&block, 1.0 / 127.0);
        let qv = q.view();
        assert_eq!(qv.scale(0), 1.0 / 127.0);
        assert_eq!(qv.scale(1), 1.0 / 127.0);
        assert_eq!(qv.row(0), &[64, -64]);
        assert_eq!(qv.row(1), &[127, 32]);
    }

    #[test]
    fn i8_zero_row_gets_zero_scale_and_codes() {
        let block = F32Block::from_flat(&[0.0, 0.0, 1.0, -1.0], 2).unwrap();
        let q = I8Block::from_f32_per_vector(&block);
        let qv = q.view();
        assert_eq!(qv.scale(0), 0.0);
        assert_eq!(qv.row(0), &[0, 0]);
        assert_eq!(qv.row(1), &[127, -127]);
    }
}
