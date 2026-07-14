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

use std::fmt;

use crate::kernels::binary_code_len;
use crate::quant;
use crate::query::CorpusView;

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

/// A block or view constructor rejected inconsistent geometry. Each variant
/// carries both sides of the mismatch, in the representation's own row units
/// (f32 elements, i8 codes, or packed-code bytes).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StorageError {
    /// The row dimension was zero.
    ZeroDim,
    /// The data length is not a whole number of rows — rows of `dim` in
    /// [`F32Block::from_flat`], rows of `stride` in the view constructors.
    Ragged { data_len: usize, row_len: usize },
    /// The stride cannot hold even one row.
    Stride { stride: usize, min_len: usize },
    /// An int8 view carries exactly one scale per row.
    ScaleCount { scales: usize, rows: usize },
    /// More rows than the `u32` id space addresses.
    TooManyRows { rows: usize },
}

impl fmt::Display for StorageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            StorageError::ZeroDim => write!(f, "dim must be > 0"),
            StorageError::Ragged { data_len, row_len } => {
                write!(f, "data length {data_len} is not whole rows of {row_len}")
            }
            StorageError::Stride { stride, min_len } => {
                write!(f, "stride {stride} cannot hold rows of length {min_len}")
            }
            StorageError::ScaleCount { scales, rows } => {
                write!(f, "expected one scale per row: {scales} vs {rows}")
            }
            StorageError::TooManyRows { rows } => {
                write!(f, "{rows} rows exceeds the u32 id space")
            }
        }
    }
}

impl std::error::Error for StorageError {}

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

// ---------------------------------------------------------------- f32

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

// ---------------------------------------------------------------- int8

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

// ---------------------------------------------------------------- binary

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
    use crate::quant::{max_abs_scale, pack_sign_bits_vec, quantize_i8_vec};
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

    #[test]
    fn storage_error_messages() {
        assert_eq!(StorageError::ZeroDim.to_string(), "dim must be > 0");
        assert_eq!(
            StorageError::Ragged { data_len: 3, row_len: 2 }.to_string(),
            "data length 3 is not whole rows of 2"
        );
        assert_eq!(
            StorageError::Stride { stride: 4, min_len: 8 }.to_string(),
            "stride 4 cannot hold rows of length 8"
        );
        assert_eq!(
            StorageError::ScaleCount { scales: 1, rows: 2 }.to_string(),
            "expected one scale per row: 1 vs 2"
        );
        assert_eq!(
            StorageError::TooManyRows { rows: 1 << 33 }.to_string(),
            "8589934592 rows exceeds the u32 id space"
        );
    }
}
