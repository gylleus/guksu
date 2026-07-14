//! Rejected block and view geometry.

use std::fmt;

/// A block or view constructor rejected inconsistent geometry. Each variant
/// carries both sides of the mismatch, in the representation's own row units
/// (f32 elements, i8 codes, or packed-code bytes).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StorageError {
    /// The row dimension was zero.
    ZeroDim,
    /// The data length is not a whole number of rows — rows of `dim` in
    /// [`F32Block::from_flat`](crate::F32Block::from_flat), rows of `stride`
    /// in the view constructors.
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

#[cfg(test)]
mod tests {
    use super::*;

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
