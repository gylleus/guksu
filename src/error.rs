//! Crate-level error type: one [`Error`] wrapping each seam's specific error.
//!
//! Fallible APIs return their own seam's error so matches stay narrow —
//! [`ScanError`] from scans, [`StorageError`] from block/view constructors.
//! [`Error`] plus the `From` impls exist for callers composing several seams
//! behind one `?`. Impls are hand-written: the library is dependency-free by
//! design, and two small enums do not justify a proc-macro dependency.

use std::fmt;

use crate::scan::ScanError;
use crate::storage::StorageError;

/// Any guksu error; each variant wraps one seam's specific error.
///
/// The wrapper is transparent: `Display` and `source` behave exactly like the
/// wrapped error's own, so routing through `Error` adds no message noise.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    /// A scan precondition violation — see [`ScanError`].
    Scan(ScanError),
    /// Rejected block/view geometry — see [`StorageError`].
    Storage(StorageError),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Error::Scan(e) => write!(f, "{e}"),
            Error::Storage(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Scan(e) => std::error::Error::source(e),
            Error::Storage(e) => std::error::Error::source(e),
        }
    }
}

impl From<ScanError> for Error {
    fn from(e: ScanError) -> Self {
        Error::Scan(e)
    }
}

impl From<StorageError> for Error {
    fn from(e: StorageError) -> Self {
        Error::Storage(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_wraps_and_display_forwards() {
        let scan = ScanError::QueryDim { query_dim: 1, view_dim: 2 };
        let e: Error = scan.into();
        assert_eq!(e, Error::Scan(scan));
        assert_eq!(e.to_string(), scan.to_string());

        let storage = StorageError::ZeroDim;
        let e: Error = storage.into();
        assert_eq!(e, Error::Storage(storage));
        assert_eq!(e.to_string(), storage.to_string());
    }

    #[test]
    fn question_mark_composes_both_seams() {
        fn scan_side() -> Result<(), Error> {
            Err(ScanError::FilterLen { filter_len: 1, rows: 2 })?;
            Ok(())
        }
        fn storage_side() -> Result<(), Error> {
            Err(StorageError::TooManyRows { rows: usize::MAX })?;
            Ok(())
        }
        assert!(matches!(scan_side(), Err(Error::Scan(ScanError::FilterLen { .. }))));
        assert!(matches!(storage_side(), Err(Error::Storage(StorageError::TooManyRows { .. }))));
    }
}
