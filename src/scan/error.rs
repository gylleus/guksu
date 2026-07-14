//! Scan precondition failures.

use std::fmt;

/// A scan precondition violation — the query or filter shape does not match
/// the view it is scored against. Returned by
/// [`Scorer::top_k`](crate::Scorer::top_k) and
/// [`Scorer::rerank`](crate::Scorer::rerank); each variant carries both sides
/// of the mismatch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScanError {
    /// The filter covers a different id universe than the view has rows.
    FilterLen { filter_len: usize, rows: usize },
    /// The query vector's dimension differs from the view's row dimension.
    QueryDim { query_dim: usize, view_dim: usize },
}

impl fmt::Display for ScanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            ScanError::FilterLen { filter_len, rows } => {
                write!(
                    f,
                    "filter covers {filter_len} ids but the view has {rows} rows"
                )
            }
            ScanError::QueryDim {
                query_dim,
                view_dim,
            } => {
                write!(f, "query dim {query_dim} vs view dim {view_dim}")
            }
        }
    }
}

impl std::error::Error for ScanError {}
