//! The (corpus view × query encoding) pairing a [`Scorer`] variant scores.
//!
//! [`ViewQuery`] bundles both sides of a scan into one field. [`CorpusView`]
//! names each view's canonical query encoding — the one shaped like the rows —
//! which is `ViewQuery`'s default `Q`, so only the asymmetric pairings (an f32
//! or int8 query against binary rows) spell their query type. [`QueryData`]
//! reports the dim the scan validates against the view; packed sign bits opt
//! out (their length is bytes while a binary view's dim is bits) and are
//! validated per row by the hamming kernel instead.
//!
//! [`Scorer`]: crate::scan::Scorer

use crate::scan::ScanError;
use crate::storage::{BinaryView, F32View, I8View};

/// A corpus representation scannable by a [`Scorer`](crate::scan::Scorer):
/// its geometry plus the canonical query encoding scored against it.
pub trait CorpusView: Copy {
    /// The same-encoding-as-the-rows query — [`ViewQuery`]'s default `Q`.
    type Query;
    /// Rows in the view.
    fn rows(&self) -> usize;
    /// Row dimension in original-vector units (a binary view reports bits).
    fn dim(&self) -> usize;
}

impl<'a> CorpusView for F32View<'a> {
    type Query = &'a [f32];
    fn rows(&self) -> usize {
        F32View::len(self)
    }
    fn dim(&self) -> usize {
        F32View::dim(self)
    }
}

impl<'a> CorpusView for I8View<'a> {
    type Query = I8Query<'a>;
    fn rows(&self) -> usize {
        I8View::len(self)
    }
    fn dim(&self) -> usize {
        I8View::dim(self)
    }
}

impl<'a> CorpusView for BinaryView<'a> {
    type Query = &'a [u8];
    fn rows(&self) -> usize {
        BinaryView::len(self)
    }
    fn dim(&self) -> usize {
        BinaryView::dim(self)
    }
}

/// A pre-quantized int8 query: the codes and the scale that returns their dot
/// products to f32 units — one is meaningless without the other. Produce with
/// [`quant::max_abs_scale`](crate::quant::max_abs_scale) +
/// [`quant::quantize_i8_vec`](crate::quant::quantize_i8_vec).
#[derive(Clone, Copy)]
pub struct I8Query<'a> {
    pub codes: &'a [i8],
    pub scale: f32,
}

/// A query encoding's validated dimension. `None` opts out of the scan-time
/// dim check: packed sign bits have a byte length, not a dim in view units,
/// and the hamming kernel validates them per row instead.
pub trait QueryData: Copy {
    fn dim(&self) -> Option<usize>;
}

impl QueryData for &[f32] {
    fn dim(&self) -> Option<usize> {
        Some(self.len())
    }
}

impl QueryData for &[i8] {
    fn dim(&self) -> Option<usize> {
        Some(self.len())
    }
}

// Packed sign bits: byte length, not a dim — the hamming kernel validates.
impl QueryData for &[u8] {
    fn dim(&self) -> Option<usize> {
        None
    }
}

impl QueryData for I8Query<'_> {
    fn dim(&self) -> Option<usize> {
        Some(self.codes.len())
    }
}

/// The two sides of a scan: a corpus view and the query scored against it.
/// `Q` defaults to the view's canonical encoding; asymmetric pairings name
/// theirs explicitly.
#[derive(Clone, Copy)]
pub struct ViewQuery<V: CorpusView, Q = <V as CorpusView>::Query> {
    pub view: V,
    pub query: Q,
}

impl<V: CorpusView, Q: QueryData> ViewQuery<V, Q> {
    pub(crate) fn rows(&self) -> usize {
        self.view.rows()
    }

    pub(crate) fn check_dim(&self) -> Result<(), ScanError> {
        match self.query.dim() {
            Some(query_dim) if query_dim != self.view.dim() => Err(ScanError::QueryDim {
                query_dim,
                view_dim: self.view.dim(),
            }),
            _ => Ok(()),
        }
    }
}
