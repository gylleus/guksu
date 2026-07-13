//! Filter bitmap over vector ids — the optional predicate a scan applies.

use crate::rng::SplitMix64;

/// Fixed-size bitmap over vector ids `[0, len)`.
///
/// Passed to [`Scorer::top_k`](crate::scan::Scorer::top_k) as
/// `Option<&Bitset>`; only set ids are scored.
/// Filtered scans iterate set bits directly ([`Bitset::iter_ones`]), so a 2%
/// selectivity filter touches 2% of rows, not 100%.
pub struct Bitset {
    words: Box<[u64]>,
    len: usize,
}

impl Bitset {
    /// All-zeros bitmap covering ids `[0, len)`.
    pub fn new(len: usize) -> Self {
        Self { words: vec![0u64; len.div_ceil(64)].into_boxed_slice(), len }
    }

    /// Number of ids covered (the corpus size), NOT the number of set bits.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Set bit `id`. Panics if `id >= len`.
    pub fn set(&mut self, id: u32) {
        assert!((id as usize) < self.len, "id {id} out of range 0..{}", self.len);
        self.words[id as usize / 64] |= 1u64 << (id % 64);
    }

    /// Test bit `id`. Panics if `id >= len`.
    pub fn contains(&self, id: u32) -> bool {
        assert!((id as usize) < self.len, "id {id} out of range 0..{}", self.len);
        self.words[id as usize / 64] & (1u64 << (id % 64)) != 0
    }

    /// Number of set bits (the selectivity numerator).
    pub fn count_ones(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Bitmap with the given ids set.
    pub fn from_ids(len: usize, ids: impl IntoIterator<Item = u32>) -> Self {
        let mut bs = Self::new(len);
        for id in ids {
            bs.set(id);
        }
        bs
    }

    /// Iterate set ids in ascending order (word-skip + trailing_zeros).
    pub fn iter_ones(&self) -> impl Iterator<Item = u32> + '_ {
        let mut word_idx = 0usize;
        let mut current = self.words.first().copied().unwrap_or(0);
        std::iter::from_fn(move || {
            while current == 0 {
                word_idx += 1;
                if word_idx >= self.words.len() {
                    return None;
                }
                current = self.words[word_idx];
            }
            let bit = current.trailing_zeros();
            current &= current - 1;
            Some(word_idx as u32 * 64 + bit)
        })
    }

    /// Deterministic Bernoulli(keep_fraction) bitmap: same `(len, keep_fraction,
    /// seed)` yields an identical bitmap on every platform and release.
    pub fn random(len: usize, keep_fraction: f64, seed: u64) -> Self {
        let mut bs = Self::new(len);
        let mut rng = SplitMix64::new(seed);
        for id in 0..len {
            if rng.next_f64() < keep_fraction {
                bs.set(id as u32);
            }
        }
        bs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_contains_across_word_boundaries() {
        let mut bs = Bitset::new(130);
        for id in [0u32, 1, 63, 64, 65, 127, 128, 129] {
            assert!(!bs.contains(id));
            bs.set(id);
            assert!(bs.contains(id));
        }
        assert!(!bs.contains(2));
        assert!(!bs.contains(126));
    }

    #[test]
    fn count_ones_with_partial_tail_word() {
        let mut bs = Bitset::new(70);
        for id in [0u32, 63, 64, 69] {
            bs.set(id);
        }
        assert_eq!(bs.count_ones(), 4);
        bs.set(69); // idempotent
        assert_eq!(bs.count_ones(), 4);
    }

    #[test]
    fn iter_ones_matches_contains() {
        let ids = [0u32, 5, 63, 64, 100, 191, 192, 250];
        let bs = Bitset::from_ids(251, ids.iter().copied());
        let via_iter: Vec<u32> = bs.iter_ones().collect();
        let via_contains: Vec<u32> = (0..251).filter(|&i| bs.contains(i)).collect();
        assert_eq!(via_iter, ids);
        assert_eq!(via_iter, via_contains);
    }

    #[test]
    fn iter_ones_empty_and_full() {
        assert_eq!(Bitset::new(100).iter_ones().count(), 0);
        assert_eq!(Bitset::new(0).iter_ones().count(), 0);
        let full = Bitset::from_ids(65, 0..65);
        assert_eq!(full.iter_ones().collect::<Vec<_>>(), (0..65).collect::<Vec<_>>());
    }

    #[test]
    fn random_is_deterministic_and_calibrated() {
        let a = Bitset::random(100_000, 0.1, 42);
        let b = Bitset::random(100_000, 0.1, 42);
        let a_ids: Vec<u32> = a.iter_ones().collect();
        assert_eq!(a_ids, b.iter_ones().collect::<Vec<u32>>());
        assert_ne!(a_ids, Bitset::random(100_000, 0.1, 43).iter_ones().collect::<Vec<u32>>());
        let rate = a.count_ones() as f64 / 100_000.0;
        assert!((rate - 0.1).abs() < 0.01, "selectivity {rate}");
        assert_eq!(Bitset::random(1000, 0.0, 1).count_ones(), 0);
        assert_eq!(Bitset::random(1000, 1.0, 1).count_ones(), 1000);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn set_out_of_range_panics() {
        Bitset::new(10).set(10);
    }
}
