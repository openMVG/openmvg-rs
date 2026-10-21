// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fixed-size store of optional per-ID values.

use crate::SfmError;

/// Fixed-size store of optional values indexed by a dense ID.
///
/// The slot count is fixed at construction; a slot is either populated or missing, and the
/// populated count is tracked so [`Self::populated_count`] is O(1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SparseStore<T> {
    slots: Vec<Option<T>>,
    populated_count: usize,
}

#[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
impl<T> SparseStore<T> {
    /// Create a store with `slot_count` slots, all missing.
    pub fn new(slot_count: usize) -> Self {
        Self {
            slots: (0..slot_count).map(|_| None).collect(),
            populated_count: 0,
        }
    }

    /// Total slot count, populated or not.
    pub fn slot_count(&self) -> usize {
        self.slots.len()
    }

    /// Returns the number of populated (non-`None`) slots in O(1).
    pub fn populated_count(&self) -> usize {
        self.populated_count
    }

    /// Whether every slot is populated.
    pub fn is_complete(&self) -> bool {
        self.populated_count() == self.slot_count()
    }

    /// Whether slot `index` is populated; `false` also when `index` is out of range.
    pub fn has(&self, index: usize) -> bool {
        self.get(index).is_some()
    }

    /// The value in slot `index`; `None` when the slot is missing or out of range.
    pub fn get(&self, index: usize) -> Option<&T> {
        self.slots.get(index).and_then(Option::as_ref)
    }

    /// Mutable access to slot `index`; `None` when the slot is missing or out of range.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        self.slots.get_mut(index).and_then(Option::as_mut)
    }

    /// Set a value in an existing slot.
    ///
    /// Store size is an invariant owned by the workspace. Out-of-range IDs are
    /// rejected instead of silently allocating up to the supplied index.
    pub fn set(&mut self, index: usize, value: T) -> Result<Option<T>, SfmError> {
        if index >= self.slots.len() {
            return Err(SfmError::InvalidId {
                kind: "sparse store slot",
                index,
            });
        }
        let old = self.slots[index].replace(value);
        if old.is_none() {
            self.populated_count += 1;
        }
        Ok(old)
    }

    /// Remove a value from an existing slot.
    pub fn take(&mut self, index: usize) -> Result<Option<T>, SfmError> {
        let slot = self.slots.get_mut(index).ok_or(SfmError::InvalidId {
            kind: "sparse store slot",
            index,
        })?;
        let taken = slot.take();
        if taken.is_some() {
            self.populated_count -= 1;
        }
        Ok(taken)
    }

    /// Same as [`Self::take`]: empty slot `index` and return its previous value.
    pub fn clear(&mut self, index: usize) -> Result<Option<T>, SfmError> {
        self.take(index)
    }

    /// Populated slots as `(index, value)`, in increasing index order.
    pub fn iter_populated(&self) -> impl Iterator<Item = (usize, &T)> {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(index, slot)| slot.as_ref().map(|value| (index, value)))
    }

    /// Indices of the missing slots, in increasing order.
    pub fn iter_missing(&self) -> impl Iterator<Item = usize> + '_ {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(index, slot)| slot.is_none().then_some(index))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn populated_count_is_maintained_through_set_take_clear() {
        let mut store = SparseStore::new(5);
        assert_eq!(store.populated_count(), 0);
        store.set(0, "a").unwrap();
        assert_eq!(store.populated_count(), 1);
        store.set(2, "b").unwrap();
        assert_eq!(store.populated_count(), 2);
        store.set(0, "a2").unwrap();
        assert_eq!(store.populated_count(), 2);
        store.take(0).unwrap();
        assert_eq!(store.populated_count(), 1);
        store.take(0).unwrap();
        assert_eq!(store.populated_count(), 1);
        store.clear(2).unwrap();
        assert_eq!(store.populated_count(), 0);
    }

    #[test]
    fn is_complete_works() {
        let mut store = SparseStore::new(2);
        assert!(!store.is_complete());
        store.set(0, "a").unwrap();
        assert!(!store.is_complete());
        store.set(1, "b").unwrap();
        assert!(store.is_complete());
    }

    #[test]
    fn set_rejects_out_of_range_index_without_growth() {
        let mut store = SparseStore::new(2);
        assert!(matches!(
            store.set(usize::MAX, "bad"),
            Err(SfmError::InvalidId { .. })
        ));
        assert_eq!(store.slot_count(), 2);
        assert_eq!(store.populated_count(), 0);
    }

    #[test]
    fn removal_rejects_out_of_range_indices() {
        let mut store: SparseStore<i32> = SparseStore::new(1);
        assert!(matches!(store.take(1), Err(SfmError::InvalidId { .. })));
        assert!(matches!(
            store.clear(usize::MAX),
            Err(SfmError::InvalidId { .. })
        ));
    }
}
