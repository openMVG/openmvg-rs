// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared [`ViewIdMapping`] for all averaging bridge layers.
//!
//! Averaging and global-positioning solvers need a bidirectional mapping between
//! sparse `ViewId`s (which may be non-contiguous) and dense solver indices. This
//! module provides their single shared implementation.

use std::collections::BTreeMap;

use openmvg_core::model::ViewId;

/// Bidirectional mapping between sparse `ViewId`s and contiguous solver indices.
///
/// Solver internals use dense `Vec<T>` indexed by `usize`, but the openMVG
/// pipeline uses potentially non-contiguous `ViewId`s.
/// This struct converts between the two representations deterministically
/// (IDs are sorted and deduplicated).
#[derive(Debug, Clone)]
pub struct ViewIdMapping {
    /// Dense solver index → `ViewId`.
    index_to_view: Vec<ViewId>,
    /// `ViewId` → dense solver index.
    view_to_index: BTreeMap<ViewId, usize>,
}

impl ViewIdMapping {
    /// Build from an iterator of `ViewId`s. Sorts and deduplicates for determinism.
    #[must_use]
    pub fn from_views(views: impl IntoIterator<Item = ViewId>) -> Self {
        let mut index_to_view: Vec<ViewId> = views.into_iter().collect();
        index_to_view.sort_unstable();
        index_to_view.dedup();
        let view_to_index = index_to_view
            .iter()
            .enumerate()
            .map(|(i, &v)| (v, i))
            .collect();
        Self {
            index_to_view,
            view_to_index,
        }
    }

    /// Number of views in the mapping.
    #[must_use]
    pub fn view_count(&self) -> usize {
        self.index_to_view.len()
    }

    /// View IDs in dense-index order.
    #[cfg(test)]
    #[must_use]
    pub fn views(&self) -> &[ViewId] {
        &self.index_to_view
    }

    /// Return the dense solver index assigned to `view`.
    #[must_use]
    pub fn index_of(&self, view: ViewId) -> Option<usize> {
        self.view_to_index.get(&view).copied()
    }

    /// Return the view ID assigned to a dense solver `index`.
    #[cfg(test)]
    #[must_use]
    pub fn view_at(&self, index: usize) -> Option<ViewId> {
        self.index_to_view.get(index).copied()
    }

    /// Iterate over `(dense_index, view_id)` pairs in dense-index order.
    pub fn iter(
        &self,
    ) -> impl ExactSizeIterator<Item = (usize, ViewId)> + DoubleEndedIterator + '_ {
        self.index_to_view.iter().copied().enumerate()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_view_id_mapping_sorted_dedup() {
        let m = ViewIdMapping::from_views(vec![
            ViewId::new(5),
            ViewId::new(2),
            ViewId::new(8),
            ViewId::new(2),
        ]);
        assert_eq!(m.view_count(), 3);
        assert_eq!(m.views(), &[ViewId::new(2), ViewId::new(5), ViewId::new(8)]);
        assert_eq!(m.index_of(ViewId::new(2)), Some(0));
        assert_eq!(m.index_of(ViewId::new(5)), Some(1));
        assert_eq!(m.index_of(ViewId::new(8)), Some(2));
        assert_eq!(m.index_of(ViewId::new(3)), None);
        assert_eq!(m.view_at(0), Some(ViewId::new(2)));
        assert_eq!(m.view_at(2), Some(ViewId::new(8)));
        assert_eq!(m.view_at(3), None);
        assert_eq!(
            m.iter().collect::<Vec<_>>(),
            vec![
                (0, ViewId::new(2)),
                (1, ViewId::new(5)),
                (2, ViewId::new(8)),
            ]
        );
    }
}
