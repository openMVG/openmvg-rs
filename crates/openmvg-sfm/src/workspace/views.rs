// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Read-only views over the workspace stores: features, matches, models, poses, and landmarks.

use crate::model::{Landmark, LandmarkStatus, Landmarks, PoseRecord};
use crate::two_view::{GeometricModel, PairMatches};
use crate::workspace::pair_graph::PairGraph;
use crate::workspace::store::SparseStore;
use crate::{LandmarkId, PairId, PoseId, ViewId, ViewPair};
use openmvg_features::model::ExtractedFeatures;

/// Read-only view over feature results keyed by image id.
///
/// ```
/// # #[cfg(feature = "unstable-internals")]
/// # {
/// use openmvg_sfm::unstable::SparseStore;
/// use openmvg_sfm::unstable::FeaturesView;
/// use openmvg_sfm::ViewId;
///
/// let store = SparseStore::new(2);
/// let view = FeaturesView::new(&store);
/// assert_eq!(view.slot_count(), 2);
/// assert!(view.get(ViewId::new(0)).is_none());
/// # }
/// ```
#[derive(Debug, Clone, Copy)]
pub struct FeaturesView<'a> {
    store: &'a SparseStore<ExtractedFeatures>,
}

/// Read-only view over pair matches keyed by stable view-pair identity.
#[derive(Debug, Clone, Copy)]
pub struct MatchesView<'a> {
    store: &'a SparseStore<PairMatches>,
    #[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
    pair_graph: &'a PairGraph,
}

/// Read-only view over registered poses.
///
/// ```
/// # #[cfg(feature = "unstable-internals")]
/// # {
/// use openmvg_sfm::unstable::SparseStore;
/// use openmvg_sfm::unstable::PoseSet;
/// use openmvg_sfm::PoseId;
///
/// let store = SparseStore::new(3);
/// let poses = PoseSet::new(&store);
/// assert_eq!(poses.populated_count(), 0);
/// assert!(poses.get(PoseId::new(1)).is_none());
/// # }
/// ```
#[derive(Debug, Clone, Copy)]
#[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
pub struct PoseSet<'a> {
    store: &'a SparseStore<PoseRecord>,
}

/// Read-only view over stored landmark records.
///
/// ```
/// # #[cfg(feature = "unstable-internals")]
/// # {
/// use openmvg_sfm::model::Landmarks;
/// use openmvg_sfm::unstable::LandmarkSet;
///
/// let landmarks = Landmarks::default();
/// let view = LandmarkSet::new(&landmarks);
/// assert!(view.is_empty());
/// # }
/// ```
#[derive(Debug, Clone, Copy)]
#[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
pub struct LandmarkSet<'a> {
    landmarks: &'a Landmarks,
}

#[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
impl<'a> FeaturesView<'a> {
    /// Wrap a per-view feature store.
    pub fn new(store: &'a SparseStore<ExtractedFeatures>) -> Self {
        Self { store }
    }

    /// Slot count of the underlying store, one per view.
    pub fn slot_count(self) -> usize {
        self.store.slot_count()
    }

    /// Number of populated slots.
    pub fn populated_count(self) -> usize {
        self.store.populated_count()
    }

    /// Whether no slot is populated.
    pub fn is_empty(self) -> bool {
        self.populated_count() == 0
    }

    /// Features of `view`; `None` when not extracted yet or the ID is out of range.
    pub fn get(self, view: ViewId) -> Option<&'a ExtractedFeatures> {
        self.store.get(view.index())
    }

    /// Views that have features, in increasing view-ID order.
    pub fn iter(self) -> impl Iterator<Item = (ViewId, &'a ExtractedFeatures)> {
        self.store
            .iter_populated()
            .map(|(index, features)| (ViewId::new(index), features))
    }
}

#[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
impl<'a> MatchesView<'a> {
    pub(crate) fn new(store: &'a SparseStore<PairMatches>, pair_graph: &'a PairGraph) -> Self {
        Self { store, pair_graph }
    }

    /// Slot count of the underlying store, one per pair-graph pair.
    pub fn slot_count(self) -> usize {
        self.store.slot_count()
    }

    /// Number of populated slots.
    pub fn populated_count(self) -> usize {
        self.store.populated_count()
    }

    /// Whether no slot is populated.
    pub fn is_empty(self) -> bool {
        self.populated_count() == 0
    }

    /// Matches of `pair`; `None` when the pair is not in the graph or not matched yet.
    pub fn get(self, pair: ViewPair) -> Option<&'a PairMatches> {
        let pair_id = self.pair_graph.pair_id(pair.first(), pair.second())?;
        self.get_by_id(pair_id)
    }

    /// Matched pairs in canonical pair order.
    pub fn iter(self) -> impl Iterator<Item = (ViewPair, &'a PairMatches)> {
        self.store.iter_populated().filter_map(|(index, matches)| {
            self.pair_graph
                .pair(PairId::new(index))
                .map(|pair| (pair, matches))
        })
    }

    pub(crate) fn get_by_id(self, pair: PairId) -> Option<&'a PairMatches> {
        self.store.get(pair.index())
    }
}

#[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
impl<'a> PoseSet<'a> {
    /// Wrap a per-pose-block store.
    pub fn new(store: &'a SparseStore<PoseRecord>) -> Self {
        Self { store }
    }

    /// Slot count of the underlying store, one per pose block.
    pub fn slot_count(self) -> usize {
        self.store.slot_count()
    }

    /// Number of registered pose blocks.
    pub fn populated_count(self) -> usize {
        self.store.populated_count()
    }

    /// Whether no pose block is registered.
    pub fn is_empty(self) -> bool {
        self.populated_count() == 0
    }

    /// Pose of block `pose`; `None` when unregistered or out of range.
    pub fn get(self, pose: PoseId) -> Option<&'a PoseRecord> {
        self.store.get(pose.index())
    }

    /// Registered pose blocks in increasing pose-ID order.
    pub fn iter(self) -> impl Iterator<Item = (PoseId, &'a PoseRecord)> {
        self.store
            .iter_populated()
            .map(|(index, pose)| (PoseId::new(index), pose))
    }
}

#[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
impl<'a> LandmarkSet<'a> {
    /// Wrap a landmark store.
    pub fn new(landmarks: &'a Landmarks) -> Self {
        Self { landmarks }
    }

    /// Number of stored records, including soft-deleted rejections.
    pub fn len(self) -> usize {
        self.landmarks.len()
    }

    /// Whether no record is stored, counting rejected ones.
    pub fn is_empty(self) -> bool {
        self.len() == 0
    }

    /// Number of active landmarks available to reconstruction consumers.
    pub fn active_count(self) -> usize {
        self.landmarks.active_count()
    }

    /// Get a stored record by stable ID, including a rejected record.
    pub fn get(self, landmark: LandmarkId) -> Option<&'a Landmark> {
        self.landmarks.get(landmark)
    }

    /// Get a landmark only when it is active.
    pub fn get_active(self, landmark: LandmarkId) -> Option<&'a Landmark> {
        self.get(landmark)
            .filter(|landmark| landmark.status == LandmarkStatus::Active)
    }

    /// Every stored record in landmark-ID order, including rejected ones.
    pub fn iter(self) -> impl Iterator<Item = &'a Landmark> {
        self.landmarks.as_slice().iter()
    }

    /// Active landmarks only, in landmark-ID order.
    pub fn iter_active(self) -> impl Iterator<Item = &'a Landmark> {
        self.landmarks.iter_active()
    }

    /// Every stored record, indexed by landmark ID, including rejected ones.
    pub fn as_slice(self) -> &'a [Landmark] {
        self.landmarks.as_slice()
    }
}

/// Read-only view over verified two-view geometric models keyed by stable view-pair identity.
#[derive(Debug, Clone, Copy)]
pub struct PairModelsView<'a> {
    store: &'a SparseStore<GeometricModel>,
    #[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
    pair_graph: &'a PairGraph,
}

#[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
impl<'a> PairModelsView<'a> {
    pub(crate) fn new(store: &'a SparseStore<GeometricModel>, pair_graph: &'a PairGraph) -> Self {
        Self { store, pair_graph }
    }

    /// Slot count of the underlying store, one per pair-graph pair.
    pub fn slot_count(self) -> usize {
        self.store.slot_count()
    }

    /// Number of populated slots.
    pub fn populated_count(self) -> usize {
        self.store.populated_count()
    }

    /// Whether no slot is populated.
    pub fn is_empty(self) -> bool {
        self.populated_count() == 0
    }

    /// Model of `pair`; `None` when the pair is not in the graph or not verified.
    pub fn get(self, pair: ViewPair) -> Option<&'a GeometricModel> {
        let pair_id = self.pair_graph.pair_id(pair.first(), pair.second())?;
        self.get_by_id(pair_id)
    }

    /// Verified pairs in canonical pair order.
    pub fn iter(self) -> impl Iterator<Item = (ViewPair, &'a GeometricModel)> {
        self.store.iter_populated().filter_map(|(index, model)| {
            self.pair_graph
                .pair(PairId::new(index))
                .map(|pair| (pair, model))
        })
    }

    pub(crate) fn get_by_id(self, pair: PairId) -> Option<&'a GeometricModel> {
        self.store.get(pair.index())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TrackId;
    use crate::model::ObservationSupport;
    use nalgebra::Point3;

    #[test]
    fn landmark_view_distinguishes_active_and_stored_records() {
        let active = Landmark::active(
            LandmarkId::new(0),
            TrackId::new(0),
            Point3::new(0.0, 0.0, 5.0),
            openmvg_core::PixelResidual::try_new(0.5).unwrap(),
            ObservationSupport::default(),
        );
        let mut rejected = Landmark::active(
            LandmarkId::new(1),
            TrackId::new(1),
            Point3::new(1.0, 0.0, 5.0),
            openmvg_core::PixelResidual::try_new(1.0).unwrap(),
            ObservationSupport::default(),
        );
        rejected.status = LandmarkStatus::Rejected;
        let landmarks = Landmarks::try_new(vec![active, rejected]).unwrap();
        let view = LandmarkSet::new(&landmarks);

        assert_eq!(view.len(), 2);
        assert!(!view.is_empty());
        assert_eq!(view.active_count(), 1);
        assert_eq!(
            view.iter_active()
                .map(|landmark| landmark.id)
                .collect::<Vec<_>>(),
            [LandmarkId::new(0)]
        );
        assert_eq!(view.iter().count(), 2);
        assert!(view.get(LandmarkId::new(1)).is_some());
        assert!(view.get_active(LandmarkId::new(1)).is_none());
        assert_eq!(view.as_slice().len(), 2);
    }
}
