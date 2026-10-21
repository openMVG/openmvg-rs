// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Plain records describing the views, poses, landmarks, and components of a reconstruction.
//!
//! [`crate::Dataset`], [`crate::ViewGraph`], and [`crate::Reconstruction`] return these values
//! from their accessors. Fields are public for reading, and the records are `#[non_exhaustive]`.
//! [`ImageMeta`], [`ViewRecord`], [`PoseRecord`] and [`ReconstructionComponent`] have
//! constructors; landmarks and their observations are built by the engines, or from
//! `ReconstructionParts` when you assemble a reconstruction yourself.

use std::path::{Path, PathBuf};

use nalgebra::{Point2, Point3};
use openmvg_core::model::WorldToCameraPose;
use openmvg_features::tracking::{Track, TrackObservation};

pub(crate) mod observation_support;
mod pair;

pub(crate) use observation_support::ObservationSupport;
pub use pair::{GraphPair, PairGeometry};

use crate::config::GaugeScalePolicy;
use crate::{
    ComponentId, FeatureId, IntrinsicId, LandmarkId, PoseId, Result, SfmError, TrackId,
    TrackObservationIndex, ViewId,
};

/// Where the pixels of a view come from.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
#[serde(rename_all = "snake_case")]
pub enum ImageSource {
    /// An image file on disk.
    Path(PathBuf),
    /// An image known only by a caller-side identifier; no pixels are read through it.
    ExternalId(String),
}

/// How far a camera model's focal length can be trusted before any view is registered.
///
/// Registration may estimate the focal of a [`Self::Guess`] group together with the pose of the
/// first view that uses it; a [`Self::Trusted`] focal is only refined by bundle adjustment.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
#[serde(rename_all = "snake_case")]
pub enum IntrinsicPrior {
    /// The focal comes from a calibration, EXIF metadata, or the caller. This is the default.
    #[default]
    Trusted,
    /// The focal is a placeholder, such as the multiple of the image size used when an image
    /// carries no focal metadata.
    Guess,
}

/// Name, source, and pixel size of one input image.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct ImageMeta {
    /// Display name: the file name for a path source, otherwise the identifier.
    pub name: String,
    /// Where the pixels come from.
    pub source: ImageSource,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
}

impl ImageMeta {
    /// Metadata for an image file of `width` x `height` pixels, named after its file name.
    pub fn path(path: impl Into<PathBuf>, width: u32, height: u32) -> Self {
        let path = path.into();
        let name = path
            .file_name()
            .and_then(|value| value.to_str())
            .map(str::to_owned)
            .unwrap_or_else(|| path.to_string_lossy().into_owned());
        Self {
            name,
            source: ImageSource::Path(path),
            width,
            height,
        }
    }

    /// Metadata for an externally identified image of `width` x `height` pixels, named `id`.
    pub fn external_id(id: impl Into<String>, width: u32, height: u32) -> Self {
        let name = id.into();
        Self {
            source: ImageSource::ExternalId(name.clone()),
            name,
            width,
            height,
        }
    }

    /// The file path of a [`ImageSource::Path`] source; `None` for other sources.
    pub fn file_path(&self) -> Option<&Path> {
        match &self.source {
            ImageSource::Path(path) => Some(path.as_path()),
            ImageSource::ExternalId(_) => None,
        }
    }
}

/// One image observation and its references into the shared intrinsic and pose tables.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct ViewRecord {
    /// Dense view index.
    pub id: ViewId,
    /// Image name, source, and size.
    pub meta: ImageMeta,
    /// Shared intrinsic block (camera model) the view uses.
    pub intrinsic_id: IntrinsicId,
    /// Shared world-to-camera pose block the view uses.
    pub pose_id: PoseId,
}

impl ViewRecord {
    /// Builds the record of view `id`, which uses camera model `intrinsic_id` and pose block
    /// `pose_id`.
    #[must_use]
    pub const fn new(
        id: ViewId,
        meta: ImageMeta,
        intrinsic_id: IntrinsicId,
        pose_id: PoseId,
    ) -> Self {
        Self {
            id,
            meta,
            intrinsic_id,
            pose_id,
        }
    }
}

/// Whether a landmark is part of the current reconstruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
#[serde(rename_all = "snake_case")]
pub enum LandmarkStatus {
    /// Part of the reconstruction.
    Active,
    /// Soft-deleted: kept so its ID stays valid, but excluded from every consumer.
    Rejected,
}

/// One triangulated 3D point and the feature track it was built from.
///
/// [`Self::supporting_observations`] lists the source-track observations that support the
/// current geometry.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct Landmark {
    /// Dense landmark index, equal to its position in [`Landmarks`].
    pub id: LandmarkId,
    /// Feature track the landmark was triangulated from.
    pub source_track: TrackId,
    /// Position in world coordinates.
    pub point: Point3<f64>,
    /// Mean reprojection error over the supporting observations; `.value()` gives it in
    /// pixels.
    pub reprojection_error: openmvg_core::PixelResidual,
    /// Source-track observations that support the current geometry.
    #[cfg(feature = "unstable-internals")]
    #[serde(default)]
    pub observation_support: ObservationSupport,
    #[cfg(not(feature = "unstable-internals"))]
    #[serde(default)]
    pub(crate) observation_support: ObservationSupport,
    /// Whether the landmark is active or rejected; a rejected landmark keeps its ID but is no
    /// longer a point of the reconstruction.
    pub status: LandmarkStatus,
}

/// One measurement of an active landmark.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct Observation {
    /// Observed landmark.
    pub landmark: LandmarkId,
    /// Registered view that observed it.
    pub view: ViewId,
    /// Feature of `view` behind the measurement.
    pub feature: FeatureId,
    /// Measured pixel, `(x = column, y = row)` in source-image pixels.
    pub pixel: Point2<f64>,
}

/// The landmarks of a reconstruction, indexed by dense [`LandmarkId`] and by source track.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize)]
pub struct Landmarks {
    landmarks: Vec<Landmark>,
    #[serde(skip)]
    landmark_by_track: Vec<(TrackId, LandmarkId)>,
}

impl<'de> serde::Deserialize<'de> for Landmarks {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        struct SerializedLandmarks {
            landmarks: Vec<Landmark>,
        }

        let serialized = SerializedLandmarks::deserialize(deserializer)?;
        Self::try_new(serialized.landmarks).map_err(serde::de::Error::custom)
    }
}

/// The world-to-camera pose of one pose block and the component it belongs to.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct PoseRecord {
    /// Pose block index.
    pub id: PoseId,
    /// World-to-camera transform.
    pub pose: WorldToCameraPose,
    /// Reconstruction component holding the pose; `None` while membership is not computed.
    pub component: Option<ComponentId>,
}

impl PoseRecord {
    /// Builds the record of pose block `id`.
    #[must_use]
    pub const fn new(id: PoseId, pose: WorldToCameraPose, component: Option<ComponentId>) -> Self {
        Self {
            id,
            pose,
            component,
        }
    }
}

/// One connected, separately gauged part of a reconstruction.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct ReconstructionComponent {
    /// Component index.
    pub id: ComponentId,
    /// Registered views in the component, sorted and unique.
    pub views: Vec<ViewId>,
    /// Landmarks in the component, sorted and unique.
    pub landmarks: Vec<LandmarkId>,
    /// How the component's similarity scale is fixed.
    pub gauge_scale_policy: GaugeScalePolicy,
}

impl Landmark {
    /// Builds an active landmark supported by the selected source-track observations.
    #[cfg(feature = "unstable-internals")]
    pub fn active(
        id: LandmarkId,
        source_track: TrackId,
        point: Point3<f64>,
        reprojection_error: openmvg_core::PixelResidual,
        observation_support: ObservationSupport,
    ) -> Self {
        Self {
            id,
            source_track,
            point,
            reprojection_error,
            observation_support,
            status: LandmarkStatus::Active,
        }
    }

    #[cfg(not(feature = "unstable-internals"))]
    pub(crate) fn active(
        id: LandmarkId,
        source_track: TrackId,
        point: Point3<f64>,
        reprojection_error: openmvg_core::PixelResidual,
        observation_support: ObservationSupport,
    ) -> Self {
        Self {
            id,
            source_track,
            point,
            reprojection_error,
            observation_support,
            status: LandmarkStatus::Active,
        }
    }

    /// Observations of `track`, the landmark's source track, that support its current geometry,
    /// with their indices in the track.
    pub fn supporting_observations<'a>(
        &'a self,
        track: &'a Track,
    ) -> impl Iterator<Item = (TrackObservationIndex, &'a TrackObservation)> + 'a {
        track
            .observations()
            .iter()
            .enumerate()
            .filter(|(index, _)| self.observation_support.contains(*index))
            .filter_map(|(index, observation)| {
                TrackObservationIndex::try_new(index).map(|index| (index, observation))
            })
    }

    /// Number of observations of `track`, the landmark's source track, that support its current
    /// geometry.
    pub fn supporting_observation_count(&self, track: &Track) -> usize {
        self.observation_support.len(track.observations().len())
    }
}

impl Landmarks {
    /// Build the store from landmarks in any order.
    ///
    /// Fails when the sorted IDs are not exactly `0..len` or when two landmarks share a source
    /// track.
    pub fn try_new(mut landmarks: Vec<Landmark>) -> Result<Self> {
        landmarks.sort_by_key(|landmark| landmark.id);
        let mut landmark_by_track = Vec::with_capacity(landmarks.len());
        for (index, landmark) in landmarks.iter().enumerate() {
            if landmark.id.index() != index {
                return Err(SfmError::InvalidId {
                    kind: LandmarkId::kind(),
                    index: landmark.id.index(),
                });
            }
            landmark_by_track.push((landmark.source_track, landmark.id));
        }
        landmark_by_track.sort_unstable_by_key(|&(track, _)| track);
        if let Some(track) = landmark_by_track
            .windows(2)
            .find(|pair| pair[0].0 == pair[1].0)
            .map(|pair| pair[0].0)
        {
            return Err(SfmError::InvalidInput(format!(
                "track {track} is linked to multiple landmarks"
            )));
        }
        Ok(Self {
            landmarks,
            landmark_by_track,
        })
    }

    /// Every record, indexed by landmark ID, including rejected ones.
    pub fn as_slice(&self) -> &[Landmark] {
        &self.landmarks
    }

    /// Mutable access to every record, for code that builds landmarks itself; IDs and source
    /// tracks must not be changed through it.
    pub(crate) fn as_mut_slice(&mut self) -> &mut [Landmark] {
        &mut self.landmarks
    }

    /// The record with `landmark_id`, active or rejected; `None` when out of range.
    pub fn get(&self, landmark_id: LandmarkId) -> Option<&Landmark> {
        self.landmarks.get(landmark_id.index())
    }

    /// Mutable access to the record with `landmark_id`; `None` when out of range.
    pub fn get_mut(&mut self, landmark_id: LandmarkId) -> Option<&mut Landmark> {
        self.landmarks.get_mut(landmark_id.index())
    }

    /// The landmark built from `track_id`, active or rejected; `None` when the track has none.
    pub fn for_track(&self, track_id: TrackId) -> Option<LandmarkId> {
        self.landmark_by_track
            .binary_search_by_key(&track_id, |&(track, _)| track)
            .ok()
            .map(|index| self.landmark_by_track[index].1)
    }

    /// The ID that the next landmark appended with [`Self::push`] must carry: landmark IDs are
    /// dense, from zero.
    pub fn next_id(&self) -> LandmarkId {
        LandmarkId::new(self.landmarks.len())
    }

    /// Append a landmark whose ID is [`Self::next_id`].
    ///
    /// # Errors
    ///
    /// [`SfmError::InvalidId`] if the ID is not [`Self::next_id`], and
    /// [`SfmError::InvalidInput`] if its source track already has a landmark.
    pub fn push(&mut self, landmark: Landmark) -> Result<()> {
        if landmark.id.index() != self.landmarks.len() {
            return Err(SfmError::InvalidId {
                kind: LandmarkId::kind(),
                index: landmark.id.index(),
            });
        }
        match self
            .landmark_by_track
            .binary_search_by_key(&landmark.source_track, |&(track, _)| track)
        {
            Ok(_) => {
                return Err(SfmError::InvalidInput(format!(
                    "track {} is linked to multiple landmarks",
                    landmark.source_track
                )));
            }
            Err(index) => {
                self.landmark_by_track
                    .insert(index, (landmark.source_track, landmark.id));
            }
        }
        self.landmarks.push(landmark);
        Ok(())
    }

    /// Append several landmarks at once, with the IDs that consecutive [`Self::push`] calls
    /// would need; faster than pushing them one by one. The whole batch is checked first, so a
    /// failure appends nothing.
    pub fn append_batch(&mut self, landmarks: Vec<Landmark>) -> Result<()> {
        if landmarks.is_empty() {
            return Ok(());
        }

        let first_id = self.landmarks.len();
        let mut appended_by_track = Vec::with_capacity(landmarks.len());
        for (offset, landmark) in landmarks.iter().enumerate() {
            let expected = first_id + offset;
            if landmark.id.index() != expected {
                return Err(SfmError::InvalidId {
                    kind: LandmarkId::kind(),
                    index: landmark.id.index(),
                });
            }
            appended_by_track.push((landmark.source_track, landmark.id));
        }
        appended_by_track.sort_unstable_by_key(|&(track, _)| track);
        if let Some(track) = appended_by_track
            .windows(2)
            .find(|pair| pair[0].0 == pair[1].0)
            .map(|pair| pair[0].0)
        {
            return Err(SfmError::InvalidInput(format!(
                "track {track} is linked to multiple landmarks"
            )));
        }

        let mut merged = Vec::with_capacity(self.landmark_by_track.len() + landmarks.len());
        let mut existing_index = 0;
        let mut appended_index = 0;
        while existing_index < self.landmark_by_track.len()
            && appended_index < appended_by_track.len()
        {
            let existing = self.landmark_by_track[existing_index];
            let appended = appended_by_track[appended_index];
            match existing.0.cmp(&appended.0) {
                std::cmp::Ordering::Less => {
                    merged.push(existing);
                    existing_index += 1;
                }
                std::cmp::Ordering::Greater => {
                    merged.push(appended);
                    appended_index += 1;
                }
                std::cmp::Ordering::Equal => {
                    return Err(SfmError::InvalidInput(format!(
                        "track {} is linked to multiple landmarks",
                        existing.0
                    )));
                }
            }
        }
        merged.extend_from_slice(&self.landmark_by_track[existing_index..]);
        merged.extend_from_slice(&appended_by_track[appended_index..]);

        self.landmarks.extend(landmarks);
        self.landmark_by_track = merged;
        Ok(())
    }

    /// Iterator over `Active` (non-`Rejected`) landmarks, the points of the reconstruction.
    /// Rejected landmarks keep their place so that IDs stay stable; skip them, as this iterator
    /// does, wherever you read points.
    pub fn iter_active(&self) -> impl Iterator<Item = &Landmark> {
        self.landmarks
            .iter()
            .filter(|landmark| landmark.status == LandmarkStatus::Active)
    }

    /// Number of `Active` landmarks.
    pub fn active_count(&self) -> usize {
        self.iter_active().count()
    }

    /// Number of stored records, including rejected ones.
    pub fn len(&self) -> usize {
        self.landmarks.len()
    }

    /// Remove an appended dense suffix while preserving the track lookup invariant.
    ///
    /// This is crate-private rollback plumbing: callers may only restore an earlier length, and
    /// existing landmark ids and track associations are unchanged by truncation.
    pub(crate) fn truncate(&mut self, len: usize) -> Result<()> {
        if len > self.landmarks.len() {
            return Err(SfmError::InvalidInput(format!(
                "cannot grow landmarks from {} to {len} through truncate",
                self.landmarks.len()
            )));
        }
        self.landmarks.truncate(len);
        self.landmark_by_track
            .retain(|&(_, landmark)| landmark.index() < len);
        Ok(())
    }

    /// Whether no record is stored, counting rejected ones.
    pub fn is_empty(&self) -> bool {
        self.landmarks.is_empty()
    }
}

impl ReconstructionComponent {
    /// Build a component; `views` and `landmarks` are sorted and deduplicated.
    pub fn new(
        id: ComponentId,
        mut views: Vec<ViewId>,
        mut landmarks: Vec<LandmarkId>,
        gauge_scale_policy: GaugeScalePolicy,
    ) -> Self {
        views.sort();
        views.dedup();
        landmarks.sort();
        landmarks.dedup();
        Self {
            id,
            views,
            landmarks,
            gauge_scale_policy,
        }
    }
}

#[cfg(test)]
mod landmark_tests {
    use super::*;

    fn landmark(id: usize, track: usize) -> Landmark {
        Landmark::active(
            LandmarkId::new(id),
            TrackId::new(track),
            Point3::new(track as f64, 0.0, 1.0),
            openmvg_core::PixelResidual::ZERO,
            ObservationSupport::default(),
        )
    }

    #[test]
    fn landmarks_reject_duplicate_source_tracks() {
        let landmarks = vec![
            Landmark::active(
                LandmarkId::new(0),
                TrackId::new(0),
                Point3::new(0.0, 0.0, 1.0),
                openmvg_core::PixelResidual::ZERO,
                ObservationSupport::default(),
            ),
            Landmark::active(
                LandmarkId::new(1),
                TrackId::new(0),
                Point3::new(1.0, 0.0, 1.0),
                openmvg_core::PixelResidual::ZERO,
                ObservationSupport::default(),
            ),
        ];

        assert!(matches!(
            Landmarks::try_new(landmarks),
            Err(SfmError::InvalidInput(_))
        ));
    }

    #[test]
    fn landmark_batch_append_matches_repeated_insertion() {
        let initial = vec![landmark(0, 1), landmark(1, 3)];
        let appended = vec![landmark(2, 0), landmark(3, 2), landmark(4, 4)];
        let mut repeated = Landmarks::try_new(initial.clone()).unwrap();
        for landmark in appended.clone() {
            repeated.push(landmark).unwrap();
        }

        let mut batched = Landmarks::try_new(initial).unwrap();
        batched.append_batch(appended).unwrap();

        assert_eq!(batched, repeated);
        for (track, landmark) in [(0, 2), (1, 0), (2, 3), (3, 1), (4, 4)] {
            assert_eq!(
                batched.for_track(TrackId::new(track)),
                Some(LandmarkId::new(landmark))
            );
        }
    }

    #[test]
    fn landmark_batch_append_is_atomic_on_invalid_input() {
        let mut landmarks = Landmarks::try_new(vec![landmark(0, 1)]).unwrap();
        let before = landmarks.clone();

        assert!(
            landmarks
                .append_batch(vec![landmark(1, 2), landmark(2, 1)])
                .is_err()
        );
        assert_eq!(landmarks, before);

        assert!(landmarks.append_batch(vec![landmark(7, 3)]).is_err());
        assert_eq!(landmarks, before);
    }
}
