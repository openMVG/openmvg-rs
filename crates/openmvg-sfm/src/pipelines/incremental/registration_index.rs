// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Per-pass index from views to their landmark observations, used to rank and prepare
//! registrations without rescanning the tracks.

use crate::model::LandmarkStatus;
use crate::{LandmarkId, Result, SfmError, SfmWorkspace, TrackId, TrackObservationIndex, ViewId};

pub(crate) const REGISTRATION_GRID_SIDE: usize = 4;
pub(crate) const REGISTRATION_GRID_CELLS: usize = REGISTRATION_GRID_SIDE * REGISTRATION_GRID_SIDE;

pub(crate) fn registration_grid_cell(x: f64, y: f64, width: u32, height: u32) -> Option<usize> {
    if width == 0
        || height == 0
        || !x.is_finite()
        || !y.is_finite()
        || x < 0.0
        || y < 0.0
        || x >= f64::from(width)
        || y >= f64::from(height)
    {
        return None;
    }
    let column = (x * REGISTRATION_GRID_SIDE as f64 / f64::from(width)) as usize;
    let row = (y * REGISTRATION_GRID_SIDE as f64 / f64::from(height)) as usize;
    Some(row * REGISTRATION_GRID_SIDE + column)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct IndexedRegistrationObservation {
    pub landmark: LandmarkId,
    pub track_observation: TrackObservationIndex,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct ImageSupportCounts {
    pub active_landmarks: usize,
    pub supporting_landmarks: usize,
}

/// Transient registration lookup reused throughout one incremental growth pass.
///
/// [`crate::TrackSet`] already owns the immutable image-to-track adjacency. Duplicating that graph
/// in a second CSR made every newly registered image trigger a full two-pass scan of all active
/// landmark observations. This index therefore stores only the mutable join from a stable
/// [`TrackId`] to its current active [`LandmarkId`], plus the per-image counts needed to rank the
/// frontier. Tracks touched by scoped retriangulation are synchronized in place; a full rebuild is
/// reserved for global operations that may change arbitrary landmarks.
#[derive(Debug, Default)]
pub(crate) struct RegistrationObservationIndex {
    active_landmark_by_track: Vec<Option<LandmarkId>>,
    visible_landmarks_by_view: Vec<usize>,
    visible_grid_cells_by_view: Vec<[u32; REGISTRATION_GRID_CELLS]>,
    registration_failures_by_view: Vec<usize>,
}

impl RegistrationObservationIndex {
    pub fn rebuild(&mut self, workspace: &SfmWorkspace) -> Result<()> {
        self.active_landmark_by_track
            .resize(workspace.tracks().len(), None);
        self.active_landmark_by_track.fill(None);
        self.visible_landmarks_by_view
            .resize(workspace.views().len(), 0);
        self.visible_landmarks_by_view.fill(0);
        self.visible_grid_cells_by_view
            .resize(workspace.views().len(), [0; REGISTRATION_GRID_CELLS]);
        self.visible_grid_cells_by_view
            .fill([0; REGISTRATION_GRID_CELLS]);
        self.registration_failures_by_view
            .resize(workspace.views().len(), 0);
        self.registration_failures_by_view.fill(0);
        for failure in workspace.registration_failures() {
            let count = self
                .registration_failures_by_view
                .get_mut(failure.view.index())
                .ok_or(SfmError::InvalidId {
                    kind: ViewId::kind(),
                    index: failure.view.index(),
                })?;
            *count = count.checked_add(1).ok_or_else(|| {
                SfmError::InvalidInput("registration failure count overflow".to_owned())
            })?;
        }

        for landmark in workspace.landmarks().iter_active() {
            let track =
                workspace
                    .tracks()
                    .get(landmark.source_track)
                    .ok_or(SfmError::InvalidId {
                        kind: TrackId::kind(),
                        index: landmark.source_track.index(),
                    })?;
            let slot = self
                .active_landmark_by_track
                .get_mut(landmark.source_track.index())
                .ok_or(SfmError::InvalidId {
                    kind: TrackId::kind(),
                    index: landmark.source_track.index(),
                })?;
            if slot.replace(landmark.id).is_some() {
                return Err(SfmError::InvalidInput(format!(
                    "track {} is linked to multiple active landmarks",
                    landmark.source_track
                )));
            }
            for observation in track.observations() {
                self.update_observation(workspace, observation, true)?;
            }
        }
        Ok(())
    }

    /// Refresh landmarks whose immutable tracks observe one newly registered image.
    ///
    /// Native incremental growth commits exactly one camera at a time, and the track collection
    /// already stores a stable unique image-to-track slice. Consuming it directly avoids retaining
    /// a second scratch adjacency solely to sort and deduplicate a one-image domain.
    pub fn sync_view(&mut self, workspace: &SfmWorkspace, view: ViewId) -> Result<()> {
        workspace
            .tracks()
            .observing_view(view)
            .iter()
            .try_for_each(|&track| self.sync_track(workspace, track))
    }

    fn sync_track(&mut self, workspace: &SfmWorkspace, track_id: TrackId) -> Result<()> {
        let track = workspace
            .tracks()
            .get(track_id)
            .ok_or(SfmError::InvalidId {
                kind: TrackId::kind(),
                index: track_id.index(),
            })?;
        let new_landmark = workspace
            .landmarks()
            .for_track(track_id)
            .and_then(|landmark_id| {
                workspace
                    .landmarks()
                    .get(landmark_id)
                    .filter(|landmark| landmark.status == LandmarkStatus::Active)
                    .map(|landmark| landmark.id)
            });
        let old_landmark = self
            .active_landmark_by_track
            .get(track_id.index())
            .copied()
            .ok_or(SfmError::InvalidId {
                kind: TrackId::kind(),
                index: track_id.index(),
            })?;
        if old_landmark == new_landmark {
            return Ok(());
        }

        for observation in track.observations() {
            if old_landmark.is_some() {
                self.update_observation(workspace, observation, false)?;
            }
            if new_landmark.is_some() {
                self.update_observation(workspace, observation, true)?;
            }
        }
        self.active_landmark_by_track[track_id.index()] = new_landmark;
        Ok(())
    }

    fn update_observation(
        &mut self,
        workspace: &SfmWorkspace,
        observation: &crate::TrackObservation,
        add: bool,
    ) -> Result<()> {
        let view = workspace
            .view(observation.view)
            .ok_or(SfmError::InvalidId {
                kind: ViewId::kind(),
                index: observation.view.index(),
            })?;
        let count = self
            .visible_landmarks_by_view
            .get_mut(observation.view.index())
            .ok_or(SfmError::InvalidId {
                kind: ViewId::kind(),
                index: observation.view.index(),
            })?;
        if add {
            *count = count.checked_add(1).ok_or_else(|| {
                SfmError::InvalidInput("registration observation count overflow".to_owned())
            })?;
        } else {
            *count = count.checked_sub(1).ok_or_else(|| {
                SfmError::InvalidInput("registration observation count underflow".to_owned())
            })?;
        }

        let pixel = workspace.track_observation_pixel(observation)?;
        let Some(cell) =
            registration_grid_cell(pixel.x, pixel.y, view.meta.width, view.meta.height)
        else {
            return Ok(());
        };
        let cells = self
            .visible_grid_cells_by_view
            .get_mut(observation.view.index())
            .ok_or(SfmError::InvalidId {
                kind: ViewId::kind(),
                index: observation.view.index(),
            })?;
        let cell_count = &mut cells[cell];
        if add {
            *cell_count = cell_count.checked_add(1).ok_or_else(|| {
                SfmError::InvalidInput("registration grid count overflow".to_owned())
            })?;
        } else {
            *cell_count = cell_count.checked_sub(1).ok_or_else(|| {
                SfmError::InvalidInput("registration grid count underflow".to_owned())
            })?;
        }
        Ok(())
    }

    pub fn observations<'a>(
        &'a self,
        workspace: &'a SfmWorkspace,
        view: ViewId,
    ) -> impl Iterator<Item = IndexedRegistrationObservation> + 'a {
        workspace
            .tracks()
            .observing_view(view)
            .iter()
            .filter_map(move |&track_id| {
                let landmark = self
                    .active_landmark_by_track
                    .get(track_id.index())
                    .copied()
                    .flatten()?;
                let track = workspace.tracks().get(track_id)?;
                let track_observation = track
                    .observations()
                    .binary_search_by_key(&view, |observation| observation.view)
                    .ok()
                    .and_then(TrackObservationIndex::try_new)?;
                Some(IndexedRegistrationObservation {
                    landmark,
                    track_observation,
                })
            })
    }

    pub fn visible_landmarks(&self, view: ViewId) -> usize {
        self.visible_landmarks_by_view
            .get(view.index())
            .copied()
            .unwrap_or(0)
    }

    pub fn occupied_image_cells(&self, view: ViewId) -> usize {
        self.visible_grid_cells_by_view
            .get(view.index())
            .map_or(0, |cells| cells.iter().filter(|&&count| count > 0).count())
    }

    /// Count how much of the active map actually retains this image after a scoped track update.
    ///
    /// Registration visibility uses immutable raw tracks. This audit instead checks
    /// the fitted landmark support, exposing poses whose proposed 2D-3D support largely disappears
    /// during robust retriangulation. The growth transaction may use this count before committing.
    pub fn support_counts(&self, workspace: &SfmWorkspace, view: ViewId) -> ImageSupportCounts {
        let mut counts = ImageSupportCounts::default();
        for observation in self.observations(workspace, view) {
            counts.active_landmarks += 1;
            if workspace
                .landmarks()
                .get(observation.landmark)
                .is_some_and(|landmark| {
                    landmark
                        .observation_support
                        .contains(observation.track_observation.index())
                })
            {
                counts.supporting_landmarks += 1;
            }
        }
        counts
    }

    pub fn previous_failures(&self, view: ViewId) -> usize {
        self.registration_failures_by_view
            .get(view.index())
            .copied()
            .unwrap_or(0)
    }

    /// Record a failure already appended to the workspace failure log.
    ///
    /// Incremental growth owns both writes, so maintaining this dense counter avoids rescanning
    /// the complete append-only log on every frontier selection.
    pub fn record_failure(&mut self, view: ViewId) -> Result<()> {
        let count = self
            .registration_failures_by_view
            .get_mut(view.index())
            .ok_or(SfmError::InvalidId {
                kind: ViewId::kind(),
                index: view.index(),
            })?;
        *count = count.checked_add(1).ok_or_else(|| {
            SfmError::InvalidInput("registration failure count overflow".to_owned())
        })?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registration_grid_is_normalized_and_does_not_clamp_outliers() {
        assert_eq!(registration_grid_cell(0.0, 0.0, 100, 80), Some(0));
        assert_eq!(registration_grid_cell(99.0, 79.0, 100, 80), Some(15));
        assert_eq!(registration_grid_cell(50.0, 40.0, 100, 80), Some(10));
        assert_eq!(registration_grid_cell(-0.1, 40.0, 100, 80), None);
        assert_eq!(registration_grid_cell(100.0, 40.0, 100, 80), None);
    }
}
