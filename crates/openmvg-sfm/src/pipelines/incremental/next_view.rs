// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Next-view selection: rank unregistered views by landmark support and image coverage.

use std::{cmp::Ordering, collections::BTreeSet};

use crate::{SfmWorkspace, ViewId};

use super::registration_index::{RegistrationObservationIndex, registration_grid_cell};

const MIN_WELL_DISTRIBUTED_REGISTRATION_CELLS: usize = 4;

/// An unregistered image eligible for resection, with the support used to rank it.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RegistrationCandidate {
    /// The unregistered image.
    pub view: ViewId,
    /// Active landmarks whose source track observes the image (potential 2D-3D matches).
    pub visible_landmarks: usize,
    /// Cells of the 4x4 image grid holding at least one of those observations (0 to 16).
    #[serde(default)]
    pub occupied_image_cells: usize,
    /// Active verified pair edges from the image to already registered images.
    pub verified_registered_neighbors: usize,
    /// Tracks observing the image, whether or not they are triangulated.
    pub track_observations: usize,
    /// Registration failures already recorded for the image in this workspace.
    pub previous_failures: usize,
}

/// Result of one frontier scan.
///
/// The counters make the cost and selectivity of the scheduling policy observable without
/// retaining the complete candidate list in the public reconstruction report.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RegistrationFrontierSelection {
    pub candidate: Option<RegistrationCandidate>,
    pub eligible_candidates: usize,
    pub support_band_candidates: usize,
}

/// Eligibility gates for the incremental registration frontier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct GrowthConfig {
    min_visible_landmarks: usize,
    min_verified_registered_neighbors: usize,
}

config_accessors!(GrowthConfig {
    /// Minimum active landmarks the candidate image must observe to enter the frontier.
    copy min_visible_landmarks, with_min_visible_landmarks: usize;
    /// Minimum active verified edges from the candidate image to cameras that are
    /// already registered. Zero permits registration from transitive track
    /// intersections alone; one prevents graph-unanchored sparse tails.
    #[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
    copy min_verified_registered_neighbors, with_min_verified_registered_neighbors: usize;
});

impl Default for GrowthConfig {
    fn default() -> Self {
        Self {
            min_visible_landmarks: 0,
            min_verified_registered_neighbors: 1,
        }
    }
}

/// Every unposed image passing `config`, ranked best first.
///
/// Ranking is by visible landmarks, then verified registered neighbors, then track
/// observations (all descending), then fewer previous failures, then image id.
#[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
pub fn registration_candidates(
    workspace: &SfmWorkspace,
    config: &GrowthConfig,
) -> Vec<RegistrationCandidate> {
    // Invert the data in ONE pass instead of scanning all landmarks+tracks per image
    // (was O(images × landmarks × obs)). `visible[img]` is the number of active
    // landmarks whose source track observes `img`. TrackCollection already owns the
    // immutable image-to-track adjacency, so no second count array is needed here.
    let view_count = workspace.views().len();
    let mut visible = vec![0usize; view_count];
    let mut occupied_cells = vec![0u16; view_count];
    // Only Active landmarks yield 2D-3D observations at registration time
    // (pose_registration_observations filters on Active); counting Rejected ones would
    // overstate visibility and re-queue images that then fail.
    for landmark in workspace.landmarks().iter_active() {
        let Some(track) = workspace.tracks().get(landmark.source_track) else {
            continue;
        };
        for observation in track.observations() {
            if let Some(slot) = visible.get_mut(observation.view.index()) {
                *slot += 1;
            }
            let Some(view) = workspace.view(observation.view) else {
                continue;
            };
            let Ok(pixel) = workspace.track_observation_pixel(observation) else {
                continue;
            };
            if let Some(cell) =
                registration_grid_cell(pixel.x, pixel.y, view.meta.width, view.meta.height)
            {
                occupied_cells[observation.view.index()] |= 1 << cell;
            }
        }
    }
    let failure_counts = registration_failure_counts(workspace);

    let mut candidates = collect_registration_candidates(
        workspace,
        config,
        |_| false,
        |view| visible[view.index()],
        |view| occupied_cells[view.index()].count_ones() as usize,
        |view| workspace.tracks().observing_view(view).len(),
        |view| failure_counts.get(view.index()).copied().unwrap_or(0),
    );
    candidates.sort_by(compare_registration_candidates);
    candidates
}

/// Select the next growth candidate with one allocation and linear ranking.
///
/// Remove candidates deferred in the current sweep,
/// retain the relative visible-support band, then prefer well-distributed image support—but it
/// avoids sorting the complete frontier twice and materializing two additional vectors.
pub(crate) fn select_registration_candidate_indexed(
    workspace: &SfmWorkspace,
    config: &GrowthConfig,
    index: &RegistrationObservationIndex,
    deferred: &BTreeSet<ViewId>,
    min_support_ratio: f64,
) -> RegistrationFrontierSelection {
    let candidates = collect_registration_candidates(
        workspace,
        config,
        |view| deferred.contains(&view),
        |view| index.visible_landmarks(view),
        |view| index.occupied_image_cells(view),
        |view| workspace.tracks().observing_view(view).len(),
        |view| index.previous_failures(view),
    );
    select_conditioned_candidate(candidates, min_support_ratio)
}

#[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
fn registration_failure_counts(workspace: &SfmWorkspace) -> Vec<usize> {
    // ImageIds are a dense workspace domain, so a flat vector is both simpler and cheaper than
    // hashing. Repeated failures remain a real scoring signal rather than write-only state.
    let mut counts = vec![0usize; workspace.views().len()];
    for failure in workspace.registration_failures() {
        if let Some(count) = counts.get_mut(failure.view.index()) {
            *count = count.saturating_add(1);
        }
    }
    counts
}

fn collect_registration_candidates<Excluded, Visible, OccupiedCells, TrackCount, PreviousFailures>(
    workspace: &SfmWorkspace,
    config: &GrowthConfig,
    excluded: Excluded,
    visible_landmarks: Visible,
    occupied_image_cells: OccupiedCells,
    track_observations: TrackCount,
    previous_failures: PreviousFailures,
) -> Vec<RegistrationCandidate>
where
    Excluded: Fn(ViewId) -> bool,
    Visible: Fn(ViewId) -> usize,
    OccupiedCells: Fn(ViewId) -> usize,
    TrackCount: Fn(ViewId) -> usize,
    PreviousFailures: Fn(ViewId) -> usize,
{
    let mut candidates = Vec::new();
    for view in workspace.views() {
        if workspace.has_pose(view.id) || excluded(view.id) {
            continue;
        }
        let visible_landmarks = visible_landmarks(view.id);
        if visible_landmarks < config.min_visible_landmarks {
            continue;
        }
        let track_observations = track_observations(view.id);
        let verified_registered_neighbors = workspace
            .pair_graph()
            .neighbors(view.id)
            .map(|neighbors| {
                neighbors
                    .iter()
                    .filter(|neighbor| {
                        workspace.pair_graph().is_active(neighbor.pair_id)
                            && workspace.has_pose(neighbor.view)
                    })
                    .count()
            })
            .unwrap_or(0);
        if verified_registered_neighbors < config.min_verified_registered_neighbors {
            continue;
        }
        let previous_failures = previous_failures(view.id);
        candidates.push(RegistrationCandidate {
            view: view.id,
            visible_landmarks,
            occupied_image_cells: occupied_image_cells(view.id),
            verified_registered_neighbors,
            track_observations,
            previous_failures,
        });
    }
    candidates
}

#[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
fn compare_registration_candidates(
    a: &RegistrationCandidate,
    b: &RegistrationCandidate,
) -> Ordering {
    b.visible_landmarks
        .cmp(&a.visible_landmarks)
        .then_with(|| {
            b.verified_registered_neighbors
                .cmp(&a.verified_registered_neighbors)
        })
        .then_with(|| b.track_observations.cmp(&a.track_observations))
        .then_with(|| a.previous_failures.cmp(&b.previous_failures))
        .then_with(|| a.view.cmp(&b.view))
}

/// Prefer spatially conditioned candidates within an already selected support band.
///
/// This is a scheduling policy, not an acceptance gate: weakly distributed
/// cameras remain eligible and are attempted once stronger frontier views are exhausted.
#[cfg(test)]
pub(crate) fn prioritize_registration_candidates(candidates: &mut [RegistrationCandidate]) {
    candidates.sort_by(compare_conditioned_candidates);
}

fn compare_conditioned_candidates(
    a: &RegistrationCandidate,
    b: &RegistrationCandidate,
) -> Ordering {
    coverage_band(b.occupied_image_cells)
        .cmp(&coverage_band(a.occupied_image_cells))
        .then_with(|| b.visible_landmarks.cmp(&a.visible_landmarks))
        .then_with(|| {
            b.verified_registered_neighbors
                .cmp(&a.verified_registered_neighbors)
        })
        .then_with(|| b.track_observations.cmp(&a.track_observations))
        .then_with(|| a.previous_failures.cmp(&b.previous_failures))
        .then_with(|| a.view.cmp(&b.view))
}

fn select_conditioned_candidate(
    mut candidates: Vec<RegistrationCandidate>,
    min_support_ratio: f64,
) -> RegistrationFrontierSelection {
    let eligible_candidates = candidates.len();
    let Some(max_visible_landmarks) = candidates
        .iter()
        .map(|candidate| candidate.visible_landmarks)
        .max()
    else {
        return RegistrationFrontierSelection {
            candidate: None,
            eligible_candidates,
            support_band_candidates: 0,
        };
    };
    let min_support = max_visible_landmarks as f64 * min_support_ratio;
    candidates.retain(|candidate| candidate.visible_landmarks as f64 >= min_support);
    let support_band_candidates = candidates.len();
    let candidate = candidates
        .into_iter()
        .min_by(compare_conditioned_candidates);
    RegistrationFrontierSelection {
        candidate,
        eligible_candidates,
        support_band_candidates,
    }
}

fn coverage_band(occupied_image_cells: usize) -> u8 {
    u8::from(occupied_image_cells >= MIN_WELL_DISTRIBUTED_REGISTRATION_CELLS)
}

/// Length of the strongest candidate prefix admitted by a relative support gate.
///
/// `registration_candidates` orders candidates by decreasing visible-landmark support,
/// so the eligible set is always a prefix and can still be bounded by a separate hard cap.
#[cfg(test)]
pub(crate) fn relative_support_prefix_len(
    candidates: &[RegistrationCandidate],
    min_support_ratio: f64,
) -> usize {
    let Some(best) = candidates.first() else {
        return 0;
    };
    let min_support = best.visible_landmarks as f64 * min_support_ratio;
    candidates
        .iter()
        .take_while(|candidate| candidate.visible_landmarks as f64 >= min_support)
        .count()
}

#[cfg(test)]
mod tests {
    use super::{RegistrationCandidate, relative_support_prefix_len, select_conditioned_candidate};
    use crate::ViewId;

    fn candidate(view: usize, visible_landmarks: usize) -> RegistrationCandidate {
        RegistrationCandidate {
            view: ViewId::new(view),
            visible_landmarks,
            occupied_image_cells: 16,
            verified_registered_neighbors: 1,
            track_observations: visible_landmarks,
            previous_failures: 0,
        }
    }

    #[test]
    fn relative_support_gate_keeps_the_76_percent_prefix() {
        let candidates = [
            candidate(0, 100),
            candidate(1, 90),
            candidate(2, 76),
            candidate(3, 75),
            candidate(4, 20),
        ];

        assert_eq!(relative_support_prefix_len(&candidates, 0.76), 3);
        assert_eq!(relative_support_prefix_len(&candidates, 1.0), 1);
        assert_eq!(relative_support_prefix_len(&[], 0.76), 0);
    }

    #[test]
    fn conditioning_ranking_defers_clustered_support_without_rejecting_it() {
        let mut candidates = [candidate(0, 100), candidate(1, 90), candidate(2, 80)];
        candidates[0].occupied_image_cells = 3;
        candidates[1].occupied_image_cells = 5;
        candidates[2].occupied_image_cells = 6;

        super::prioritize_registration_candidates(&mut candidates);

        assert_eq!(candidates.len(), 3);
        assert_eq!(
            candidates.map(|candidate| candidate.view.index()),
            [1, 2, 0]
        );
    }

    #[test]
    fn linear_frontier_selection_matches_relative_support_then_conditioning_policy() {
        let mut candidates = vec![
            candidate(0, 100),
            candidate(1, 90),
            candidate(2, 76),
            candidate(3, 75),
        ];
        candidates[0].occupied_image_cells = 3;
        candidates[1].occupied_image_cells = 5;
        candidates[2].occupied_image_cells = 6;
        candidates[3].occupied_image_cells = 16;

        let selected = select_conditioned_candidate(candidates, 0.76);

        assert_eq!(selected.eligible_candidates, 4);
        assert_eq!(selected.support_band_candidates, 3);
        assert_eq!(selected.candidate.unwrap().view, ViewId::new(1));
    }
}
