// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Initial-pair selection: score verified pairs by triangulation angle, rotation-cycle
//! consistency, and support.

use std::collections::VecDeque;

use nalgebra::{Rotation3, Vector3};
use openmvg_core::camera::CameraModel;
use openmvg_core::model::rotation_angle_deg;

use crate::config::{
    DEFAULT_INCREMENTAL_REGISTRATION_SUPPORT, IncrementalSfmConfig, InitialPairConfig,
    InitialPairPolicy,
};
use crate::{PairId, SfmError, SfmWorkspace, ViewId};

/// Minimum median triangulation angle (degrees) for a pair to be considered a
/// *well-conditioned* initial pair. Low-parallax pairs have the most inlier matches but
/// their triangulated structure is ill-conditioned (points near infinity), which
/// breaks P3P growth and cannot be recovered by bundle adjustment, so they are
/// not chosen as the initial pair even when they have the most inliers.
const MIN_SEED_TRIANGULATION_ANGLE_DEG: f64 = 3.0;

/// Maximum median triangle-closure error for an edge to enter the preferred initial-pair
/// score band. Triangle-supported edges above this are not discarded: they remain as
/// fallback candidates below cycle-consistent edges, while bridges (`None`) stay eligible.
const MAX_SEED_ROTATION_CYCLE_ERROR_DEG: f64 = 5.0;

fn preferred_pair_seed_band(well_conditioned: bool, rotation_cycle_error_deg: Option<f64>) -> bool {
    well_conditioned
        && rotation_cycle_error_deg.is_none_or(|error| error <= MAX_SEED_ROTATION_CYCLE_ERROR_DEG)
}

/// Median triangulation angle (degrees) over the shared tracks of a pair, using
/// its verified relative pose at unit baseline. `None` if it can't be computed
/// (no relative pose / intrinsics / points in front of both cameras).
fn median_triangulation_angle_deg(
    workspace: &SfmWorkspace,
    pair_id: PairId,
    view_a: ViewId,
    view_b: ViewId,
    shared_track_indices: &[usize],
) -> Option<f64> {
    let model = workspace.geometric_model_store().get(pair_id.index())?;
    let rel = model.relative_pose()?;
    let r_b = *rel.rotation_source_to_target();
    let center_b: Vector3<f64> = rel.baseline_direction_in_source().into_inner();

    let ia = workspace.view(view_a)?.intrinsic_id;
    let ib = workspace.view(view_b)?.intrinsic_id;
    let camera_a = workspace.camera_model(ia)?;
    let camera_b = workspace.camera_model(ib)?;

    let mut angles = Vec::new();
    for &track_index in shared_track_indices {
        if angles.len() >= 300 {
            break;
        }
        let track = workspace.tracks().as_slice().get(track_index)?;
        let mut a_xy = None;
        let mut b_xy = None;
        for obs in track.observations() {
            if obs.view == view_a {
                a_xy = workspace.track_observation_pixel(obs).ok();
            } else if obs.view == view_b {
                b_xy = workspace.track_observation_pixel(obs).ok();
            }
        }
        let (Some(pa), Some(pb)) = (a_xy, b_xy) else {
            continue;
        };
        let (Some(d1), Some(d2)) = (
            camera_a
                .unproject(&pa)
                .ok()
                .and_then(|bearing| bearing.try_normalize(1e-12)),
            camera_b
                .unproject(&pb)
                .ok()
                .and_then(|bearing| bearing.try_normalize(1e-12)),
        ) else {
            continue;
        };
        let d2 = r_b.inverse() * d2; // world-frame ray from center_b
        // Closest approach of ray(0, d1) and ray(center_b, d2).
        let w0 = -center_b;
        let (a, b, c) = (d1.dot(&d1), d1.dot(&d2), d2.dot(&d2));
        let (dd, ee) = (d1.dot(&w0), d2.dot(&w0));
        let denom = a * c - b * b;
        if denom.abs() < 1e-12 {
            continue;
        }
        let s = (b * ee - c * dd) / denom;
        let t = (a * ee - b * dd) / denom;
        if s <= 0.0 || t <= 0.0 {
            continue; // point behind a camera
        }
        let x = (d1 * s + (center_b + d2 * t)) * 0.5;
        let (ray_a, ray_b) = (x, x - center_b);
        let (na, nb) = (ray_a.norm(), ray_b.norm());
        if na < 1e-9 || nb < 1e-9 {
            continue;
        }
        let cos = (ray_a.dot(&ray_b) / (na * nb)).clamp(-1.0, 1.0);
        angles.push(cos.acos().to_degrees());
    }
    if angles.is_empty() {
        return None;
    }
    angles.sort_by(f64::total_cmp);
    Some(angles[angles.len() / 2])
}

/// Median rotation closure error over every verified triangle containing an edge.
///
/// For directed relative rotations `R_ij = R_j R_i^-1`, a consistent triangle obeys
/// `R_ab = R_bc^-1 R_ac`.  This is a GT-free confidence signal for initial-pair selection: a
/// high-support essential can still be the catastrophic edge in an otherwise coherent
/// component, while a bridge edge (no triangle) correctly returns `None` rather than
/// being labeled inconsistent.
fn median_rotation_cycle_error_deg(
    workspace: &SfmWorkspace,
    view_a: ViewId,
    view_b: ViewId,
) -> Option<f64> {
    let r_ab = directed_relative_rotation(workspace, view_a, view_b)?;
    let mut errors = Vec::new();
    for neighbor in workspace.pair_graph().neighbors(view_a).ok()? {
        let view_c = neighbor.view;
        if view_c == view_b || !workspace.pair_graph().is_active(neighbor.pair_id) {
            continue;
        }
        let (Some(r_ac), Some(r_bc)) = (
            directed_relative_rotation(workspace, view_a, view_c),
            directed_relative_rotation(workspace, view_b, view_c),
        ) else {
            continue;
        };
        let error = triangle_closure_error_deg(&r_ab, &r_ac, &r_bc);
        if error.is_finite() {
            errors.push(error);
        }
    }
    if errors.is_empty() {
        return None;
    }
    errors.sort_by(f64::total_cmp);
    Some(errors[errors.len() / 2])
}

/// Disagreement, in degrees, between `R_ab` and the rotation `R_bc^-1 R_ac` the other two edges of
/// the triangle imply.
fn triangle_closure_error_deg(
    r_ab: &Rotation3<f64>,
    r_ac: &Rotation3<f64>,
    r_bc: &Rotation3<f64>,
) -> f64 {
    rotation_angle_deg(&(r_ab.inverse() * (r_bc.inverse() * r_ac)))
}

fn directed_relative_rotation(
    workspace: &SfmWorkspace,
    view_from: ViewId,
    view_to: ViewId,
) -> Option<Rotation3<f64>> {
    let pair_id = workspace.pair_graph().pair_id(view_from, view_to)?;
    if !workspace.pair_graph().is_active(pair_id) {
        return None;
    }
    let pair = workspace.pair_graph().pair(pair_id)?;
    let rotation = workspace
        .geometric_model_store()
        .get(pair_id.index())?
        .relative_pose()?
        .rotation_source_to_target();
    Some(if pair.first() == view_from {
        *rotation
    } else {
        rotation.inverse()
    })
}

/// A candidate initial pair for incremental reconstruction, with its ranking score.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct InitialPairHypothesis {
    /// The two views of the initial pair.
    pub views: Vec<ViewId>,
    /// Verified pairs the initial pair is built from.
    pub pairs: Vec<crate::ViewPair>,
    /// Finite ranking score, higher is better. Lexicographic bands packed into one number:
    /// parallax and cycle consistency first, then component size, frontier, and pair support.
    pub score: f64,
    /// The evidence behind `score`.
    pub breakdown: InitialPairScoreBreakdown,
}

/// Evidence used to score one initial-pair hypothesis.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct InitialPairScoreBreakdown {
    /// Verified geometric inliers of the initial pair.
    pub inliers: usize,
    /// Tracks observed in both views of the initial pair.
    pub shared_tracks: usize,
    /// Size of the active verified-pair connected component containing the initial pair.
    #[serde(default)]
    pub component_size: usize,
    /// Views outside the initial pair that observe enough tracks triangulatable from it for P3P.
    #[serde(default)]
    pub frontier_views: usize,
    /// Observations of tracks triangulatable from the initial pair, across the other views.
    #[serde(default)]
    pub frontier_tracks: usize,
    /// Median triangulation angle in degrees over up to 300 shared tracks, using the verified
    /// relative pose at unit baseline. `None` when it cannot be computed (no relative pose or
    /// no point in front of both cameras).
    pub triangulation_angle_deg: Option<f64>,
    /// Median rotation closure error in degrees over the verified triangles containing the
    /// pair. `None` for a bridge edge with no triangle.
    pub rotation_cycle_error_deg: Option<f64>,
    /// Qualitative properties the initial pair satisfies, in evaluation order.
    pub reasons: Vec<InitialPairReasonCode>,
}

/// A qualitative property recorded for an initial-pair hypothesis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InitialPairReasonCode {
    /// The pair meets the minimum verified inlier count.
    StrongInlierSupport,
    /// The pair shares at least one track.
    HasSharedTracks,
    /// The pair lies in the largest active verified-pair connected component.
    InLargestConnectedComponent,
    /// At least one other view observes enough of the pair's tracks for registration.
    HasRegistrationFrontier,
    /// The pair has triangle evidence and its median closure error is at most 5 degrees.
    RotationCycleConsistent,
}

#[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
pub(crate) fn seed_candidates(
    workspace: &SfmWorkspace,
    strategy: &InitialPairPolicy,
) -> crate::Result<Vec<InitialPairHypothesis>> {
    seed_candidates_with_registration_support(
        workspace,
        strategy,
        DEFAULT_INCREMENTAL_REGISTRATION_SUPPORT,
    )
}

fn seed_candidates_with_registration_support(
    workspace: &SfmWorkspace,
    strategy: &InitialPairPolicy,
    min_registration_support: usize,
) -> crate::Result<Vec<InitialPairHypothesis>> {
    match strategy {
        InitialPairPolicy::ExplicitPair(pair) => {
            if pair.first() == pair.second() {
                return Err(SfmError::InvalidInput(
                    "an explicit initial pair must contain two distinct views".to_owned(),
                ));
            }
            let pair_id = workspace
                .pair_graph()
                .pair_id(pair.first(), pair.second())
                .ok_or(SfmError::NoInitialPairCandidate)?;
            let mut candidates = best_pair_candidates(
                workspace,
                InitialPairConfig::default().with_min_inliers(0),
                min_registration_support,
            )?;
            let pair = workspace
                .pair_graph()
                .pair(pair_id)
                .ok_or(SfmError::NoInitialPairCandidate)?;
            candidates.retain(|candidate| candidate.pairs == [pair]);
            if candidates.is_empty() {
                return Err(SfmError::NoInitialPairCandidate);
            }
            Ok(candidates)
        }
        InitialPairPolicy::BestPair(config) => {
            best_pair_candidates(workspace, *config, min_registration_support)
        }
    }
}

/// Return the best-scoring initial-pair hypothesis for `strategy`.
///
/// Registration-frontier counts use the default initial-pair screening support floor, not the
/// configured incremental `min_registration_support`.
///
/// # Errors
///
/// [`SfmError::NoInitialPairCandidate`] if no pair qualifies (or the explicit pair is not an active
/// verified pair), and [`SfmError::InvalidInput`] if an explicit pair repeats one view.
#[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
pub fn select_initial_pair(
    workspace: &SfmWorkspace,
    strategy: &InitialPairPolicy,
) -> crate::Result<InitialPairHypothesis> {
    seed_candidates(workspace, strategy)?
        .into_iter()
        .next()
        .ok_or(SfmError::NoInitialPairCandidate)
}

pub(crate) fn select_initial_pair_for_incremental(
    workspace: &SfmWorkspace,
    config: &IncrementalSfmConfig,
) -> crate::Result<InitialPairHypothesis> {
    seed_candidates_with_registration_support(
        workspace,
        config.initial_pair(),
        config.min_registration_support(),
    )?
    .into_iter()
    .next()
    .ok_or(SfmError::NoInitialPairCandidate)
}

fn best_pair_candidates(
    workspace: &SfmWorkspace,
    config: InitialPairConfig,
    min_registration_support: usize,
) -> crate::Result<Vec<InitialPairHypothesis>> {
    let component_sizes = active_component_sizes(workspace)?;
    let pair_tracks = PairTrackIndex::build(workspace);
    let largest_component = component_sizes.iter().copied().max().unwrap_or(0);
    let view_count = workspace.views().len().max(1);
    let mut candidates = Vec::new();
    for pair_id in workspace.pair_graph().active_pair_ids() {
        let Some(pair) = workspace.pair_graph().pair(pair_id) else {
            continue;
        };
        let Some(model) = workspace.geometric_model_store().get(pair_id.index()) else {
            continue;
        };
        let inliers = model.evidence().inlier_count;
        if inliers < config.min_inliers() {
            continue;
        }
        let shared_track_indices = pair_tracks.get(pair_id);
        let shared_tracks = shared_track_indices.len();
        let mut reasons = vec![InitialPairReasonCode::StrongInlierSupport];
        if shared_tracks > 0 {
            reasons.push(InitialPairReasonCode::HasSharedTracks);
        }
        let component_size = component_sizes[pair.first().index()];
        if component_size == largest_component {
            reasons.push(InitialPairReasonCode::InLargestConnectedComponent);
        }
        let (frontier_views, frontier_tracks) = pair_seed_frontier_counts(
            workspace,
            pair.first(),
            pair.second(),
            shared_track_indices,
            min_registration_support,
        );
        if frontier_views > 0 {
            reasons.push(InitialPairReasonCode::HasRegistrationFrontier);
        }
        // Parallax gate: a well-conditioned (adequate-parallax) pair forms a
        // strictly higher scoring band than any low-parallax pair, so it is
        // always preferred even if a degenerate near-zero-baseline pair has more
        // inliers. Inliers + shared tracks rank within a band. If no pair clears
        // the gate the whole set stays in the low band (graceful fallback).
        let tri_angle = median_triangulation_angle_deg(
            workspace,
            pair_id,
            pair.first(),
            pair.second(),
            shared_track_indices,
        );
        let rotation_cycle_error_deg =
            median_rotation_cycle_error_deg(workspace, pair.first(), pair.second());
        let well_conditioned = tri_angle.is_some_and(|a| a >= MIN_SEED_TRIANGULATION_ANGLE_DEG);
        let cycle_consistent =
            rotation_cycle_error_deg.is_none_or(|error| error <= MAX_SEED_ROTATION_CYCLE_ERROR_DEG);
        if rotation_cycle_error_deg.is_some() && cycle_consistent {
            reasons.push(InitialPairReasonCode::RotationCycleConsistent);
        }
        // Lexicographic score bands, expressed as one finite public score:
        // parallax + cycle consistency first, then reachable component, immediately
        // registrable frontier, and finally pair support. Every term is
        // normalized/bounded so a lower-priority term cannot leak into the next band.
        // A cycle-inconsistent edge remains a fallback in case every triangle-supported
        // candidate is inconsistent; bridges have no cycle evidence and remain eligible.
        let component_fraction = component_size as f64 / view_count as f64;
        let frontier_fraction = frontier_views as f64 / view_count as f64;
        let support = (inliers as f64 + shared_tracks as f64 * 0.25).ln_1p();
        let score = if preferred_pair_seed_band(well_conditioned, rotation_cycle_error_deg) {
            1.0e12
        } else {
            0.0
        } + component_fraction * 1.0e9
            + frontier_fraction * 1.0e6
            + (frontier_tracks as f64).ln_1p() * 1.0e3
            + support;
        candidates.push(InitialPairHypothesis {
            views: vec![pair.first(), pair.second()],
            pairs: vec![pair],
            score,
            breakdown: InitialPairScoreBreakdown {
                inliers,
                shared_tracks,
                component_size,
                frontier_views,
                frontier_tracks,
                triangulation_angle_deg: tri_angle,
                rotation_cycle_error_deg,
                reasons,
            },
        });
    }
    sort_candidates(&mut candidates);
    Ok(candidates)
}

/// Stable track lookup for initial-pair scoring.
///
/// The index materializes the pair-to-track relation once instead of rescanning the track table
/// for every active pair. Track indices are appended in source order, so the first-300 parallax
/// sample and every score are deterministic.
struct PairTrackIndex {
    by_pair: Vec<Vec<usize>>,
}

impl PairTrackIndex {
    fn build(workspace: &SfmWorkspace) -> Self {
        let mut by_pair = vec![Vec::new(); workspace.pair_graph().pairs().len()];
        for (track_index, track) in workspace.tracks().as_slice().iter().enumerate() {
            let observations = track.observations();
            for left in 0..observations.len() {
                for right in (left + 1)..observations.len() {
                    let view_a = observations[left].view;
                    let view_b = observations[right].view;
                    let Some(pair_id) = workspace.pair_graph().pair_id(view_a, view_b) else {
                        continue;
                    };
                    if workspace.pair_graph().is_active(pair_id) {
                        by_pair[pair_id.index()].push(track_index);
                    }
                }
            }
        }
        Self { by_pair }
    }

    fn get(&self, pair: PairId) -> &[usize] {
        self.by_pair
            .get(pair.index())
            .map(Vec::as_slice)
            .unwrap_or_default()
    }
}

fn pair_seed_frontier_counts(
    workspace: &SfmWorkspace,
    view_a: ViewId,
    view_b: ViewId,
    shared_track_indices: &[usize],
    min_registration_support: usize,
) -> (usize, usize) {
    let mut observations_per_view = vec![0usize; workspace.views().len()];
    for &track_index in shared_track_indices {
        let Some(track) = workspace.tracks().as_slice().get(track_index) else {
            continue;
        };
        for observation in track.observations() {
            if observation.view != view_a && observation.view != view_b {
                observations_per_view[observation.view.index()] += 1;
            }
        }
    }
    let frontier_views = observations_per_view
        .iter()
        .enumerate()
        .filter(|(image, count)| {
            let image = ViewId::new(*image);
            image != view_a && image != view_b && **count >= min_registration_support
        })
        .count();
    let frontier_tracks = observations_per_view.iter().sum();
    (frontier_views, frontier_tracks)
}

fn active_component_sizes(workspace: &SfmWorkspace) -> crate::Result<Vec<usize>> {
    let mut sizes = vec![0usize; workspace.views().len()];
    let mut seen = vec![false; workspace.views().len()];
    for seed_index in 0..seen.len() {
        if seen[seed_index] {
            continue;
        }
        let mut component = Vec::new();
        let mut queue = VecDeque::from([ViewId::new(seed_index)]);
        seen[seed_index] = true;
        while let Some(image) = queue.pop_front() {
            component.push(image);
            for neighbor in workspace.pair_graph().neighbors(image)? {
                if !workspace.pair_graph().is_active(neighbor.pair_id) {
                    continue;
                }
                let index = neighbor.view.index();
                if !seen[index] {
                    seen[index] = true;
                    queue.push_back(neighbor.view);
                }
            }
        }
        let size = component.len();
        for image in component {
            sizes[image.index()] = size;
        }
    }
    Ok(sizes)
}

fn sort_candidates(candidates: &mut [InitialPairHypothesis]) {
    candidates.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.views.cmp(&b.views))
            .then_with(|| a.pairs.cmp(&b.pairs))
    });
}

#[cfg(test)]
mod tests {
    use super::{
        PairTrackIndex, pair_seed_frontier_counts, preferred_pair_seed_band,
        triangle_closure_error_deg,
    };
    use crate::config::DEFAULT_INCREMENTAL_REGISTRATION_SUPPORT;
    use crate::model::ImageMeta;
    use crate::{
        Camera, FeatureId, IntrinsicId, Track, TrackId, TrackObservation, TrackSet, ViewId,
    };

    #[test]
    fn preferred_pair_seed_band_demotes_cycle_inconsistent_edges_but_keeps_bridges() {
        assert!(preferred_pair_seed_band(true, None));
        assert!(preferred_pair_seed_band(true, Some(5.0)));
        assert!(!preferred_pair_seed_band(true, Some(5.000_001)));
        assert!(!preferred_pair_seed_band(false, None));
    }

    #[test]
    fn an_exactly_closing_triangle_has_zero_error_not_a_dropped_one() {
        let r_ab = nalgebra::Rotation3::from_axis_angle(&nalgebra::Vector3::y_axis(), 0.3);
        let r_bc = nalgebra::Rotation3::from_axis_angle(&nalgebra::Vector3::x_axis(), -0.2);
        let r_ac = r_bc * r_ab;
        let error = triangle_closure_error_deg(&r_ab, &r_ac, &r_bc);
        assert!(error.is_finite() && error < 1e-6, "{error}");
        let bent = nalgebra::Rotation3::from_axis_angle(&nalgebra::Vector3::z_axis(), 0.1) * r_ac;
        assert!(
            (triangle_closure_error_deg(&r_ab, &bent, &r_bc) - 0.1_f64.to_degrees()).abs() < 1e-9
        );
    }

    #[test]
    fn pair_track_index_preserves_shared_track_and_frontier_counts() {
        let mut workspace = crate::SfmWorkspace::builder()
            .image_meta(
                (0..4).map(|index| ImageMeta::external_id(format!("image-{index}"), 64, 48)),
            )
            .camera_models([Camera::try_pinhole(100.0, 100.0, 32.0, 24.0).unwrap()])
            .shared_intrinsic(IntrinsicId::new(0))
            .exhaustive_pairs()
            .build()
            .expect("valid workspace");
        let make_track = |id, images: &[usize]| {
            Track::try_new(
                TrackId::new(id),
                images
                    .iter()
                    .map(|&image| TrackObservation {
                        view: ViewId::new(image),
                        feature: FeatureId::new(id),
                    })
                    .collect(),
            )
            .expect("valid track")
        };
        workspace.tracks = TrackSet::try_new(vec![
            make_track(0, &[0, 1, 2]),
            make_track(1, &[0, 1, 3]),
            make_track(2, &[0, 2, 3]),
            make_track(3, &[0, 1, 2, 3]),
        ])
        .expect("valid tracks");

        let pair = workspace
            .pair_graph()
            .pair_id(ViewId::new(0), ViewId::new(1))
            .expect("pair exists");
        let index = PairTrackIndex::build(&workspace);
        assert_eq!(index.get(pair).len(), 3);
        assert_eq!(
            pair_seed_frontier_counts(
                &workspace,
                ViewId::new(0),
                ViewId::new(1),
                index.get(pair),
                DEFAULT_INCREMENTAL_REGISTRATION_SUPPORT,
            ),
            (0, 4)
        );
        assert_eq!(
            pair_seed_frontier_counts(
                &workspace,
                ViewId::new(0),
                ViewId::new(1),
                index.get(pair),
                2,
            ),
            (2, 4)
        );
    }
}
