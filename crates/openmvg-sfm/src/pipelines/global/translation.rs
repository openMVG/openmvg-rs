// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Candidate translation estimation from checked rotation estimates.
//!
//! Direction and mixed-positioning candidates share the same validation and
//! triangulation boundary before entering transactional refinement.

use crate::backends::Triangulator;
use crate::config::GlobalAveragingConfig;
use crate::determinism::GLOBAL_MIXED_POSITIONING_SEED_DOMAIN;
use crate::ops::averaging::GlobalAveragingReport;
use crate::ops::probes::{GraphProbes, active_edge_component_count, graph_probes};
use crate::progress::{SfmProgressSink, SfmProgressStage, cancellation_point};
use crate::{Result, SfmError, SfmWorkspace, ViewId};
use nalgebra::{Rotation3, Vector3};
use openmvg_core::camera::CameraModel;
use openmvg_geometry::averaging::global_positioning::{GlobalPositioningConfig, position_globally};
use std::collections::{BTreeMap, BTreeSet};

type PositioningTrack = Vec<(ViewId, Vector3<f64>)>;
type RawPositioningTracks = BTreeMap<usize, PositioningTrack>;

const MAX_DEFERRED_POSITIONING_VIEWS: usize = 4;
const MAX_DEFERRED_POSITIONING_FRACTION: f64 = 0.005;

/// One global translation candidate: averaging, triangulation, and the resulting view graph.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct GlobalTranslationReport {
    /// Averaging configuration the candidate was solved with.
    pub config: GlobalAveragingConfig,
    /// Active-pair graph statistics measured after triangulation.
    pub probes: GraphProbes,
    /// Rotation and translation averaging diagnostics.
    pub averaging: GlobalAveragingReport,
    /// Views holding features when the candidate started.
    pub input_featured_view_count: usize,
    /// Image pairs holding putative matches when the candidate started.
    pub input_match_pair_count: usize,
    /// Pairs holding a verified geometric model when the candidate started.
    pub input_verified_pair_count: usize,
    /// Tracks offered to triangulation.
    pub input_track_count: usize,
    /// Track triangulation outcome under the candidate poses.
    pub triangulation: crate::ops::reconstruction::TriangulationReport,
    /// Active landmarks left after triangulation.
    pub final_active_landmarks: usize,
}

/// Data budget for the pretracked global positioning candidate.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PretrackedGlobalPositioningConfig {
    max_observations: usize,
    min_track_views: usize,
    direction_reweight: f64,
}

config_accessors!(PretrackedGlobalPositioningConfig {
    /// Maximum admitted camera-point observations. Zero consumes all stable tracks.
    #[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
    copy max_observations, with_max_observations: usize;
    /// Minimum views per admitted track. Values below two are clamped to two. The default is
    /// three, matching GLOMAP and avoiding weak two-view depth constraints.
    copy min_track_views, with_min_track_views: usize;
    /// Aggregate relative-direction mass versus bearing-track mass in mixed mode.
    copy direction_reweight, with_direction_reweight: f64;
});

impl Default for PretrackedGlobalPositioningConfig {
    fn default() -> Self {
        Self {
            max_observations: 0,
            min_track_views: 3,
            direction_reweight: 1.0,
        }
    }
}

/// Direction-based translation from one shared rotation solution checked against the view graph.
pub(crate) fn solve_direction_translation(
    workspace: &mut SfmWorkspace,
    config: &GlobalAveragingConfig,
    translation_scope: crate::config::GlobalTranslationScope,
    rotation_preparation: &crate::ops::averaging::GlobalRotationPreparation,
    translation_solver: crate::backends::TranslationAveragingFn,
    triangulator: &dyn Triangulator,
    progress: &dyn SfmProgressSink,
) -> Result<GlobalTranslationReport> {
    cancellation_point(progress)?;
    progress.begin(SfmProgressStage::TranslationAveraging, 1);
    let rotations = validate_pretracked_rotations(
        workspace,
        config,
        translation_scope,
        rotation_preparation,
        None,
    )?;
    let averaging = crate::ops::averaging::run_global_translation_from_validated_rotations(
        workspace,
        translation_solver,
        rotations,
        config
            .min_translation_parallax()
            .map(|threshold| threshold.value().to_degrees()),
        translation_scope,
        config.min_edge_isotropy(),
    )?;
    progress.advance(SfmProgressStage::TranslationAveraging, 1);
    progress.finish(SfmProgressStage::TranslationAveraging);
    finish_global_reconstruction(
        workspace,
        config,
        triangulator,
        workspace.keypoint_store().populated_count(),
        workspace.match_store().populated_count(),
        workspace.geometric_model_store().populated_count(),
        averaging,
        progress,
    )
}

/// Global positioning translation from one shared rotation solution checked against the view
/// graph.
///
/// This path does not accept or invoke a direction-translation solver, because global
/// positioning estimates centers directly from raw tracks, active directions, and the prepared
/// rotations.
pub(crate) fn solve_mixed_translation(
    workspace: &mut SfmWorkspace,
    config: &GlobalAveragingConfig,
    translation_scope: crate::config::GlobalTranslationScope,
    positioning: &PretrackedGlobalPositioningConfig,
    rotation_preparation: &crate::ops::averaging::GlobalRotationPreparation,
    triangulator: &dyn Triangulator,
    progress: &dyn SfmProgressSink,
) -> Result<GlobalTranslationReport> {
    cancellation_point(progress)?;
    progress.begin(SfmProgressStage::TranslationAveraging, 1);
    let rotations = validate_pretracked_rotations(
        workspace,
        config,
        translation_scope,
        rotation_preparation,
        Some(positioning),
    )?;
    let rotations = rotations.estimates();
    let mut averaging = crate::ops::averaging::rotation_only_averaging_report(
        workspace.pair_graph().active_count(),
        rotations,
    );
    apply_pretracked_global_positioning_from_rotations(
        workspace,
        positioning,
        rotations,
        &mut averaging,
    )?;
    progress.advance(SfmProgressStage::TranslationAveraging, 1);
    progress.finish(SfmProgressStage::TranslationAveraging);
    finish_global_reconstruction(
        workspace,
        config,
        triangulator,
        workspace.keypoint_store().populated_count(),
        workspace.match_store().populated_count(),
        workspace.geometric_model_store().populated_count(),
        averaging,
        progress,
    )
}

fn validate_pretracked_rotations<'a>(
    workspace: &SfmWorkspace,
    config: &GlobalAveragingConfig,
    translation_scope: crate::config::GlobalTranslationScope,
    rotation_preparation: &'a crate::ops::averaging::GlobalRotationPreparation,
    positioning: Option<&PretrackedGlobalPositioningConfig>,
) -> Result<crate::ops::averaging::ValidatedGlobalRotations<'a>> {
    config.validate()?;
    if workspace.tracks().is_empty() {
        return Err(SfmError::MissingPrerequisite {
            prerequisite: "precomputed tracks",
            method: "build_tracks",
        });
    }
    if active_edge_component_count(workspace)? != 1 {
        return Err(SfmError::InvalidInput(
            "prepared global rotations require one already conditioned active component".to_owned(),
        ));
    }
    let rotations = rotation_preparation.validate_for(workspace)?;
    if let Some(positioning) = positioning {
        if !matches!(
            translation_scope,
            crate::config::GlobalTranslationScope::AllActiveEdges
        ) {
            return Err(SfmError::InvalidConfig(
                "pretracked global positioning requires AllActiveEdges translation scope"
                    .to_owned(),
            ));
        }
        if !positioning.direction_reweight.is_finite() || positioning.direction_reweight < 0.0 {
            return Err(SfmError::InvalidConfig(
                "global positioning direction reweight must be finite and non-negative".to_owned(),
            ));
        }
    }
    Ok(rotations)
}

fn raw_positioning_tracks(
    workspace: &SfmWorkspace,
    rotations: &BTreeMap<crate::ViewId, Rotation3<f64>>,
    config: &PretrackedGlobalPositioningConfig,
) -> (RawPositioningTracks, usize) {
    let minimum_views = config.min_track_views.max(2);
    let mut candidates = Vec::<(usize, PositioningTrack)>::new();
    for track in workspace.tracks().as_slice() {
        let mut bearings = Vec::with_capacity(track.observations().len());
        for observation in track.observations() {
            if !rotations.contains_key(&observation.view) {
                continue;
            }
            let Some(image) = workspace.view(observation.view) else {
                continue;
            };
            let Some(camera) = workspace.camera_model(image.intrinsic_id) else {
                continue;
            };
            let Ok(pixel) = workspace.track_observation_pixel(observation) else {
                continue;
            };
            let Ok(bearing) = camera.unproject(&pixel) else {
                continue;
            };
            let norm = bearing.norm();
            if norm.is_finite() && norm > 1e-12 {
                bearings.push((observation.view, bearing / norm));
            }
        }
        if bearings.len() >= minimum_views {
            candidates.push((track.id().index(), bearings));
        }
    }

    let total_observations = candidates
        .iter()
        .map(|(_, observations)| observations.len())
        .sum::<usize>();
    if config.max_observations == 0 || total_observations <= config.max_observations {
        return (candidates.into_iter().collect(), total_observations);
    }

    let mut best_by_camera = BTreeMap::<ViewId, (usize, usize)>::new();
    for (track_id, observations) in &candidates {
        for &(camera, _) in observations {
            let candidate = (observations.len(), *track_id);
            best_by_camera
                .entry(camera)
                .and_modify(|current| {
                    if candidate.0 > current.0
                        || (candidate.0 == current.0 && candidate.1 < current.1)
                    {
                        *current = candidate;
                    }
                })
                .or_insert(candidate);
        }
    }
    let by_id = candidates.into_iter().collect::<BTreeMap<_, _>>();
    let mut selected = best_by_camera
        .values()
        .map(|(_, track_id)| *track_id)
        .collect::<BTreeSet<_>>();
    let mut used = selected
        .iter()
        .filter_map(|track_id| by_id.get(track_id))
        .map(Vec::len)
        .sum::<usize>();
    let mut remaining = by_id.keys().copied().collect::<Vec<_>>();
    remaining.sort_by_key(|track_id| {
        // Keep the pseudo-random ordering identical on 32- and 64-bit targets.
        (*track_id as u64)
            .wrapping_mul(0x9e37_79b9_7f4a_7c15)
            .rotate_left(17)
    });
    for track_id in remaining {
        if selected.contains(&track_id) {
            continue;
        }
        let observation_count = by_id[&track_id].len();
        if used.saturating_add(observation_count) > config.max_observations {
            continue;
        }
        selected.insert(track_id);
        used += observation_count;
    }
    let tracks = selected
        .into_iter()
        .filter_map(|track_id| {
            by_id
                .get(&track_id)
                .cloned()
                .map(|observations| (track_id, observations))
        })
        .collect();
    (tracks, used)
}

fn active_world_directions(
    workspace: &SfmWorkspace,
    rotations: &BTreeMap<crate::ViewId, Rotation3<f64>>,
) -> Vec<(ViewId, ViewId, Vector3<f64>)> {
    workspace
        .pair_graph()
        .active_pair_ids()
        .filter_map(|pair_id| {
            let pair = workspace.pair_graph().pair(pair_id)?;
            if pair.first() == pair.second() || !rotations.contains_key(&pair.second()) {
                return None;
            }
            let rotation = rotations.get(&pair.first())?;
            let relative_pose = workspace
                .geometric_model_store()
                .get(pair_id.index())?
                .relative_pose()?;
            let direction =
                rotation.inverse() * relative_pose.baseline_direction_in_source().as_ref();
            let norm = direction.norm();
            (norm.is_finite() && norm > 1e-12).then_some((
                pair.first(),
                pair.second(),
                direction / norm,
            ))
        })
        .collect()
}

fn apply_pretracked_global_positioning_from_rotations(
    workspace: &mut SfmWorkspace,
    config: &PretrackedGlobalPositioningConfig,
    rotation_estimates: &crate::backends::RotationEstimates,
    averaging: &mut GlobalAveragingReport,
) -> Result<()> {
    let rotations = rotation_estimates
        .rotations()
        .iter()
        .copied()
        .collect::<BTreeMap<_, _>>();
    let rotations_by_view = rotations
        .iter()
        .map(|(image, rotation)| (*image, *rotation))
        .collect::<BTreeMap<_, _>>();
    let (mut tracks, observation_count) = raw_positioning_tracks(workspace, &rotations, config);
    if tracks.is_empty() {
        return Err(SfmError::MissingPrerequisite {
            prerequisite: "multi-view raw bearing tracks",
            method: "build_tracks",
        });
    }
    let mut directions = active_world_directions(workspace, &rotations);
    // A no-op on this evidence (both builders already apply the same rules); it keeps the
    // solver's strict input contract satisfied without changing what it admits.
    crate::ops::averaging::retain_global_positioning_input(
        &rotations_by_view,
        &mut tracks,
        &mut directions,
    );
    let positioning_config = global_positioning_solver_config(workspace.root_seed(), config);
    let result = position_globally(
        &rotations_by_view,
        &tracks,
        &directions,
        &positioning_config,
    )
    .map_err(|error| {
        SfmError::InvalidInput(format!("raw-track global positioning failed: {error}"))
    })?;
    let missing_count = rotations
        .keys()
        .filter(|image| !result.translations.contains_key(*image))
        .count();
    let allowed_missing = allowed_pretracked_positioning_omissions(rotations.len());
    if missing_count > allowed_missing {
        return Err(SfmError::InvalidInput(format!(
            "raw-track global positioning covered {} of {} rotated views; {} omissions exceed the allowance of {}",
            result.translations.len(),
            rotations.len(),
            missing_count,
            allowed_missing,
        )));
    }

    let centers = result
        .translations
        .iter()
        .map(|(view, center)| {
            let image = *view;
            rotations
                .contains_key(&image)
                .then_some((image, *center))
                .ok_or_else(|| {
                    SfmError::InvalidInput(format!(
                        "raw-track global positioning returned unexpected view {image}"
                    ))
                })
        })
        .collect::<Result<Vec<_>>>()?;
    let camera_centers =
        crate::backends::CameraCenters::try_new(centers, result.iteration_count, result.residual)?;
    crate::ops::averaging::populate_pose_records(workspace, rotation_estimates, &camera_centers)?;
    workspace.set_global_averaging(rotation_estimates.clone(), camera_centers);
    averaging.method = crate::ops::averaging::GlobalAveragingMethod::GlobalPositioning;
    averaging.center_count = result.translations.len();
    averaging.translation_iterations = result.iteration_count;
    averaging.translation_residual = result.residual.value();
    averaging.positioning_track_count = Some(tracks.len());
    averaging.positioning_observation_count = Some(observation_count);
    Ok(())
}

fn global_positioning_solver_config(
    root_seed: crate::RunSeed,
    config: &PretrackedGlobalPositioningConfig,
) -> GlobalPositioningConfig {
    GlobalPositioningConfig::default()
        .with_min_track_views(config.min_track_views.max(2))
        .with_direction_reweight(config.direction_reweight)
        .with_seed(root_seed.derive(GLOBAL_MIXED_POSITIONING_SEED_DOMAIN, &[]))
}

fn allowed_pretracked_positioning_omissions(rotated_views: usize) -> usize {
    super::bounded_deferred_view_count(
        rotated_views,
        MAX_DEFERRED_POSITIONING_VIEWS,
        MAX_DEFERRED_POSITIONING_FRACTION,
    )
}

#[allow(clippy::too_many_arguments)]
fn finish_global_reconstruction(
    workspace: &mut SfmWorkspace,
    config: &GlobalAveragingConfig,
    triangulator: &dyn Triangulator,
    input_featured_view_count: usize,
    input_match_pair_count: usize,
    input_verified_pair_count: usize,
    averaging: GlobalAveragingReport,
    progress: &dyn SfmProgressSink,
) -> Result<GlobalTranslationReport> {
    let input_track_count = workspace.tracks().len();
    cancellation_point(progress)?;
    progress.begin(SfmProgressStage::Triangulation, input_track_count);
    let tri_report = crate::ops::reconstruction::triangulate_global_translation(
        workspace,
        triangulator,
        config.gauge_scale_policy(),
    )?;
    progress.advance(SfmProgressStage::Triangulation, input_track_count);
    progress.finish(SfmProgressStage::Triangulation);

    let probes = graph_probes(workspace)?;
    Ok(GlobalTranslationReport {
        config: *config,
        probes,
        averaging,
        input_featured_view_count,
        input_match_pair_count,
        input_verified_pair_count,
        input_track_count,
        triangulation: tri_report,
        final_active_landmarks: workspace.landmarks().active_count(),
    })
}

#[cfg(test)]
mod tests {
    use super::{
        GLOBAL_MIXED_POSITIONING_SEED_DOMAIN, PretrackedGlobalPositioningConfig,
        allowed_pretracked_positioning_omissions, global_positioning_solver_config,
    };
    use crate::RunSeed;

    #[test]
    fn global_positioning_omission_cap_is_small_and_bounded() {
        assert_eq!(allowed_pretracked_positioning_omissions(74), 1);
        assert_eq!(allowed_pretracked_positioning_omissions(308), 1);
        assert_eq!(allowed_pretracked_positioning_omissions(1_000), 4);
        assert_eq!(allowed_pretracked_positioning_omissions(10_000), 4);
    }

    #[test]
    fn global_positioning_uses_the_workspace_seed_domain() {
        let root = RunSeed::from_u64(42);
        let config =
            global_positioning_solver_config(root, &PretrackedGlobalPositioningConfig::default());

        assert_eq!(
            config.seed(),
            root.derive(GLOBAL_MIXED_POSITIONING_SEED_DOMAIN, &[])
        );
        assert_ne!(
            config.seed(),
            global_positioning_solver_config(
                RunSeed::from_u64(43),
                &PretrackedGlobalPositioningConfig::default(),
            )
            .seed()
        );
    }
}
