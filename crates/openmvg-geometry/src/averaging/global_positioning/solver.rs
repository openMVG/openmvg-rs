// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Global positioning solver adapter (BATA-style bounded objective, independent implementation).
//!
//! Its teaching docs are on [`position_globally`], where docs.rs shows them.

use std::collections::{BTreeMap, BTreeSet};

use super::kernel::{
    BataOptions, GlobalPositioningError, GlobalPositioningPlan, GlobalPositioningProblem,
    GlobalPositioningReport, PositioningObservation,
};
use nalgebra::{Rotation3, Vector3};
use openmvg_core::model::{EstimationError, EstimationResult, ViewId};

use super::GlobalPositioningConfig;
use crate::averaging::TranslationEstimate;
use crate::averaging::view_id_mapping::ViewIdMapping;

/// Global positioning with the positive-scale, BATA-style bounded objective.
#[derive(Clone, Debug)]
struct GlobalPositioningSolver {
    config: GlobalPositioningConfig,
}

/// Position cameras globally from bearing tracks plus optional oriented world-frame
/// directions, with the rotations held fixed.
///
/// `absolute_rotations` holds the fixed world-to-camera rotations. `tracks`
/// maps a track id to its camera-frame bearings, one per view; tracks are
/// processed in increasing id order and each bearing is rotated into the world
/// frame and normalized. `relative_directions` holds oriented world-frame
/// `source -> target` baseline directions, normalized before use; they are
/// ignored by the objective when the configured direction reweight is zero,
/// but still validated.
///
/// Every input must be usable: callers filter their evidence first.
///
/// A track with fewer views than the configured minimum (at least two) is
/// dropped without error. Only cameras participating in an admitted
/// constraint are lowered, so a rotation without any admitted constraint is
/// omitted from the returned partial solution rather than making the
/// problem invalid.
///
/// The returned residual is an angular RMS diagnostic. Undefined baselines
/// contribute a right-angle penalty rather than a measured angle. Polish
/// minimizes its projected surrogate, whose descent can increase this RMS.
///
/// # How it works
///
/// The function checks the view IDs and tracks, turns each admitted
/// observation into a BATA factor, and solves for the camera positions.
///
/// ```text
/// ┌──────────────────────────────────────────────────────────────────────┐
/// │                      BATA global positioning                         │
/// └──────────────────────────────────────────────────────────────────────┘
/// Input: fixed rotations, bearing tracks, optional relative directions
///                                  │
///                                  ▼
/// ┌──────────────────────────────────────────────────────────────────────┐
/// │ Step 1: Validate IDs, tracks, weights, and gauge                     │
/// └──────────────────────────────────────────────────────────────────────┘
///                                  │
///                                  ▼
/// ┌──────────────────────────────────────────────────────────────────────┐
/// │ Step 2: Build BATA plan and eliminate point/scale blocks             │
/// └──────────────────────────────────────────────────────────────────────┘
///                                  │
///                                  ▼
/// ┌──────────────────────────────────────────────────────────────────────┐
/// │ Step 3: Solve reduced camera system and recover translations         │
/// └──────────────────────────────────────────────────────────────────────┘
///                                  │
///                                  ▼
/// Output: camera centers, iteration count, and angular RMS residual
/// ```
///
/// # Errors
///
/// [`EstimationError::InvalidInput`], naming the track or direction and the
/// view, when a track observation's view has no absolute rotation, a track
/// contains a view more than once, a world-frame bearing is non-finite or
/// zero-length, a direction's source or target has no absolute rotation, a
/// direction joins a view to itself, a direction is non-finite or
/// zero-length, or the same unordered pair carries two weighted directions;
/// [`EstimationError::InsufficientData`] when fewer than two rotations are
/// given or the admitted constraints touch fewer than two cameras; and the
/// configuration, degeneracy and numerical errors of the dense solve.
pub fn position_globally(
    absolute_rotations: &BTreeMap<ViewId, Rotation3<f64>>,
    tracks: &BTreeMap<usize, Vec<(ViewId, Vector3<f64>)>>,
    relative_directions: &[(ViewId, ViewId, Vector3<f64>)],
    config: &GlobalPositioningConfig,
) -> EstimationResult<TranslationEstimate> {
    GlobalPositioningSolver {
        config: config.clone(),
    }
    .solve(absolute_rotations, tracks, relative_directions)
}

impl GlobalPositioningSolver {
    fn solve(
        &self,
        absolute_rotations: &BTreeMap<ViewId, Rotation3<f64>>,
        tracks: &BTreeMap<usize, Vec<(ViewId, Vector3<f64>)>>,
        relative_directions: &[(ViewId, ViewId, Vector3<f64>)],
    ) -> EstimationResult<TranslationEstimate> {
        self.validate_config()?;
        if absolute_rotations.len() < 2 {
            return Err(EstimationError::InsufficientData {
                required: 2,
                provided: absolute_rotations.len(),
            });
        }

        let mut track_ids: Vec<_> = tracks.keys().copied().collect();
        track_ids.sort_unstable();
        let mut admitted_tracks = Vec::new();
        let mut active_views = BTreeSet::new();
        for track_id in track_ids {
            let track = &tracks[&track_id];
            let mut seen_views = Vec::with_capacity(track.len());
            let mut admitted = Vec::with_capacity(track.len());
            for &(view, ray) in track {
                let Some(rotation) = absolute_rotations.get(&view) else {
                    return Err(EstimationError::InvalidInput(format!(
                        "track {track_id} observes view {view}, which has no absolute rotation"
                    )));
                };
                if seen_views.contains(&view) {
                    return Err(EstimationError::InvalidInput(format!(
                        "track {track_id} contains view {view} more than once"
                    )));
                }
                seen_views.push(view);
                let world_bearing = rotation.inverse() * ray;
                let norm = world_bearing.norm();
                if !is_usable_norm(norm) {
                    return Err(EstimationError::InvalidInput(format!(
                        "track {track_id} has a non-finite or zero-length bearing in view {view}"
                    )));
                }
                admitted.push((view, world_bearing / norm));
            }
            if admitted.len() >= self.config.min_track_views.max(2) {
                active_views.extend(admitted.iter().map(|(view, _)| *view));
                admitted_tracks.push(admitted);
            }
        }

        let mut admitted_directions = Vec::new();
        let mut direction_pairs = BTreeSet::new();
        for (edge, &(source, target, direction)) in relative_directions.iter().enumerate() {
            if source == target {
                return Err(EstimationError::InvalidInput(format!(
                    "relative direction {edge} joins view {source} to itself"
                )));
            }
            if let Some(view) = [source, target]
                .into_iter()
                .find(|view| !absolute_rotations.contains_key(view))
            {
                return Err(EstimationError::InvalidInput(format!(
                    "relative direction {edge} ({source} -> {target}) references view {view}, \
                     which has no absolute rotation"
                )));
            }
            let norm = direction.norm();
            if !is_usable_norm(norm) {
                return Err(EstimationError::InvalidInput(format!(
                    "relative direction {edge} ({source} -> {target}) is non-finite or zero-length"
                )));
            }
            // Checked even when the directions carry no weight, so that the input contract does
            // not depend on `direction_reweight`.
            if !direction_pairs.insert((source.min(target), source.max(target))) {
                return Err(EstimationError::InvalidInput(format!(
                    "relative direction {edge} ({source} -> {target}) repeats a view pair"
                )));
            }
            if self.config.direction_reweight != 0.0 {
                active_views.insert(source);
                active_views.insert(target);
                admitted_directions.push((source, target, direction / norm));
            }
        }

        if admitted_tracks.is_empty() && admitted_directions.is_empty() {
            return Err(EstimationError::InsufficientData {
                required: 1,
                provided: 0,
            });
        }
        let mapping = ViewIdMapping::from_views(active_views);
        if mapping.view_count() < 2 {
            return Err(EstimationError::InsufficientData {
                required: 2,
                provided: mapping.view_count(),
            });
        }

        let point_count = admitted_tracks.len();
        let observation_count: usize = admitted_tracks.iter().map(Vec::len).sum();
        let mut observations = Vec::with_capacity(observation_count + admitted_directions.len());
        for (point, track) in admitted_tracks.into_iter().enumerate() {
            for (view, world_bearing) in track {
                let camera = mapping.index_of(view).ok_or_else(|| {
                    EstimationError::InvalidInput("active view mapping is inconsistent".into())
                })?;
                observations.push(PositioningObservation::bearing(
                    camera,
                    point,
                    world_bearing,
                    1.0,
                ));
            }
        }
        for (source, target, direction) in admitted_directions {
            let source = mapping.index_of(source).ok_or_else(|| {
                EstimationError::InvalidInput("active source mapping is inconsistent".into())
            })?;
            let target = mapping.index_of(target).ok_or_else(|| {
                EstimationError::InvalidInput("active target mapping is inconsistent".into())
            })?;
            observations.push(PositioningObservation::relative_direction(
                source, target, direction, 1.0,
            ));
        }

        let (camera_centers, report) = self.solve_dense(GlobalPositioningProblem {
            camera_count: mapping.view_count(),
            point_count,
            observations: &observations,
        })?;
        let translations = mapping
            .iter()
            .zip(camera_centers)
            .map(|((_, view), center)| (view, center))
            .collect();
        Ok(TranslationEstimate {
            translations,
            iteration_count: report.iteration_count,
            residual: crate::averaging::checked_angular_residual(report.rms_angular_error)?,
        })
    }

    /// Solve a checked dense-index problem; returns the camera centers and the
    /// solve report (the point buffer is solver scratch).
    fn solve_dense(
        &self,
        problem: GlobalPositioningProblem<'_>,
    ) -> EstimationResult<(Vec<Vector3<f64>>, GlobalPositioningReport)> {
        let camera_count = problem.camera_count;
        let point_count = problem.point_count;
        let plan = GlobalPositioningPlan::try_new(problem, self.options()).map_err(map_error)?;
        let mut workspace = plan.try_new_workspace().map_err(map_error)?;
        let mut camera_centers = vec![Vector3::zeros(); camera_count];
        let mut points = vec![Vector3::zeros(); point_count];
        let report = plan
            .solve_into(&mut camera_centers, &mut points, &mut workspace)
            .map_err(map_error)?;
        Ok((camera_centers, report))
    }

    fn options(&self) -> BataOptions {
        BataOptions {
            max_outer_iterations: self.config.max_outer_iterations,
            warmup_iterations: self.config.warmup_iterations,
            initial_huber_delta_radians: self.config.initial_huber_delta_rad,
            final_huber_delta_radians: self.config.final_huber_delta_rad,
            hard_cut_start_iteration: self.config.hard_cut_start_iteration,
            hard_cut_deg: self.config.hard_cut_deg,
            seed: self.config.seed,
            direction_reweight: self.config.direction_reweight,
            ..BataOptions::default()
        }
    }

    fn validate_config(&self) -> EstimationResult<()> {
        self.options().validate().map_err(map_error)?;
        Ok(())
    }
}

/// Whether a ray or direction norm is finite and long enough to normalize.
fn is_usable_norm(norm: f64) -> bool {
    norm.is_finite() && norm > 1e-12
}

fn map_error(error: GlobalPositioningError) -> EstimationError {
    use GlobalPositioningError as Error;
    match error {
        Error::TooFewCameras { provided } => EstimationError::InsufficientData {
            required: 2,
            provided,
        },
        Error::EmptyProblem => EstimationError::InsufficientData {
            required: 1,
            provided: 0,
        },
        Error::NotParallelRigid | Error::Disconnected | Error::DegenerateSolution => {
            EstimationError::DegenerateConfiguration {
                reason: error.to_string(),
            }
        }
        Error::DirectionInitializationFailed
        | Error::LinearSolveFailed
        | Error::NonFiniteSolution
        | Error::WorkspaceAllocationFailed => EstimationError::Numerical(error.to_string()),
        _ => EstimationError::InvalidInput(error.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type Tracks = BTreeMap<usize, Vec<(ViewId, Vector3<f64>)>>;

    /// Four identity-rotation cameras observing eight points with every view.
    fn problem() -> (BTreeMap<ViewId, Rotation3<f64>>, Tracks) {
        let centers = [
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(1.0, 1.0, 0.2),
        ];
        let rotations = (0..centers.len())
            .map(|index| (ViewId::new(index), Rotation3::identity()))
            .collect();
        let tracks = (0..8)
            .map(|point| {
                let offset = point as f64;
                let world = Vector3::new(
                    -1.0 + 0.4 * offset,
                    0.5 * (0.9 * offset).sin(),
                    5.0 + 0.3 * offset,
                );
                let observations = centers
                    .iter()
                    .enumerate()
                    .map(|(view, center)| (ViewId::new(view), world - center))
                    .collect();
                (point, observations)
            })
            .collect();
        (rotations, tracks)
    }

    fn invalid_input(result: EstimationResult<TranslationEstimate>) -> String {
        match result {
            Err(EstimationError::InvalidInput(message)) => message,
            other => panic!("expected InvalidInput, got {other:?}"),
        }
    }

    #[test]
    fn valid_input_solves_and_short_tracks_are_dropped() {
        let (rotations, mut tracks) = problem();
        tracks.insert(100, vec![(ViewId::new(0), Vector3::new(0.0, 0.0, 1.0))]);

        let result = position_globally(
            &rotations,
            &tracks,
            &[],
            &GlobalPositioningConfig::default(),
        )
        .expect("a well-posed problem with one short track solves");

        assert_eq!(result.translations.len(), 4);
    }

    #[test]
    fn observation_of_a_view_without_rotation_is_rejected() {
        let (rotations, mut tracks) = problem();
        tracks
            .get_mut(&3)
            .unwrap()
            .push((ViewId::new(9), Vector3::new(0.0, 0.0, 1.0)));

        let message = invalid_input(position_globally(
            &rotations,
            &tracks,
            &[],
            &GlobalPositioningConfig::default(),
        ));

        assert!(message.contains("track 3"), "{message}");
        assert!(message.contains("view 9"), "{message}");
        assert!(message.contains("no absolute rotation"), "{message}");
    }

    #[test]
    fn degenerate_bearing_is_rejected() {
        let (rotations, mut tracks) = problem();
        for ray in [Vector3::zeros(), Vector3::new(f64::NAN, 0.0, 1.0)] {
            tracks.get_mut(&2).unwrap()[1].1 = ray;

            let message = invalid_input(position_globally(
                &rotations,
                &tracks,
                &[],
                &GlobalPositioningConfig::default(),
            ));

            assert!(message.contains("track 2"), "{message}");
            assert!(message.contains("non-finite or zero-length"), "{message}");
        }
    }

    #[test]
    fn a_repeated_direction_pair_is_rejected_even_when_unweighted() {
        let (rotations, tracks) = problem();
        let directions = [
            (ViewId::new(0), ViewId::new(1), Vector3::new(1.0, 0.0, 0.0)),
            (ViewId::new(1), ViewId::new(0), Vector3::new(-1.0, 0.0, 0.0)),
        ];
        for config in [
            GlobalPositioningConfig::default(),
            GlobalPositioningConfig::default().with_direction_reweight(0.0),
        ] {
            let message =
                invalid_input(position_globally(&rotations, &tracks, &directions, &config));

            assert!(message.contains("relative direction 1"), "{message}");
            assert!(message.contains("repeats a view pair"), "{message}");
        }
    }

    #[test]
    fn direction_endpoint_without_rotation_is_rejected() {
        let (rotations, tracks) = problem();
        let unknown = ViewId::new(7);
        for (source, target) in [(ViewId::new(0), unknown), (unknown, ViewId::new(1))] {
            let directions = [
                (ViewId::new(0), ViewId::new(1), Vector3::new(1.0, 0.0, 0.0)),
                (source, target, Vector3::new(0.0, 1.0, 0.0)),
            ];

            let message = invalid_input(position_globally(
                &rotations,
                &tracks,
                &directions,
                &GlobalPositioningConfig::default(),
            ));

            assert!(message.contains("relative direction 1"), "{message}");
            assert!(message.contains("no absolute rotation"), "{message}");
            assert!(message.contains(&format!("view {unknown}")), "{message}");
        }
    }

    #[test]
    fn degenerate_or_self_directions_are_rejected_even_when_unweighted() {
        let (rotations, tracks) = problem();
        let unweighted = GlobalPositioningConfig::default().with_direction_reweight(0.0);
        for (config, direction, expected) in [
            (
                GlobalPositioningConfig::default(),
                (
                    ViewId::new(0),
                    ViewId::new(2),
                    Vector3::new(f64::INFINITY, 0.0, 0.0),
                ),
                "non-finite or zero-length",
            ),
            (
                unweighted.clone(),
                (ViewId::new(0), ViewId::new(2), Vector3::zeros()),
                "non-finite or zero-length",
            ),
            (
                unweighted,
                (ViewId::new(1), ViewId::new(1), Vector3::new(1.0, 0.0, 0.0)),
                "to itself",
            ),
        ] {
            let message = invalid_input(position_globally(
                &rotations,
                &tracks,
                &[direction],
                &config,
            ));

            assert!(message.contains("relative direction 0"), "{message}");
            assert!(message.contains(expected), "{message}");
        }
    }
}
