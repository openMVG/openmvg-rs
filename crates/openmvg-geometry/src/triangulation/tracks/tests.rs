// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, HashSet};

use nalgebra::{Point3, Rotation3, Vector2};
use openmvg_core::AngularResidual;
use openmvg_core::camera::{Camera, CameraModel, PinholeCamera, SphericalCamera};
use openmvg_core::model::{ViewId, WorldToCameraPose};

use super::config::TriangulationPreset;
use super::observation::prepare_observations;
use super::triangulator::select_pair_hypotheses;
use super::*;

fn pinhole_scene(
    point: Point3<f64>,
    centers: &[Point3<f64>],
) -> (
    Vec<TriangulationObservation>,
    BTreeMap<ViewId, WorldToCameraPose>,
    BTreeMap<ViewId, Camera>,
) {
    let mut observations = Vec::new();
    let mut poses = BTreeMap::new();
    let mut cameras = BTreeMap::new();
    for (view_index, &center) in centers.iter().enumerate() {
        let view_id = ViewId::new(view_index);
        let pose = WorldToCameraPose::from_rotation_center(Rotation3::identity(), center).unwrap();
        let camera = Camera::Pinhole(
            PinholeCamera::try_with_focal(500.0, 320.0, 240.0).expect("valid camera"),
        );
        let camera_point = pose.project_to_camera(&point);
        let pixel = camera
            .project(&camera_point.coords)
            .expect("fixture point is visible");
        observations.push(TriangulationObservation { view_id, pixel });
        poses.insert(view_id, pose);
        cameras.insert(view_id, camera);
    }
    (observations, poses, cameras)
}

fn circular_centers(count: usize) -> Vec<Point3<f64>> {
    (0..count)
        .map(|index| {
            let angle = std::f64::consts::TAU * index as f64 / count as f64;
            Point3::new(1.5 * angle.cos(), 1.5 * angle.sin(), 0.0)
        })
        .collect()
}

#[test]
fn zero_minimum_parallax_remains_a_valid_disabled_cutoff() {
    let config = TrackTriangulationConfig {
        hypothesis_min_parallax: AngularResidual::ZERO,
        acceptance: AcceptanceConfig {
            min_parallax: AngularResidual::ZERO,
            ..AcceptanceConfig::default()
        },
        ..TrackTriangulationConfig::default()
    };

    assert!(TrackTriangulator::try_new(config).is_ok());
}

#[test]
fn conservative_pair_budget_is_exhaustive_then_bounded() {
    let budget = PairHypothesisBudget::CONSERVATIVE;
    let expected = [0, 0, 1, 3, 6, 10, 12, 14, 16, 16, 16];
    for (observations, hypotheses) in expected.into_iter().enumerate() {
        assert_eq!(budget.for_observation_count(observations), hypotheses);
    }
    assert_eq!(budget.for_observation_count(10_000), 16);
    assert_eq!(budget.for_observation_count(usize::MAX), 16);
}

#[test]
fn exhaustive_pair_budget_requests_every_unique_pair() {
    let budget = PairHypothesisBudget::EXHAUSTIVE;
    for observations in 0..100 {
        assert_eq!(
            budget.for_observation_count(observations),
            observations * observations.saturating_sub(1) / 2
        );
    }
    assert_eq!(budget.for_observation_count(usize::MAX), usize::MAX / 2);
}

#[test]
fn pair_budget_validation_rejects_zero_and_inverted_bounds() {
    for pair_budget in [
        PairHypothesisBudget {
            per_observation: 0,
            min: 1,
            max: 1,
        },
        PairHypothesisBudget {
            per_observation: 1,
            min: 0,
            max: 1,
        },
        PairHypothesisBudget {
            per_observation: 1,
            min: 3,
            max: 2,
        },
    ] {
        let mut config = TrackTriangulationConfig::default();
        let TriangulationPolicy::Consensus(ref mut consensus) = config.policy else {
            unreachable!("robust preset uses consensus")
        };
        consensus.pair_budget = pair_budget;
        assert!(matches!(
            TrackTriangulator::try_new(config),
            Err(TriangulationError::InvalidConfig(_))
        ));
    }
}

#[test]
fn two_view_track_skips_consensus_sampling() {
    let expected = Point3::new(0.2, -0.1, 6.0);
    let (observations, poses, cameras) =
        pinhole_scene(expected, &[Point3::origin(), Point3::new(1.0, 0.0, 0.0)]);
    let result = TrackTriangulator::robust()
        .triangulate(
            &observations,
            TriangulationContext {
                poses: &poses,
                cameras: &cameras,
            },
        )
        .expect("two-view track should be fitted directly");

    assert_eq!(result.quality.tested_hypothesis_count, 0);
    assert!((result.point - expected).norm() < 1e-9);
}

#[test]
fn long_track_pair_schedule_is_unique_bounded_and_balanced() {
    let expected = Point3::new(0.2, -0.1, 6.0);
    let centers = circular_centers(31);
    let (observations, poses, cameras) = pinhole_scene(expected, &centers);
    let context = TriangulationContext {
        poses: &poses,
        cameras: &cameras,
    };
    let prepared = prepare_observations(&observations, context).expect("valid fixture");
    let hypotheses = select_pair_hypotheses(&prepared, 0.0, 16);
    let pairs: HashSet<_> = hypotheses
        .iter()
        .map(|hypothesis| (hypothesis.first, hypothesis.second))
        .collect();
    let mut usage = vec![0usize; prepared.len()];
    for hypothesis in &hypotheses {
        usage[hypothesis.first] += 1;
        usage[hypothesis.second] += 1;
    }

    assert_eq!(hypotheses.len(), 16);
    assert_eq!(pairs.len(), hypotheses.len());
    assert!(usage.iter().all(|&count| count <= 2));

    let result = TrackTriangulator::robust()
        .triangulate(&observations, context)
        .expect("clean long track should triangulate");
    assert_eq!(result.quality.tested_hypothesis_count, 16);
}

#[test]
fn first_schedule_round_uses_disjoint_observations() {
    let expected = Point3::new(0.2, -0.1, 6.0);
    for count in 6..=64 {
        let centers = circular_centers(count);
        let (observations, poses, cameras) = pinhole_scene(expected, &centers);
        let prepared = prepare_observations(
            &observations,
            TriangulationContext {
                poses: &poses,
                cameras: &cameras,
            },
        )
        .expect("valid fixture");
        let budget = PairHypothesisBudget::CONSERVATIVE.for_observation_count(count);
        let hypotheses = select_pair_hypotheses(&prepared, 0.0, budget);
        let first_round = (count / 2).min(budget);
        let mut used = HashSet::with_capacity(first_round * 2);
        for hypothesis in hypotheses.iter().take(first_round) {
            assert!(used.insert(hypothesis.first), "count={count}");
            assert!(used.insert(hypothesis.second), "count={count}");
        }
    }
}

#[test]
fn all_observations_refits_clean_track() {
    let expected = Point3::new(0.2, -0.1, 6.0);
    let (observations, poses, cameras) = pinhole_scene(
        expected,
        &[
            Point3::origin(),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        ],
    );
    let triangulator = TrackTriangulator::from_preset(TriangulationPreset::AllObservations);
    let result = triangulator
        .triangulate(
            &observations,
            TriangulationContext {
                poses: &poses,
                cameras: &cameras,
            },
        )
        .expect("clean track should triangulate");

    assert!((result.point - expected).norm() < 1e-9);
    assert_eq!(result.inlier_indices, vec![0, 1, 2]);
}

#[test]
fn pair_consensus_removes_one_outlier_from_three_views() {
    let expected = Point3::new(0.2, -0.1, 6.0);
    let (mut observations, poses, cameras) = pinhole_scene(
        expected,
        &[
            Point3::origin(),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        ],
    );
    observations[2].pixel.coords += Vector2::new(80.0, -50.0);
    let triangulator = TrackTriangulator::from_preset(TriangulationPreset::Robust);
    let result = triangulator
        .triangulate(
            &observations,
            TriangulationContext {
                poses: &poses,
                cameras: &cameras,
            },
        )
        .expect("clean pair should recover from one outlier");

    assert!((result.point - expected).norm() < 1e-8);
    assert_eq!(result.inlier_indices, vec![0, 1]);
    let supporting_views: Vec<_> = result
        .supporting_observations(&observations)
        .expect("compatible source observations")
        .map(|observation| observation.view_id)
        .collect();
    assert_eq!(supporting_views, vec![ViewId::new(0), ViewId::new(1)]);
}

#[test]
fn hypothesis_size_and_required_support_are_independent() {
    let expected = Point3::new(0.2, -0.1, 6.0);
    let (mut observations, poses, cameras) = pinhole_scene(
        expected,
        &[
            Point3::origin(),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        ],
    );
    observations[2].pixel.coords += Vector2::new(80.0, -50.0);
    let mut config = TrackTriangulationConfig::from_preset(TriangulationPreset::Robust);
    let TriangulationPolicy::Consensus(ref mut consensus) = config.policy else {
        unreachable!("robust preset uses consensus")
    };
    consensus.min_inliers = 3;
    let triangulator = TrackTriangulator::try_new(config).expect("valid config");

    assert!(matches!(
        triangulator.triangulate(
            &observations,
            TriangulationContext {
                poses: &poses,
                cameras: &cameras,
            }
        ),
        Err(TriangulationError::InsufficientConsensus {
            inliers: 2,
            observations: 3
        })
    ));
}

#[test]
fn robust_result_is_invariant_to_observation_order() {
    let expected = Point3::new(0.2, -0.1, 6.0);
    let (mut observations, poses, cameras) = pinhole_scene(
        expected,
        &[
            Point3::origin(),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(-1.0, 0.0, 0.0),
        ],
    );
    observations[2].pixel.coords += Vector2::new(80.0, -50.0);
    let triangulator = TrackTriangulator::default();
    let context = TriangulationContext {
        poses: &poses,
        cameras: &cameras,
    };
    let forward = triangulator
        .triangulate(&observations, context)
        .expect("forward order should triangulate");
    let forward_views: HashSet<_> = forward
        .inlier_indices
        .iter()
        .map(|&index| observations[index].view_id)
        .collect();
    observations.reverse();
    let reverse = triangulator
        .triangulate(&observations, context)
        .expect("reverse order should triangulate");

    assert!((forward.point - reverse.point).norm() < 1e-12);
    let reverse_views: HashSet<_> = reverse
        .inlier_indices
        .iter()
        .map(|&index| observations[index].view_id)
        .collect();
    assert_eq!(forward_views, reverse_views);
}

#[test]
fn valid_negative_z_spherical_ray_is_not_rejected_as_behind_camera() {
    let point = Point3::new(1.0, 0.0, -5.0);
    let centers = [Point3::origin(), Point3::new(1.0, 0.0, 0.0)];
    let mut poses = BTreeMap::new();
    let mut cameras = BTreeMap::new();
    let mut observations = Vec::new();
    for (view_index, center) in centers.into_iter().enumerate() {
        let view_id = ViewId::new(view_index);
        let pose = WorldToCameraPose::from_rotation_center(Rotation3::identity(), center).unwrap();
        let camera = Camera::Spherical(SphericalCamera::try_new(1024).expect("valid camera"));
        let camera_point = pose.project_to_camera(&point);
        let pixel = camera
            .project(&camera_point.coords)
            .expect("spherical camera sees the ray");
        poses.insert(view_id, pose);
        cameras.insert(view_id, camera);
        observations.push(TriangulationObservation { view_id, pixel });
    }

    let result = TrackTriangulator::from_preset(TriangulationPreset::Robust)
        .triangulate(
            &observations,
            TriangulationContext {
                poses: &poses,
                cameras: &cameras,
            },
        )
        .expect("oriented ray depth, not camera Z, defines visibility");
    assert!((result.point - point).norm() < 1e-8);
}
