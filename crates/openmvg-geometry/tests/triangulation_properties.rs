// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, HashSet};

use nalgebra::{Point2, Point3, Rotation3, Vector2};
use openmvg_core::camera::{Camera, CameraModel, PinholeCamera};
use openmvg_core::model::{ViewId, WorldToCameraPose};
use openmvg_geometry::triangulation::{
    PairHypothesisBudget, TrackTriangulationRejection, TrackTriangulator, TriangulationContext,
    TriangulationError, TriangulationObservation,
};
use proptest::prelude::*;

fn circular_scene(
    count: usize,
    point: Point3<f64>,
    phase: f64,
) -> (
    Vec<TriangulationObservation>,
    BTreeMap<ViewId, WorldToCameraPose>,
    BTreeMap<ViewId, Camera>,
) {
    let mut observations = Vec::with_capacity(count);
    let mut poses = BTreeMap::new();
    let mut cameras = BTreeMap::new();
    for view_index in 0..count {
        let view_id = ViewId::new(view_index);
        let angle = std::f64::consts::TAU * view_index as f64 / count as f64 + phase;
        let center = Point3::new(1.5 * angle.cos(), 1.5 * angle.sin(), 0.0);
        let pose = WorldToCameraPose::from_rotation_center(Rotation3::identity(), center).unwrap();
        let camera = Camera::Pinhole(
            PinholeCamera::try_with_focal(700.0, 640.0, 480.0).expect("valid camera"),
        );
        let pixel = camera
            .project(&pose.project_to_camera(&point).coords)
            .expect("generated point is visible");
        observations.push(TriangulationObservation::new(view_id, pixel));
        poses.insert(view_id, pose);
        cameras.insert(view_id, camera);
    }
    (observations, poses, cameras)
}

fn supporting_views(
    observations: &[TriangulationObservation],
    inlier_indices: &[usize],
) -> HashSet<ViewId> {
    inlier_indices
        .iter()
        .map(|&index| observations[index].view_id)
        .collect()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn clean_tracks_are_accurate_bounded_and_permutation_invariant(
        count in 2usize..25,
        x in -0.8f64..0.8,
        y in -0.8f64..0.8,
        z in 4.0f64..20.0,
        phase in 0.0f64..std::f64::consts::TAU,
        rotation in 0usize..64,
    ) {
        let expected = Point3::new(x, y, z);
        let (mut observations, poses, cameras) = circular_scene(count, expected, phase);
        let context = TriangulationContext::new(&poses, &cameras);
        let triangulator = TrackTriangulator::robust();
        let forward = triangulator
            .triangulate(&observations, context)
            .expect("well-conditioned clean track should triangulate");
        let forward_views = supporting_views(&observations, &forward.inlier_indices);

        observations.rotate_left(rotation % count);
        let permuted = triangulator
            .triangulate(&observations, context)
            .expect("permuted clean track should triangulate");
        let permuted_views = supporting_views(&observations, &permuted.inlier_indices);
        let expected_hypotheses = if count == 2 {
            0
        } else {
            PairHypothesisBudget::CONSERVATIVE.for_observation_count(count)
        };

        prop_assert!((forward.point - expected).norm() < 1e-8);
        prop_assert!((permuted.point - expected).norm() < 1e-8);
        prop_assert!((forward.point - permuted.point).norm() < 1e-12);
        prop_assert_eq!(forward_views, permuted_views);
        prop_assert_eq!(forward.quality.inlier_observation_count, count);
        prop_assert_eq!(forward.quality.tested_hypothesis_count, expected_hypotheses);
        prop_assert!(forward.quality.tested_hypothesis_count <= 16);
    }

    /// Robust track triangulation: one contaminated observation is rejected across supported track
    /// lengths; the recovered point and supporting-view set remain correct; the deterministic
    /// hypothesis budget remains bounded.
    #[test]
    fn one_contaminated_observation_is_removed_across_track_lengths(
        count in 3usize..25,
        x in -0.5f64..0.5,
        y in -0.5f64..0.5,
        z in 4.0f64..15.0,
        phase in 0.0f64..std::f64::consts::TAU,
        outlier_seed in 0usize..64,
        rotation in 0usize..64,
    ) {
        let expected = Point3::new(x, y, z);
        let (mut observations, poses, cameras) = circular_scene(count, expected, phase);
        let outlier_view = outlier_seed % count;
        observations[outlier_view].pixel.coords += Vector2::new(120.0, -90.0);
        observations.rotate_left(rotation % count);
        let context = TriangulationContext::new(&poses, &cameras);
        let result = TrackTriangulator::robust()
            .triangulate(&observations, context)
            .expect("one contaminated observation should not destroy a track");
        let inlier_views = supporting_views(&observations, &result.inlier_indices);

        prop_assert!((result.point - expected).norm() < 1e-8);
        prop_assert_eq!(result.quality.inlier_observation_count, count - 1);
        prop_assert!(!inlier_views.contains(&ViewId::new(outlier_view)));
        prop_assert!(result.quality.tested_hypothesis_count <= 16);
    }

    #[test]
    fn two_contaminated_observations_are_removed_from_longer_tracks(
        count in 6usize..25,
        x in -0.5f64..0.5,
        y in -0.5f64..0.5,
        z in 4.0f64..15.0,
        phase in 0.0f64..std::f64::consts::TAU,
        first_seed in 0usize..64,
        second_offset in 1usize..64,
        rotation in 0usize..64,
    ) {
        let expected = Point3::new(x, y, z);
        let (mut observations, poses, cameras) = circular_scene(count, expected, phase);
        let first_outlier = first_seed % count;
        let second_outlier = (first_outlier + second_offset % (count - 1) + 1) % count;
        observations[first_outlier].pixel.coords += Vector2::new(120.0, -90.0);
        observations[second_outlier].pixel.coords += Vector2::new(-100.0, 130.0);
        observations.rotate_left(rotation % count);
        let context = TriangulationContext::new(&poses, &cameras);
        let result = TrackTriangulator::robust()
            .triangulate(&observations, context)
            .expect("two contaminated observations should not destroy a longer track");
        let inlier_views = supporting_views(&observations, &result.inlier_indices);

        prop_assert!((result.point - expected).norm() < 1e-8);
        prop_assert_eq!(result.quality.inlier_observation_count, count - 2);
        prop_assert!(!inlier_views.contains(&ViewId::new(first_outlier)));
        prop_assert!(!inlier_views.contains(&ViewId::new(second_outlier)));
        prop_assert!(result.quality.tested_hypothesis_count <= 16);
    }

    #[test]
    fn low_parallax_two_view_tracks_are_rejected(
        x in -0.5f64..0.5,
        y in -0.5f64..0.5,
        z in 5.0f64..20.0,
        baseline in 1e-5f64..0.02,
    ) {
        let point = Point3::new(x, y, z);
        let camera =
            Camera::Pinhole(PinholeCamera::try_with_focal(700.0, 640.0, 480.0).expect("valid camera"));
        let centers = [Point3::origin(), Point3::new(baseline, 0.0, 0.0)];
        let mut observations = Vec::with_capacity(2);
        let mut poses = BTreeMap::new();
        let mut cameras = BTreeMap::new();
        for (view_index, center) in centers.into_iter().enumerate() {
            let view_id = ViewId::new(view_index);
            let pose = WorldToCameraPose::from_rotation_center(Rotation3::identity(), center).unwrap();
            let pixel: Point2<f64> = camera
                .project(&pose.project_to_camera(&point).coords)
                .expect("generated point is visible");
            observations.push(TriangulationObservation::new(view_id, pixel));
            poses.insert(view_id, pose);
            cameras.insert(view_id, camera.clone());
        }

        let result = TrackTriangulator::robust().triangulate(
            &observations,
            TriangulationContext::new(&poses, &cameras),
        );
        prop_assert!(matches!(
            result,
            Err(TriangulationError::Rejected(
                TrackTriangulationRejection::InsufficientParallax
            ))
        ));
    }
}
