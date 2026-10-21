// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::api::{
    FixedRotationError, FixedRotationGauge, FixedRotationIntrinsics, FixedRotationObservation,
    FixedRotationProblem,
};
use super::api::{
    FixedRotationObservabilityProbeOptions, FixedRotationProbeReason, FixedRotationProbeStatus,
};
use super::observability::probe_fixed_rotation_observability;
use crate::numerics::refine::Loss;

struct SyntheticProblem {
    rotations: Vec<[[f64; 3]; 3]>,
    centers: Vec<[f64; 3]>,
    points: Vec<[f64; 3]>,
    intrinsics: Vec<FixedRotationIntrinsics>,
    observations: Vec<Vec<FixedRotationObservation>>,
    fixed_cameras: Vec<bool>,
}

impl SyntheticProblem {
    fn problem(&self) -> FixedRotationProblem<'_> {
        FixedRotationProblem {
            rotations: &self.rotations,
            centers: &self.centers,
            points: &self.points,
            intrinsics: &self.intrinsics,
            observations: &self.observations,
            fixed_cameras: &self.fixed_cameras,
            gauge: FixedRotationGauge::Free {
                anchor: 0,
                scale_camera: 1,
            },
        }
    }
}

fn synthetic_problem() -> SyntheticProblem {
    let centers = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ];
    let points = (0..24)
        .map(|index| {
            let column = (index % 6) as f64;
            let row = (index / 6) as f64;
            [
                -0.5 + 0.35 * column,
                -0.4 + 0.3 * row,
                4.0 + 0.15 * ((index * 7) % 5) as f64,
            ]
        })
        .collect::<Vec<_>>();
    let intrinsics = vec![
        FixedRotationIntrinsics {
            focal_x: 800.0,
            focal_y: 800.0,
            principal_x: 0.0,
            principal_y: 0.0,
            k1: 0.0,
            k2: 0.0,
            k3: 0.0,
            p1: 0.0,
            p2: 0.0,
        };
        centers.len()
    ];
    let observations = points
        .iter()
        .map(|point| {
            centers
                .iter()
                .enumerate()
                .map(|(camera, center)| {
                    let depth = point[2] - center[2];
                    FixedRotationObservation {
                        camera,
                        xy: [
                            800.0 * (point[0] - center[0]) / depth,
                            800.0 * (point[1] - center[1]) / depth,
                        ],
                    }
                })
                .collect()
        })
        .collect();
    SyntheticProblem {
        rotations: vec![[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]; centers.len()],
        fixed_cameras: vec![false; centers.len()],
        centers,
        points,
        intrinsics,
        observations,
    }
}

#[test]
fn capped_sparse_probe_certifies_a_well_constrained_subset() {
    let problem = synthetic_problem();
    let report = probe_fixed_rotation_observability(
        problem.problem(),
        Loss::L2,
        FixedRotationObservabilityProbeOptions {
            max_tracks: 8,
            max_observations: 32,
            ..FixedRotationObservabilityProbeOptions::default()
        },
    )
    .expect("synthetic problem is valid");

    assert_eq!(report.status, FixedRotationProbeStatus::FullRankCertified);
    assert_eq!(report.pose_dimension, 8);
    assert_eq!(report.selected_tracks, 8);
    assert_eq!(report.covered_free_cameras, 3);
    assert!(report.reduced_nonzeros > 0);
    assert!(report.factor_nonzeros > 0);
}

#[test]
fn an_incomplete_cap_is_not_mislabeled_rank_deficient() {
    let problem = synthetic_problem();
    let report = probe_fixed_rotation_observability(
        problem.problem(),
        Loss::L2,
        FixedRotationObservabilityProbeOptions {
            max_observations: 2,
            ..FixedRotationObservabilityProbeOptions::default()
        },
    )
    .expect("synthetic problem is valid");

    assert_eq!(
        report.status,
        FixedRotationProbeStatus::NotCertified(
            FixedRotationProbeReason::InsufficientCameraCoverage
        )
    );
    assert_eq!(report.selected_tracks, 0);
}

#[test]
fn bounded_long_track_exception_covers_free_cameras() {
    let problem = synthetic_problem();
    let report = probe_fixed_rotation_observability(
        problem.problem(),
        Loss::L2,
        FixedRotationObservabilityProbeOptions {
            max_observations_per_track: 3,
            max_coverage_observations_per_track: 4,
            max_tracks: 8,
            max_observations: 32,
            ..FixedRotationObservabilityProbeOptions::default()
        },
    )
    .expect("bounded long tracks are a valid coverage source");

    assert_eq!(report.covered_free_cameras, 3);
    assert!(report.selected_tracks > 0);
    assert!(!matches!(
        report.status,
        FixedRotationProbeStatus::NotCertified(
            FixedRotationProbeReason::InsufficientCameraCoverage
        )
    ));
}

#[test]
fn an_unprojectable_point_is_weak_instead_of_aborting_the_probe() {
    let mut problem = synthetic_problem();
    // A converged solve can leave an outlier point on a camera center.
    problem.points[0] = problem.centers[0];
    let report = probe_fixed_rotation_observability(
        problem.problem(),
        Loss::L2,
        FixedRotationObservabilityProbeOptions::default(),
    )
    .expect("one unprojectable point must not invalidate the probe");

    assert_eq!(report.weak_point_count, 1);
    assert_eq!(report.covered_free_cameras, 3);
}

#[test]
fn a_numerically_singular_point_is_weak_instead_of_corrupting_the_rank() {
    let mut problem = synthetic_problem();
    // Forward motion: a point near the ray through both centers has almost no depth
    // information, yet its block is still invertible in floating point.
    problem.centers[3] = [0.0, 0.0, 1.0];
    let observe = |camera: usize, point: [f64; 3], center: [f64; 3]| {
        let depth = point[2] - center[2];
        FixedRotationObservation {
            camera,
            xy: [
                800.0 * (point[0] - center[0]) / depth,
                800.0 * (point[1] - center[1]) / depth,
            ],
        }
    };
    for (track, point) in problem.observations.iter_mut().zip(&problem.points) {
        track[3] = observe(3, *point, problem.centers[3]);
    }
    let point = [1e-7, 0.0, 5.0];
    problem.points.push(point);
    problem.observations.push(vec![
        observe(0, point, problem.centers[0]),
        observe(3, point, problem.centers[3]),
    ]);

    let probe = probe_fixed_rotation_observability(
        problem.problem(),
        Loss::L2,
        FixedRotationObservabilityProbeOptions::default(),
    )
    .expect("synthetic problem is valid");

    assert_eq!(probe.weak_point_count, 1);
    assert_eq!(probe.status, FixedRotationProbeStatus::FullRankCertified);
}

#[test]
fn the_probe_certifies_through_a_point_collapsed_onto_a_camera_centre() {
    let mut problem = synthetic_problem();
    problem.centers[3] = [0.5, 0.5, -0.1];
    for (track, point) in problem.observations.iter_mut().zip(&problem.points) {
        let depth = point[2] + 0.1;
        track[3].xy = [
            800.0 * (point[0] - 0.5) / depth,
            800.0 * (point[1] - 0.5) / depth,
        ];
    }
    // The track of the refinement test above: its point block passes the weak test but is
    // ~10^20 stiffer than the rest of camera 0's evidence. Eliminating it with an explicit
    // inverse leaves a negative camera diagonal and a false weak-geometry verdict.
    problem.points.push([0.0; 3]);
    problem.observations.push(vec![
        FixedRotationObservation {
            camera: 0,
            xy: [40.0, -30.0],
        },
        FixedRotationObservation {
            camera: 3,
            xy: [-60.0, 20.0],
        },
    ]);
    for step in 0..20 {
        *problem.points.last_mut().expect("collapsed point") =
            [1e-7, 1e-7, 1e-7 * (1.0 + 0.01 * f64::from(step))];

        let probe = probe_fixed_rotation_observability(
            problem.problem(),
            Loss::L2,
            FixedRotationObservabilityProbeOptions::default(),
        )
        .expect("synthetic problem is valid");

        assert_eq!(
            probe.status,
            FixedRotationProbeStatus::FullRankCertified,
            "step {step}"
        );
    }
}

#[test]
fn a_distant_camera_is_not_mislabeled_rank_deficient() {
    let mut problem = synthetic_problem();
    // Camera 3 sees every point from far behind: its weakest direction is ~1e-15 of the
    // stiffest camera's, below an unequilibrated rank tolerance, but its center is determined.
    let distance = 1.0e4;
    problem.centers[3] = [0.5, 0.5, -distance];
    for (track, point) in problem.observations.iter_mut().zip(&problem.points) {
        let depth = point[2] + distance;
        track[3].xy = [
            800.0 * (point[0] - 0.5) / depth,
            800.0 * (point[1] - 0.5) / depth,
        ];
    }

    let probe = probe_fixed_rotation_observability(
        problem.problem(),
        Loss::L2,
        FixedRotationObservabilityProbeOptions::default(),
    )
    .expect("synthetic problem is valid");

    assert_eq!(probe.weak_point_count, 0);
    assert_eq!(probe.status, FixedRotationProbeStatus::FullRankCertified);
}

#[test]
fn a_camera_linked_by_two_view_tracks_to_one_neighbour_is_not_certified() {
    let mut problem = synthetic_problem();
    for track in &mut problem.observations {
        track.retain(|observation| observation.camera != 3);
    }
    // Tracks shared only with camera 2 fix the direction from camera 2, not the distance.
    for index in 0..6 {
        let point = [
            0.2 + 0.1 * f64::from(index),
            0.3 - 0.05 * f64::from(index),
            3.0 + 0.2 * f64::from(index % 3),
        ];
        problem.points.push(point);
        problem.observations.push(
            [2, 3]
                .into_iter()
                .map(|camera| {
                    let center = problem.centers[camera];
                    let depth = point[2] - center[2];
                    FixedRotationObservation {
                        camera,
                        xy: [
                            800.0 * (point[0] - center[0]) / depth,
                            800.0 * (point[1] - center[1]) / depth,
                        ],
                    }
                })
                .collect(),
        );
    }

    let probe = probe_fixed_rotation_observability(
        problem.problem(),
        Loss::L2,
        FixedRotationObservabilityProbeOptions::default(),
    )
    .expect("synthetic problem is valid");

    assert_eq!(probe.weak_point_count, 0);
    assert!(
        matches!(probe.status, FixedRotationProbeStatus::NotCertified(_)),
        "{:?}",
        probe.status
    );
}

#[test]
fn a_camera_without_any_track_fails_closed() {
    let mut problem = synthetic_problem();
    for track in &mut problem.observations {
        track.retain(|observation| observation.camera != 3);
    }
    let report = probe_fixed_rotation_observability(
        problem.problem(),
        Loss::L2,
        FixedRotationObservabilityProbeOptions::default(),
    )
    .expect("synthetic problem is valid");

    assert_eq!(
        report.status,
        FixedRotationProbeStatus::NotCertified(
            FixedRotationProbeReason::InsufficientCameraCoverage
        )
    );
}

#[test]
fn invalid_probe_limits_are_rejected() {
    let problem = synthetic_problem();
    let error = probe_fixed_rotation_observability(
        problem.problem(),
        Loss::L2,
        FixedRotationObservabilityProbeOptions {
            max_factor_nonzeros: 0,
            ..FixedRotationObservabilityProbeOptions::default()
        },
    )
    .expect_err("zero factor cap is invalid");

    assert_eq!(
        error,
        FixedRotationError::InvalidOptions("invalid fixed-rotation observability probe limits")
    );
}

#[test]
fn large_sparse_chain_is_bounded_and_fails_closed() {
    const CAMERA_COUNT: usize = 2_307;
    let centers = (0..CAMERA_COUNT)
        .map(|camera| [camera as f64, 0.0, 0.0])
        .collect::<Vec<_>>();
    let intrinsics = vec![
        FixedRotationIntrinsics {
            focal_x: 800.0,
            focal_y: 800.0,
            principal_x: 0.0,
            principal_y: 0.0,
            k1: 0.0,
            k2: 0.0,
            k3: 0.0,
            p1: 0.0,
            p2: 0.0,
        };
        CAMERA_COUNT
    ];
    let mut points = Vec::with_capacity(3 * (CAMERA_COUNT - 1));
    let mut observations = Vec::with_capacity(3 * (CAMERA_COUNT - 1));
    for camera in 0..CAMERA_COUNT - 1 {
        for sample in 0..3 {
            let point = [
                camera as f64 + 0.25 + 0.2 * sample as f64,
                -0.3 + 0.3 * sample as f64,
                5.0 + 0.2 * sample as f64,
            ];
            points.push(point);
            observations.push(
                [camera, camera + 1]
                    .into_iter()
                    .map(|view| FixedRotationObservation {
                        camera: view,
                        xy: [
                            800.0 * (point[0] - centers[view][0]) / point[2],
                            800.0 * point[1] / point[2],
                        ],
                    })
                    .collect(),
            );
        }
    }
    let rotations = vec![[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]; CAMERA_COUNT];
    let fixed_cameras = vec![false; CAMERA_COUNT];
    let report = probe_fixed_rotation_observability(
        FixedRotationProblem {
            rotations: &rotations,
            centers: &centers,
            points: &points,
            intrinsics: &intrinsics,
            observations: &observations,
            fixed_cameras: &fixed_cameras,
            gauge: FixedRotationGauge::Free {
                anchor: 0,
                scale_camera: 1,
            },
        },
        Loss::L2,
        FixedRotationObservabilityProbeOptions::default(),
    )
    .expect("large chain is a valid problem");

    assert_eq!(report.pose_dimension, 3 * CAMERA_COUNT - 4);
    assert!(matches!(
        report.status,
        FixedRotationProbeStatus::NotCertified(
            FixedRotationProbeReason::SparseFactorizationFailed
                | FixedRotationProbeReason::WeakPointGeometry
        )
    ));
    assert!(report.reduced_nonzeros < 100_000);
    assert!(report.factor_nonzeros < 100_000);
}

#[test]
fn a_chain_of_two_view_tracks_is_not_certified_at_any_length() {
    // Tracks seen by two adjacent cameras fix each baseline's direction, never its length, so
    // the chain keeps one free length per link beyond the gauge. The probe must not certify it
    // from the rounding noise of coordinates whose information the points absorbed.
    for camera_count in [3_usize, 4, 8, 40] {
        let centers = (0..camera_count)
            .map(|camera| [camera as f64, 0.0, 0.0])
            .collect::<Vec<_>>();
        let intrinsics = vec![
            FixedRotationIntrinsics {
                focal_x: 800.0,
                focal_y: 800.0,
                principal_x: 0.0,
                principal_y: 0.0,
                k1: 0.0,
                k2: 0.0,
                k3: 0.0,
                p1: 0.0,
                p2: 0.0,
            };
            camera_count
        ];
        let mut points = Vec::new();
        let mut observations = Vec::new();
        for camera in 0..camera_count - 1 {
            for sample in 0..3 {
                let point = [
                    camera as f64 + 0.25 + 0.2 * sample as f64,
                    -0.3 + 0.3 * sample as f64,
                    5.0 + 0.2 * sample as f64,
                ];
                points.push(point);
                observations.push(
                    [camera, camera + 1]
                        .into_iter()
                        .map(|view| FixedRotationObservation {
                            camera: view,
                            xy: [
                                800.0 * (point[0] - centers[view][0]) / point[2],
                                800.0 * point[1] / point[2],
                            ],
                        })
                        .collect(),
                );
            }
        }
        let rotations = vec![[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]; camera_count];
        let fixed_cameras = vec![false; camera_count];
        let problem = || FixedRotationProblem {
            rotations: &rotations,
            centers: &centers,
            points: &points,
            intrinsics: &intrinsics,
            observations: &observations,
            fixed_cameras: &fixed_cameras,
            gauge: FixedRotationGauge::Free {
                anchor: 0,
                scale_camera: 1,
            },
        };

        let probe = probe_fixed_rotation_observability(
            problem(),
            Loss::L2,
            FixedRotationObservabilityProbeOptions::default(),
        )
        .expect("chain is a valid problem");
        assert!(
            matches!(probe.status, FixedRotationProbeStatus::NotCertified(_)),
            "{camera_count} cameras: {:?}",
            probe.status
        );
    }
}
