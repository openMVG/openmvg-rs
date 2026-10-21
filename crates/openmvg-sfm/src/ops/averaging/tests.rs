// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::dispatch::solve_rotation_on_active_graph;
use super::dispatch::{
    largest_translation_connected_domain, populate_pose_records, solve_translation_on_active_graph,
    translation_geometric_models,
};
use super::rigid_core::largest_rigid_biconnected_translation_core;
use super::*;
use crate::RelativePose;
use crate::model::ImageMeta;
use crate::two_view::{InlierMask, ModelEvidence, TerminationReason};
use crate::workspace::quality::ErrorStats;
use crate::{Camera, IntrinsicId, ViewPair};
use nalgebra::{Matrix3, Rotation3, UnitVector3, Vector3};
use openmvg_core::model::{EstimationResult, RelativeRotation, RelativeTranslation};
use openmvg_geometry::averaging::{
    RotationConfig, RotationEstimate, TranslationConfig, TranslationEstimate,
    average_rotations_dense,
};
use std::collections::BTreeMap;

fn dense_id_asserting_solver(
    rotations: &[RelativeRotation],
    _config: &RotationConfig,
) -> EstimationResult<RotationEstimate> {
    assert_eq!(rotations[0].view_i(), ViewId::new(0));
    assert_eq!(rotations[0].view_j(), ViewId::new(1));
    Ok(RotationEstimate::new(
        [
            (ViewId::new(0), nalgebra::Rotation3::identity()),
            (ViewId::new(1), nalgebra::Rotation3::identity()),
        ]
        .into_iter()
        .collect(),
        1,
        openmvg_core::AngularResidual::ZERO,
    ))
}

fn wrong_dense_rotation_solver(
    _rotations: &[RelativeRotation],
    _config: &RotationConfig,
) -> EstimationResult<RotationEstimate> {
    Ok(RotationEstimate::new(
        [
            (ViewId::new(0), Rotation3::identity()),
            (ViewId::new(2), Rotation3::identity()),
        ]
        .into_iter()
        .collect(),
        1,
        openmvg_core::AngularResidual::ZERO,
    ))
}

fn wrong_dense_translation_solver(
    _translations: &[RelativeTranslation],
    _rotations: &BTreeMap<ViewId, Rotation3<f64>>,
    _config: &TranslationConfig,
) -> EstimationResult<TranslationEstimate> {
    Ok(TranslationEstimate::new(
        [
            (ViewId::new(0), Vector3::zeros()),
            (ViewId::new(2), Vector3::x()),
        ]
        .into_iter()
        .collect(),
        1,
        openmvg_core::AngularResidual::ZERO,
    ))
}

#[test]
fn averaging_boundaries_reject_wrong_dense_image_domains() {
    let edges = [PairEdge {
        pair: ViewPair::new(ViewId::new(2), ViewId::new(5)),
        view_i: ViewId::new(2),
        view_j: ViewId::new(5),
        confidence: 1.0,
    }];
    let models = [test_relative_pose_model(Vector3::x())];
    let rotations = RotationEstimates::try_new(
        vec![
            (ViewId::new(2), Rotation3::identity()),
            (ViewId::new(5), Rotation3::identity()),
        ],
        1,
        openmvg_core::AngularResidual::ZERO,
    )
    .unwrap();

    let rotation_error =
        solve_rotation_on_active_graph(wrong_dense_rotation_solver, &edges, &models).unwrap_err();
    let translation_error = solve_translation_on_active_graph(
        wrong_dense_translation_solver,
        &edges,
        &models,
        &rotations,
    )
    .unwrap_err();

    assert!(rotation_error.to_string().contains("unexpected view"));
    assert!(translation_error.to_string().contains("unexpected view"));
}

#[test]
fn numerical_rotation_boundary_compacts_and_restores_sparse_image_ids() {
    let edges = [PairEdge {
        pair: ViewPair::new(ViewId::new(2), ViewId::new(5)),
        view_i: ViewId::new(2),
        view_j: ViewId::new(5),
        confidence: 0.75,
    }];

    let models = [GeometricModel::RotationOnly {
        rotation: Rotation3::identity(),
        evidence: test_evidence("dense-id-test"),
    }];
    let estimates =
        solve_rotation_on_active_graph(dense_id_asserting_solver, &edges, &models).unwrap();

    assert_eq!(estimates.rotations()[0].0, ViewId::new(2));
    assert_eq!(estimates.rotations()[1].0, ViewId::new(5));
}

#[test]
fn rotation_only_evidence_reaches_rotation_but_not_translation() {
    let edges = vec![
        PairEdge {
            pair: ViewPair::new(ViewId::new(0), ViewId::new(1)),
            view_i: ViewId::new(0),
            view_j: ViewId::new(1),
            confidence: 1.0,
        },
        PairEdge {
            pair: ViewPair::new(ViewId::new(1), ViewId::new(2)),
            view_i: ViewId::new(1),
            view_j: ViewId::new(2),
            confidence: 1.0,
        },
    ];
    let models = vec![
        test_relative_pose_model(Vector3::x()),
        GeometricModel::RotationOnly {
            rotation: Rotation3::identity(),
            evidence: test_evidence("rotation-only-test"),
        },
    ];

    let rotations =
        solve_rotation_on_active_graph(average_rotations_dense, &edges, &models).unwrap();
    assert_eq!(rotations.rotations().len(), 3);

    let (translation_edges, translation_models) =
        translation_geometric_models(edges, models).unwrap();
    assert_eq!(translation_edges.len(), 1);
    assert_eq!(
        translation_edges[0].pair,
        ViewPair::new(ViewId::new(0), ViewId::new(1))
    );
    assert_eq!(translation_models.len(), 1);
    assert!(translation_models[0].relative_pose().is_some());
}

#[test]
fn translation_uses_largest_connected_metric_domain_with_stable_tie_breaking() {
    let edges = vec![
        PairEdge {
            pair: ViewPair::new(ViewId::new(0), ViewId::new(1)),
            view_i: ViewId::new(0),
            view_j: ViewId::new(1),
            confidence: 1.0,
        },
        PairEdge {
            pair: ViewPair::new(ViewId::new(2), ViewId::new(3)),
            view_i: ViewId::new(2),
            view_j: ViewId::new(3),
            confidence: 1.0,
        },
        PairEdge {
            pair: ViewPair::new(ViewId::new(3), ViewId::new(4)),
            view_i: ViewId::new(3),
            view_j: ViewId::new(4),
            confidence: 1.0,
        },
    ];
    let models = vec![
        test_relative_pose_model(Vector3::x()),
        test_relative_pose_model(Vector3::x()),
        test_relative_pose_model(Vector3::x()),
    ];

    let (retained_edges, retained_models) =
        largest_translation_connected_domain(edges, models).unwrap();

    assert_eq!(
        retained_edges
            .iter()
            .map(|edge| edge.pair)
            .collect::<Vec<_>>(),
        vec![
            ViewPair::new(ViewId::new(2), ViewId::new(3)),
            ViewPair::new(ViewId::new(3), ViewId::new(4)),
        ]
    );
    assert_eq!(retained_models.len(), 2);
}

#[test]
fn translation_component_ties_prefer_the_lowest_image_id() {
    let edges = vec![
        PairEdge {
            pair: ViewPair::new(ViewId::new(2), ViewId::new(3)),
            view_i: ViewId::new(2),
            view_j: ViewId::new(3),
            confidence: 1.0,
        },
        PairEdge {
            pair: ViewPair::new(ViewId::new(0), ViewId::new(1)),
            view_i: ViewId::new(0),
            view_j: ViewId::new(1),
            confidence: 1.0,
        },
    ];
    let models = vec![
        test_relative_pose_model(Vector3::x()),
        test_relative_pose_model(Vector3::x()),
    ];

    let (retained_edges, _) = largest_translation_connected_domain(edges, models).unwrap();

    assert_eq!(retained_edges.len(), 1);
    assert_eq!(
        retained_edges[0].pair,
        ViewPair::new(ViewId::new(0), ViewId::new(1))
    );
}

#[test]
fn publishing_global_poses_replaces_stale_pose_slots() {
    let mut workspace = SfmWorkspace::builder()
        .image_meta((0..3).map(|index| ImageMeta::external_id(format!("image-{index}"), 64, 48)))
        .camera_models([Camera::try_pinhole(100.0, 100.0, 32.0, 24.0).unwrap()])
        .shared_intrinsic(IntrinsicId::new(0))
        .build()
        .unwrap();
    workspace
        .set_pose(PoseRecord {
            id: crate::PoseId::new(2),
            pose: WorldToCameraPose::from_rotation_center(
                Rotation3::identity(),
                Point3::new(9.0, 0.0, 0.0),
            )
            .unwrap(),
            component: Some(ComponentId::new(0)),
        })
        .unwrap();
    let rotations = RotationEstimates::try_new(
        vec![
            (ViewId::new(0), Rotation3::identity()),
            (ViewId::new(1), Rotation3::identity()),
            (ViewId::new(2), Rotation3::identity()),
        ],
        1,
        openmvg_core::AngularResidual::ZERO,
    )
    .unwrap();
    let centers = CameraCenters::try_new(
        vec![
            (ViewId::new(0), Vector3::zeros()),
            (ViewId::new(1), Vector3::x()),
        ],
        1,
        openmvg_core::AngularResidual::ZERO,
    )
    .unwrap();

    populate_pose_records(&mut workspace, &rotations, &centers).unwrap();

    assert_eq!(workspace.pose_store().populated_count(), 2);
    assert!(workspace.pose_store().has(0));
    assert!(workspace.pose_store().has(1));
    assert!(!workspace.pose_store().has(2));
    assert_eq!(
        workspace.components()[0].views,
        vec![ViewId::new(0), ViewId::new(1)]
    );
}

#[test]
fn stable_core_search_considers_sibling_biconnected_blocks() {
    let centers = [
        Vector3::new(0.0, 0.0, 0.0),
        Vector3::new(1.0, 0.0, 0.0),
        Vector3::new(0.0, 1.0, 0.0),
        Vector3::new(0.0, 0.0, 1.0),
        Vector3::new(-1.0, 0.0, 0.0),
        Vector3::new(0.0, -1.0, 0.0),
        Vector3::new(0.0, 0.0, -1.0),
    ];
    let first_block = complete_block(&[0, 1, 2, 3]);
    let second_block = complete_block(&[0, 4, 5, 6]);
    let endpoints = first_block
        .iter()
        .chain(&second_block)
        .copied()
        .collect::<Vec<_>>();
    let edges = endpoints
        .iter()
        .map(|&(view_i, view_j)| PairEdge {
            pair: ViewPair::new(ViewId::new(view_i), ViewId::new(view_j)),
            view_i: ViewId::new(view_i),
            view_j: ViewId::new(view_j),
            confidence: if view_i < 4 && view_j < 4 && view_i != 0 {
                1e-6
            } else {
                1.0
            },
        })
        .collect::<Vec<_>>();
    let models = endpoints
        .iter()
        .map(|&(view_i, view_j)| test_relative_pose_model(centers[view_j] - centers[view_i]))
        .collect::<Vec<_>>();
    let rotations = RotationEstimates::try_new(
        (0..centers.len())
            .map(|image| (ViewId::new(image), Rotation3::identity()))
            .collect(),
        1,
        openmvg_core::AngularResidual::ZERO,
    )
    .unwrap();

    let (selected, _) =
        largest_rigid_biconnected_translation_core(&edges, &models, &rotations, 1e-4).unwrap();
    let selected_views = selected
        .iter()
        .flat_map(|edge| [edge.view_i, edge.view_j])
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();

    assert_eq!(
        selected_views,
        vec![
            ViewId::new(0),
            ViewId::new(4),
            ViewId::new(5),
            ViewId::new(6)
        ]
    );
}

fn complete_block(images: &[usize]) -> Vec<(usize, usize)> {
    images
        .iter()
        .enumerate()
        .flat_map(|(index, &view_i)| {
            images[(index + 1)..]
                .iter()
                .map(move |&view_j| (view_i, view_j))
        })
        .collect()
}

fn test_relative_pose_model(direction: Vector3<f64>) -> GeometricModel {
    GeometricModel::Essential {
        matrix: Matrix3::identity(),
        relative_pose: RelativePose::from_baseline_direction(
            Rotation3::identity(),
            UnitVector3::new_normalize(direction),
        )
        .unwrap(),
        evidence: test_evidence("stable-core-test"),
    }
}

fn test_evidence(solver: &str) -> ModelEvidence {
    ModelEvidence::try_new(
        InlierMask::all_inliers(1),
        ErrorStats::from_values(&[0.0], &[1.0]),
        openmvg_core::PixelThreshold::ONE_PIXEL.into(),
        solver,
        1,
        TerminationReason::Converged,
        Vec::new(),
    )
    .unwrap()
}
