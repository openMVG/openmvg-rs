// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use std::f64::consts::PI;

#[test]
fn test_l1_irls_translation_simple() {
    let mut absolute_rotations = BTreeMap::new();
    absolute_rotations.insert(ViewId::new(0), Rotation3::identity());
    absolute_rotations.insert(ViewId::new(1), Rotation3::identity());

    let relative_translations = vec![
        RelativeTranslation::try_new(
            ViewId::new(0),
            ViewId::new(1),
            Vector3::new(1.0, 0.0, 0.0),
            1.0,
        )
        .unwrap(),
    ];

    let solver = L1IrlsTranslationSolver::new();
    let config = TranslationConfig::default();
    let result = solver
        .solve(&relative_translations, &absolute_rotations, &config)
        .unwrap();

    let c0 = result.translations.get(&ViewId::new(0)).unwrap();
    assert!(c0.norm() < 0.1, "Camera 0 should be near origin");

    let c1 = result.translations.get(&ViewId::new(1)).unwrap();
    let dir = (c1 - c0).normalize();
    assert!(
        dir.x > 0.9,
        "Camera 1 should be along +x from camera 0, got direction {:?}",
        dir
    );
}

/// L1-IRLS translation averaging: translation directions are interpreted in the documented endpoint
/// frames; the retained L1-IRLS policy recovers camera positions with non-identity rotations.
#[test]
fn test_l1_irls_with_rotations() {
    let mut absolute_rotations = BTreeMap::new();
    absolute_rotations.insert(ViewId::new(0), Rotation3::identity());
    absolute_rotations.insert(
        ViewId::new(1),
        Rotation3::from_euler_angles(0.0, PI / 4.0, 0.0),
    );
    absolute_rotations.insert(
        ViewId::new(2),
        Rotation3::from_euler_angles(0.0, -PI / 4.0, 0.0),
    );

    let positions_gt = vec![
        (0, Vector3::new(0.0, 0.0, 0.0)),
        (1, Vector3::new(2.0, 0.0, 0.0)),
        (2, Vector3::new(1.0, 0.0, 2.0)),
    ];

    let mut relative_translations = Vec::new();
    for &(i, ref c_i) in &positions_gt {
        for &(j, ref c_j) in &positions_gt {
            if i < j {
                let r_i = absolute_rotations.get(&ViewId::new(i)).unwrap();
                let dir_world = (c_j - c_i).normalize();
                // Module convention: d_ij = R_w2c · (t_j − t_i) — apply
                // r_i directly (w2c), not its inverse (c2w). With the
                // inverse applied here, `solve()`'s own
                // `r_i.transpose() * rel.direction()` transform (correctly
                // implementing the documented convention) ends up
                // applying r_i's inverse *twice*, corrupting every edge
                // whose source view has a non-identity rotation — exactly
                // edge (1, 2) below, which is why this test failed for
                // that pair specifically before this fix.
                let dir_cam = *r_i * dir_world;
                relative_translations.push(
                    RelativeTranslation::try_new(ViewId::new(i), ViewId::new(j), dir_cam, 1.0)
                        .unwrap(),
                );
            }
        }
    }

    let solver = L1IrlsTranslationSolver::new();
    let config = TranslationConfig::default();
    let result = solver
        .solve(&relative_translations, &absolute_rotations, &config)
        .unwrap();

    assert_eq!(result.translations.len(), 3);

    for &(i, ref c_i) in &positions_gt {
        for &(j, ref c_j) in &positions_gt {
            if i < j {
                let gt_dir = (c_j - c_i).normalize();
                let est_i = result.translations.get(&ViewId::new(i)).unwrap();
                let est_j = result.translations.get(&ViewId::new(j)).unwrap();
                let est_dir = (est_j - est_i).normalize();
                let angle = gt_dir.dot(&est_dir).clamp(-1.0, 1.0).acos().to_degrees();
                assert!(
                    angle < 5.0,
                    "Direction error for ({}, {}): {:.1}° (should be < 5°)",
                    i,
                    j,
                    angle
                );
            }
        }
    }
}

#[test]
fn test_l1_irls_insufficient_data() {
    let solver = L1IrlsTranslationSolver::new();
    let config = TranslationConfig::default();
    let absolute_rotations = BTreeMap::new();

    let result = solver.solve(&[], &absolute_rotations, &config);
    assert!(result.is_err());
}

#[test]
fn test_l1_irls_rejects_disconnected_inputs() {
    let solver = L1IrlsTranslationSolver::new();
    let config = TranslationConfig::default();
    let rotations: BTreeMap<_, _> = (0..4)
        .map(|view| (ViewId::new(view), Rotation3::identity()))
        .collect();

    assert!(
        RelativeTranslation::try_new(ViewId::new(0), ViewId::new(1), Vector3::zeros(), 1.0,)
            .is_err()
    );

    let disconnected = [
        RelativeTranslation::try_new(ViewId::new(0), ViewId::new(1), Vector3::x(), 0.25).unwrap(),
        RelativeTranslation::try_new(ViewId::new(2), ViewId::new(3), Vector3::x(), 1.0).unwrap(),
    ];
    assert!(solver.solve(&disconnected, &rotations, &config).is_err());

    let missing_rotation =
        [
            RelativeTranslation::try_new(ViewId::new(0), ViewId::new(4), Vector3::x(), 1.0)
                .unwrap(),
        ];
    assert!(
        solver
            .solve(&missing_rotation, &rotations, &config)
            .is_err()
    );
}

#[test]
fn test_l1_irls_rejects_invalid_solver_config() {
    let rotations: BTreeMap<_, _> = (0..2)
        .map(|view| (ViewId::new(view), Rotation3::identity()))
        .collect();
    let relative =
        [
            RelativeTranslation::try_new(ViewId::new(0), ViewId::new(1), Vector3::x(), 1.0)
                .unwrap(),
        ];
    let solver = L1IrlsTranslationSolver::with_config(L1IrlsConfig { huber_delta: 0.0 });
    assert!(
        solver
            .solve(&relative, &rotations, &TranslationConfig::default())
            .is_err()
    );
}

/// Test that outliers are down-weighted by the Huber loss and do not
/// corrupt the solution.
#[test]
fn test_l1_irls_outlier_rejection() {
    let mut absolute_rotations = BTreeMap::new();
    for i in 0..4usize {
        absolute_rotations.insert(ViewId::new(i), Rotation3::identity());
    }

    let gt = [
        Vector3::new(0.0, 0.0, 0.0),
        Vector3::new(1.0, 0.0, 0.0),
        Vector3::new(2.0, 0.0, 0.0),
        Vector3::new(3.0, 0.0, 0.0),
    ];

    let mut relative_translations = Vec::new();
    for i in 0..4usize {
        for j in (i + 1)..4usize {
            let dir = (gt[j] - gt[i]).normalize();
            relative_translations.push(
                RelativeTranslation::try_new(ViewId::new(i), ViewId::new(j), dir, 1.0).unwrap(),
            );
        }
    }

    // Inject a gross outlier: edge (0,3) pointing in the wrong direction
    relative_translations.push(
        RelativeTranslation::try_new(
            ViewId::new(0),
            ViewId::new(3),
            Vector3::new(0.0, 1.0, 0.0),
            1.0,
        )
        .unwrap(),
    );

    let solver = L1IrlsTranslationSolver::new();
    let config = TranslationConfig::default()
        .with_max_iterations(200)
        .with_tolerance(openmvg_core::AngularThreshold::try_new(1e-8).unwrap());
    let result = solver
        .solve(&relative_translations, &absolute_rotations, &config)
        .unwrap();

    let est_0 = result.translations.get(&ViewId::new(0)).unwrap();
    let est_3 = result.translations.get(&ViewId::new(3)).unwrap();
    let est_dir = (est_3 - est_0).normalize();
    let angle = est_dir
        .dot(&Vector3::x())
        .clamp(-1.0, 1.0)
        .acos()
        .to_degrees();
    assert!(
        angle < 15.0,
        "Outlier corrupted solution: direction error {:.1}° (should be < 15°)",
        angle,
    );
}

/// Exercise the matrix-free CG path by creating a synthetic scene
/// large enough to exceed `MATRIX_FREE_THRESHOLD`.
#[test]
fn test_l1_irls_large_scale_cg() {
    let n_views: usize = 120;
    let mut absolute_rotations = BTreeMap::new();
    for i in 0..n_views {
        absolute_rotations.insert(ViewId::new(i), Rotation3::identity());
    }

    let gt: Vec<Vector3<f64>> = (0..n_views)
        .map(|i| Vector3::new(i as f64, 0.0, 0.0))
        .collect();

    let mut relative_translations = Vec::new();
    // Chain of consecutive edges
    for i in 0..n_views - 1 {
        let dir = (gt[i + 1] - gt[i]).normalize();
        relative_translations.push(
            RelativeTranslation::try_new(ViewId::new(i), ViewId::new(i + 1), dir, 1.0).unwrap(),
        );
    }
    // Skip-2 edges for redundancy
    for i in 0..n_views.saturating_sub(2) {
        let dir = (gt[i + 2] - gt[i]).normalize();
        relative_translations.push(
            RelativeTranslation::try_new(ViewId::new(i), ViewId::new(i + 2), dir, 1.0).unwrap(),
        );
    }

    let solver = L1IrlsTranslationSolver::new();
    let config = TranslationConfig::default()
        .with_max_iterations(50)
        .with_tolerance(openmvg_core::AngularThreshold::ONE_MICRORADIAN);
    let result = solver
        .solve(&relative_translations, &absolute_rotations, &config)
        .unwrap();

    assert_eq!(result.translations.len(), n_views);

    // Verify first→last direction is roughly +x
    let est_0 = result.translations.get(&ViewId::new(0)).unwrap();
    let est_last = result.translations.get(&ViewId::new(n_views - 1)).unwrap();
    let est_dir = (est_last - est_0).normalize();
    let angle = est_dir
        .dot(&Vector3::x())
        .clamp(-1.0, 1.0)
        .acos()
        .to_degrees();
    assert!(
        angle < 5.0,
        "Large-scale CG: direction error {:.1}° (should be < 5°)",
        angle,
    );
}

/// A long chain made of two heavily weighted halves joined by one weak edge is the
/// ill-conditioned Laplacian of a weakly linked camera block. The matrix-free path must
/// certify the same solution as the dense factorization instead of stopping at a cap.
#[test]
fn matrix_free_solve_matches_dense_on_clustered_baselines() {
    // Twenty tight camera clusters (baselines ~1e-3) chained by unit-length bridges: the
    // IRLS system weights w/s² then span ~six orders of magnitude, as in a real scene with
    // short-baseline bursts. Plain CG capped at min(500, n) leaves an O(1) error here.
    let (clusters, per_cluster, tight) = (20, 10, 1e-3);
    let n_views = clusters * per_cluster;
    let n_free = 3 * (n_views - 1);
    let centres: Vec<Vector3<f64>> = (0..n_views)
        .map(|view| {
            let (cluster, k) = (view / per_cluster, view % per_cluster);
            let base = Vector3::new(
                (0.9 * cluster as f64).cos(),
                (0.9 * cluster as f64).sin(),
                0.3 * cluster as f64,
            );
            let angle = 2.399963 * k as f64;
            let z = 2.0 * k as f64 / (per_cluster - 1) as f64 - 1.0;
            let rho = (1.0 - z * z).max(0.0).sqrt();
            base + tight
                * (1 + k % 3) as f64
                * Vector3::new(rho * angle.cos(), rho * angle.sin(), z)
        })
        .collect();
    let mut pairs = Vec::new();
    for cluster in 0..clusters {
        for k in 0..per_cluster {
            let i = cluster * per_cluster + k;
            for k2 in k + 1..(k + 4).min(per_cluster) {
                pairs.push((
                    i,
                    cluster * per_cluster + k2,
                    (60 + (i * 7919 + k2 * 104729) % 2940) as f64,
                ));
            }
        }
        if cluster + 1 < clusters {
            for k in (0..per_cluster).step_by(per_cluster / 3) {
                let j = (cluster + 1) * per_cluster + (k * 5) % per_cluster;
                pairs.push((cluster * per_cluster + k, j, (60 + (k * 31) % 200) as f64));
            }
        }
    }
    let edges: Vec<Edge> = pairs
        .iter()
        .map(|&(i, j, weight)| Edge {
            i,
            j,
            direction: (centres[j] - centres[i]).normalize(),
            weight,
        })
        .collect();
    let weights: Vec<f64> = edges.iter().map(|edge| edge.weight).collect();
    let scales: Vec<f64> = pairs
        .iter()
        .map(|&(i, j, _)| (centres[j] - centres[i]).norm())
        .collect();

    let matrix_free = L1IrlsTranslationSolver::solve_cg_matrix_free(
        &edges,
        &weights,
        &scales,
        TIKHONOV_LAMBDA,
        &vec![0.0; n_free],
    )
    .expect("preconditioned CG must certify the clustered system");
    let dense = L1IrlsTranslationSolver::solve_normal_equations(
        &mut DMatrix::zeros(n_free, n_free),
        &mut DVector::zeros(n_free),
        &edges,
        &weights,
        &scales,
        TIKHONOV_LAMBDA,
    )
    .unwrap();

    let scale = dense.iter().map(|value| value * value).sum::<f64>().sqrt();
    let error = matrix_free
        .iter()
        .zip(&dense)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        .sqrt();
    assert!(
        error <= 1e-6 * scale,
        "matrix-free solve deviates from dense by {error:.3e} (scale {scale:.3e})"
    );

    let warm = L1IrlsTranslationSolver::solve_cg_matrix_free(
        &edges,
        &weights,
        &scales,
        TIKHONOV_LAMBDA,
        &dense,
    )
    .expect("a warm start at the solution must certify");
    let warm_error = warm
        .iter()
        .zip(&dense)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        .sqrt();
    assert!(
        warm_error <= 1e-6 * scale,
        "warm-started solve leaves the solution by {warm_error:.3e} (scale {scale:.3e})"
    );
}
