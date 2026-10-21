// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Rigid 3D-to-3D alignment policy, colocated with the solver family.
//!
//! Its teaching docs are on [`estimate_rigid_alignment`], where docs.rs shows them.

use crate::alignment::RigidAlignmentSolver;
use crate::correspondence::Point3Correspondence;
use nalgebra::Isometry3;

use crate::robust::error::RobustError;
use crate::robust::result::RobustEstimate;
use crate::robust::threshold::WorldDistanceThreshold;

use crate::robust::genransac::config::{GenRansacConfig, RobustOptions};
use crate::robust::genransac::estimator::{GenRansacEstimator, RobustGuidance};
use crate::robust::genransac::refiner::SolverRefiner;

/// Estimate a source-to-target rigid transform from 3D correspondences.
///
/// # How it works
///
/// The same proper-rotation Kabsch solver fits minimal three-point hypotheses
/// and overdetermined consensuses. Residuals and thresholds use the input
/// points' world-distance unit.
///
/// ```text
/// 3D pairs -> Kabsch minimal hypotheses -> distance score
///         -> Kabsch consensus refit -> rigid transform result
/// ```
pub fn estimate_rigid_alignment(
    data: &[Point3Correspondence],
    threshold: WorldDistanceThreshold,
    options: &RobustOptions,
) -> Result<RobustEstimate<Isometry3<f64>>, RobustError> {
    estimate_rigid_alignment_guided(data, threshold, options, RobustGuidance::default())
}

/// Estimate a source-to-target rigid transform with sampling guidance.
///
/// A compatibility clique can be supplied as a best-first `order`; calibrated
/// clique or match scores can additionally be supplied as inlier `priors`.
pub fn estimate_rigid_alignment_guided(
    data: &[Point3Correspondence],
    threshold: WorldDistanceThreshold,
    options: &RobustOptions,
    guidance: RobustGuidance<'_>,
) -> Result<RobustEstimate<Isometry3<f64>>, RobustError> {
    let search = GenRansacConfig::from_options(GenRansacConfig::builder(), options);
    search.validate()?;
    if data.iter().any(|correspondence| {
        !correspondence
            .source
            .coords
            .iter()
            .chain(correspondence.target.coords.iter())
            .all(|value| value.is_finite())
    }) {
        return Err(crate::SolverError::InvalidInput {
            reason: "alignment points must be finite",
        }
        .into());
    }
    let refiner = SolverRefiner::<RigidAlignmentSolver>::new(());
    GenRansacEstimator::estimate_guided::<RigidAlignmentSolver, _>(
        data,
        threshold.into(),
        &search,
        &(),
        guidance,
        &refiner,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::robust::error::RobustError;
    use nalgebra::{Isometry3, Point3, Translation3, UnitQuaternion, Vector3};

    fn synthetic_problem() -> (Isometry3<f64>, Vec<Point3Correspondence>, usize) {
        let transform = Isometry3::from_parts(
            Translation3::new(0.42, -0.31, 0.18),
            UnitQuaternion::from_euler_angles(0.13, -0.19, 0.07),
        );
        let mut data = (0..28)
            .map(|index| {
                let u = index as f64;
                let source = Point3::new(
                    1.7 * (0.61 * u + 0.2).sin(),
                    1.3 * (1.07 * u - 0.4).cos(),
                    0.9 * (0.43 * u + 0.1).sin() + 0.08 * u,
                );
                let noise = Vector3::new(
                    8.0e-5 * (1.31 * u).sin(),
                    8.0e-5 * (0.83 * u).cos(),
                    8.0e-5 * (1.73 * u + 0.5).sin(),
                );
                Point3Correspondence::new(source, transform.transform_point(&source) + noise)
            })
            .collect::<Vec<_>>();
        let inlier_count = data.len();
        for index in 0..8 {
            let source = data[index + 3].source;
            let wrong_target =
                data[(index + 15) % inlier_count].target + Vector3::new(0.8, -0.6, 0.5);
            data.push(Point3Correspondence::new(source, wrong_target));
        }
        (transform, data, inlier_count)
    }

    fn test_options() -> RobustOptions {
        RobustOptions::default().with_max_draws(256).with_seed(101)
    }

    fn assert_recovery(
        expected: &Isometry3<f64>,
        result: &RobustEstimate<Isometry3<f64>>,
        inlier_count: usize,
    ) {
        let rotation_error = (result.model.rotation.inverse() * expected.rotation).angle();
        let translation_error =
            (result.model.translation.vector - expected.translation.vector).norm();
        assert!(rotation_error < 5.0e-4, "rotation error: {rotation_error}");
        assert!(
            translation_error < 5.0e-4,
            "translation error: {translation_error}"
        );
        assert!(
            result
                .inliers
                .iter()
                .filter(|&&index| index < inlier_count)
                .count()
                >= inlier_count - 1
        );
        assert!(result.inliers.iter().all(|&index| index < inlier_count));
    }

    /// Robust rigid alignment: the typed world-distance facade recovers a proper rigid transform
    /// with outliers; minimal fitting and consensus refitting preserve source-to-target direction.
    #[test]
    fn facade_recovers_a_rigid_transform_with_outliers() {
        let (expected, data, inlier_count) = synthetic_problem();
        let result = estimate_rigid_alignment(
            &data,
            WorldDistanceThreshold::try_new(1.0e-3).unwrap(),
            &test_options(),
        )
        .expect("rigid-alignment policy should recover the dominant transform");
        assert_recovery(&expected, &result, inlier_count);
    }

    #[test]
    fn guided_facade_is_deterministic_with_outliers() {
        let (expected, data, inlier_count) = synthetic_problem();
        let order = (0..data.len()).collect::<Vec<_>>();
        let solve = || {
            estimate_rigid_alignment_guided(
                &data,
                WorldDistanceThreshold::try_new(1.0e-3).unwrap(),
                &test_options(),
                RobustGuidance::ranked(&order),
            )
            .expect("guided rigid-alignment policy should recover the dominant transform")
        };
        let first = solve();
        let second = solve();
        assert_eq!(first.inliers, second.inliers);
        assert_eq!(first.model, second.model);
        assert_recovery(&expected, &first, inlier_count);
    }

    #[test]
    fn facade_rejects_non_finite_correspondence_before_sampling() {
        let mut data: Vec<_> = [
            Point3::origin(),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        ]
        .into_iter()
        .map(|point| Point3Correspondence::new(point, point))
        .collect();
        data.push(Point3Correspondence::new(
            Point3::new(f64::NAN, 0.0, 0.0),
            Point3::origin(),
        ));
        let result = estimate_rigid_alignment(
            &data,
            WorldDistanceThreshold::try_new(1e-3).unwrap(),
            &RobustOptions::default().with_max_draws(64).with_seed(101),
        );
        assert!(
            matches!(result, Err(RobustError::SolverFailed { reason }) if reason.contains("must be finite"))
        );
    }

    #[test]
    fn facade_preserves_the_three_point_contract() {
        let result = estimate_rigid_alignment(
            &[],
            WorldDistanceThreshold::try_new(0.01).unwrap(),
            &test_options(),
        );
        assert!(matches!(
            result,
            Err(RobustError::InsufficientData {
                required: 3,
                provided: 0
            })
        ));
    }
}
