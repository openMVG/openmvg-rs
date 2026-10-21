// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Robust calibrated absolute-pose estimation, exported from [`crate::pose`].
//!
//! Its teaching docs are on [`estimate_absolute_pose`], where docs.rs shows them.

use super::{
    DingP3pSolver, FusedP3pConfig, FusedP3pSolver, KeP3pConfig, KeP3pSolver, KneipP3pSolver,
    LambdaTwistP3pSolver, NakanoP3pSolver, P3pMethod, WuP3pSolver,
};
use crate::pose::{
    BearingToPoint, ResectionConfig, ResectionSolver, SolverPose, WorldToCameraPose,
    camera_pose_from_world_to_camera, world_to_camera_from_matrix,
};
use nalgebra::{Matrix3, Point3, Vector3};

use crate::pose::resection::{refine_bearing_solver_pose, refine_perspective_solver_pose};
use crate::robust::error::RobustError;
use crate::robust::result::{Diagnostics, EstimateStats, RobustEstimate};
use crate::robust::threshold::AngularThreshold;
use crate::solver_traits::MultiSolutionSolver;

use crate::robust::genransac::config::{GenRansacConfig, RobustOptions};
use crate::robust::genransac::estimator::{GenRansacEstimator, RobustGuidance};
use crate::robust::genransac::refiner::SolverRefiner;

/// Options for robust calibrated absolute-pose estimation.
///
/// The default uses [`P3pMethod::Fused`] and the default [`RobustOptions`], and does not polish
/// the result. To polish it, prefer [`Self::with_refinement`] over calling
/// [`refine_perspective_pose`](crate::pose::refine_perspective_pose) yourself: the option
/// refines on the inliers and then recomputes the inliers and residuals of the returned
/// estimate.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct AbsolutePoseOptions {
    method: P3pMethod,
    robust_options: RobustOptions,
    refinement: Option<PoseRefinement>,
}

/// Observation model used for optional nonlinear polishing after robust P3P.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum PoseRefinement {
    /// Normalized perspective residuals for pinhole-like cameras.
    Perspective,
    /// Angular tangent-space residuals for full-sphere bearing cameras.
    Bearing,
}

/// Robust absolute-pose estimate.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct AbsolutePoseEstimate {
    model: WorldToCameraPose,
    inliers: Vec<usize>,
    residuals: Vec<f64>,
    stats: EstimateStats,
    diagnostics: Option<Diagnostics>,
}

impl AbsolutePoseEstimate {
    /// Return the world-to-camera pose.
    #[must_use]
    pub const fn model(&self) -> &WorldToCameraPose {
        &self.model
    }

    /// Return the final inlier indices.
    #[must_use]
    pub fn inliers(&self) -> &[usize] {
        &self.inliers
    }

    /// Number of inliers in the final consensus.
    #[must_use]
    pub fn inlier_count(&self) -> usize {
        self.inliers.len()
    }

    /// The angular residual, in radians, of every input correspondence, in input order, against
    /// the returned pose, which is the refined pose when
    /// [`AbsolutePoseOptions::with_refinement`] is set. With refinement, the inliers are
    /// recomputed from these residuals.
    #[must_use]
    pub fn residuals(&self) -> &[f64] {
        &self.residuals
    }

    /// Return robust-search statistics updated for the final model.
    #[must_use]
    pub const fn stats(&self) -> &EstimateStats {
        &self.stats
    }

    /// Return optional robust-search diagnostics.
    #[must_use]
    pub const fn diagnostics(&self) -> Option<&Diagnostics> {
        self.diagnostics.as_ref()
    }

    /// Consume the estimate and return its world-to-camera pose.
    #[must_use]
    pub fn into_model(self) -> WorldToCameraPose {
        self.model
    }
}

fn solver_pose_components(pose: &SolverPose<f64>) -> (Matrix3<f64>, Vector3<f64>) {
    (
        *pose.rotation.to_rotation_matrix().matrix(),
        pose.translation.vector,
    )
}

fn angular_residual(
    rotation: &Matrix3<f64>,
    translation: &Vector3<f64>,
    datum: &BearingToPoint,
) -> f64 {
    let camera_point = rotation * datum.world_point.coords + translation;
    let bearing = datum.bearing.as_ref();
    let ray_depth = camera_point.dot(bearing);
    if !ray_depth.is_finite() || ray_depth <= 0.0 {
        return f64::INFINITY;
    }
    camera_point.cross(bearing).norm().atan2(ray_depth)
}

impl AbsolutePoseOptions {
    /// Use a specific P3P minimal solver; [`P3pMethod::Fused`] is the default. Keep it unless you
    /// want to compare formulations.
    #[must_use]
    pub fn with_method(mut self, method: P3pMethod) -> Self {
        self.method = method;
        self
    }

    /// Set the robust-search options: the seed and the iteration and confidence controls, as
    /// the other robust estimators take them.
    #[must_use]
    pub fn with_robust_options(mut self, options: RobustOptions) -> Self {
        self.robust_options = options;
        self
    }

    /// Polish the final robust estimate on its inliers: [`PoseRefinement::Perspective`] for
    /// pinhole and Brown-Conrady cameras, [`PoseRefinement::Bearing`] for fisheye and spherical
    /// ones. Without this call the pose is not polished; a polish that would raise the error is
    /// discarded.
    #[must_use]
    pub fn with_refinement(mut self, refinement: PoseRefinement) -> Self {
        self.refinement = Some(refinement);
        self
    }

    /// Return the selected P3P method.
    #[must_use]
    pub fn method(self) -> P3pMethod {
        self.method
    }

    /// Return the robust-search options.
    #[must_use]
    pub fn robust_options(self) -> RobustOptions {
        self.robust_options
    }

    /// Return the requested final nonlinear refinement policy.
    #[must_use]
    pub fn refinement(self) -> Option<PoseRefinement> {
        self.refinement
    }
}

/// Estimate a calibrated camera pose from 3D-to-bearing correspondences.
///
/// The threshold is an angular error in radians. Convert a pixel threshold with
/// [`AngularThreshold::from_pixels`] and the focal length. Triangulated 3D points carry their
/// own error on top of the image noise, so choose a looser value than for exact points, a few
/// pixels rather than a bound fitted to the image noise alone; a tight threshold drops good
/// matches.
///
/// The minimal solver selected by [`AbsolutePoseOptions::with_method`] ([`P3pMethod::Fused`]
/// by default) supplies three-point hypotheses and [`ResectionSolver`] refits consensuses
/// containing at least six correspondences.
///
/// # How it works
///
/// The default path combines FusedP3P minimal hypotheses with the
/// overdetermined calibrated resection solver. Both use angular residuals in
/// radians, so one typed threshold governs hypothesis scoring and refinement.
///
/// ```text
/// 3D/bearing pairs -> FusedP3P hypotheses -> angular score
///                  -> calibrated resection refit -> pose result
/// ```
pub fn estimate_absolute_pose(
    data: &[BearingToPoint],
    threshold: AngularThreshold,
    options: &AbsolutePoseOptions,
) -> Result<AbsolutePoseEstimate, RobustError> {
    estimate_absolute_pose_guided(data, threshold, options, RobustGuidance::default())
}

/// Estimate a calibrated camera pose with optional sampling guidance.
pub fn estimate_absolute_pose_guided(
    data: &[BearingToPoint],
    threshold: AngularThreshold,
    options: &AbsolutePoseOptions,
    guidance: RobustGuidance<'_>,
) -> Result<AbsolutePoseEstimate, RobustError> {
    let refiner = SolverRefiner::<ResectionSolver>::new(ResectionConfig::default());
    let search = GenRansacConfig::from_options(GenRansacConfig::builder(), &options.robust_options);
    let result = match options.method {
        P3pMethod::Fused => estimate_with::<FusedP3pSolver>(
            data,
            threshold,
            &search,
            &FusedP3pConfig::default(),
            guidance,
            &refiner,
        ),
        P3pMethod::LambdaTwist => {
            estimate_with::<LambdaTwistP3pSolver>(data, threshold, &search, &(), guidance, &refiner)
        }
        P3pMethod::Ding => {
            estimate_with::<DingP3pSolver>(data, threshold, &search, &(), guidance, &refiner)
        }
        P3pMethod::KeOrtho => {
            estimate_with::<KeP3pSolver>(data, threshold, &search, &KeP3pConfig, guidance, &refiner)
        }
        P3pMethod::Kneip => {
            estimate_with::<KneipP3pSolver>(data, threshold, &search, &(), guidance, &refiner)
        }
        P3pMethod::Nakano => {
            estimate_with::<NakanoP3pSolver>(data, threshold, &search, &(), guidance, &refiner)
        }
        P3pMethod::Wu => {
            estimate_with::<WuP3pSolver>(data, threshold, &search, &(), guidance, &refiner)
        }
    }?;
    finalize_estimate(result, data, threshold, options.refinement)
}

fn finalize_estimate(
    mut result: RobustEstimate<SolverPose<f64>>,
    data: &[BearingToPoint],
    threshold: AngularThreshold,
    refinement: Option<PoseRefinement>,
) -> Result<AbsolutePoseEstimate, RobustError> {
    if let Some(refinement) = refinement {
        // Recover `(R, t)` from the winning isometry through its camera center
        // before the final polish. Keep this round trip: changing it changes
        // the polished pose in the last bits.
        let rotation = *result.model.rotation.to_rotation_matrix().matrix();
        let translation = -(rotation
            * result
                .model
                .inverse_transform_point(&Point3::origin())
                .coords);
        let initial = world_to_camera_from_matrix(&rotation, translation);
        let inlier_data = data
            .iter()
            .copied()
            .filter(|datum| angular_residual(&rotation, &translation, datum) <= threshold.value())
            .collect::<Vec<_>>();
        result.model = match refinement {
            PoseRefinement::Perspective => refine_perspective_solver_pose(&initial, &inlier_data),
            PoseRefinement::Bearing => refine_bearing_solver_pose(&initial, &inlier_data),
        }
        .unwrap_or(initial);
    }

    let (rotation, translation) = solver_pose_components(&result.model);
    let residuals = data
        .iter()
        .map(|datum| angular_residual(&rotation, &translation, datum))
        .collect::<Vec<_>>();
    if refinement.is_some() {
        let threshold_squared = threshold.value() * threshold.value();
        result.inliers = residuals
            .iter()
            .enumerate()
            .filter_map(|(index, &residual)| (residual <= threshold.value()).then_some(index))
            .collect();
        result.stats.score = residuals
            .iter()
            .map(|residual| threshold_squared - residual * residual)
            .filter(|gain| gain.is_finite() && *gain >= 0.0)
            .sum();
        result.stats.inlier_ratio = result.inliers.len() as f64 / data.len() as f64;
    }
    if let Some(diagnostics) = result.diagnostics.as_mut() {
        diagnostics.residuals.clone_from(&residuals);
    }

    let model = camera_pose_from_world_to_camera(&result.model).map_err(|error| {
        RobustError::Numerical {
            reason: error.to_string(),
        }
    })?;
    Ok(AbsolutePoseEstimate {
        model,
        inliers: result.inliers,
        residuals,
        stats: result.stats,
        diagnostics: result.diagnostics,
    })
}

fn estimate_with<S>(
    data: &[BearingToPoint],
    threshold: AngularThreshold,
    search: &GenRansacConfig,
    solver_config: &S::Config,
    guidance: RobustGuidance<'_>,
    refiner: &SolverRefiner<ResectionSolver>,
) -> Result<RobustEstimate<SolverPose<f64>>, RobustError>
where
    S: MultiSolutionSolver<Datum = BearingToPoint, Model = SolverPose<f64>>,
{
    GenRansacEstimator::estimate_guided::<S, _>(
        data,
        threshold.into(),
        search,
        solver_config,
        guidance,
        refiner,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::robust::error::RobustError;
    use nalgebra::{Isometry3, Point3, Translation3, UnitQuaternion, Vector3};

    fn synthetic_pose_problem() -> (Isometry3<f64>, Vec<BearingToPoint>, usize) {
        let pose = Isometry3::from_parts(
            Translation3::new(0.31, -0.18, 0.24),
            UnitQuaternion::from_euler_angles(0.08, -0.13, 0.05),
        );
        let mut data = (0..32)
            .map(|index| {
                let u = index as f64;
                let world = Point3::new(
                    1.8 * (0.71 * u + 0.2).sin(),
                    1.5 * (1.17 * u - 0.3).cos(),
                    4.0 + 0.12 * u + 0.5 * (0.43 * u).sin().abs(),
                );
                let camera = pose.transform_point(&world);
                let perturbation =
                    Vector3::new(2.0e-5 * (1.31 * u).sin(), 2.0e-5 * (0.83 * u).cos(), 0.0);
                BearingToPoint::from_direction(world, camera.coords + perturbation)
            })
            .collect::<Vec<_>>();
        let inlier_count = data.len();

        for index in 0..6 {
            let source = index + 2;
            let wrong_bearing = data[(index + 13) % inlier_count].bearing;
            data.push(BearingToPoint::new(data[source].world_point, wrong_bearing));
        }
        (pose, data, inlier_count)
    }

    fn test_options() -> AbsolutePoseOptions {
        AbsolutePoseOptions::default().with_robust_options(
            RobustOptions::default()
                .with_max_draws(256)
                .with_seed(73)
                .with_diagnostics(true),
        )
    }

    fn assert_pose_recovery(
        expected: &SolverPose<f64>,
        data: &[BearingToPoint],
        inlier_count: usize,
        result: &AbsolutePoseEstimate,
        threshold: AngularThreshold,
    ) {
        let rotation_error =
            (result.model.rotation().inverse() * expected.rotation.to_rotation_matrix()).angle();
        let translation_error = (result.model.translation() - expected.translation.vector).norm();
        assert!(rotation_error < 2.0e-3, "rotation error: {rotation_error}");
        assert!(
            translation_error < 1.0e-2,
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

        let maximum_true_residual = data[..inlier_count]
            .iter()
            .map(|datum| ResectionSolver::residual(&result.model, datum))
            .fold(0.0_f64, f64::max);
        assert!(maximum_true_residual < threshold.value());
    }

    /// Robust calibrated absolute-pose estimation: the robust facade defaults to Fused P3P
    /// hypotheses and calibrated resection refinement; the returned world-to-camera pose recovers
    /// the dominant angular consensus.
    #[test]
    fn ordinary_facade_recovers_calibrated_pose_end_to_end() {
        let (expected, data, inlier_count) = synthetic_pose_problem();
        let threshold = AngularThreshold::try_new(5.0e-4).unwrap();
        assert_eq!(AbsolutePoseOptions::default().method(), P3pMethod::Fused);
        let options = test_options().with_refinement(PoseRefinement::Perspective);
        let result = estimate_absolute_pose(&data, threshold, &options)
            .expect("ordinary P3P policy should recover the dominant pose");

        assert_pose_recovery(&expected, &data, inlier_count, &result, threshold);
        assert_eq!(result.diagnostics().unwrap().residuals, result.residuals());
        assert_eq!(
            result.stats().inlier_ratio,
            result.inlier_count() as f64 / data.len() as f64
        );
    }

    #[test]
    fn guided_facade_is_deterministic_end_to_end() {
        let (expected, data, inlier_count) = synthetic_pose_problem();
        let threshold = AngularThreshold::try_new(5.0e-4).unwrap();
        let order = (0..data.len()).collect::<Vec<_>>();
        let solve = || {
            estimate_absolute_pose_guided(
                &data,
                threshold,
                &test_options(),
                RobustGuidance::ranked(&order),
            )
            .expect("guided P3P policy should recover the dominant pose")
        };
        let first = solve();
        let second = solve();

        assert_eq!(first.inliers, second.inliers);
        assert_eq!(first.model, second.model);
        assert_pose_recovery(&expected, &data, inlier_count, &first, threshold);
    }

    #[test]
    fn every_method_preserves_the_three_point_contract() {
        for method in [
            P3pMethod::Fused,
            P3pMethod::LambdaTwist,
            P3pMethod::Ding,
            P3pMethod::KeOrtho,
            P3pMethod::Kneip,
            P3pMethod::Nakano,
            P3pMethod::Wu,
        ] {
            let options = test_options().with_method(method);
            let result =
                estimate_absolute_pose(&[], AngularThreshold::try_new(1.0e-3).unwrap(), &options);
            assert!(matches!(
                result,
                Err(RobustError::InsufficientData {
                    required: 3,
                    provided: 0
                })
            ));
        }
    }

    #[test]
    fn facade_refines_full_sphere_bearings_end_to_end() {
        let mut data = (0..32)
            .map(|index| {
                let fraction = (index as f64 + 0.5) / 32.0;
                let z = -0.95 + 1.9 * fraction;
                let azimuth = index as f64 * 2.399_963_229_728_653;
                let radial = (1.0 - z * z).sqrt();
                let direction = Vector3::new(radial * azimuth.cos(), radial * azimuth.sin(), z);
                let world = Point3::from(direction * (3.0 + 0.07 * index as f64));
                BearingToPoint::from_direction(world, direction)
            })
            .collect::<Vec<_>>();
        data.push(BearingToPoint::from_direction(
            Point3::new(4.0, 0.0, 0.0),
            Vector3::x(),
        ));
        data.push(BearingToPoint::from_direction(
            Point3::new(0.0, -4.5, 0.0),
            -Vector3::y(),
        ));
        let inlier_count = data.len();
        for index in 0..6 {
            let wrong_bearing = data[(index + 11) % inlier_count].bearing;
            data.push(BearingToPoint::new(data[index].world_point, wrong_bearing));
        }

        let threshold = AngularThreshold::try_new(1.0e-5).unwrap();
        let options = test_options().with_refinement(PoseRefinement::Bearing);
        let result = estimate_absolute_pose(&data, threshold, &options)
            .expect("P3P policy should refine a full-sphere consensus");

        assert!(result.model.rotation().angle() < 1.0e-5);
        assert!(result.model.translation().norm() < 1.0e-4);
        assert!(result.inliers.iter().any(|&index| {
            index < inlier_count
                && data[index].bearing.z <= 0.0
                && ResectionSolver::residual(&result.model, &data[index]).is_finite()
        }));
        assert!(result.inliers.iter().any(|&index| {
            index < inlier_count
                && data[index].bearing.z.abs() <= f64::EPSILON
                && ResectionSolver::residual(&result.model, &data[index]).is_finite()
        }));
    }
}
