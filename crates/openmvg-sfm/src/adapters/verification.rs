// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Calibrated two-view geometric verification: five-point Essential GenRANSAC with a
//! homography degeneracy check.

use nalgebra::Matrix3;
use openmvg_core::{
    NormalizedImageThreshold,
    camera::{Camera, CameraModel, PinholeCamera},
};
use openmvg_geometry::correspondence::{BearingCorrespondence, PixelCorrespondence};
use openmvg_geometry::essential::EssentialMatrix;
use openmvg_geometry::essential::{
    estimate_essential_angular_guided as estimate_essential_genransac_angular,
    estimate_essential_guided as estimate_essential_genransac,
};
use openmvg_geometry::homography::estimate_homography as estimate_homography_genransac;
use openmvg_geometry::residuals::{
    sampson_error_essential, sampson_error_essential_pinhole_squared,
};
use openmvg_geometry::robust::RobustError;
pub use openmvg_geometry::robust::threshold::{AngularThreshold, PixelThreshold};
use openmvg_geometry::robust::{RobustGuidance, RobustOptions};

use crate::backends::{GeometricVerifier, VerificationContext};
use crate::two_view::EvidenceThreshold;
use crate::two_view::{
    GeometricModel, InlierMask, ModelEvidence, PairMatches, PairVerificationReport,
    TerminationReason, VerificationRejectReason, VerificationVerdict,
};
use crate::workspace::quality::ErrorStats;
use crate::{Result, SfmError};
use openmvg_features::model::ExtractedFeatures;

const MIN_ESSENTIAL_MATCHES: usize = 8;
const CANONICAL_MIN_INLIERS: usize = 16;
const CANONICAL_MIN_INLIER_RATIO: f64 = 0.10;
const HOMOGRAPHY_DOMINANCE_RATIO: f64 = 0.95;
const HOMOGRAPHY_MAX_ITERATIONS: usize = 256;
/// Stable identifier for the calibrated verification policy used by public pipelines.
pub const CALIBRATED_VERIFIER_POLICY_ID: &str = "calibrated_genransac_e5_h_degeneracy_v1";
/// Stable identifier for the calibrated angular-residual verification policy.
const CALIBRATED_ANGULAR_VERIFIER_POLICY_ID: &str =
    "calibrated_genransac_angular_e5_rotation_degeneracy_v1";

/// Calibrated five-point verification backed by GenRANSAC, the front end's default verifier.
#[derive(Debug, Clone)]
pub struct CalibratedVerifier {
    threshold_px: PixelThreshold,
    angular_threshold: AngularThreshold,
    min_inliers: usize,
    min_inlier_ratio: f64,
    options: RobustOptions,
}

#[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
impl CalibratedVerifier {
    /// Construct a GenRANSAC verifier with an explicit pixel threshold.
    pub fn try_new(threshold_px: PixelThreshold, min_inliers: usize) -> Result<Self> {
        validate_min_inliers(min_inliers)?;
        Ok(Self {
            threshold_px,
            angular_threshold: AngularThreshold::FIVE_MILLIRADIANS,
            min_inliers,
            min_inlier_ratio: CANONICAL_MIN_INLIER_RATIO,
            options: RobustOptions::default(),
        })
    }

    /// Override stable GenRANSAC controls. The pair context still owns the seed.
    pub fn with_options(mut self, options: RobustOptions) -> Self {
        self.options = options;
        self
    }

    /// Stable identifier of the pixel-threshold policy, recorded in pair verification reports.
    pub fn policy_id(&self) -> &'static str {
        CALIBRATED_VERIFIER_POLICY_ID
    }

    /// Inlier threshold in pixels, used when both cameras have a perspective model; it is
    /// divided by the smallest focal length of the pair to get a normalized-image threshold.
    pub fn threshold_px(&self) -> PixelThreshold {
        self.threshold_px
    }

    /// Replaces the angular inlier threshold (default 5 mrad), used instead of the pixel
    /// threshold when either camera has no perspective model.
    pub fn with_angular_threshold(mut self, threshold: AngularThreshold) -> Self {
        self.angular_threshold = threshold;
        self
    }
}

impl Default for CalibratedVerifier {
    fn default() -> Self {
        Self {
            threshold_px: PixelThreshold::FOUR_PIXELS,
            angular_threshold: AngularThreshold::FIVE_MILLIRADIANS,
            min_inliers: CANONICAL_MIN_INLIERS,
            min_inlier_ratio: CANONICAL_MIN_INLIER_RATIO,
            options: RobustOptions::default(),
        }
    }
}

impl GeometricVerifier for CalibratedVerifier {
    fn verify_pair(
        &self,
        ctx: &VerificationContext,
        fi: &ExtractedFeatures,
        fj: &ExtractedFeatures,
        matches: &PairMatches,
        camera_i: &Camera,
        camera_j: &Camera,
    ) -> Result<VerificationVerdict> {
        if let Some(outcome) = reject_low_match_count(ctx, matches) {
            return Ok(outcome);
        }
        let prepared = prepare_calibrated_correspondences(fi, fj, matches, camera_i, camera_j)?;
        let ranked_order = has_match_quality(matches)
            .then(|| (0..prepared.bearing_corrs.len()).collect::<Vec<_>>());
        let guidance = ranked_order
            .as_deref()
            .map_or_else(RobustGuidance::default, RobustGuidance::ranked);
        let options = self.options.with_seed(ctx.seed);
        let (result, threshold, residual) = match prepared.min_focal {
            Some(min_focal) => {
                let normalized_threshold =
                    NormalizedImageThreshold::from_pixels(self.threshold_px, min_focal)
                        .map_err(|error| SfmError::InvalidInput(error.to_string()))?;
                let result = estimate_essential_genransac(
                    &prepared.bearing_corrs,
                    normalized_threshold,
                    &options,
                    guidance,
                );
                (
                    result,
                    normalized_threshold.into(),
                    EssentialResidual::NormalizedImage,
                )
            }
            None => {
                let result = estimate_essential_genransac_angular(
                    &prepared.bearing_corrs,
                    self.angular_threshold,
                    &options,
                    guidance,
                );
                (
                    result,
                    self.angular_threshold.into(),
                    EssentialResidual::Angular,
                )
            }
        };
        let result = match result {
            Ok(result) => result,
            Err(RobustError::InvalidConfig { reason }) => {
                return Err(SfmError::InvalidInput(format!(
                    "invalid GenRANSAC Essential configuration: {reason}"
                )));
            }
            Err(RobustError::InsufficientData { required, provided }) => {
                return Err(SfmError::InvalidInput(format!(
                    "GenRANSAC Essential requires {required} correspondences, got {provided}"
                )));
            }
            Err(error) => {
                return Ok(rejected(
                    ctx,
                    matches,
                    VerificationRejectReason::SolverDidNotConverge,
                    0,
                    format!("GenRANSAC Essential returned no model: {error}"),
                ));
            }
        };

        finish_calibrated_verification(
            ctx,
            matches,
            &prepared,
            CalibratedEstimate {
                model: result.model(),
                inliers: result.inliers(),
                iterations: result.stats().iteration_count,
            },
            self.min_inliers,
            self.min_inlier_ratio,
            CalibratedPostprocessPolicy {
                policy_id: if prepared.min_focal.is_some() {
                    CALIBRATED_VERIFIER_POLICY_ID
                } else {
                    CALIBRATED_ANGULAR_VERIFIER_POLICY_ID
                },
                estimator_name: "GenRANSAC Essential",
                threshold,
                residual,
            },
            self.threshold_px,
        )
    }
}

struct PreparedCalibratedCorrespondences {
    ordered_match_indices: Vec<usize>,
    perspective_pixel_corrs: Option<Vec<PixelCorrespondence>>,
    bearing_corrs: Vec<BearingCorrespondence>,
    min_focal: Option<f64>,
}

#[derive(Clone, Copy)]
struct CalibratedEstimate<'a> {
    model: &'a EssentialMatrix,
    inliers: &'a [usize],
    iterations: usize,
}

#[derive(Debug, Clone, Copy)]
enum EssentialResidual {
    Angular,
    NormalizedImage,
}

impl EssentialResidual {
    fn evaluate(self, model: &EssentialMatrix, correspondence: &BearingCorrespondence) -> f64 {
        match self {
            Self::Angular => sampson_error_essential(model.matrix(), correspondence),
            Self::NormalizedImage => {
                sampson_error_essential_pinhole_squared(model.matrix(), correspondence).sqrt()
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct CalibratedPostprocessPolicy {
    policy_id: &'static str,
    estimator_name: &'static str,
    threshold: EvidenceThreshold,
    residual: EssentialResidual,
}

fn prepare_calibrated_correspondences(
    fi: &ExtractedFeatures,
    fj: &ExtractedFeatures,
    matches: &PairMatches,
    camera_i: &Camera,
    camera_j: &Camera,
) -> Result<PreparedCalibratedCorrespondences> {
    let ordered_match_indices = match_indices_by_quality(matches);
    let perspective = perspective_pinhole(camera_i).zip(perspective_pinhole(camera_j));
    let mut bearing_corrs = Vec::with_capacity(ordered_match_indices.len());
    let mut perspective_pixel_corrs = perspective
        .as_ref()
        .map(|_| Vec::with_capacity(ordered_match_indices.len()));
    for &match_index in &ordered_match_indices {
        let matched = &matches.as_slice()[match_index];
        let key_i = fi
            .keypoints()
            .get(matched.first().index())
            .ok_or(SfmError::InvalidId {
                kind: crate::FeatureId::kind(),
                index: matched.first().index(),
            })?;
        let key_j = fj
            .keypoints()
            .get(matched.second().index())
            .ok_or(SfmError::InvalidId {
                kind: crate::FeatureId::kind(),
                index: matched.second().index(),
            })?;
        let pixel_i = nalgebra::Point2::new(key_i.x() as f64, key_i.y() as f64);
        let pixel_j = nalgebra::Point2::new(key_j.x() as f64, key_j.y() as f64);
        let bearing_i = unit_bearing(camera_i, pixel_i)?;
        let bearing_j = unit_bearing(camera_j, pixel_j)?;
        bearing_corrs.push(
            BearingCorrespondence::try_new(bearing_i, bearing_j).map_err(|error| {
                SfmError::InvalidInput(format!("failed to form bearing correspondence: {error}"))
            })?,
        );
        if let (Some(pixel_corrs), Some((pinhole_i, pinhole_j))) =
            (&mut perspective_pixel_corrs, perspective)
        {
            let virtual_i = pinhole_i.project(&bearing_i).map_err(|error| {
                SfmError::InvalidInput(format!(
                    "failed to form perspective verification pixel: {error}"
                ))
            })?;
            let virtual_j = pinhole_j.project(&bearing_j).map_err(|error| {
                SfmError::InvalidInput(format!(
                    "failed to form perspective verification pixel: {error}"
                ))
            })?;
            pixel_corrs.push(PixelCorrespondence::new(virtual_i, virtual_j));
        }
    }
    let min_focal = perspective.map(|(pinhole_i, pinhole_j)| {
        pinhole_i
            .focal_x()
            .min(pinhole_i.focal_y())
            .min(pinhole_j.focal_x())
            .min(pinhole_j.focal_y())
    });
    Ok(PreparedCalibratedCorrespondences {
        ordered_match_indices,
        perspective_pixel_corrs,
        bearing_corrs,
        min_focal,
    })
}

fn perspective_pinhole(camera: &Camera) -> Option<&PinholeCamera> {
    match camera {
        Camera::Pinhole(pinhole) => Some(pinhole),
        Camera::BrownConrady(brown) => Some(brown.pinhole()),
        Camera::Fisheye(_) | Camera::Spherical(_) => None,
    }
}

pub(crate) fn unit_bearing(
    camera: &Camera,
    pixel: nalgebra::Point2<f64>,
) -> Result<nalgebra::Vector3<f64>> {
    camera
        .unproject(&pixel)
        .map_err(|error| {
            SfmError::InvalidInput(format!("failed to unproject verified pixel: {error}"))
        })?
        .try_normalize(1e-12)
        .ok_or_else(|| SfmError::InvalidInput("camera returned a degenerate bearing".to_owned()))
}

#[allow(clippy::too_many_arguments)]
fn finish_calibrated_verification(
    ctx: &VerificationContext,
    matches: &PairMatches,
    prepared: &PreparedCalibratedCorrespondences,
    estimate: CalibratedEstimate<'_>,
    min_inliers: usize,
    min_inlier_ratio: f64,
    policy: CalibratedPostprocessPolicy,
    threshold_px: PixelThreshold,
) -> Result<VerificationVerdict> {
    if estimate
        .inliers
        .iter()
        .any(|&index| index >= prepared.bearing_corrs.len())
    {
        return Err(SfmError::InvalidInput(format!(
            "{} returned an out-of-range inlier index",
            policy.estimator_name
        )));
    }
    let inlier_ratio = estimate.inliers.len() as f64 / matches.len() as f64;
    if estimate.inliers.len() < min_inliers {
        return Ok(rejected(
            ctx,
            matches,
            VerificationRejectReason::InsufficientInliers {
                required: min_inliers,
                actual: estimate.inliers.len(),
            },
            estimate.inliers.len(),
            format!("{} produced too few inliers", policy.estimator_name),
        ));
    }
    if inlier_ratio < min_inlier_ratio {
        return Ok(rejected(
            ctx,
            matches,
            VerificationRejectReason::LowInlierRatio,
            estimate.inliers.len(),
            format!(
                "{} inlier ratio is below the calibrated policy",
                policy.estimator_name
            ),
        ));
    }
    if let Some(pixel_corrs) = &prepared.perspective_pixel_corrs {
        if homography_dominates(pixel_corrs, estimate.inliers.len(), threshold_px, ctx.seed)? {
            return Ok(rejected(
                ctx,
                matches,
                VerificationRejectReason::HomographyDominant,
                estimate.inliers.len(),
                "homography support is indistinguishable from epipolar support",
            ));
        }
    } else if bearing_rotation_dominates(
        &prepared.bearing_corrs,
        estimate.inliers,
        policy.threshold.value(),
    ) {
        return Ok(rejected(
            ctx,
            matches,
            VerificationRejectReason::RotationOnlyDominant,
            estimate.inliers.len(),
            "one bearing rotation explains as much support as the epipolar model",
        ));
    }

    let inlier_bearings = estimate
        .inliers
        .iter()
        .map(|&index| prepared.bearing_corrs[index])
        .collect::<Vec<_>>();
    let Some(solver_pose) = estimate.model.decompose_with_cheirality(&inlier_bearings) else {
        return Ok(rejected(
            ctx,
            matches,
            VerificationRejectReason::FailedChirality,
            estimate.inliers.len(),
            "essential-matrix decomposition failed the cheirality check",
        ));
    };
    let mut inlier_mask = vec![false; matches.len()];
    for &index in estimate.inliers {
        let original_index = prepared.ordered_match_indices[index];
        inlier_mask[original_index] = true;
    }
    let residuals = estimate
        .inliers
        .iter()
        .map(|&index| {
            policy
                .residual
                .evaluate(estimate.model, &prepared.bearing_corrs[index])
        })
        .collect::<Vec<_>>();
    let evidence = ModelEvidence::try_new(
        InlierMask::new(inlier_mask),
        ErrorStats::try_from_values(&residuals, &[policy.threshold.value()])?,
        policy.threshold,
        policy.policy_id,
        estimate.iterations,
        TerminationReason::Converged,
        Vec::new(),
    )?;
    let model = GeometricModel::Essential {
        matrix: *estimate.model.matrix(),
        relative_pose: solver_pose,
        evidence,
    };
    model.validate_for_matches(matches)?;

    Ok(VerificationVerdict::Accepted {
        report: PairVerificationReport {
            pair: ctx.pair,
            input_match_count: matches.len(),
            inlier_count: estimate.inliers.len(),
            message: None,
        },
        model,
    })
}

fn reject_low_match_count(
    ctx: &VerificationContext,
    matches: &PairMatches,
) -> Option<VerificationVerdict> {
    (matches.len() < MIN_ESSENTIAL_MATCHES).then(|| {
        rejected(
            ctx,
            matches,
            VerificationRejectReason::LowMatchCount {
                required: MIN_ESSENTIAL_MATCHES,
                actual: matches.len(),
            },
            0,
            "not enough matches for essential-matrix estimation",
        )
    })
}

#[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
fn validate_min_inliers(min_inliers: usize) -> Result<()> {
    if min_inliers < 5 {
        return Err(SfmError::InvalidInput(
            "calibrated verification requires at least five inliers".to_owned(),
        ));
    }
    Ok(())
}

fn rejected(
    ctx: &VerificationContext,
    matches: &PairMatches,
    reason: VerificationRejectReason,
    inlier_count: usize,
    message: impl Into<String>,
) -> VerificationVerdict {
    VerificationVerdict::Rejected {
        reason,
        report: PairVerificationReport {
            pair: ctx.pair,
            input_match_count: matches.len(),
            inlier_count,
            message: Some(message.into()),
        },
    }
}

fn match_indices_by_quality(matches: &PairMatches) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..matches.len()).collect();
    indices.sort_unstable_by(|&lhs, &rhs| {
        let lhs_distance = matches.as_slice()[lhs].distance().unwrap_or(f32::INFINITY);
        let rhs_distance = matches.as_slice()[rhs].distance().unwrap_or(f32::INFINITY);
        lhs_distance
            .total_cmp(&rhs_distance)
            .then_with(|| lhs.cmp(&rhs))
    });
    indices
}

fn has_match_quality(matches: &PairMatches) -> bool {
    matches
        .as_slice()
        .iter()
        .any(|matched| matched.distance().is_some())
}

fn bearing_rotation_dominates(
    correspondences: &[BearingCorrespondence],
    essential_inliers: &[usize],
    angular_threshold: f64,
) -> bool {
    if essential_inliers.len() < 3 || !angular_threshold.is_finite() || angular_threshold <= 0.0 {
        return false;
    }
    let mut cross_covariance = Matrix3::zeros();
    for &index in essential_inliers {
        let Some(correspondence) = correspondences.get(index) else {
            return false;
        };
        cross_covariance += correspondence.bearing2() * correspondence.bearing1().transpose();
    }
    let svd = cross_covariance.svd(true, true);
    let (Some(u), Some(v_t)) = (svd.u, svd.v_t) else {
        return false;
    };
    let mut correction = Matrix3::identity();
    correction[(2, 2)] = (u * v_t).determinant().signum();
    let rotation = u * correction * v_t;
    if !rotation.iter().all(|value| value.is_finite()) || rotation.determinant() <= 0.0 {
        return false;
    }
    let rotation_support = correspondences
        .iter()
        .filter(|correspondence| {
            let predicted = rotation * correspondence.bearing1();
            predicted
                .cross(correspondence.bearing2())
                .norm()
                .atan2(predicted.dot(correspondence.bearing2()))
                <= angular_threshold
        })
        .count();
    rotation_support as f64 >= HOMOGRAPHY_DOMINANCE_RATIO * essential_inliers.len() as f64
}

fn homography_dominates(
    correspondences: &[PixelCorrespondence],
    essential_inliers: usize,
    threshold_px: PixelThreshold,
    seed: crate::RunSeed,
) -> Result<bool> {
    if correspondences.len() < 4 || essential_inliers == 0 {
        return Ok(false);
    }

    // Homography residual combines forward and backward transfer distances.
    // Multiplying by sqrt(2) gives each direction the same nominal pixel bound
    // as the calibrated epipolar test.
    let homography_threshold =
        PixelThreshold::try_new(threshold_px.value() * std::f64::consts::SQRT_2)
            .map_err(|error| SfmError::InvalidInput(error.to_string()))?;
    let options = RobustOptions::default()
        .with_confidence(0.99)
        .with_max_draws(HOMOGRAPHY_MAX_ITERATIONS)
        .with_seed(seed.derive("sfm/geometric-verification/homography-dominance/v1", &[]));
    let result = estimate_homography_genransac(correspondences, homography_threshold, &options);
    match result {
        Ok(result) => Ok((result.inliers().len() as f64)
            >= HOMOGRAPHY_DOMINANCE_RATIO * essential_inliers as f64),
        Err(RobustError::NoValidModel { .. } | RobustError::AllSamplesDegenerate { .. }) => {
            Ok(false)
        }
        Err(error) => Err(SfmError::InvalidInput(format!(
            "homography degeneracy check failed: {error}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Point2, Rotation3, Vector3};
    use openmvg_core::camera::{PinholeCamera, SphericalCamera};

    fn feature_fixture(pixels: &[Point2<f64>]) -> ExtractedFeatures {
        let keypoints = pixels
            .iter()
            .map(|pixel| {
                openmvg_features::model::Keypoint::try_new(pixel.x as f32, pixel.y as f32, 1.0, 0.0)
                    .unwrap()
            })
            .collect::<Vec<_>>();
        ExtractedFeatures::try_new(
            keypoints,
            openmvg_features::model::DescriptorSet::try_f32(vec![0.0; pixels.len()], 1).unwrap(),
        )
        .unwrap()
    }

    fn identity_matches(count: usize) -> PairMatches {
        PairMatches::try_new(
            crate::ViewPair::new(crate::ViewId::new(0), crate::ViewId::new(1)),
            (0..count)
                .map(|index| {
                    openmvg_features::model::FeatureMatch::try_new(
                        crate::FeatureId::new(index),
                        crate::FeatureId::new(index),
                        Some(index as f32),
                    )
                    .unwrap()
                })
                .collect(),
        )
        .unwrap()
    }

    fn planar_correspondences() -> Vec<PixelCorrespondence> {
        (0..8)
            .flat_map(|row| {
                (0..8).map(move |column| {
                    let first = Point2::new(40.0 + column as f64 * 60.0, 30.0 + row as f64 * 45.0);
                    let second = Point2::new(first.x + 12.0 + 0.002 * first.y, first.y - 7.0);
                    PixelCorrespondence::new(first, second)
                })
            })
            .collect()
    }

    fn varying_depth_correspondences() -> Vec<PixelCorrespondence> {
        let focal = 800.0;
        (0..8)
            .flat_map(|row| {
                (0..8).map(move |column| {
                    let x = -1.2 + column as f64 * 0.35;
                    let y = -0.9 + row as f64 * 0.28;
                    let z = 3.0 + ((row * 5 + column * 3) % 11) as f64 * 0.45;
                    let first = Point2::new(focal * x / z + 320.0, focal * y / z + 240.0);
                    let second = Point2::new(focal * (x - 0.5) / z + 320.0, focal * y / z + 240.0);
                    PixelCorrespondence::new(first, second)
                })
            })
            .collect()
    }

    fn pure_rotation_correspondences() -> Vec<PixelCorrespondence> {
        let focal = 800.0;
        let rotation = Rotation3::from_axis_angle(&Vector3::y_axis(), 0.12);
        (0..8)
            .flat_map(|row| {
                (0..8).map(move |column| {
                    let first = Point2::new(80.0 + column as f64 * 65.0, 60.0 + row as f64 * 45.0);
                    let bearing =
                        Vector3::new((first.x - 320.0) / focal, (first.y - 240.0) / focal, 1.0);
                    let rotated = rotation * bearing;
                    let second = Point2::new(
                        focal * rotated.x / rotated.z + 320.0,
                        focal * rotated.y / rotated.z + 240.0,
                    );
                    PixelCorrespondence::new(first, second)
                })
            })
            .collect()
    }

    #[test]
    fn canonical_policy_has_stable_explicit_units() {
        let verifier = CalibratedVerifier::default();
        assert_eq!(verifier.policy_id(), CALIBRATED_VERIFIER_POLICY_ID);
        assert_eq!(verifier.threshold_px().value(), 4.0);
        assert!(PixelThreshold::try_new(f64::NAN).is_err());
        assert!(CalibratedVerifier::try_new(PixelThreshold::FOUR_PIXELS, 4).is_err());
    }

    #[test]
    fn missing_match_distances_do_not_claim_a_quality_order() {
        let matches = PairMatches::try_new(
            crate::ViewPair::new(crate::ViewId::new(0), crate::ViewId::new(1)),
            (0..8)
                .map(|index| {
                    openmvg_features::model::FeatureMatch::try_new(
                        crate::FeatureId::new(index),
                        crate::FeatureId::new(index),
                        None,
                    )
                    .unwrap()
                })
                .collect(),
        )
        .unwrap();

        assert!(!has_match_quality(&matches));
        assert!(has_match_quality(&identity_matches(8)));
    }

    #[test]
    fn pixel_to_angular_conversion_uses_radians() {
        let pixels = PixelThreshold::try_new(4.0).unwrap();
        let angular = AngularThreshold::from_pixels(pixels, 800.0).unwrap();
        assert!((angular.value() - (4.0_f64 / 800.0).atan()).abs() < 1e-15);
    }

    #[test]
    fn planar_support_is_detected_deterministically() {
        let correspondences = planar_correspondences();
        let threshold = PixelThreshold::try_new(2.0).unwrap();
        let first = homography_dominates(
            &correspondences,
            correspondences.len(),
            threshold,
            17.into(),
        )
        .unwrap();
        let second = homography_dominates(
            &correspondences,
            correspondences.len(),
            threshold,
            17.into(),
        )
        .unwrap();
        assert!(first);
        assert_eq!(first, second);
    }

    #[test]
    fn varying_depth_is_not_homography_dominant() {
        let correspondences = varying_depth_correspondences();
        assert!(
            !homography_dominates(
                &correspondences,
                correspondences.len(),
                PixelThreshold::try_new(2.0).unwrap(),
                17.into(),
            )
            .unwrap()
        );
    }

    #[test]
    fn pure_rotation_is_homography_dominant() {
        let correspondences = pure_rotation_correspondences();
        assert!(
            homography_dominates(
                &correspondences,
                correspondences.len(),
                PixelThreshold::try_new(2.0).unwrap(),
                23.into(),
            )
            .unwrap()
        );
    }

    #[test]
    fn rescued_candidate_cannot_bypass_planar_homography_guard() {
        let correspondences = planar_correspondences();
        let pixels_i = correspondences
            .iter()
            .map(|correspondence| correspondence.point1)
            .collect::<Vec<_>>();
        let pixels_j = correspondences
            .iter()
            .map(|correspondence| correspondence.point2)
            .collect::<Vec<_>>();
        let features_i = feature_fixture(&pixels_i);
        let features_j = feature_fixture(&pixels_j);
        let matches = identity_matches(correspondences.len());
        let camera = Camera::Pinhole(PinholeCamera::try_new(800.0, 800.0, 320.0, 240.0).unwrap());
        let prepared = prepare_calibrated_correspondences(
            &features_i,
            &features_j,
            &matches,
            &camera,
            &camera,
        )
        .unwrap();
        let model =
            EssentialMatrix::try_new(Matrix3::new(0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0))
                .unwrap();
        let inliers = (0..correspondences.len()).collect::<Vec<_>>();
        let outcome = finish_calibrated_verification(
            &VerificationContext {
                pair: crate::ViewPair::new(crate::ViewId::new(0), crate::ViewId::new(1)),
                view_i: crate::ViewId::new(0),
                view_j: crate::ViewId::new(1),
                seed: 23.into(),
            },
            &matches,
            &prepared,
            CalibratedEstimate {
                model: &model,
                inliers: &inliers,
                iterations: 1,
            },
            16,
            0.1,
            CalibratedPostprocessPolicy {
                policy_id: CALIBRATED_VERIFIER_POLICY_ID,
                estimator_name: "GenRANSAC Essential test",
                threshold: AngularThreshold::try_new(2.0 / 800.0).unwrap().into(),
                residual: EssentialResidual::Angular,
            },
            PixelThreshold::try_new(2.0).unwrap(),
        )
        .unwrap();

        assert!(matches!(
            outcome,
            VerificationVerdict::Rejected {
                reason: VerificationRejectReason::HomographyDominant,
                ..
            }
        ));
    }

    #[test]
    fn rescued_candidate_cannot_bypass_rotation_only_guard() {
        let rotation = Rotation3::from_euler_angles(0.11, -0.06, 0.04);
        let bearing_corrs = (0..32)
            .map(|index| {
                let value = index as f64;
                let first = Vector3::new(
                    (0.71 * value + 0.2).sin(),
                    (1.13 * value - 0.4).cos(),
                    -0.7 + 0.05 * value,
                )
                .normalize();
                BearingCorrespondence::try_new(first, rotation * first).unwrap()
            })
            .collect::<Vec<_>>();
        let matches = identity_matches(bearing_corrs.len());
        let prepared = PreparedCalibratedCorrespondences {
            ordered_match_indices: (0..bearing_corrs.len()).collect(),
            perspective_pixel_corrs: None,
            bearing_corrs,
            min_focal: None,
        };
        let model =
            EssentialMatrix::try_new(Matrix3::new(0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0))
                .unwrap();
        let inliers = (0..prepared.bearing_corrs.len()).collect::<Vec<_>>();
        let outcome = finish_calibrated_verification(
            &VerificationContext {
                pair: crate::ViewPair::new(crate::ViewId::new(0), crate::ViewId::new(1)),
                view_i: crate::ViewId::new(0),
                view_j: crate::ViewId::new(1),
                seed: 29.into(),
            },
            &matches,
            &prepared,
            CalibratedEstimate {
                model: &model,
                inliers: &inliers,
                iterations: 1,
            },
            16,
            0.1,
            CalibratedPostprocessPolicy {
                policy_id: CALIBRATED_ANGULAR_VERIFIER_POLICY_ID,
                estimator_name: "GenRANSAC Essential test",
                threshold: AngularThreshold::try_new(1e-8).unwrap().into(),
                residual: EssentialResidual::Angular,
            },
            PixelThreshold::FOUR_PIXELS,
        )
        .unwrap();

        assert!(matches!(
            outcome,
            VerificationVerdict::Rejected {
                reason: VerificationRejectReason::RotationOnlyDominant,
                ..
            }
        ));
    }

    #[test]
    fn spherical_seam_and_poles_prepare_finite_unit_bearings() {
        let pixels = [
            Point2::new(-0.375, 255.5),
            Point2::new(1023.375, 255.5),
            Point2::new(511.5, -0.5),
            Point2::new(511.5, 511.375),
        ];
        let features = feature_fixture(&pixels);
        let matches = identity_matches(pixels.len());
        let camera = Camera::Spherical(SphericalCamera::try_new(1024).unwrap());

        let prepared =
            prepare_calibrated_correspondences(&features, &features, &matches, &camera, &camera)
                .unwrap();

        assert!(prepared.min_focal.is_none());
        assert!(prepared.perspective_pixel_corrs.is_none());
        assert!(prepared.bearing_corrs.iter().all(|correspondence| {
            correspondence
                .bearing1()
                .iter()
                .all(|value| value.is_finite())
                && correspondence
                    .bearing2()
                    .iter()
                    .all(|value| value.is_finite())
                && (correspondence.bearing1().norm() - 1.0).abs() < 1e-12
                && (correspondence.bearing2().norm() - 1.0).abs() < 1e-12
        }));
        assert!(
            prepared.bearing_corrs[0]
                .bearing1()
                .dot(prepared.bearing_corrs[1].bearing1())
                > 0.999_99
        );
    }

    #[test]
    fn mixed_pinhole_spherical_pair_uses_generalized_route() {
        let pixels = [Point2::new(320.0, 240.0), Point2::new(400.0, 200.0)];
        let features = feature_fixture(&pixels);
        let matches = identity_matches(pixels.len());
        let pinhole = Camera::Pinhole(PinholeCamera::try_new(800.0, 800.0, 320.0, 240.0).unwrap());
        let spherical = Camera::Spherical(SphericalCamera::try_new(1024).unwrap());

        let prepared = prepare_calibrated_correspondences(
            &features, &features, &matches, &pinhole, &spherical,
        )
        .unwrap();

        assert!(prepared.min_focal.is_none());
        assert!(prepared.perspective_pixel_corrs.is_none());
    }

    #[test]
    fn bearing_rotation_dominance_is_permutation_invariant() {
        let rotation = Rotation3::from_euler_angles(0.2, -0.1, 0.07);
        let correspondences = (0..16)
            .map(|index| {
                let value = index as f64;
                let first = Vector3::new(
                    (0.7 * value).sin(),
                    (1.1 * value).cos(),
                    -0.8 + 0.11 * value,
                )
                .normalize();
                BearingCorrespondence::try_new(first, rotation * first).unwrap()
            })
            .collect::<Vec<_>>();
        let inliers = (0..correspondences.len()).collect::<Vec<_>>();
        assert!(bearing_rotation_dominates(&correspondences, &inliers, 1e-8));

        let mut reversed = correspondences.clone();
        reversed.reverse();
        assert!(bearing_rotation_dominates(&reversed, &inliers, 1e-8));
    }

    #[test]
    fn angular_verification_reports_angular_policy_and_units() {
        let camera = Camera::Spherical(SphericalCamera::try_new(1024).unwrap());
        let rotation = Rotation3::from_euler_angles(0.07, -0.11, 0.04);
        let center = Vector3::new(1.1, -0.25, 0.35);
        let points = (0..40)
            .map(|index| {
                let value = index as f64;
                Vector3::new(
                    2.1 * (0.63 * value).sin(),
                    1.7 * (1.07 * value).cos(),
                    -4.0 + (index % 11) as f64 * 0.85,
                )
            })
            .collect::<Vec<_>>();
        let pixels_i = points
            .iter()
            .map(|point| camera.project(point).unwrap())
            .collect::<Vec<_>>();
        let pixels_j = points
            .iter()
            .map(|point| camera.project(&(rotation * (point - center))).unwrap())
            .collect::<Vec<_>>();
        let features_i = feature_fixture(&pixels_i);
        let features_j = feature_fixture(&pixels_j);
        let matches = identity_matches(points.len());
        let verifier = CalibratedVerifier::try_new(PixelThreshold::FOUR_PIXELS, 16)
            .unwrap()
            .with_angular_threshold(AngularThreshold::try_new(1e-4).unwrap());

        let outcome = verifier
            .verify_pair(
                &VerificationContext {
                    pair: crate::ViewPair::new(crate::ViewId::new(0), crate::ViewId::new(1)),
                    view_i: crate::ViewId::new(0),
                    view_j: crate::ViewId::new(1),
                    seed: 47.into(),
                },
                &features_i,
                &features_j,
                &matches,
                &camera,
                &camera,
            )
            .unwrap();
        let VerificationVerdict::Accepted { model, .. } = outcome else {
            panic!("exact generalized geometry should verify: {outcome:?}");
        };
        let diagnostics = model.evidence().diagnostics.as_ref().unwrap();

        assert_eq!(
            diagnostics.solver_name,
            CALIBRATED_ANGULAR_VERIFIER_POLICY_ID
        );
        assert_eq!(
            diagnostics.inlier_threshold.residual_units(),
            crate::two_view::ResidualUnit::AngularRadians
        );
        assert!((diagnostics.inlier_threshold.value() - 1e-4).abs() < 1e-15);
    }
}
