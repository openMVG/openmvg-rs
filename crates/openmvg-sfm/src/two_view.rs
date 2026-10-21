// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Two-view values shared by matching and verification: putative matches, inlier masks, the
//! verified geometric model with its evidence, and the verification reports.

use nalgebra::{Matrix3, Rotation3};
use openmvg_core::model::RelativePose;
use openmvg_features::model::{FeatureMatch, MatchSet};

use crate::workspace::quality::ErrorStats;
use crate::{FeatureId, Result, SfmError, ViewPair};

/// Putative feature matches of one view pair.
///
/// Each match pairs a feature of `pair.first()` with a feature of `pair.second()`.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PairMatches {
    pair: ViewPair,
    matches: MatchSet,
}

impl PairMatches {
    /// Wrap the matches of `pair`, sorted by feature pair.
    ///
    /// Fails when two matches share the same feature pair or a distance is negative or non-finite.
    pub fn try_new(pair: ViewPair, matches: Vec<FeatureMatch>) -> Result<Self> {
        Ok(Self {
            pair,
            matches: MatchSet::try_new(matches)?,
        })
    }

    /// The view pair the matches belong to.
    pub fn pair(&self) -> ViewPair {
        self.pair
    }

    /// The matches, sorted by `(first-view feature, second-view feature)`.
    pub fn as_slice(&self) -> &[FeatureMatch] {
        self.matches.as_slice()
    }

    /// Number of matches.
    pub fn len(&self) -> usize {
        self.matches.len()
    }

    /// Whether the pair has no matches.
    pub fn is_empty(&self) -> bool {
        self.matches.is_empty()
    }

    pub(crate) fn validate_for_pair(
        &self,
        expected_pair: ViewPair,
        feature_count_i: usize,
        feature_count_j: usize,
    ) -> Result<()> {
        if self.pair != expected_pair {
            return Err(SfmError::InvalidInput(format!(
                "matcher returned pair {} while processing {expected_pair}",
                self.pair
            )));
        }
        self.matches.validate()?;
        for matched in self.matches.as_slice() {
            if matched.first().index() >= feature_count_i {
                return Err(SfmError::InvalidId {
                    kind: FeatureId::kind(),
                    index: matched.first().index(),
                });
            }
            if matched.second().index() >= feature_count_j {
                return Err(SfmError::InvalidId {
                    kind: FeatureId::kind(),
                    index: matched.second().index(),
                });
            }
        }
        Ok(())
    }
}

/// Per-match inlier flags of a verified model, in the order of the pair's matches.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct InlierMask {
    bits: Vec<bool>,
}

#[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
impl InlierMask {
    /// Wrap one flag per match; `true` marks an inlier.
    pub fn new(bits: Vec<bool>) -> Self {
        Self { bits }
    }

    /// A mask of `len` matches that are all inliers.
    pub fn all_inliers(len: usize) -> Self {
        Self {
            bits: vec![true; len],
        }
    }

    /// Number of flags, equal to the pair's match count.
    pub fn len(&self) -> usize {
        self.bits.len()
    }

    /// Whether the mask covers no match.
    pub fn is_empty(&self) -> bool {
        self.bits.is_empty()
    }

    /// Number of `true` flags.
    pub fn inlier_count(&self) -> usize {
        self.bits.iter().filter(|is_inlier| **is_inlier).count()
    }

    /// The flags, one per match.
    pub fn as_slice(&self) -> &[bool] {
        &self.bits
    }
}

/// Physical unit of a residual or threshold value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
#[serde(rename_all = "snake_case")]
pub enum ResidualUnit {
    /// Image-plane distance in pixels.
    Pixels,
    /// Distance on the normalized image plane (pixels divided by the focal length).
    NormalizedImage,
    /// Angle in radians, for example between bearing vectors.
    AngularRadians,
    /// Angle in degrees; evidence read in this unit is converted to radians.
    AngularDegrees,
}

/// Why the robust estimator behind a [`ModelEvidence`] stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TerminationReason {
    /// The estimator reached its confidence target and returned a model.
    Converged,
    /// The estimator stopped at its iteration limit.
    MaxIterations,
    /// Every distinct minimal sample was tried.
    ExhaustedSamples,
    /// The data supports no unique model.
    Degenerate,
    /// The estimator failed for another reason.
    Failed,
}

/// A degenerate configuration detected in a pair's correspondences.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GeometryDegeneracy {
    /// Several correspondences repeat the same feature pairing.
    DuplicateCorrespondences,
    /// The baseline is too short for the viewing rays to constrain depth.
    LowParallax,
    /// The points lie close to one plane, so a homography explains them.
    PlanarScene,
    /// The views differ by a rotation only, so the translation is unobservable.
    PureRotation,
    /// The solver ran into ill-conditioned numerics.
    NumericalInstability,
}

/// Inlier threshold whose physical unit cannot disagree with its value.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
#[serde(tag = "units", content = "value", rename_all = "snake_case")]
pub enum EvidenceThreshold {
    /// Pixel-domain residual threshold.
    Pixels(openmvg_core::PixelThreshold),
    /// Normalized-image residual threshold.
    NormalizedImage(openmvg_core::NormalizedImageThreshold),
    /// Angular residual threshold, stored in radians.
    Radians(openmvg_core::AngularThreshold),
}

impl EvidenceThreshold {
    /// Return the threshold in the unit named by its variant.
    #[must_use]
    pub const fn value(self) -> f64 {
        match self {
            Self::Pixels(value) => value.value(),
            Self::NormalizedImage(value) => value.value(),
            Self::Radians(value) => value.value(),
        }
    }

    /// Return the residual unit paired with this threshold.
    // `ResidualUnit` is internal; the variant already names the unit for stable callers.
    #[cfg(feature = "unstable-internals")]
    #[must_use]
    pub const fn units(self) -> ResidualUnit {
        self.residual_units()
    }

    /// Residual unit paired with this threshold.
    pub(crate) const fn residual_units(self) -> ResidualUnit {
        match self {
            Self::Pixels(_) => ResidualUnit::Pixels,
            Self::NormalizedImage(_) => ResidualUnit::NormalizedImage,
            Self::Radians(_) => ResidualUnit::AngularRadians,
        }
    }
}

impl From<openmvg_core::PixelThreshold> for EvidenceThreshold {
    fn from(value: openmvg_core::PixelThreshold) -> Self {
        Self::Pixels(value)
    }
}

impl From<openmvg_core::NormalizedImageThreshold> for EvidenceThreshold {
    fn from(value: openmvg_core::NormalizedImageThreshold) -> Self {
        Self::NormalizedImage(value)
    }
}

impl From<openmvg_core::AngularThreshold> for EvidenceThreshold {
    fn from(value: openmvg_core::AngularThreshold) -> Self {
        Self::Radians(value)
    }
}

/// Diagnostics of the robust estimator that produced a two-view model.
#[derive(Debug, Clone, PartialEq)]
pub struct SolverDiagnostics {
    /// Residual distribution over the accepted inliers.
    pub residual_stats: ErrorStats,
    /// Threshold, with its unit, that separated inliers from outliers.
    pub inlier_threshold: EvidenceThreshold,
    /// Name of the estimator or policy that produced the model; never blank.
    pub solver_name: String,
    /// Robust-estimation iterations used.
    pub iterations: usize,
    /// Why the estimator stopped.
    pub termination_reason: TerminationReason,
    /// Degenerate configurations detected; empty when none was flagged.
    pub degeneracy: Vec<GeometryDegeneracy>,
}

impl SolverDiagnostics {
    /// Check the residual statistics are valid and the solver name is not blank.
    fn validate(&self) -> Result<()> {
        self.residual_stats.validate()?;
        if self.solver_name.trim().is_empty() {
            return Err(SfmError::InvalidInput(
                "solver name must not be empty".to_owned(),
            ));
        }
        Ok(())
    }

    /// Check the diagnostics are valid and count one residual per inlier.
    fn validate_for_inlier_count(&self, inlier_count: usize) -> Result<()> {
        self.validate()?;
        if self.residual_stats.count != inlier_count {
            return Err(SfmError::InvalidInput(format!(
                "residual count {} does not match inlier count {inlier_count}",
                self.residual_stats.count
            )));
        }
        Ok(())
    }
}

impl serde::Serialize for SolverDiagnostics {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;

        // Write the threshold and its unit as two plain fields so that non-self-describing
        // formats such as the binary checkpoints can read them back.
        let mut state = serializer.serialize_struct("SolverDiagnostics", 7)?;
        state.serialize_field("residual_stats", &self.residual_stats)?;
        state.serialize_field("inlier_threshold", &self.inlier_threshold.value())?;
        state.serialize_field("threshold_units", &self.inlier_threshold.residual_units())?;
        state.serialize_field("solver_name", &self.solver_name)?;
        state.serialize_field("iterations", &self.iterations)?;
        state.serialize_field("termination_reason", &self.termination_reason)?;
        state.serialize_field("degeneracy", &self.degeneracy)?;
        state.end()
    }
}

#[derive(serde::Deserialize)]
struct SolverDiagnosticsWire {
    residual_stats: ErrorStats,
    inlier_threshold: f64,
    threshold_units: ResidualUnit,
    solver_name: String,
    iterations: usize,
    termination_reason: TerminationReason,
    degeneracy: Vec<GeometryDegeneracy>,
}

impl<'de> serde::Deserialize<'de> for SolverDiagnostics {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let wire = SolverDiagnosticsWire::deserialize(deserializer)?;
        let inlier_threshold = match wire.threshold_units {
            ResidualUnit::Pixels => openmvg_core::PixelThreshold::try_new(wire.inlier_threshold)
                .map(EvidenceThreshold::Pixels),
            ResidualUnit::NormalizedImage => {
                openmvg_core::NormalizedImageThreshold::try_new(wire.inlier_threshold)
                    .map(EvidenceThreshold::NormalizedImage)
            }
            ResidualUnit::AngularRadians => {
                openmvg_core::AngularThreshold::try_new(wire.inlier_threshold)
                    .map(EvidenceThreshold::Radians)
            }
            ResidualUnit::AngularDegrees => {
                openmvg_core::AngularThreshold::from_degrees(wire.inlier_threshold)
                    .map(EvidenceThreshold::Radians)
            }
        }
        .map_err(serde::de::Error::custom)?;
        let diagnostics = Self {
            residual_stats: wire.residual_stats,
            inlier_threshold,
            solver_name: wire.solver_name,
            iterations: wire.iterations,
            termination_reason: wire.termination_reason,
            degeneracy: wire.degeneracy,
        };
        diagnostics.validate().map_err(serde::de::Error::custom)?;
        Ok(diagnostics)
    }
}

/// Inlier support behind an accepted two-view model.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct ModelEvidence {
    /// Inlier flags, one per match of the pair.
    pub inliers: InlierMask,
    /// Number of inliers, equal to `inliers.inlier_count()`.
    pub inlier_count: usize,
    /// Inlier fraction of the matches in `[0, 1]`; `0` for an empty mask.
    pub inlier_ratio: f64,
    /// Diagnostics of the estimator that produced the model; `None` when the model was
    /// verified outside this crate, for example in a view graph read from a file.
    pub diagnostics: Option<SolverDiagnostics>,
}

#[derive(serde::Deserialize)]
struct ModelEvidenceWire {
    inliers: InlierMask,
    inlier_count: usize,
    inlier_ratio: f64,
    diagnostics: Option<SolverDiagnostics>,
}

impl<'de> serde::Deserialize<'de> for ModelEvidence {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let wire = ModelEvidenceWire::deserialize(deserializer)?;
        let evidence = Self {
            inliers: wire.inliers,
            inlier_count: wire.inlier_count,
            inlier_ratio: wire.inlier_ratio,
            diagnostics: wire.diagnostics,
        };
        evidence.validate().map_err(serde::de::Error::custom)?;
        Ok(evidence)
    }
}

impl ModelEvidence {
    /// Build evidence from an estimator run, deriving the inlier count and ratio from `inliers`.
    ///
    /// Fails when `residual_stats` is invalid or does not count exactly one residual per inlier,
    /// or when `solver_name` is blank.
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        inliers: InlierMask,
        residual_stats: ErrorStats,
        inlier_threshold: EvidenceThreshold,
        solver_name: impl Into<String>,
        iterations: usize,
        termination_reason: TerminationReason,
        degeneracy: Vec<GeometryDegeneracy>,
    ) -> Result<Self> {
        let diagnostics = SolverDiagnostics {
            residual_stats,
            inlier_threshold,
            solver_name: solver_name.into(),
            iterations,
            termination_reason,
            degeneracy,
        };
        diagnostics.validate_for_inlier_count(inliers.inlier_count())?;
        Ok(Self {
            diagnostics: Some(diagnostics),
            ..Self::from_inliers(inliers)
        })
    }

    /// Build evidence without solver diagnostics, for a model verified outside this crate.
    #[must_use]
    pub fn from_inliers(inliers: InlierMask) -> Self {
        let inlier_count = inliers.inlier_count();
        let inlier_ratio = if inliers.is_empty() {
            0.0
        } else {
            inlier_count as f64 / inliers.len() as f64
        };
        Self {
            inliers,
            inlier_count,
            inlier_ratio,
            diagnostics: None,
        }
    }

    /// Check the evidence is internally consistent and has one inlier flag per match.
    pub fn validate_for_matches(&self, matches: &PairMatches) -> Result<()> {
        self.validate()?;
        if self.inliers.len() != matches.len() {
            return Err(SfmError::InvalidInput(format!(
                "inlier mask length {} does not match match count {}",
                self.inliers.len(),
                matches.len()
            )));
        }
        Ok(())
    }

    /// Check the inlier count and ratio agree with the mask, and that any diagnostics are valid
    /// and count one residual per inlier.
    pub(crate) fn validate(&self) -> Result<()> {
        let counted_inliers = self.inliers.inlier_count();
        if self.inlier_count != counted_inliers {
            return Err(SfmError::InvalidInput(format!(
                "model evidence reports {} inliers, but its mask contains {counted_inliers}",
                self.inlier_count
            )));
        }
        let expected_ratio = if self.inliers.is_empty() {
            0.0
        } else {
            counted_inliers as f64 / self.inliers.len() as f64
        };
        if !self.inlier_ratio.is_finite()
            || (self.inlier_ratio - expected_ratio).abs() > f64::EPSILON
        {
            return Err(SfmError::InvalidInput(format!(
                "model evidence inlier ratio {} does not match mask ratio {expected_ratio}",
                self.inlier_ratio
            )));
        }
        if let Some(diagnostics) = &self.diagnostics {
            diagnostics.validate_for_inlier_count(counted_inliers)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod model_evidence_tests {
    use super::*;
    use crate::ViewId;

    #[test]
    fn evidence_requires_one_residual_per_inlier_and_a_solver_name() {
        let mask = InlierMask::new(vec![true, false, true]);
        let wrong_count = ModelEvidence::try_new(
            mask.clone(),
            ErrorStats::from_values(&[0.0], &[1.0]),
            openmvg_core::PixelThreshold::ONE_PIXEL.into(),
            "solver",
            1,
            TerminationReason::Converged,
            Vec::new(),
        );
        assert!(wrong_count.is_err());

        let unnamed = ModelEvidence::try_new(
            mask,
            ErrorStats::from_values(&[0.0, 0.0], &[1.0]),
            openmvg_core::PixelThreshold::ONE_PIXEL.into(),
            " ",
            1,
            TerminationReason::Converged,
            Vec::new(),
        );
        assert!(unnamed.is_err());
    }

    #[test]
    fn evidence_revalidates_public_fields_at_the_backend_boundary() {
        let matches =
            PairMatches::try_new(ViewPair::new(ViewId::new(0), ViewId::new(1)), Vec::new())
                .unwrap();
        let mut evidence = ModelEvidence::try_new(
            InlierMask::new(Vec::new()),
            ErrorStats::from_values(&[], &[]),
            openmvg_core::PixelThreshold::ONE_PIXEL.into(),
            "solver",
            1,
            TerminationReason::Converged,
            Vec::new(),
        )
        .unwrap();
        evidence.inlier_count = 1;

        assert!(evidence.validate_for_matches(&matches).is_err());
    }

    #[test]
    fn typed_threshold_round_trips_with_its_unit() {
        let evidence = ModelEvidence::try_new(
            InlierMask::all_inliers(1),
            ErrorStats::from_values(&[0.25], &[1.0]),
            openmvg_core::PixelThreshold::ONE_PIXEL.into(),
            "solver",
            1,
            TerminationReason::Converged,
            Vec::new(),
        )
        .unwrap();

        let json = serde_json::to_value(&evidence).unwrap();
        assert_eq!(json["diagnostics"]["inlier_threshold"], 1.0);
        assert_eq!(json["diagnostics"]["threshold_units"], "pixels");
        assert_eq!(
            serde_json::from_value::<ModelEvidence>(json).unwrap(),
            evidence
        );
    }

    #[test]
    fn evidence_without_diagnostics_derives_its_counts_and_round_trips() {
        let evidence = ModelEvidence::from_inliers(InlierMask::new(vec![true, false, true, true]));

        assert_eq!(evidence.inlier_count, 3);
        assert_eq!(evidence.inlier_ratio, 0.75);
        assert_eq!(evidence.diagnostics, None);
        assert_eq!(
            ModelEvidence::from_inliers(InlierMask::new(Vec::new())).inlier_ratio,
            0.0
        );
        let json = serde_json::to_value(&evidence).unwrap();
        assert!(json["diagnostics"].is_null());
        assert_eq!(
            serde_json::from_value::<ModelEvidence>(json).unwrap(),
            evidence
        );
    }

    #[test]
    fn deserialization_rejects_inconsistent_evidence() {
        let evidence = ModelEvidence::try_new(
            InlierMask::new(vec![true, false]),
            ErrorStats::from_values(&[0.25], &[1.0]),
            openmvg_core::PixelThreshold::ONE_PIXEL.into(),
            "solver",
            1,
            TerminationReason::Converged,
            Vec::new(),
        )
        .unwrap();
        let json = serde_json::to_value(&evidence).unwrap();
        let reject = |path: &[&str], value: serde_json::Value| {
            let mut tampered = json.clone();
            let mut slot = &mut tampered;
            for key in path {
                slot = &mut slot[*key];
            }
            *slot = value;
            assert!(
                serde_json::from_value::<ModelEvidence>(tampered).is_err(),
                "accepted {path:?}"
            );
        };

        reject(&["inlier_count"], 2.into());
        reject(&["inlier_ratio"], 1.0.into());
        reject(&["diagnostics", "solver_name"], " ".into());
        reject(&["diagnostics", "residual_stats", "count"], 2.into());
        reject(&["diagnostics", "inlier_threshold"], (-1.0).into());
    }
}

/// Verified two-view model of a pair, from its first view (source) to its second (target).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GeometricModel {
    /// Calibrated epipolar geometry with a decomposed relative pose.
    Essential {
        /// Essential matrix `E` with `x_second^T E x_first = 0` on bearing vectors.
        matrix: Matrix3<f64>,
        /// Relative pose decomposed from `E` with the cheirality check; unit translation.
        relative_pose: RelativePose,
        /// Inlier support behind the model.
        evidence: ModelEvidence,
    },
    /// Uncalibrated epipolar geometry.
    Fundamental {
        /// Fundamental matrix `F` with `x_second^T F x_first = 0` on pixel coordinates.
        matrix: Matrix3<f64>,
        /// Relative pose recovered from `F`; `None` when no pose was recovered.
        relative_pose: Option<RelativePose>,
        /// Inlier support behind the model.
        evidence: ModelEvidence,
    },
    /// Plane-induced or rotation-only homography.
    Homography {
        /// Homography `H` mapping first-view points to second-view points.
        matrix: Matrix3<f64>,
        /// Relative pose decomposed from `H`; `None` when no pose was recovered.
        relative_pose: Option<RelativePose>,
        /// Inlier support behind the model.
        evidence: ModelEvidence,
    },
    /// Verified relative rotation with no translation claim. Global rotation
    /// averaging may consume it; metric and incremental stages cannot.
    RotationOnly {
        /// Rotation from the first view's camera frame to the second's.
        rotation: Rotation3<f64>,
        /// Inlier support behind the model.
        evidence: ModelEvidence,
    },
}

impl GeometricModel {
    /// Inlier support behind the model, whatever its kind.
    pub fn evidence(&self) -> &ModelEvidence {
        match self {
            Self::Essential { evidence, .. }
            | Self::Fundamental { evidence, .. }
            | Self::Homography { evidence, .. }
            | Self::RotationOnly { evidence, .. } => evidence,
        }
    }

    /// The relative pose; `None` for rotation-only models and models without a recovered pose.
    pub fn relative_pose(&self) -> Option<&RelativePose> {
        match self {
            Self::Essential { relative_pose, .. } => Some(relative_pose),
            Self::Fundamental { relative_pose, .. } | Self::Homography { relative_pose, .. } => {
                relative_pose.as_ref()
            }
            Self::RotationOnly { .. } => None,
        }
    }

    /// Return every relative-rotation constraint, including explicitly
    /// rotation-only evidence.
    pub fn relative_rotation(&self) -> Option<&Rotation3<f64>> {
        match self {
            Self::Essential { relative_pose, .. } => {
                Some(relative_pose.rotation_source_to_target())
            }
            Self::Fundamental { relative_pose, .. } | Self::Homography { relative_pose, .. } => {
                relative_pose
                    .as_ref()
                    .map(RelativePose::rotation_source_to_target)
            }
            Self::RotationOnly { rotation, .. } => Some(rotation),
        }
    }

    /// Check the evidence against `matches` and that the matrix and pose or rotation are finite
    /// and valid.
    pub fn validate_for_matches(&self, matches: &PairMatches) -> Result<()> {
        self.evidence().validate_for_matches(matches)?;
        match self {
            Self::Essential {
                matrix,
                relative_pose,
                ..
            } => {
                validate_finite_matrix(matrix)?;
                relative_pose.validate()?;
            }
            Self::Fundamental {
                matrix,
                relative_pose,
                ..
            }
            | Self::Homography {
                matrix,
                relative_pose,
                ..
            } => {
                validate_finite_matrix(matrix)?;
                if let Some(relative_pose) = relative_pose {
                    relative_pose.validate()?;
                }
            }
            Self::RotationOnly { rotation, .. } => {
                validate_relative_rotation(rotation)?;
            }
        }
        Ok(())
    }
}

fn validate_relative_rotation(rotation: &Rotation3<f64>) -> Result<()> {
    let matrix = rotation.matrix();
    let orthogonality_error = (matrix.transpose() * matrix - Matrix3::identity()).norm();
    if !matrix.iter().all(|value| value.is_finite())
        || !orthogonality_error.is_finite()
        || orthogonality_error > 1e-6
        || (matrix.determinant() - 1.0).abs() > 1e-6
    {
        return Err(SfmError::InvalidInput(
            "relative rotation must be a finite proper rotation".to_owned(),
        ));
    }
    Ok(())
}

fn validate_finite_matrix(matrix: &Matrix3<f64>) -> Result<()> {
    if matrix.iter().any(|value| !value.is_finite()) {
        return Err(SfmError::InvalidInput(
            "geometric model contains non-finite values".to_owned(),
        ));
    }
    if matrix.iter().all(|value| *value == 0.0) {
        return Err(SfmError::InvalidInput(
            "geometric model contains an all-zero matrix".to_owned(),
        ));
    }
    Ok(())
}

/// Match and inlier counts of one pair's geometric verification.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PairVerificationReport {
    /// The verified pair.
    pub pair: ViewPair,
    /// Putative matches given to the verifier.
    pub input_match_count: usize,
    /// Inliers of the best model; `0` when no model was estimated.
    pub inlier_count: usize,
    /// Human-readable detail, usually set on rejection; `None` when there is nothing to add.
    pub message: Option<String>,
}

/// Why geometric verification rejected a pair.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VerificationRejectReason {
    /// Too few putative matches to run the estimator.
    LowMatchCount {
        /// Minimum match count.
        required: usize,
        /// Match count of the pair.
        actual: usize,
    },
    /// The best model has too few inliers.
    InsufficientInliers {
        /// Minimum inlier count.
        required: usize,
        /// Inlier count of the best model.
        actual: usize,
    },
    /// The inlier fraction is below the verifier's minimum ratio.
    LowInlierRatio,
    /// The correspondences are degenerate for the model.
    DegenerateGeometry,
    /// A homography explains the matches as well as the epipolar model (planar scene or pure
    /// rotation), so the relative pose is ambiguous.
    HomographyDominant,
    /// A single bearing rotation explains as much support as the epipolar model.
    RotationOnlyDominant,
    /// No decomposition of the model puts the inliers in front of both cameras.
    FailedChirality,
    /// The verifier does not support the pair's camera models.
    UnsupportedIntrinsics,
    /// The robust estimator returned no model.
    SolverDidNotConverge,
}

/// Result of verifying one pair.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VerificationVerdict {
    /// The pair has a verified model.
    Accepted {
        /// The verified model.
        model: GeometricModel,
        /// Counts of the verification.
        report: PairVerificationReport,
    },
    /// The pair was rejected.
    Rejected {
        /// Why the pair was rejected.
        reason: VerificationRejectReason,
        /// Counts of the verification.
        report: PairVerificationReport,
    },
}

/// A rejected pair in a [`VerificationBatchReport`].
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RejectedPairVerification {
    /// Why the pair was rejected.
    pub reason: VerificationRejectReason,
    /// Counts of the verification.
    pub report: PairVerificationReport,
}

/// Deterministic per-pair accounting for one verification operation.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct VerificationBatchReport {
    /// Accepted pairs, in verification order.
    pub accepted: Vec<PairVerificationReport>,
    /// Rejected pairs with their reasons, in verification order.
    pub rejected: Vec<RejectedPairVerification>,
}

#[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
impl VerificationBatchReport {
    /// Number of pairs verified, accepted or rejected.
    pub fn attempted_pairs(&self) -> usize {
        self.accepted.len() + self.rejected.len()
    }

    /// Number of accepted pairs.
    pub fn accepted_pairs(&self) -> usize {
        self.accepted.len()
    }

    /// Number of rejected pairs.
    pub fn rejected_pairs(&self) -> usize {
        self.rejected.len()
    }
}
