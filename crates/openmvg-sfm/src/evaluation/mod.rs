// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Reusable reconstruction evaluation metrics.
//!
//! Pose and trajectory metrics shared by reconstruction tests and custom applications.

pub(crate) mod ground_truth;

use std::collections::BTreeMap;

use nalgebra::{Matrix3, Point3, Rotation3, Vector3};
use openmvg_core::model::WorldToCameraPose;
use openmvg_geometry::SolverError;
use openmvg_geometry::alignment::{self, AlignmentScale};
use openmvg_geometry::correspondence::Point3Correspondence;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::ViewId;
pub use crate::evaluation::ground_truth::{GroundTruthReconstruction, GroundTruthScene};
use crate::workspace::quality::ErrorStats;

/// Result type for reconstruction evaluation.
pub type EvaluationResult<T> = Result<T, EvaluationError>;

/// Evaluation failures.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum EvaluationError {
    /// Evaluation thresholds or options violate their documented contract.
    #[error("invalid evaluation configuration: {0}")]
    InvalidConfig(String),

    /// Estimated reconstruction violates identifier or numerical invariants.
    #[error("invalid reconstruction: {0}")]
    InvalidReconstruction(String),

    /// Ground-truth scene violates identifier, geometry, or pair invariants.
    #[error("invalid ground truth: {0}")]
    InvalidGroundTruth(String),

    /// A derived metric was non-finite or violated error-statistic invariants.
    #[error("invalid evaluation metric: {0}")]
    InvalidMetric(String),

    /// No camera poses have matching view ids between reconstruction and GT.
    #[error("no matching camera poses between reconstruction and ground truth")]
    NoMatchingPoses,

    /// Alignment needs more correspondences than were available.
    #[error("alignment requires at least {required} matched poses, got {actual}")]
    NotEnoughPoses {
        /// Minimum number of matched camera centers the fit needs.
        required: usize,
        /// Number of matched camera centers that were available.
        actual: usize,
    },

    /// Camera centers are degenerate for alignment.
    #[error("cannot align degenerate camera centers")]
    DegenerateAlignment,

    /// SVD did not return the matrices needed by the closed-form alignment.
    #[error("alignment SVD failed")]
    SvdFailed,
}

/// How estimated camera centers are aligned to ground truth before ATE.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AlignmentMethod {
    /// No alignment. Estimated and GT poses are assumed to be in the same frame.
    None,
    /// Best-fit SE(3): rotation + translation, scale fixed to 1.
    RigidSe3,
    /// Best-fit Sim(3): scale + rotation + translation.
    ///
    /// This is the default for monocular SfM because global scale is ambiguous.
    #[default]
    SimilaritySim3,
}

/// Direction angular error semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DirectionErrorMethod {
    /// Compare the ordered direction as-is.
    Directed,
    /// Treat `d` and `-d` as equivalent.
    SignInvariant,
}

/// Alignment from estimated world coordinates into ground-truth world coordinates.
///
/// Applies as `p_gt ~= scale * rotation * p_est + translation`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Alignment {
    /// Alignment model that produced this transform; `None` for the identity.
    pub mode: AlignmentMethod,
    /// Scale `s` applied to estimated coordinates; exactly 1 unless `mode` is Sim(3).
    pub scale: f64,
    /// Rotation `Q` from the estimated world frame into the ground-truth world frame.
    pub rotation: Rotation3<f64>,
    /// Translation `t`, in ground-truth scene units.
    pub translation: Vector3<f64>,
}

impl Alignment {
    /// Identity alignment.
    pub fn identity() -> Self {
        Self {
            mode: AlignmentMethod::None,
            scale: 1.0,
            rotation: Rotation3::identity(),
            translation: Vector3::zeros(),
        }
    }

    /// Apply alignment to a world point.
    #[inline]
    pub fn transform_point(&self, point: &Point3<f64>) -> Point3<f64> {
        Point3::from(self.scale * (self.rotation * point.coords) + self.translation)
    }

    /// Apply alignment to a world-to-camera pose.
    ///
    /// If `C' = s * Q * C + t`, then the aligned world-to-camera rotation is
    /// `R' = R * Q^T`.
    #[inline]
    pub fn transform_pose(
        &self,
        pose: &WorldToCameraPose,
    ) -> Result<WorldToCameraPose, openmvg_core::model::StateValidationError> {
        WorldToCameraPose::from_rotation_center(
            pose.rotation() * self.rotation.inverse(),
            self.transform_point(pose.center()),
        )
    }
}

/// Ground-truth pairs left out of the relative rotation/translation metrics, by cause.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RelativePairSkipCounts {
    /// Pairs where either view lacks an estimated or a ground-truth pose.
    pub missing_pose: usize,
    /// Pairs whose estimated or ground-truth baseline is shorter than `1e-12`; they still count
    /// in the relative rotation error, only the translation direction is skipped.
    pub degenerate_baseline: usize,
    /// Pairs whose relative rotation or translation-direction error was not finite.
    pub non_finite: usize,
}

/// Evaluation configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvaluationConfig {
    alignment: AlignmentMethod,
    include_pairwise_relative_errors: bool,
    rotation_thresholds_deg: Vec<f64>,
    translation_thresholds_deg: Vec<f64>,
    ate_thresholds: Vec<f64>,
    relative_translation_mode: DirectionErrorMethod,
}

config_accessors!(EvaluationConfig {
    /// How estimated camera centers are aligned to ground truth before ATE.
    copy alignment, with_alignment: AlignmentMethod;
    /// Compute pairwise relative rotation and translation-direction errors.
    copy include_pairwise_relative_errors, with_include_pairwise_relative_errors: bool;
    /// Rotation accuracy-at-threshold levels, in degrees, strictly increasing.
    ref rotation_thresholds_deg, with_rotation_thresholds_deg: Vec<f64>;
    /// Translation-direction accuracy-at-threshold levels, in degrees, strictly increasing.
    ref translation_thresholds_deg, with_translation_thresholds_deg: Vec<f64>;
    /// ATE accuracy-at-threshold levels, in scene units, strictly increasing.
    ref ate_thresholds, with_ate_thresholds: Vec<f64>;
    /// Sign semantics of the relative translation-direction error.
    copy relative_translation_mode, with_relative_translation_mode: DirectionErrorMethod;
});

impl Default for EvaluationConfig {
    fn default() -> Self {
        Self {
            alignment: AlignmentMethod::SimilaritySim3,
            include_pairwise_relative_errors: true,
            rotation_thresholds_deg: vec![1.0, 2.0, 5.0, 10.0, 20.0],
            translation_thresholds_deg: vec![1.0, 2.0, 5.0, 10.0, 20.0],
            ate_thresholds: Vec::new(),
            relative_translation_mode: DirectionErrorMethod::Directed,
        }
    }
}

impl EvaluationConfig {
    /// Checks that every threshold list is finite, non-negative, and strictly increasing.
    ///
    /// # Errors
    ///
    /// Returns [`EvaluationError::InvalidConfig`] naming the first offending list.
    pub fn validate(&self) -> EvaluationResult<()> {
        validate_thresholds("rotation", &self.rotation_thresholds_deg)?;
        validate_thresholds("translation", &self.translation_thresholds_deg)?;
        validate_thresholds("ATE", &self.ate_thresholds)
    }
}

fn validate_thresholds(label: &str, thresholds: &[f64]) -> EvaluationResult<()> {
    if thresholds
        .iter()
        .any(|threshold| !threshold.is_finite() || *threshold < 0.0)
        || thresholds.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err(EvaluationError::InvalidConfig(format!(
            "{label} thresholds must be finite, non-negative, and strictly increasing"
        )));
    }
    Ok(())
}

fn metric_stats(label: &str, values: &[f64], thresholds: &[f64]) -> EvaluationResult<ErrorStats> {
    ErrorStats::try_from_values(values, thresholds)
        .map_err(|error| EvaluationError::InvalidMetric(format!("{label}: {error}")))
}

/// Full reconstruction evaluation report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvaluationReport {
    /// Center alignment applied before ATE; `None` when the configured mode is
    /// [`AlignmentMethod::None`].
    pub alignment: Option<Alignment>,
    /// Number of evaluated views: those with both an estimated and a ground-truth pose.
    pub pose_count: usize,
    /// Number of estimated poses matched to a ground-truth pose by view id.
    pub matched_gt_pose_count: usize,
    /// Number of matched camera centers used to fit the alignment; 0 without alignment.
    pub used_for_alignment_count: usize,
    /// Absolute trajectory error on aligned camera centers (standard L2 Sim3 alignment).
    pub ate: ErrorStats,
    /// "Core quality" ATE: RMSE over the inlier bulk after a robust (Huber-IRLS) Sim3
    /// alignment. `core << ate` ⇒ good bulk + a drifted tail; `core ≈ ate` ⇒ uniform
    /// drift. `None` when no alignment is requested or the robust fit is degenerate.
    pub ate_robust_core: Option<ErrorStats>,
    /// Absolute rotation angular error in degrees after applying alignment.
    pub absolute_rotation_are: ErrorStats,
    /// Pairwise relative rotation angular error in degrees.
    pub relative_rotation_are: Option<ErrorStats>,
    /// Pairwise relative translation-direction angular error in degrees.
    pub relative_translation_are: Option<ErrorStats>,
    /// Ground-truth pairs excluded from the relative metrics; all zero when they are disabled.
    pub relative_pair_skip_counts: RelativePairSkipCounts,
}

/// Rotation angular error in degrees.
pub fn rotation_angular_error_deg(r_est: &Rotation3<f64>, r_gt: &Rotation3<f64>) -> f64 {
    let r_diff = r_est * r_gt.inverse();
    let cos_angle = ((r_diff.matrix().trace() - 1.0) * 0.5).clamp(-1.0, 1.0);
    cos_angle.acos().to_degrees()
}

/// Direction angular error in degrees.
pub fn direction_angular_error_deg(
    estimated: &Vector3<f64>,
    ground_truth: &Vector3<f64>,
    mode: DirectionErrorMethod,
) -> f64 {
    let n_est = estimated.norm();
    let n_gt = ground_truth.norm();
    if n_est < 1e-12 || n_gt < 1e-12 {
        return 180.0;
    }

    let mut cos_angle = (estimated / n_est)
        .dot(&(ground_truth / n_gt))
        .clamp(-1.0, 1.0);
    if mode == DirectionErrorMethod::SignInvariant {
        cos_angle = cos_angle.abs();
    }
    cos_angle.acos().to_degrees()
}

/// Best-fit global gauge rotation `G` (est world → GT world) from matched camera
/// orientations, via a single Procrustes solve on SO(3).
///
/// Returns `G` such that the gauge-aligned world-to-camera rotation `R_est · Gᵀ`
/// best matches `R_gt` across all views. Unlike a center-derived rotation, this
/// is well-posed even when the camera centers are (near-)collinear or
/// (near-)coplanar (sequential rigs, turntable/arc datasets like fountain-P11),
/// so it gives a meaningful absolute-rotation error in those degenerate layouts.
///
/// This is a non-robust least-squares fit over all matched
/// views: a single grossly mis-estimated camera shifts `G` and therefore raises
/// the reported error for the other views too. For an *evaluation* metric that
/// is the desired behavior — a badly mis-oriented camera should not be hidden
/// by a robust gauge — so the metric stays a faithful, conservative summary of
/// orientation accuracy rather than masking outliers.
///
/// `pairs` holds `(estimated_pose, gt_pose)` for matched views.
pub fn gauge_rotation_from_poses(
    pairs: &[(WorldToCameraPose, WorldToCameraPose)],
) -> EvaluationResult<Rotation3<f64>> {
    if pairs.is_empty() {
        return Err(EvaluationError::NoMatchingPoses);
    }
    // R_gt = R_est · Gᵀ  ⇒  Gᵀ = ProjectSO3(Σ R_est_iᵀ R_gt_i),  G = (Gᵀ)ᵀ.
    let mut m = Matrix3::zeros();
    for (est, gt) in pairs {
        m += est.rotation().matrix().transpose() * gt.rotation().matrix();
    }
    let svd = m.svd(true, true);
    let u = svd.u.ok_or(EvaluationError::SvdFailed)?;
    let vt = svd.v_t.ok_or(EvaluationError::SvdFailed)?;
    let mut reflection_fix = Matrix3::identity();
    if (u * vt).determinant() < 0.0 {
        reflection_fix[(2, 2)] = -1.0;
    }
    let g_transpose = u * reflection_fix * vt;
    Ok(Rotation3::from_matrix_unchecked(g_transpose.transpose()))
}

/// Estimate an alignment from estimated camera centers to ground-truth centers.
pub(crate) fn estimate_alignment(
    estimated_centers: &[(ViewId, Point3<f64>)],
    gt_centers: &[(ViewId, Point3<f64>)],
    mode: AlignmentMethod,
) -> EvaluationResult<Option<Alignment>> {
    if mode == AlignmentMethod::None {
        return Ok(None);
    }

    let pairs = matched_points(estimated_centers, gt_centers);
    if pairs.len() < 3 {
        return Err(EvaluationError::NotEnoughPoses {
            required: 3,
            actual: pairs.len(),
        });
    }

    fit_alignment(&point_correspondences(&pairs), None, mode).map(Some)
}

/// Closed-form Umeyama Sim(3)/SE(3) fit of `estimated -> GT` centers, optionally
/// weighted (the robust IRLS below), delegated to
/// [`openmvg_geometry::alignment::fit_alignment`].
fn fit_alignment(
    correspondences: &[Point3Correspondence],
    weights: Option<&[f64]>,
    mode: AlignmentMethod,
) -> EvaluationResult<Alignment> {
    let scale = match mode {
        AlignmentMethod::SimilaritySim3 => AlignmentScale::Estimated,
        AlignmentMethod::None | AlignmentMethod::RigidSe3 => AlignmentScale::Fixed,
    };
    let fit =
        alignment::fit_alignment(correspondences, weights, scale).map_err(|error| match error {
            SolverError::SvdFailed => EvaluationError::SvdFailed,
            _ => EvaluationError::DegenerateAlignment,
        })?;
    Ok(Alignment {
        mode,
        scale: fit.scaling(),
        rotation: fit.isometry.rotation.to_rotation_matrix(),
        translation: fit.isometry.translation.vector,
    })
}

/// `(estimated, GT)` center pairs as source-to-target alignment data.
fn point_correspondences(pairs: &[(Point3<f64>, Point3<f64>)]) -> Vec<Point3Correspondence> {
    pairs
        .iter()
        .map(|&(estimated, ground_truth)| Point3Correspondence::new(estimated, ground_truth))
        .collect()
}

/// Median of a slice (non-mutating; small n, so a copy+sort is fine).
fn median_of(values: &[f64]) -> f64 {
    let mut v: Vec<f64> = values.iter().copied().filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    if n.is_multiple_of(2) {
        let lower = v[n / 2 - 1];
        lower + (v[n / 2] - lower) * 0.5
    } else {
        v[n / 2]
    }
}

/// Huber threshold `δ = 1.345 · (1.4826 · MAD)` from a residual set (MAD = median absolute
/// deviation → a robust estimate of the residual scale). Residuals ≤ δ are inliers.
pub(crate) fn huber_delta(residuals: &[f64]) -> f64 {
    let median = median_of(residuals);
    let dev: Vec<f64> = residuals.iter().map(|r| (r - median).abs()).collect();
    let mad = median_of(&dev);
    (1.345 * 1.4826 * mad).max(1e-9)
}

/// Robust Sim(3)/SE(3) alignment via **Huber-IRLS reweighting** of Umeyama. Starts from the
/// L2 fit ([`estimate_alignment`]), then iteratively down-weights cameras whose residual
/// exceeds a MAD-derived Huber threshold, so a drifted subset no longer skews the global
/// scale/rotation. Pair this with the standard L2 ATE to separate "core quality" (how good
/// the inlier bulk is) from a drifted tail. Deterministic (no sampling).
pub(crate) fn estimate_alignment_robust(
    estimated_centers: &[(ViewId, Point3<f64>)],
    gt_centers: &[(ViewId, Point3<f64>)],
    mode: AlignmentMethod,
    max_iters: usize,
) -> EvaluationResult<Option<Alignment>> {
    if mode == AlignmentMethod::None {
        return Ok(None);
    }
    let pairs = matched_points(estimated_centers, gt_centers);
    if pairs.len() < 3 {
        return Err(EvaluationError::NotEnoughPoses {
            required: 3,
            actual: pairs.len(),
        });
    }
    let Some(mut align) = estimate_alignment(estimated_centers, gt_centers, mode)? else {
        return Ok(None);
    };
    let correspondences = point_correspondences(&pairs);
    let mut weights = vec![1.0; pairs.len()];
    for _ in 0..max_iters.max(1) {
        let residuals: Vec<f64> = pairs
            .iter()
            .map(|(est, gt)| (align.transform_point(est).coords - gt.coords).norm())
            .collect();
        let delta = huber_delta(&residuals);
        for (w, &r) in weights.iter_mut().zip(&residuals) {
            *w = if r <= delta {
                1.0
            } else {
                delta / r.max(1e-12)
            };
        }
        if weights.iter().sum::<f64>() < 1e-12 {
            return Err(EvaluationError::DegenerateAlignment);
        }
        let next = fit_alignment(&correspondences, Some(&weights), mode)?;
        let converged = (next.scale - align.scale).abs() < 1e-10
            && (next.translation - align.translation).norm() < 1e-10;
        align = next;
        if converged {
            break;
        }
    }
    Ok(Some(align))
}

/// "Core quality" ATE: robust-align, then RMSE over the inlier bulk (Huber weight = 1).
/// `core << L2` ⇒ good bulk + few drifters; `core ≈ L2` ⇒ uniform drift/scale error.
/// Returns `None` when no alignment is requested or the robust fit is degenerate.
pub(crate) fn robust_core_ate(
    estimated_centers: &[(ViewId, Point3<f64>)],
    gt_centers: &[(ViewId, Point3<f64>)],
    mode: AlignmentMethod,
    thresholds: &[f64],
) -> EvaluationResult<Option<ErrorStats>> {
    let align = match estimate_alignment_robust(estimated_centers, gt_centers, mode, 12) {
        Ok(Some(align)) => align,
        Ok(None) | Err(EvaluationError::DegenerateAlignment) => return Ok(None),
        Err(error) => return Err(error),
    };
    let pairs = matched_points(estimated_centers, gt_centers);
    let residuals: Vec<f64> = pairs
        .iter()
        .map(|(est, gt)| (align.transform_point(est).coords - gt.coords).norm())
        .collect();
    let delta = huber_delta(&residuals);
    let inliers: Vec<f64> = residuals.iter().copied().filter(|&r| r <= delta).collect();
    let src = if inliers.len() >= 3 {
        &inliers
    } else {
        &residuals
    };
    Ok(Some(metric_stats("robust-core ATE", src, thresholds)?))
}

/// Aligns `reconstruction` to `ground_truth` and computes ATE, absolute rotation error, and
/// (if enabled) relative errors over the ground-truth pairs.
///
/// Views are matched by [`ViewId`]; views present on only one side are ignored.
///
/// # Errors
///
/// Returns an [`EvaluationError`] when the configuration, reconstruction, or ground truth is
/// invalid, when no view ids match, when the alignment needs more matched poses or is
/// degenerate, or when a metric is not finite.
pub fn evaluate_reconstruction_against_gt(
    reconstruction: &GroundTruthReconstruction,
    ground_truth: &GroundTruthScene,
    config: &EvaluationConfig,
) -> EvaluationResult<EvaluationReport> {
    config.validate()?;
    reconstruction
        .validate()
        .map_err(|error| EvaluationError::InvalidReconstruction(error.to_string()))?;
    ground_truth
        .validate()
        .map_err(|error| EvaluationError::InvalidGroundTruth(error.to_string()))?;

    let mut est_poses = reconstruction.poses.clone();
    let mut gt_poses = ground_truth.poses.clone();
    est_poses.sort_by_key(|(view_id, _)| *view_id);
    gt_poses.sort_by_key(|(view_id, _)| *view_id);

    let view_ids = matching_view_ids(&est_poses, &gt_poses);
    if view_ids.is_empty() {
        return Err(EvaluationError::NoMatchingPoses);
    }
    let est_pose_by_id: BTreeMap<_, _> = est_poses.iter().map(|(id, pose)| (*id, pose)).collect();
    let gt_pose_by_id: BTreeMap<_, _> = gt_poses.iter().map(|(id, pose)| (*id, pose)).collect();

    let estimated_centers: Vec<_> = view_ids
        .iter()
        .map(|&id| {
            est_pose_by_id
                .get(&id)
                .ok_or(EvaluationError::NoMatchingPoses)
                .map(|pose| (id, *pose.center()))
        })
        .collect::<EvaluationResult<Vec<_>>>()?;
    let gt_centers: Vec<_> = view_ids
        .iter()
        .map(|&id| {
            gt_pose_by_id
                .get(&id)
                .ok_or(EvaluationError::NoMatchingPoses)
                .map(|pose| (id, *pose.center()))
        })
        .collect::<EvaluationResult<Vec<_>>>()?;

    // Center-based Sim(3)/SE(3) alignment drives ATE and the (gauge-free)
    // relative metrics.
    let alignment = estimate_alignment(&estimated_centers, &gt_centers, config.alignment)?;

    // The absolute-rotation gauge is fixed separately from the camera
    // orientations (a Procrustes solve on SO(3)). This is robust to the
    // rotational ambiguity of a center-only fit when camera centers are
    // near-collinear/coplanar, so absolute rotation error stays meaningful on
    // sequential / arc datasets. Falls back to the center alignment rotation
    // when no alignment is requested or the gauge solve fails.
    let abs_rotation_gauge = if config.alignment == AlignmentMethod::None {
        None
    } else {
        let pose_pairs: Vec<(WorldToCameraPose, WorldToCameraPose)> = view_ids
            .iter()
            .map(|id| {
                let est = est_pose_by_id
                    .get(id)
                    .ok_or(EvaluationError::NoMatchingPoses)?;
                let gt = gt_pose_by_id
                    .get(id)
                    .ok_or(EvaluationError::NoMatchingPoses)?;
                Ok(((*est).clone(), (*gt).clone()))
            })
            .collect::<EvaluationResult<Vec<_>>>()?;
        gauge_rotation_from_poses(&pose_pairs).ok()
    };
    let used_for_alignment_count = if config.alignment == AlignmentMethod::None {
        0
    } else {
        view_ids.len()
    };
    let identity = Alignment::identity();
    let alignment_ref = alignment.as_ref().unwrap_or(&identity);

    let mut ate_values = Vec::with_capacity(view_ids.len());
    let mut absolute_rotation_values = Vec::with_capacity(view_ids.len());
    let mut aligned_poses = Vec::with_capacity(view_ids.len());

    for view_id in &view_ids {
        let est_pose = est_pose_by_id
            .get(view_id)
            .ok_or(EvaluationError::NoMatchingPoses)?;
        let gt_pose = gt_pose_by_id
            .get(view_id)
            .ok_or(EvaluationError::NoMatchingPoses)?;
        let aligned_pose = alignment_ref.transform_pose(est_pose).map_err(|error| {
            EvaluationError::InvalidMetric(format!("alignment produced an invalid pose: {error}"))
        })?;

        ate_values.push((aligned_pose.center() - gt_pose.center()).norm());
        // Absolute rotation error uses the orientation-derived gauge `G`
        // (R_est · Gᵀ vs R_gt) so it stays correct under coplanar/collinear
        // camera layouts; fall back to the center-alignment rotation otherwise.
        let abs_rotation_est = match &abs_rotation_gauge {
            Some(g) => est_pose.rotation() * g.inverse(),
            None => *aligned_pose.rotation(),
        };
        absolute_rotation_values.push(rotation_angular_error_deg(
            &abs_rotation_est,
            gt_pose.rotation(),
        ));
        aligned_poses.push((*view_id, aligned_pose));
    }

    let (relative_rotation_are, relative_translation_are, relative_pair_skip_counts) =
        if config.include_pairwise_relative_errors {
            let (r, t, skip_counts) = explicit_pair_relative_errors(
                &aligned_poses,
                &gt_poses,
                &ground_truth.pairs,
                config.relative_translation_mode,
            );
            (
                Some(metric_stats(
                    "relative rotation",
                    &r,
                    &config.rotation_thresholds_deg,
                )?),
                Some(metric_stats(
                    "relative translation",
                    &t,
                    &config.translation_thresholds_deg,
                )?),
                skip_counts,
            )
        } else {
            (None, None, RelativePairSkipCounts::default())
        };

    Ok(EvaluationReport {
        alignment,
        pose_count: view_ids.len(),
        matched_gt_pose_count: view_ids.len(),
        used_for_alignment_count,
        ate: metric_stats("ATE", &ate_values, &config.ate_thresholds)?,
        ate_robust_core: robust_core_ate(
            &estimated_centers,
            &gt_centers,
            config.alignment,
            &config.ate_thresholds,
        )?,
        absolute_rotation_are: metric_stats(
            "absolute rotation",
            &absolute_rotation_values,
            &config.rotation_thresholds_deg,
        )?,
        relative_rotation_are,
        relative_translation_are,
        relative_pair_skip_counts,
    })
}

fn matched_points(
    estimated: &[(ViewId, Point3<f64>)],
    ground_truth: &[(ViewId, Point3<f64>)],
) -> Vec<(Point3<f64>, Point3<f64>)> {
    let mut pairs = Vec::new();
    for (est_id, est_point) in estimated {
        if let Some((_, gt_point)) = ground_truth.iter().find(|(gt_id, _)| gt_id == est_id) {
            pairs.push((*est_point, *gt_point));
        }
    }
    pairs
}

fn matching_view_ids(
    reconstruction: &[(ViewId, WorldToCameraPose)],
    ground_truth: &[(ViewId, WorldToCameraPose)],
) -> Vec<ViewId> {
    let mut ids: Vec<_> = reconstruction
        .iter()
        .filter(|(id, _)| ground_truth.iter().any(|(gt_id, _)| gt_id == id))
        .map(|(id, _)| *id)
        .collect();
    ids.sort();
    ids
}

fn explicit_pair_relative_errors(
    aligned_estimated: &[(ViewId, WorldToCameraPose)],
    ground_truth: &[(ViewId, WorldToCameraPose)],
    pairs: &[crate::ViewPair],
    translation_mode: DirectionErrorMethod,
) -> (Vec<f64>, Vec<f64>, RelativePairSkipCounts) {
    let mut rotation_errors = Vec::new();
    let mut translation_errors = Vec::new();
    let mut skip_counts = RelativePairSkipCounts::default();

    for pair in pairs {
        let view_i = pair.first();
        let view_j = pair.second();
        let Some((_, est_i)) = aligned_estimated.iter().find(|(id, _)| *id == view_i) else {
            skip_counts.missing_pose += 1;
            continue;
        };
        let Some((_, est_j)) = aligned_estimated.iter().find(|(id, _)| *id == view_j) else {
            skip_counts.missing_pose += 1;
            continue;
        };
        let Some((_, gt_i)) = ground_truth.iter().find(|(id, _)| *id == view_i) else {
            skip_counts.missing_pose += 1;
            continue;
        };
        let Some((_, gt_j)) = ground_truth.iter().find(|(id, _)| *id == view_j) else {
            skip_counts.missing_pose += 1;
            continue;
        };

        let r_est = est_j.rotation() * est_i.rotation().inverse();
        let r_gt = gt_j.rotation() * gt_i.rotation().inverse();
        let rotation_error = rotation_angular_error_deg(&r_est, &r_gt);
        if !rotation_error.is_finite() {
            skip_counts.non_finite += 1;
            continue;
        }
        rotation_errors.push(rotation_error);

        let t_est = est_i.rotation() * (est_j.center() - est_i.center());
        let t_gt = gt_i.rotation() * (gt_j.center() - gt_i.center());
        if t_est.norm() < 1e-12 || t_gt.norm() < 1e-12 {
            skip_counts.degenerate_baseline += 1;
            continue;
        }
        let translation_error = direction_angular_error_deg(&t_est, &t_gt, translation_mode);
        if !translation_error.is_finite() {
            skip_counts.non_finite += 1;
            continue;
        }
        translation_errors.push(translation_error);
    }

    (rotation_errors, translation_errors, skip_counts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Unit;

    #[test]
    fn evaluation_thresholds_are_canonical_and_error_stats_are_a_valid_cdf() {
        let config = EvaluationConfig {
            rotation_thresholds_deg: vec![1.0, 1.0],
            ..EvaluationConfig::default()
        };
        assert!(matches!(
            config.validate(),
            Err(EvaluationError::InvalidConfig(_))
        ));

        let mut empty = ErrorStats::from_values(&[], &[1.0]);
        empty.threshold_pass_rates[0].pass_rate = 1.0;
        assert!(empty.validate().is_err());

        let mut non_monotonic = ErrorStats::from_values(&[0.5, 1.5], &[1.0, 2.0]);
        non_monotonic.threshold_pass_rates[1].pass_rate = 0.0;
        assert!(non_monotonic.validate().is_err());

        assert!(ErrorStats::try_from_values(&[f64::INFINITY], &[]).is_err());
        let huge = ErrorStats::try_from_values(&[1e308, 1e308], &[]).unwrap();
        assert!(huge.mean.is_finite());
        assert!(huge.rmse.is_finite());
    }

    fn pose_at(
        id: ViewId,
        center: Point3<f64>,
        rotation: Rotation3<f64>,
    ) -> (ViewId, WorldToCameraPose) {
        (
            id,
            WorldToCameraPose::from_rotation_center(rotation, center).unwrap(),
        )
    }

    fn poses(
        poses: impl IntoIterator<Item = (ViewId, WorldToCameraPose)>,
    ) -> Vec<(ViewId, WorldToCameraPose)> {
        poses.into_iter().collect()
    }

    fn evaluate_poses(
        reconstruction: &[(ViewId, WorldToCameraPose)],
        ground_truth: &[(ViewId, WorldToCameraPose)],
        config: &EvaluationConfig,
    ) -> EvaluationResult<EvaluationReport> {
        let pairs = ground_truth
            .iter()
            .enumerate()
            .flat_map(|(left, (left_id, _))| {
                ground_truth[(left + 1)..]
                    .iter()
                    .map(move |(right_id, _)| crate::ViewPair::new(*left_id, *right_id))
            })
            .collect();
        evaluate_reconstruction_against_gt(
            &GroundTruthReconstruction {
                poses: reconstruction.to_vec(),
                points: Vec::new(),
            },
            &GroundTruthScene {
                image_meta: Vec::new(),
                camera_models: Vec::new(),
                poses: ground_truth.to_vec(),
                points: Vec::new(),
                pairs,
            },
            config,
        )
    }

    fn assert_close(a: f64, b: f64, eps: f64) {
        assert!(
            (a - b).abs() <= eps,
            "expected {a:.12} to be within {eps} of {b:.12}"
        );
    }

    #[test]
    fn gauge_rotation_recovers_known_gauge_with_coplanar_centers() {
        use nalgebra::Unit;
        // Known world gauge applied to "true" poses to create the estimate.
        let g0 = Rotation3::from_axis_angle(&Vector3::x_axis(), std::f64::consts::FRAC_PI_2);
        // GT poses with VARIED rotations but COPLANAR centers (z = 0 plane) —
        // the degenerate layout that breaks center-only rotation recovery.
        let gt_poses: Vec<(WorldToCameraPose, WorldToCameraPose)> = (0..6)
            .map(|i| {
                let angle = 0.3 * i as f64;
                let axis = Unit::new_normalize(Vector3::new(0.2, 1.0, 0.4 + 0.1 * i as f64));
                let r_gt = Rotation3::from_axis_angle(&axis, angle);
                let c_gt = Point3::new(i as f64, (i as f64 * 0.5).sin(), 0.0);
                // R_est · G0ᵀ = R_gt  ⇒  R_est = R_gt · G0; C_gt = G0 · C_est.
                let r_est = r_gt * g0;
                let c_est = Point3::from(g0.transpose() * c_gt.coords);
                (
                    WorldToCameraPose::from_rotation_center(r_est, c_est).unwrap(),
                    WorldToCameraPose::from_rotation_center(r_gt, c_gt).unwrap(),
                )
            })
            .collect();

        let g = gauge_rotation_from_poses(&gt_poses).unwrap();
        // The recovered gauge must align every estimated rotation to its GT.
        for (est, gt) in &gt_poses {
            let aligned = est.rotation() * g.inverse();
            assert_close(
                rotation_angular_error_deg(&aligned, gt.rotation()),
                0.0,
                1e-4,
            );
        }
    }

    #[test]
    fn evaluation_resolves_absolute_rotation_with_collinear_centers() {
        // Strongest gauge degeneracy: camera centers lie *exactly* on a line (the
        // X axis, a rank-1 layout). A center-only Sim(3) fit leaves the rotation
        // about that line completely free, so it cannot recover the absolute
        // rotation gauge. The orientation Procrustes gauge still can, because each
        // camera carries a distinct, well-separated orientation. This tests the
        // collinear case (e.g. a straight camera dolly) the way the coplanar case
        // (turntable/arc, fountain-P11) is tested above.
        let g0 = Rotation3::from_axis_angle(&Unit::new_normalize(Vector3::new(0.3, 0.7, 0.5)), 0.9);
        let scale = 1.7;
        let t = Vector3::new(2.0, -1.0, 0.5);

        let gt: Vec<(ViewId, Rotation3<f64>, Point3<f64>)> = (0..6)
            .map(|i| {
                let angle = 0.4 + 0.25 * i as f64;
                let axis = Unit::new_normalize(Vector3::new(0.1 + 0.05 * i as f64, 0.3, 1.0));
                let r_gt = Rotation3::from_axis_angle(&axis, angle);
                let c_gt = Point3::new(i as f64, 0.0, 0.0); // strictly collinear on X
                (ViewId::new(i), r_gt, c_gt)
            })
            .collect();

        // est = gt re-gauged by (scale, g0, t):
        //   R_gt = R_est · g0ᵀ      ⇒  R_est = R_gt · g0
        //   C_gt = scale·g0·C_est + t ⇒ C_est = g0ᵀ·(C_gt − t)/scale
        let est_poses = poses(gt.iter().map(|&(id, r_gt, c_gt)| {
            let r_est = r_gt * g0;
            let c_est = Point3::from(g0.transpose() * (c_gt.coords - t) / scale);
            pose_at(id, c_est, r_est)
        }));
        let gt_poses = poses(gt.iter().map(|&(id, r_gt, c_gt)| pose_at(id, c_gt, r_gt)));

        let report = evaluate_poses(&est_poses, &gt_poses, &EvaluationConfig::default()).unwrap();

        // The gauge resolves absolute rotation to machine precision despite the
        // rank-1 center layout; relative rotation is gauge-free and also exact.
        assert_close(report.absolute_rotation_are.max, 0.0, 1e-3);
        assert_close(report.relative_rotation_are.unwrap().max, 0.0, 1e-4);
    }

    #[test]
    fn identity_evaluation_has_zero_error() {
        let gt = poses([
            pose_at(
                ViewId::new(0),
                Point3::new(0.0, 0.0, 0.0),
                Rotation3::identity(),
            ),
            pose_at(
                ViewId::new(1),
                Point3::new(1.0, 0.0, 0.0),
                Rotation3::identity(),
            ),
            pose_at(
                ViewId::new(2),
                Point3::new(0.0, 1.0, 0.0),
                Rotation3::identity(),
            ),
        ]);
        let report = evaluate_poses(&gt, &gt, &EvaluationConfig::default()).unwrap();

        assert_eq!(report.pose_count, 3);
        assert_close(report.ate.rmse, 0.0, 1e-5);
        assert_close(report.absolute_rotation_are.max, 0.0, 1e-5);
        assert_close(report.relative_rotation_are.unwrap().max, 0.0, 1e-5);
        assert_close(report.relative_translation_are.unwrap().max, 0.0, 1e-5);
    }

    #[test]
    fn sim3_alignment_recovers_scale_rotation_and_translation() {
        let q = Rotation3::from_axis_angle(&Vector3::z_axis(), 0.4);
        let t = Vector3::new(3.0, -2.0, 0.5);
        let scale = 2.5;
        let gt_centers = [
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 2.0, 0.0),
            Point3::new(0.5, 0.5, 1.0),
        ];
        let est_centers: Vec<_> = gt_centers
            .iter()
            .map(|c_gt| Point3::from((q.transpose() * (c_gt.coords - t)) / scale))
            .collect();

        let gt = poses(gt_centers.iter().enumerate().map(|(i, &center)| {
            let r_gt = Rotation3::identity() * q.inverse();
            pose_at(ViewId::new(i), center, r_gt)
        }));
        let est = poses(
            est_centers
                .iter()
                .enumerate()
                .map(|(i, &center)| pose_at(ViewId::new(i), center, Rotation3::identity())),
        );

        let report = evaluate_poses(&est, &gt, &EvaluationConfig::default()).unwrap();

        let alignment = report.alignment.unwrap();
        assert_close(alignment.scale, scale, 1e-10);
        assert_close(report.ate.rmse, 0.0, 1e-5);
        assert_close(report.absolute_rotation_are.max, 0.0, 1e-5);
    }

    #[test]
    fn robust_alignment_downweights_outlier_and_core_ate_isolates_bulk() {
        // Realistic regime (mirrors courtyard): a well-fit majority + a drifted minority.
        // 8 cameras span a ~7-unit cube; 6 match GT (with sub-cm jitter), 2 are drifted by a
        // scene-scale offset. The L2 fit is a compromise pulled by the drifters; the robust
        // IRLS recovers the 6-camera bulk and the core ATE isolates its (tiny) error.
        let gt_centers: Vec<Point3<f64>> = [
            (0.0, 0.0, 0.0),
            (5.0, 0.0, 0.0),
            (0.0, 5.0, 0.0),
            (5.0, 5.0, 0.0),
            (2.0, 3.0, 4.0),
            (4.0, 1.0, 2.0),
            (1.0, 4.0, 3.0),
            (3.0, 2.0, 5.0),
        ]
        .iter()
        .map(|&(x, y, z)| Point3::new(x, y, z))
        .collect();
        let mut est_centers = gt_centers.clone();
        for (i, c) in est_centers.iter_mut().enumerate().take(6) {
            let d = ((i as f64) * 0.7).sin() * 0.01; // sub-cm deterministic jitter on the bulk
            c.coords += Vector3::new(d, -d, 0.5 * d);
        }
        // Two drifted cameras (scene-scale offset — a minority, so L2 stays a compromise).
        est_centers[6].coords += Vector3::new(3.0, 3.0, 0.0);
        est_centers[7].coords += Vector3::new(-3.0, 2.5, 3.0);

        let est: Vec<(ViewId, Point3<f64>)> = est_centers
            .iter()
            .enumerate()
            .map(|(i, &c)| (ViewId::new(i), c))
            .collect();
        let gtc: Vec<(ViewId, Point3<f64>)> = gt_centers
            .iter()
            .enumerate()
            .map(|(i, &c)| (ViewId::new(i), c))
            .collect();

        let l2 = estimate_alignment(&est, &gtc, AlignmentMethod::SimilaritySim3)
            .unwrap()
            .unwrap();
        let rob = estimate_alignment_robust(&est, &gtc, AlignmentMethod::SimilaritySim3, 12)
            .unwrap()
            .unwrap();
        let core = robust_core_ate(&est, &gtc, AlignmentMethod::SimilaritySim3, &[])
            .unwrap()
            .unwrap();
        let l2_ate = (est
            .iter()
            .zip(gtc.iter())
            .map(|((_, c), (_, g))| (l2.transform_point(c).coords - g.coords).norm_squared())
            .sum::<f64>()
            / est.len() as f64)
            .sqrt();

        // Robust recovers the near-identity Sim3 of the bulk; core ATE ≈ jitter ≪ L2 ATE.
        assert!(
            (rob.scale - 1.0).abs() < 0.05,
            "robust scale {} not ~1",
            rob.scale
        );
        assert!(core.rmse < 0.1, "core rmse {} not small", core.rmse);
        assert!(
            l2_ate > 1.0,
            "L2 ATE {} should be inflated by the drifters",
            l2_ate
        );
        assert!(
            core.rmse < 0.2 * l2_ate,
            "core ({}) should be far below L2 ATE ({})",
            core.rmse,
            l2_ate
        );
    }

    #[test]
    fn rigid_alignment_keeps_scale_fixed() {
        let gt = poses([
            pose_at(
                ViewId::new(0),
                Point3::new(0.0, 0.0, 0.0),
                Rotation3::identity(),
            ),
            pose_at(
                ViewId::new(1),
                Point3::new(2.0, 0.0, 0.0),
                Rotation3::identity(),
            ),
            pose_at(
                ViewId::new(2),
                Point3::new(0.0, 2.0, 0.0),
                Rotation3::identity(),
            ),
        ]);
        let est = poses([
            pose_at(
                ViewId::new(0),
                Point3::new(0.0, 0.0, 0.0),
                Rotation3::identity(),
            ),
            pose_at(
                ViewId::new(1),
                Point3::new(1.0, 0.0, 0.0),
                Rotation3::identity(),
            ),
            pose_at(
                ViewId::new(2),
                Point3::new(0.0, 1.0, 0.0),
                Rotation3::identity(),
            ),
        ]);
        let cfg = EvaluationConfig {
            alignment: AlignmentMethod::RigidSe3,
            ..EvaluationConfig::default()
        };

        let report = evaluate_poses(&est, &gt, &cfg).unwrap();

        assert_close(report.alignment.unwrap().scale, 1.0, 1e-12);
        assert!(report.ate.rmse > 0.1);
    }

    #[test]
    fn directed_translation_error_distinguishes_opposite_directions() {
        let a = Vector3::x();
        let b = -Vector3::x();

        assert_close(
            direction_angular_error_deg(&a, &b, DirectionErrorMethod::Directed),
            180.0,
            1e-12,
        );
        assert_close(
            direction_angular_error_deg(&a, &b, DirectionErrorMethod::SignInvariant),
            0.0,
            1e-12,
        );
    }

    #[test]
    fn alignment_requires_three_poses() {
        let estimated = [
            (ViewId::new(0), Point3::origin()),
            (ViewId::new(1), Point3::new(1.0, 0.0, 0.0)),
        ];
        let gt = estimated;

        let err = estimate_alignment(&estimated, &gt, AlignmentMethod::SimilaritySim3).unwrap_err();

        assert_eq!(
            err,
            EvaluationError::NotEnoughPoses {
                required: 3,
                actual: 2
            }
        );
    }

    #[test]
    fn reflection_is_not_returned_as_alignment_rotation() {
        let estimated = [
            (ViewId::new(0), Point3::new(0.0, 0.0, 0.0)),
            (ViewId::new(1), Point3::new(1.0, 0.0, 0.0)),
            (ViewId::new(2), Point3::new(0.0, 1.0, 0.0)),
            (ViewId::new(3), Point3::new(0.0, 0.0, 1.0)),
        ];
        let gt = [
            (ViewId::new(0), Point3::new(0.0, 0.0, 0.0)),
            (ViewId::new(1), Point3::new(-1.0, 0.0, 0.0)),
            (ViewId::new(2), Point3::new(0.0, 1.0, 0.0)),
            (ViewId::new(3), Point3::new(0.0, 0.0, 1.0)),
        ];

        let alignment =
            estimate_alignment(&estimated, &gt, AlignmentMethod::SimilaritySim3).unwrap();

        let det = alignment.unwrap().rotation.matrix().determinant();
        assert_close(det, 1.0, 1e-12);
    }

    #[test]
    fn absolute_rotation_error_uses_world_alignment_rotation() {
        let q = Rotation3::from_axis_angle(&Unit::new_normalize(Vector3::z()), 0.7);
        let gt = poses([
            pose_at(
                ViewId::new(0),
                Point3::new(0.0, 0.0, 0.0),
                Rotation3::identity() * q.inverse(),
            ),
            pose_at(
                ViewId::new(1),
                Point3::new(1.0, 0.0, 0.0),
                Rotation3::identity() * q.inverse(),
            ),
            pose_at(
                ViewId::new(2),
                Point3::new(0.0, 1.0, 0.0),
                Rotation3::identity() * q.inverse(),
            ),
        ]);
        let est = poses([
            pose_at(
                ViewId::new(0),
                Point3::new(0.0, 0.0, 0.0),
                Rotation3::identity(),
            ),
            pose_at(
                ViewId::new(1),
                Point3::from(q.transpose() * Vector3::x()),
                Rotation3::identity(),
            ),
            pose_at(
                ViewId::new(2),
                Point3::from(q.transpose() * Vector3::y()),
                Rotation3::identity(),
            ),
        ]);

        let report = evaluate_poses(&est, &gt, &EvaluationConfig::default()).unwrap();

        assert_close(report.absolute_rotation_are.max, 0.0, 1e-10);
    }

    #[test]
    fn reconstruction_evaluation_rejects_duplicate_pose_ids_before_counting() {
        let duplicate = GroundTruthReconstruction {
            poses: vec![
                (
                    ViewId::new(0),
                    WorldToCameraPose::from_rotation_center(
                        Rotation3::identity(),
                        Point3::origin(),
                    )
                    .unwrap(),
                ),
                (
                    ViewId::new(0),
                    WorldToCameraPose::from_rotation_center(
                        Rotation3::identity(),
                        Point3::new(1.0, 0.0, 0.0),
                    )
                    .unwrap(),
                ),
            ],
            points: Vec::new(),
        };
        let ground_truth = GroundTruthScene {
            image_meta: Vec::new(),
            camera_models: Vec::new(),
            poses: vec![(
                ViewId::new(0),
                WorldToCameraPose::from_rotation_center(Rotation3::identity(), Point3::origin())
                    .unwrap(),
            )],
            points: Vec::new(),
            pairs: Vec::new(),
        };

        assert!(matches!(
            evaluate_reconstruction_against_gt(
                &duplicate,
                &ground_truth,
                &EvaluationConfig {
                    alignment: AlignmentMethod::None,
                    ..EvaluationConfig::default()
                }
            ),
            Err(EvaluationError::InvalidReconstruction(_))
        ));
    }
}
