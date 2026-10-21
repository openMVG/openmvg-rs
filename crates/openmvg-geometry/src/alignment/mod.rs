// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Closed-form 3D-to-3D alignment: proper-rotation Kabsch and Umeyama.
//!
//! One least-squares core serves every alignment fit. [`fit_alignment`]
//! estimates a rigid (scale fixed at 1) or similarity (estimated scale)
//! transform from optionally weighted correspondences; [`estimate_rigid_alignment`]
//! uses the same core with unit weights and a fixed scale for minimal
//! hypotheses and consensus refits. Models map source
//! points into the target frame.
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │                Kabsch / Umeyama closed-form alignment                │
//! └──────────────────────────────────────────────────────────────────────┘
//! 3D correspondences (sᵢ → tᵢ), weights wᵢ ≥ 0 (1 when unweighted)
//!        │
//!        ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 1: Center correspondences                                       │
//! └──────────────────────────────────────────────────────────────────────┘
//! μₛ = Σwᵢsᵢ / Σwᵢ, μₜ = Σwᵢtᵢ / Σwᵢ; subtract centroids from every point.
//!                                  │
//!                                  ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 2: Kabsch rotation                                              │
//! └──────────────────────────────────────────────────────────────────────┘
//! H = Σwᵢ(tᵢ−μₜ)(sᵢ−μₛ)ᵀ; H = UΣVᵀ; D = diag(1,1,det(UVᵀ)); R = U D Vᵀ.
//!                                  │
//!                                  ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 3: Umeyama scale (similarity only)                              │
//! └──────────────────────────────────────────────────────────────────────┘
//! s = trace(ΣD) / Σwᵢ‖sᵢ−μₛ‖²; s = 1 for a rigid fit.
//!                                  │
//!                                  ▼
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │ Step 4: Translation and validation                                   │
//! └──────────────────────────────────────────────────────────────────────┘
//! t = μₜ − s·Rμₛ; reject degenerate support and non-finite results.
//! ```
//!
//! # References
//!
//! - W. Kabsch, "A Solution for the Best Rotation to Relate Two Sets of Vectors", Acta
//!   Crystallographica A32, 1976.
//! - S. Umeyama, "Least-Squares Estimation of Transformation Parameters Between Two Point
//!   Patterns", IEEE TPAMI 13(4), 1991.

use crate::correspondence::Point3Correspondence;
use crate::{SolverError, SolverResult};
use nalgebra::{Isometry3, Matrix3, Rotation3, Similarity3, Translation3, UnitQuaternion, Vector3};

use crate::solver_traits::{SingleSolutionSolver, Solver};

mod robust;

pub use robust::{estimate_rigid_alignment, estimate_rigid_alignment_guided};

const MIN_CORRESPONDENCES: usize = 3;
const RELATIVE_RANK_TOLERANCE: f64 = 1.0e-12;
/// Smallest weighted mean squared source spread accepted by [`fit_alignment`].
const MIN_MEAN_SQUARED_SPREAD: f64 = 1.0e-15;
/// Smallest similarity scale accepted by [`fit_alignment`].
const MIN_SCALE: f64 = 1.0e-15;

/// Whether [`fit_alignment`] estimates a global scale.
// Closed set: a global scale is either held fixed or estimated.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum AlignmentScale {
    /// Rigid SE(3) fit (Kabsch): the scale is fixed at 1.
    #[default]
    Fixed,
    /// Similarity Sim(3) fit (Umeyama): a positive global scale is estimated.
    Estimated,
}

/// Fit the closed-form least-squares source-to-target alignment.
///
/// Minimizes `Σ wᵢ ‖s·R·sourceᵢ + t − targetᵢ‖²` over a proper rotation `R`,
/// a translation `t` and, with [`AlignmentScale::Estimated`], a scale `s`
/// (Umeyama, 1991); [`AlignmentScale::Fixed`] keeps `s = 1` (Kabsch). `None`
/// weights every correspondence equally; otherwise `weights` holds one finite,
/// non-negative weight per correspondence. The returned
/// [`Similarity3`] has `scaling() == 1.0` for a rigid fit.
///
/// Collinear support is accepted: every rotation about the common line is then
/// a minimizer and one of them is returned. Use [`estimate_rigid_alignment`]
/// (or its rank-checked solver) when a rotation must be fully determined.
///
/// # Errors
///
/// - [`SolverError::InsufficientData`] for fewer than three correspondences.
/// - [`SolverError::InvalidInput`] for a weight-count mismatch or a negative
///   or non-finite weight.
/// - [`SolverError::Numerical`] for non-finite points or overflow.
/// - [`SolverError::DegenerateConfiguration`] when the total weight is zero,
///   the weighted mean squared source spread is below `1e-15`, or the
///   estimated scale is below `1e-15`.
/// - [`SolverError::SvdFailed`] when the cross-covariance SVD fails.
///
/// # Examples
///
/// ```
/// use nalgebra::{Point3, Similarity3, Vector3};
/// use openmvg_geometry::correspondence::Point3Correspondence;
/// use openmvg_geometry::alignment::{AlignmentScale, fit_alignment};
///
/// let truth = Similarity3::new(Vector3::new(0.5, -1.0, 2.0), Vector3::new(0.1, 0.2, -0.3), 2.5);
/// let correspondences: Vec<_> = [
///     Point3::new(0.0, 0.0, 0.0),
///     Point3::new(1.0, 0.0, 0.0),
///     Point3::new(0.0, 1.0, 0.0),
///     Point3::new(0.0, 0.0, 1.0),
/// ]
/// .into_iter()
/// .map(|source| Point3Correspondence::new(source, truth.transform_point(&source)))
/// .collect();
///
/// let fitted = fit_alignment(&correspondences, None, AlignmentScale::Estimated)?;
/// assert!((fitted.scaling() - 2.5).abs() < 1e-12);
/// # Ok::<(), openmvg_geometry::SolverError>(())
/// ```
pub fn fit_alignment(
    correspondences: &[Point3Correspondence],
    weights: Option<&[f64]>,
    scale: AlignmentScale,
) -> SolverResult<Similarity3<f64>> {
    if let Some(weights) = weights {
        if weights.len() != correspondences.len() {
            return Err(SolverError::InvalidInput {
                reason: "alignment weights must match the correspondence count",
            });
        }
        if weights
            .iter()
            .any(|weight| !weight.is_finite() || *weight < 0.0)
        {
            return Err(SolverError::InvalidInput {
                reason: "alignment weights must be finite and non-negative",
            });
        }
    }
    let fit = fit_closed_form(
        correspondences.len(),
        |index| {
            let weight = weights.map_or(1.0, |weights| weights[index]);
            Ok((correspondences[index], weight))
        },
        scale,
        SupportCheck::Spread,
    )?;
    Ok(Similarity3::from_parts(
        Translation3::from(fit.translation),
        UnitQuaternion::from_rotation_matrix(&fit.rotation),
        fit.scale,
    ))
}

/// Closed-form least-squares rigid-alignment solver.
///
/// The proper-rotation Kabsch solution minimizes the sum of squared Euclidean
/// source-to-target errors. At least three non-collinear correspondences are
/// required; the same implementation accepts larger consensus sets.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct RigidAlignmentSolver;

impl Solver for RigidAlignmentSolver {
    type Datum = Point3Correspondence;
    type Model = Isometry3<f64>;
    type Config = ();
    type Unit = openmvg_core::WorldUnits;

    fn min_sample_size() -> usize {
        MIN_CORRESPONDENCES
    }

    fn residual(model: &Self::Model, datum: &Self::Datum) -> f64 {
        Self::squared_residual(model, datum).sqrt()
    }

    fn squared_residual(model: &Self::Model, datum: &Self::Datum) -> f64 {
        (model.transform_point(&datum.source) - datum.target).norm_squared()
    }

    fn is_valid(model: &Self::Model) -> bool {
        model.to_homogeneous().iter().all(|value| value.is_finite())
    }
}

impl SingleSolutionSolver for RigidAlignmentSolver {
    fn solve_single(
        data: &[Self::Datum],
        _config: &Self::Config,
    ) -> SolverResult<Option<Self::Model>> {
        fit_rigid_transform(data.len(), |index| Ok(data[index]))
    }

    fn solve_single_from_indices(
        indices: &[usize],
        data: &[Self::Datum],
        _config: &Self::Config,
    ) -> SolverResult<Option<Self::Model>> {
        fit_rigid_transform(indices.len(), |sample_index| {
            let data_index = indices[sample_index];
            data.get(data_index)
                .copied()
                .ok_or(SolverError::InvalidDataIndex {
                    index: data_index,
                    data_len: data.len(),
                })
        })
    }
}

fn fit_rigid_transform(
    count: usize,
    mut correspondence_at: impl FnMut(usize) -> SolverResult<Point3Correspondence>,
) -> SolverResult<Option<Isometry3<f64>>> {
    let fit = fit_closed_form(
        count,
        |index| Ok((correspondence_at(index)?, 1.0)),
        AlignmentScale::Fixed,
        SupportCheck::NonCollinear,
    )?;
    let model = Isometry3::from_parts(
        Translation3::from(fit.translation),
        UnitQuaternion::from_rotation_matrix(&fit.rotation),
    );
    Ok(RigidAlignmentSolver::is_valid(&model).then_some(model))
}

/// Support requirement enforced by [`fit_closed_form`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SupportCheck {
    /// Consensus hypotheses need a fully determined rotation: reject
    /// rank-deficient (collinear) cross-covariance.
    NonCollinear,
    /// General fits accept collinear support and only reject a source spread
    /// that leaves the rotation or the scale undefined.
    Spread,
}

/// Rotation, scale and translation of one closed-form alignment fit.
struct ClosedFormFit {
    rotation: Rotation3<f64>,
    scale: f64,
    translation: Vector3<f64>,
}

/// The shared weighted Kabsch/Umeyama core.
///
/// `sample_at` yields each correspondence with its (already validated,
/// non-negative) weight. Unit weights reproduce the unweighted sums exactly.
fn fit_closed_form(
    count: usize,
    mut sample_at: impl FnMut(usize) -> SolverResult<(Point3Correspondence, f64)>,
    scale_mode: AlignmentScale,
    support: SupportCheck,
) -> SolverResult<ClosedFormFit> {
    if count < MIN_CORRESPONDENCES {
        return Err(SolverError::InsufficientData {
            required: MIN_CORRESPONDENCES,
            provided: count,
        });
    }

    let mut weight_sum = 0.0;
    let mut source_centroid = Vector3::zeros();
    let mut target_centroid = Vector3::zeros();
    for index in 0..count {
        let (correspondence, weight) = sample_at(index)?;
        if !correspondence
            .source
            .coords
            .iter()
            .chain(correspondence.target.coords.iter())
            .all(|value| value.is_finite())
        {
            return Err(SolverError::Numerical {
                reason: "alignment points must be finite".to_owned(),
            });
        }
        weight_sum += weight;
        source_centroid += correspondence.source.coords * weight;
        target_centroid += correspondence.target.coords * weight;
    }
    if weight_sum <= 0.0 {
        return Err(SolverError::DegenerateConfiguration {
            reason: "alignment requires a positive total weight".to_owned(),
        });
    }
    let inverse_weight_sum = 1.0 / weight_sum;
    source_centroid *= inverse_weight_sum;
    target_centroid *= inverse_weight_sum;
    if !source_centroid
        .iter()
        .chain(target_centroid.iter())
        .all(|value| value.is_finite())
    {
        return Err(SolverError::Numerical {
            reason: "alignment centroids overflowed".to_owned(),
        });
    }

    let mut covariance = Matrix3::zeros();
    let mut source_spread = 0.0;
    for index in 0..count {
        let (correspondence, weight) = sample_at(index)?;
        let source = correspondence.source.coords - source_centroid;
        let target = correspondence.target.coords - target_centroid;
        covariance += (target * weight) * source.transpose();
        source_spread += weight * source.norm_squared();
    }
    if !covariance.iter().all(|value| value.is_finite()) {
        return Err(SolverError::Numerical {
            reason: "alignment covariance is not finite".to_owned(),
        });
    }
    // The spread only gates general fits and feeds the similarity scale; the
    // rank-checked consensus path keeps its own checks.
    let mean_spread = source_spread * inverse_weight_sum;
    if support == SupportCheck::Spread
        && (mean_spread.is_nan() || mean_spread < MIN_MEAN_SQUARED_SPREAD)
    {
        return Err(SolverError::DegenerateConfiguration {
            reason: "alignment requires spread-out source support".to_owned(),
        });
    }

    let decomposition = covariance.svd(true, true);
    if !decomposition
        .singular_values
        .iter()
        .all(|value| value.is_finite())
    {
        return Err(SolverError::SvdFailed);
    }
    if support == SupportCheck::NonCollinear {
        let leading = decomposition.singular_values[0];
        let second = decomposition.singular_values[1];
        if leading <= 0.0 || second <= RELATIVE_RANK_TOLERANCE * leading {
            return Err(SolverError::DegenerateConfiguration {
                reason: "rigid alignment requires non-collinear source and target support"
                    .to_owned(),
            });
        }
    }

    let u = decomposition.u.ok_or(SolverError::SvdFailed)?;
    let v_t = decomposition.v_t.ok_or(SolverError::SvdFailed)?;
    let mut handedness = Matrix3::identity();
    if (u * v_t).determinant() < 0.0 {
        handedness[(2, 2)] = -1.0;
    }
    let rotation_matrix = u * handedness * v_t;
    if !rotation_matrix.iter().all(|value| value.is_finite())
        || rotation_matrix.determinant() <= 0.0
    {
        return Err(SolverError::Numerical {
            reason: "Kabsch projection did not produce a finite proper rotation".to_owned(),
        });
    }

    let scale = match scale_mode {
        AlignmentScale::Fixed => 1.0,
        AlignmentScale::Estimated => {
            let singular_values = decomposition.singular_values;
            let signed_trace = singular_values[0] * handedness[(0, 0)]
                + singular_values[1] * handedness[(1, 1)]
                + singular_values[2] * handedness[(2, 2)];
            signed_trace / source_spread
        }
    };
    if !scale.is_finite() || scale < MIN_SCALE {
        return Err(SolverError::DegenerateConfiguration {
            reason: "similarity alignment requires spread-out target support".to_owned(),
        });
    }

    let rotation = Rotation3::from_matrix_unchecked(rotation_matrix);
    let translation = target_centroid - scale * (rotation * source_centroid);
    Ok(ClosedFormFit {
        rotation,
        scale,
        translation,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Point3;

    fn exact_data(transform: &Isometry3<f64>) -> Vec<Point3Correspondence> {
        [
            Point3::new(-1.0, 0.2, 0.5),
            Point3::new(0.7, -0.8, 1.2),
            Point3::new(0.3, 1.4, -0.4),
            Point3::new(1.6, 0.6, 0.9),
            Point3::new(-0.5, -1.1, -0.7),
        ]
        .into_iter()
        .map(|source| Point3Correspondence::new(source, transform.transform_point(&source)))
        .collect()
    }

    /// Rigid 3D alignment solver: minimal and overdetermined fits recover the same proper rigid
    /// transform; source-to-target transform direction remains fixed.
    #[test]
    fn minimal_and_overdetermined_fits_recover_the_same_transform() {
        let expected = Isometry3::from_parts(
            Translation3::new(0.4, -0.2, 0.7),
            UnitQuaternion::from_euler_angles(0.17, -0.23, 0.08),
        );
        let data = exact_data(&expected);
        for subset in [&data[..3], data.as_slice()] {
            let fitted = RigidAlignmentSolver::solve_single(subset, &())
                .unwrap()
                .unwrap();
            let rotation_error = (fitted.rotation.inverse() * expected.rotation).angle();
            let translation_error =
                (fitted.translation.vector - expected.translation.vector).norm();
            assert!(rotation_error < 1.0e-12, "rotation error: {rotation_error}");
            assert!(
                translation_error < 1.0e-12,
                "translation error: {translation_error}"
            );
        }
    }

    #[test]
    fn indexed_fit_is_allocation_free_and_checks_indices() {
        let expected = Isometry3::from_parts(
            Translation3::new(-0.1, 0.5, 0.2),
            UnitQuaternion::from_euler_angles(-0.09, 0.11, 0.21),
        );
        let data = exact_data(&expected);
        let fitted = RigidAlignmentSolver::solve_single_from_indices(&[4, 1, 3], &data, &())
            .unwrap()
            .unwrap();
        assert!((fitted.translation.vector - expected.translation.vector).norm() < 1.0e-12);
        assert!(matches!(
            RigidAlignmentSolver::solve_single_from_indices(&[0, 1, 99], &data, &()),
            Err(SolverError::InvalidDataIndex { index: 99, .. })
        ));
    }

    #[test]
    fn rejects_collinear_and_non_finite_inputs() {
        let collinear = (0..4)
            .map(|index| {
                let source = Point3::new(index as f64, 0.0, 0.0);
                Point3Correspondence::new(source, source)
            })
            .collect::<Vec<_>>();
        assert!(matches!(
            RigidAlignmentSolver::solve_single(&collinear, &()),
            Err(SolverError::DegenerateConfiguration { .. })
        ));

        let mut non_finite = collinear;
        non_finite[2].source.x = f64::NAN;
        assert!(matches!(
            RigidAlignmentSolver::solve_single(&non_finite, &()),
            Err(SolverError::Numerical { .. })
        ));
    }

    #[test]
    fn reflection_input_still_returns_a_proper_rotation() {
        let data = [
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(0.0, 0.0, 1.0),
        ]
        .into_iter()
        .map(|source| {
            let target = Point3::new(-source.x, source.y, source.z);
            Point3Correspondence::new(source, target)
        })
        .collect::<Vec<_>>();
        let fitted = RigidAlignmentSolver::solve_single(&data, &())
            .unwrap()
            .unwrap();
        assert!(fitted.rotation.to_rotation_matrix().matrix().determinant() > 0.0);
    }

    #[test]
    fn unweighted_rigid_fit_matches_the_consensus_solver_exactly() {
        let expected = Isometry3::from_parts(
            Translation3::new(0.4, -0.2, 0.7),
            UnitQuaternion::from_euler_angles(0.17, -0.23, 0.08),
        );
        let mut data = exact_data(&expected);
        for (index, correspondence) in data.iter_mut().enumerate() {
            correspondence.target.x += 1.0e-3 * (index as f64 * 1.3).sin();
        }
        let solver = RigidAlignmentSolver::solve_single(&data, &())
            .unwrap()
            .unwrap();
        let fitted = fit_alignment(&data, None, AlignmentScale::Fixed).unwrap();
        assert_eq!(fitted.scaling(), 1.0);
        assert_eq!(fitted.isometry, solver);
        let unit_weights = vec![1.0; data.len()];
        let weighted = fit_alignment(&data, Some(&unit_weights), AlignmentScale::Fixed).unwrap();
        assert_eq!(weighted, fitted);
    }

    #[test]
    fn similarity_fit_recovers_scale_rotation_and_translation() {
        let expected = Similarity3::from_parts(
            Translation3::new(3.0, -2.0, 0.5),
            UnitQuaternion::from_euler_angles(-0.31, 0.12, 0.4),
            2.5,
        );
        let data = [
            Point3::new(-1.0, 0.2, 0.5),
            Point3::new(0.7, -0.8, 1.2),
            Point3::new(0.3, 1.4, -0.4),
            Point3::new(1.6, 0.6, 0.9),
        ]
        .into_iter()
        .map(|source| Point3Correspondence::new(source, expected.transform_point(&source)))
        .collect::<Vec<_>>();
        let fitted = fit_alignment(&data, None, AlignmentScale::Estimated).unwrap();
        assert!((fitted.scaling() - 2.5).abs() < 1.0e-12);
        assert!(
            (fitted.isometry.rotation.inverse() * expected.isometry.rotation).angle() < 1.0e-12
        );
        assert!(
            (fitted.isometry.translation.vector - expected.isometry.translation.vector).norm()
                < 1.0e-12
        );

        let rigid = fit_alignment(&data, None, AlignmentScale::Fixed).unwrap();
        assert_eq!(rigid.scaling(), 1.0);
        assert!((rigid.isometry.rotation.inverse() * expected.isometry.rotation).angle() < 1.0e-12);
    }

    #[test]
    fn weights_select_and_scale_the_support() {
        let expected = Similarity3::from_parts(
            Translation3::new(-0.1, 0.5, 0.2),
            UnitQuaternion::from_euler_angles(-0.09, 0.11, 0.21),
            0.8,
        );
        let mut data = exact_data(&expected.isometry)
            .into_iter()
            .map(|correspondence| {
                Point3Correspondence::new(
                    correspondence.source,
                    expected.transform_point(&correspondence.source),
                )
            })
            .collect::<Vec<_>>();
        data.push(Point3Correspondence::new(
            Point3::new(0.2, 0.2, 0.2),
            Point3::new(50.0, -40.0, 30.0),
        ));

        // A zero weight removes the gross outlier from the fit.
        let mut weights = vec![1.0; data.len()];
        weights[5] = 0.0;
        let fitted = fit_alignment(&data, Some(&weights), AlignmentScale::Estimated).unwrap();
        assert!((fitted.scaling() - 0.8).abs() < 1.0e-12);
        assert!(
            (fitted.isometry.translation.vector - expected.isometry.translation.vector).norm()
                < 1.0e-12
        );

        // An integer weight equals duplicating the correspondence, and a common
        // factor on every weight leaves the fit unchanged.
        let mut noisy = data[..5].to_vec();
        noisy[1].target.y += 0.05;
        noisy[3].target.z -= 0.04;
        let duplicated = [noisy.as_slice(), &noisy[1..2]].concat();
        let by_duplicate = fit_alignment(&duplicated, None, AlignmentScale::Estimated).unwrap();
        let by_weight = fit_alignment(
            &noisy,
            Some(&[1.0, 2.0, 1.0, 1.0, 1.0]),
            AlignmentScale::Estimated,
        )
        .unwrap();
        let by_scaled_weight = fit_alignment(
            &noisy,
            Some(&[3.0, 6.0, 3.0, 3.0, 3.0]),
            AlignmentScale::Estimated,
        )
        .unwrap();
        for other in [by_weight, by_scaled_weight] {
            assert!((other.scaling() - by_duplicate.scaling()).abs() < 1.0e-12);
            assert!(
                (other.isometry.rotation.inverse() * by_duplicate.isometry.rotation).angle()
                    < 1.0e-12
            );
            assert!(
                (other.isometry.translation.vector - by_duplicate.isometry.translation.vector)
                    .norm()
                    < 1.0e-12
            );
        }
    }

    #[test]
    fn general_fit_accepts_collinear_support_that_the_solver_rejects() {
        let expected = Similarity3::from_parts(
            Translation3::new(1.0, 2.0, -1.0),
            UnitQuaternion::from_euler_angles(0.2, 0.3, -0.1),
            1.7,
        );
        let collinear = (0..5)
            .map(|index| {
                let source = Point3::new(index as f64, 0.0, 0.0);
                Point3Correspondence::new(source, expected.transform_point(&source))
            })
            .collect::<Vec<_>>();
        assert!(matches!(
            RigidAlignmentSolver::solve_single(&collinear, &()),
            Err(SolverError::DegenerateConfiguration { .. })
        ));
        let fitted = fit_alignment(&collinear, None, AlignmentScale::Estimated).unwrap();
        assert!((fitted.scaling() - 1.7).abs() < 1.0e-12);
        for correspondence in &collinear {
            assert!(
                (fitted.transform_point(&correspondence.source) - correspondence.target).norm()
                    < 1.0e-12
            );
        }
    }

    #[test]
    fn general_fit_rejects_invalid_weights_and_degenerate_support() {
        let data = exact_data(&Isometry3::identity());
        assert!(matches!(
            fit_alignment(&data[..2], None, AlignmentScale::Fixed),
            Err(SolverError::InsufficientData {
                required: 3,
                provided: 2
            })
        ));
        for weights in [
            vec![1.0; 4],
            vec![1.0, 1.0, -1.0, 1.0, 1.0],
            vec![1.0, f64::NAN, 1.0, 1.0, 1.0],
        ] {
            assert!(matches!(
                fit_alignment(&data, Some(&weights), AlignmentScale::Fixed),
                Err(SolverError::InvalidInput { .. })
            ));
        }
        assert!(matches!(
            fit_alignment(&data, Some(&[0.0; 5]), AlignmentScale::Fixed),
            Err(SolverError::DegenerateConfiguration { .. })
        ));

        let coincident_source = (0..4)
            .map(|index| {
                Point3Correspondence::new(
                    Point3::new(1.0, 2.0, 3.0),
                    Point3::new(index as f64, 0.0, 0.0),
                )
            })
            .collect::<Vec<_>>();
        for scale in [AlignmentScale::Fixed, AlignmentScale::Estimated] {
            assert!(matches!(
                fit_alignment(&coincident_source, None, scale),
                Err(SolverError::DegenerateConfiguration { .. })
            ));
        }

        let collapsed_target = data
            .iter()
            .map(|correspondence| {
                Point3Correspondence::new(correspondence.source, Point3::origin())
            })
            .collect::<Vec<_>>();
        assert!(matches!(
            fit_alignment(&collapsed_target, None, AlignmentScale::Estimated),
            Err(SolverError::DegenerateConfiguration { .. })
        ));

        let mut non_finite = data;
        non_finite[0].target.z = f64::INFINITY;
        assert!(matches!(
            fit_alignment(&non_finite, None, AlignmentScale::Estimated),
            Err(SolverError::Numerical { .. })
        ));
    }

    #[test]
    fn similarity_fit_of_reflected_input_keeps_a_proper_rotation_and_positive_scale() {
        let data = [
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(0.0, 0.0, 1.0),
        ]
        .into_iter()
        .map(|source| Point3Correspondence::new(source, Point3::new(-source.x, source.y, source.z)))
        .collect::<Vec<_>>();
        let fitted = fit_alignment(&data, None, AlignmentScale::Estimated).unwrap();
        assert!(fitted.scaling() > 0.0);
        assert!(
            fitted
                .isometry
                .rotation
                .to_rotation_matrix()
                .matrix()
                .determinant()
                > 0.0
        );
    }

    #[test]
    fn residual_and_squared_residual_agree() {
        let model = Isometry3::identity();
        let datum = Point3Correspondence::new(Point3::origin(), Point3::new(0.1, -0.2, 0.3));
        let residual = RigidAlignmentSolver::residual(&model, &datum);
        assert_eq!(
            residual * residual,
            RigidAlignmentSolver::squared_residual(&model, &datum)
        );
    }
}
