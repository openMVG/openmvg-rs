// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Calibrated five-point essential matrix solver.
//!
//! Its teaching docs are on [`FivePointSolver`], where docs.rs shows them.

use crate::correspondence::BearingCorrespondence;
use crate::{SolverError, SolverResult};
use arrayvec::ArrayVec;
use nalgebra::Matrix3;

use crate::residuals::{sampson_error_essential, sampson_error_essential_squared};
use crate::solver_traits::{MultiSolutionSolver, Solver};

use super::EssentialMatrix;
use super::five_point_kernel::{self, MAX_SOLUTIONS, Pose, Poses};

/// Fixed-capacity result of a calibrated five-point minimal solve.
///
/// Crate-private: the public five-point surface is [`FivePointSolver::solve`]
/// and the robust `estimate_essential*` entry points. It is reachable from
/// the allocation-contract test through `essential::test_support`.
#[derive(Debug, Clone, Default, PartialEq)]
#[must_use]
pub struct FivePointSolutions {
    models: ArrayVec<EssentialMatrix, MAX_SOLUTIONS>,
}

impl FivePointSolutions {
    /// Number of admitted essential matrices.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.models.len()
    }

    /// Whether no valid candidate was found.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
    }

    /// Candidate matrices in deterministic solver order.
    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[EssentialMatrix] {
        self.models.as_slice()
    }

    /// Iterate over the candidates without allocating.
    #[inline]
    pub fn iter(&self) -> std::slice::Iter<'_, EssentialMatrix> {
        self.models.iter()
    }

    /// Convert to the workspace-wide multi-solution representation.
    #[must_use]
    pub fn into_vec(self) -> Vec<EssentialMatrix> {
        self.models.into_iter().collect()
    }
}

impl std::ops::Index<usize> for FivePointSolutions {
    type Output = EssentialMatrix;

    fn index(&self, index: usize) -> &Self::Output {
        &self.as_slice()[index]
    }
}

impl<'a> IntoIterator for &'a FivePointSolutions {
    type Item = &'a EssentialMatrix;
    type IntoIter = std::slice::Iter<'a, EssentialMatrix>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl IntoIterator for FivePointSolutions {
    type Item = EssentialMatrix;
    type IntoIter = arrayvec::IntoIter<EssentialMatrix, MAX_SOLUTIONS>;

    fn into_iter(self) -> Self::IntoIter {
        self.models.into_iter()
    }
}

/// Calibrated five-point solver.
///
/// Returns, in a deterministic order, the essential matrices `E = [t]ₓR/√2`
/// (unit Frobenius norm) of the real roots whose relative pose places at least
/// three of the five points in front of both cameras, so the bearing
/// directions (not only their lines) matter. The RANSAC path never allocates.
///
/// # How it works
///
/// Each call solves one exact sample of five bearing correspondences, without heap allocation.
///
/// ```text
/// ┌──────────────────────────────────────────────────────────────────────┐
/// │                    Five-point essential solver                       │
/// └──────────────────────────────────────────────────────────────────────┘
/// Input: 5 bearing correspondences
///        │
///        ▼
/// ┌──────────────────────────────────────────────────────────────────────┐
/// │ Step 1: Epipolar null space                                          │
/// └──────────────────────────────────────────────────────────────────────┘
/// Each pair gives b₂ᵀ E b₁ = 0. Householder QR of the 5×9 system yields
/// the null space E = x E₁ + y E₂ + z E₃ + E₄.
///                                  │
///                                  ▼
/// ┌──────────────────────────────────────────────────────────────────────┐
/// │ Step 2: Degree-10 polynomial                                         │
/// └──────────────────────────────────────────────────────────────────────┘
/// det E = 0 and 2 E Eᵀ E − tr(E Eᵀ) E = 0 give a 10×20 cubic system;
/// Nistér elimination reduces it to a degree-10 polynomial in z (redone
/// in a fixed rotated frame when a root sits near z = ∞).
///                                  │
///                                  ▼
/// ┌──────────────────────────────────────────────────────────────────────┐
/// │ Step 3: Real roots                                                   │
/// └──────────────────────────────────────────────────────────────────────┘
/// Sturm isolation and safeguarded Newton; near-multiple roots add the
/// roots of a compensated (double-double) determinant.
///                                  │
///                                  ▼
/// ┌──────────────────────────────────────────────────────────────────────┐
/// │ Step 4: Pose per root                                                │
/// └──────────────────────────────────────────────────────────────────────┘
/// Keep roots with at least 3 of the 5 points in front of both cameras,
/// then polish (R, t) by Gauss-Newton on the rotation manifold. When the
/// five matches fit a pure rotation to within about 0.6°, the compensated
/// roots and a Procrustes candidate are added.
///                                  │
///                                  ▼
///              Output: E = [t]ₓR/√2 candidates
/// ```
///
/// # References
///
/// - D. Nistér, "An Efficient Solution to the Five-Point Relative Pose Problem", IEEE TPAMI
///   26(6), 2004.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FivePointSolver;

impl FivePointSolver {
    /// Solve one exact sample without allocating the result container.
    ///
    /// Evidence helper for the unit tests and the `unstable-test-support`
    /// allocation-contract test; library code reaches the kernel through
    /// the solver traits.
    #[cfg(any(test, feature = "unstable-test-support"))]
    pub(crate) fn solve_minimal(data: &[BearingCorrespondence; 5]) -> FivePointSolutions {
        let (bearings1, bearings2) = split_bearings(data);
        let mut poses = Poses::new();
        five_point_kernel::solve(&bearings1, &bearings2, &mut poses);
        // Same capacity as the kernel buffer, so the collection cannot overflow.
        FivePointSolutions {
            models: poses.iter().map(essential).collect(),
        }
    }
}

crate::solver_traits::expose_solver_api!(FivePointSolver, BearingCorrespondence, EssentialMatrix);

type Bearings = [[f64; 3]; 5];

fn split_bearings(data: &[BearingCorrespondence; 5]) -> (Bearings, Bearings) {
    (
        std::array::from_fn(|index| (*data[index].bearing1()).into()),
        std::array::from_fn(|index| (*data[index].bearing2()).into()),
    )
}

fn sample_size_error(provided: usize) -> SolverError {
    if provided < 5 {
        SolverError::InsufficientData {
            required: 5,
            provided,
        }
    } else {
        SolverError::InvalidSampleSize {
            expected: 5,
            provided,
        }
    }
}

fn essential(pose: &Pose) -> EssentialMatrix {
    let e = &pose.e;
    EssentialMatrix::new(Matrix3::new(
        e[0], e[1], e[2], e[3], e[4], e[5], e[6], e[7], e[8],
    ))
}

impl Solver for FivePointSolver {
    type Datum = BearingCorrespondence;
    type Model = EssentialMatrix;
    type Config = ();
    type Unit = openmvg_core::Radians;

    #[inline]
    fn residual(model: &Self::Model, datum: &Self::Datum) -> f64 {
        sampson_error_essential(&model.0, datum)
    }

    #[inline]
    fn squared_residual(model: &Self::Model, datum: &Self::Datum) -> f64 {
        sampson_error_essential_squared(&model.0, datum)
    }

    fn min_sample_size() -> usize {
        5
    }

    fn is_valid(model: &Self::Model) -> bool {
        model.is_valid()
    }
}

impl MultiSolutionSolver for FivePointSolver {
    fn solve(data: &[Self::Datum], _config: &Self::Config) -> SolverResult<Vec<Self::Model>> {
        let sample: &[BearingCorrespondence; 5] =
            data.try_into().map_err(|_| sample_size_error(data.len()))?;
        let (bearings1, bearings2) = split_bearings(sample);
        let mut poses = Poses::new();
        five_point_kernel::solve(&bearings1, &bearings2, &mut poses);
        Ok(poses.iter().map(essential).collect())
    }

    fn for_each_solution_from_indices<F>(
        indices: &[usize],
        data: &[Self::Datum],
        _config: &Self::Config,
        mut emit: F,
    ) -> SolverResult<usize>
    where
        F: FnMut(Self::Model),
    {
        let indices: &[usize; 5] = indices
            .try_into()
            .map_err(|_| sample_size_error(indices.len()))?;
        let mut bearings1 = [[0.0; 3]; 5];
        let mut bearings2 = [[0.0; 3]; 5];
        for (slot, &index) in indices.iter().enumerate() {
            let datum = data.get(index).ok_or(SolverError::InvalidDataIndex {
                index,
                data_len: data.len(),
            })?;
            bearings1[slot] = (*datum.bearing1()).into();
            bearings2[slot] = (*datum.bearing2()).into();
        }
        let mut poses = Poses::new();
        five_point_kernel::solve(&bearings1, &bearings2, &mut poses);
        poses.iter().map(essential).for_each(&mut emit);
        Ok(poses.len())
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn five_point_classic_contract_uses_angular_thresholds() {
        use crate::robust::classic::ClassicConfig;
        use openmvg_core::AngularThreshold;

        let estimator = ClassicConfig::ransac(AngularThreshold::HALF_DEGREE).build();
        let result = estimator.estimate::<FivePointSolver>(&[], &());
        assert!(matches!(
            result,
            Err(crate::robust::RobustError::InsufficientData {
                required: 5,
                provided: 0
            })
        ));
    }

    use super::*;
    use nalgebra::{Rotation3, Vector3};

    /// Correspondences of `points` seen from the identity camera and from `(rotation, translation)`.
    fn observe(
        rotation: &Rotation3<f64>,
        translation: &Vector3<f64>,
        points: &[Vector3<f64>],
    ) -> Vec<BearingCorrespondence> {
        points
            .iter()
            .map(|point| {
                BearingCorrespondence::try_new(*point, rotation * point + translation).unwrap()
            })
            .collect()
    }

    fn generic_scene() -> (Rotation3<f64>, Vector3<f64>, Vec<BearingCorrespondence>) {
        let rotation = Rotation3::from_euler_angles(0.15, -0.1, 0.08);
        let translation = Vector3::new(-0.3, 0.6, 0.1).normalize();
        let points = [
            Vector3::new(0.4, -0.2, 3.0),
            Vector3::new(-0.7, 0.3, 4.5),
            Vector3::new(0.1, 0.8, 2.5),
            Vector3::new(-0.3, -0.6, 5.0),
            Vector3::new(0.9, 0.5, 3.8),
        ];
        let correspondences = observe(&rotation, &translation, &points);
        (rotation, translation, correspondences)
    }

    /// Projective distance between two essential matrices (sign and scale free).
    fn projective_distance(a: &Matrix3<f64>, b: &Matrix3<f64>) -> f64 {
        let (a, b) = (a.normalize(), b.normalize());
        (a - b).norm().min((a + b).norm())
    }

    #[test]
    fn exact_sample_contract_is_explicit() {
        let (_, _, correspondences) = generic_scene();
        assert_eq!(FivePointSolver::min_sample_size(), 5);
        assert_eq!(
            FivePointSolver::solve(&correspondences[..4]),
            Err(SolverError::InsufficientData {
                required: 5,
                provided: 4,
            })
        );
        let six = [correspondences.as_slice(), &correspondences[..1]].concat();
        assert_eq!(
            FivePointSolver::solve(&six),
            Err(SolverError::InvalidSampleSize {
                expected: 5,
                provided: 6,
            })
        );
        assert_eq!(
            <FivePointSolver as MultiSolutionSolver>::for_each_solution_from_indices(
                &[0, 1, 2, 3, 5],
                &correspondences,
                &(),
                |_| {},
            ),
            Err(SolverError::InvalidDataIndex {
                index: 5,
                data_len: 5,
            })
        );
    }

    #[test]
    fn recovers_the_generating_essential_matrix_with_unit_norm() {
        let (rotation, translation, correspondences) = generic_scene();
        let truth = translation.cross_matrix() * rotation.matrix();
        let candidates = FivePointSolver::solve(&correspondences).unwrap();
        assert!(!candidates.is_empty());
        for candidate in &candidates {
            assert!((candidate.matrix().norm() - 1.0).abs() < 1e-12);
            assert!(FivePointSolver::is_valid(candidate));
        }
        let best = candidates
            .iter()
            .map(|candidate| projective_distance(candidate.matrix(), &truth))
            .fold(f64::INFINITY, f64::min);
        assert!(best < 1e-10, "closest candidate is {best:e} away");
    }

    #[test]
    fn fixed_and_indexed_paths_match_vec_adapter() {
        let (_, _, correspondences) = generic_scene();
        let fixed = FivePointSolver::solve_minimal(correspondences.as_slice().try_into().unwrap());
        let adapted = FivePointSolver::solve(&correspondences).unwrap();
        assert_eq!(fixed.as_slice(), adapted.as_slice());

        let mut emitted = Vec::new();
        let count = <FivePointSolver as MultiSolutionSolver>::for_each_solution_from_indices(
            &[0, 1, 2, 3, 4],
            &correspondences,
            &(),
            |model| emitted.push(model),
        )
        .unwrap();
        assert_eq!(count, fixed.len());
        assert_eq!(emitted.as_slice(), fixed.as_slice());
    }

    #[test]
    fn invalid_bearings_are_rejected_and_degenerate_samples_return_no_model() {
        assert!(
            BearingCorrespondence::try_new(Vector3::new(f64::NAN, 0.0, 1.0), Vector3::z()).is_err()
        );

        let repeated = vec![BearingCorrespondence::try_new(Vector3::z(), Vector3::z()).unwrap(); 5];
        assert!(FivePointSolver::solve(&repeated).unwrap().is_empty());
    }
}
