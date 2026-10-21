// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fixed-size `SO(3)` values and differential operations.

use std::error::Error;
use std::fmt;

use nalgebra::{Matrix3, Vector3};

use super::{Mat3, from_na, mat_mul, skew, to_na};

const SMALL_ANGLE: f64 = 1e-10;
// Jacobian coefficients subtract nearly equal trigonometric values and need a
// wider series branch than the exponential itself.
const JACOBIAN_SERIES_SQUARED: f64 = 1e-4;
// 64 * sqrt(f64::EPSILON), matching the solver input-validation contract.
const ROTATION_VALIDATION_TOLERANCE: f64 = 9.536_743_164_062_5e-7;
const ROTATION_PRESERVATION_TOLERANCE: f64 = 64.0 * f64::EPSILON;

/// Failure to construct an invariant-preserving rotation.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rotation3Error {
    /// An input component or derived rotation value is NaN or infinite.
    NonFinite,
    /// The supplied matrix is not orthogonal to floating-point tolerance.
    NotOrthogonal,
    /// The supplied orthogonal matrix is a reflection rather than a rotation.
    ImproperDeterminant,
}

impl fmt::Display for Rotation3Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFinite => formatter.write_str("rotation contains a non-finite value"),
            Self::NotOrthogonal => formatter.write_str("rotation matrix is not orthogonal"),
            Self::ImproperDeterminant => {
                formatter.write_str("rotation matrix determinant is not positive")
            }
        }
    }
}

impl Error for Rotation3Error {}

/// A finite, proper 3D rotation stored as a row-major matrix.
///
/// Construction validates the `SO(3)` constraints and removes admitted
/// orthogonality drift with a polar projection. Composition controls the same
/// drift, and the inner matrix is private.
/// This crate uses the matrix representation because geometry kernels can
/// borrow it directly without conversion or allocation.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rotation3(Mat3);

impl Rotation3 {
    /// Identity rotation.
    pub const IDENTITY: Self = Self([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);

    /// Validate and construct a rotation, projecting tolerated orthogonality drift.
    ///
    /// Matrices orthogonal within `64 * f64::EPSILON` retain their original bits.
    pub fn try_from_matrix(matrix: Mat3) -> Result<Self, Rotation3Error> {
        validate_rotation_matrix(&matrix)?;
        let matrix = project_rotation_drift(matrix);
        validate_rotation_matrix(&matrix)?;
        Ok(Self(matrix))
    }

    /// Construct a rotation from a finite scaled axis.
    pub fn try_from_rotation_vector(vector: &[f64; 3]) -> Result<Self, Rotation3Error> {
        let matrix = try_exp_so3(vector).ok_or(Rotation3Error::NonFinite)?;
        Self::try_from_matrix(matrix)
    }

    /// Borrow the row-major rotation matrix.
    pub const fn as_matrix(&self) -> &Mat3 {
        &self.0
    }

    /// Consume this rotation and return its row-major matrix.
    pub const fn into_matrix(self) -> Mat3 {
        self.0
    }

    /// Return the inverse rotation.
    pub fn inverse(self) -> Self {
        Self(std::array::from_fn(|row| {
            std::array::from_fn(|column| self.0[column][row])
        }))
    }

    /// Compose this rotation with `other`, returning `self * other` with drift controlled.
    pub fn compose(self, other: Self) -> Self {
        Self(project_rotation_drift(mat_mul(&self.0, &other.0)))
    }

    /// Apply this rotation to a three-dimensional vector.
    pub fn act(self, vector: &[f64; 3]) -> [f64; 3] {
        std::array::from_fn(|row| {
            self.0[row]
                .iter()
                .zip(vector)
                .map(|(coefficient, value)| coefficient * value)
                .sum()
        })
    }

    /// Principal-branch rotation vector in radians.
    pub fn log(self) -> [f64; 3] {
        log_so3(&self.0)
    }

    /// Adjoint matrix mapping tangent vectors through this rotation.
    pub const fn adjoint(self) -> Mat3 {
        self.0
    }
}

impl TryFrom<Mat3> for Rotation3 {
    type Error = Rotation3Error;

    fn try_from(matrix: Mat3) -> Result<Self, Self::Error> {
        Self::try_from_matrix(matrix)
    }
}

impl From<Rotation3> for Mat3 {
    fn from(rotation: Rotation3) -> Self {
        rotation.into_matrix()
    }
}

impl AsRef<Mat3> for Rotation3 {
    fn as_ref(&self) -> &Mat3 {
        self.as_matrix()
    }
}

/// Failure to construct or operate on an invariant-preserving 3D pose.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pose3Error {
    /// The rotation is not a finite member of `SO(3)`.
    Rotation(Rotation3Error),
    /// At least one translation component is NaN or infinite.
    NonFiniteTranslation,
}

impl fmt::Display for Pose3Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Rotation(error) => error.fmt(formatter),
            Self::NonFiniteTranslation => {
                formatter.write_str("pose translation contains a non-finite value")
            }
        }
    }
}

impl Error for Pose3Error {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Rotation(error) => Some(error),
            Self::NonFiniteTranslation => None,
        }
    }
}

impl From<Rotation3Error> for Pose3Error {
    fn from(error: Rotation3Error) -> Self {
        Self::Rotation(error)
    }
}

/// A finite rigid transform acting as `x_out = rotation * x_in + translation`.
///
/// This type represents the group element only. It does not prescribe the
/// optimization chart: bundle adjustment uses a product-manifold
/// translation update, while pose-graph refinement uses the coupled `SE(3)`
/// exponential.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pose3 {
    rotation: Rotation3,
    translation: [f64; 3],
}

impl Pose3 {
    /// Identity rigid transform.
    pub const IDENTITY: Self = Self {
        rotation: Rotation3::IDENTITY,
        translation: [0.0; 3],
    };

    /// Construct a pose from a checked rotation and finite translation.
    pub fn try_new(rotation: Rotation3, translation: [f64; 3]) -> Result<Self, Pose3Error> {
        if translation.iter().any(|value| !value.is_finite()) {
            return Err(Pose3Error::NonFiniteTranslation);
        }
        Ok(Self {
            rotation,
            translation,
        })
    }

    /// Validate matrix and translation parts and construct a pose.
    pub fn try_from_parts(rotation: Mat3, translation: [f64; 3]) -> Result<Self, Pose3Error> {
        Self::try_new(Rotation3::try_from_matrix(rotation)?, translation)
    }

    /// Rotation component.
    pub const fn rotation(self) -> Rotation3 {
        self.rotation
    }

    /// Translation component.
    pub const fn translation(self) -> [f64; 3] {
        self.translation
    }

    /// Apply the rigid transform to a point.
    pub fn act(self, point: &[f64; 3]) -> [f64; 3] {
        let rotated = self.rotation.act(point);
        std::array::from_fn(|coordinate| rotated[coordinate] + self.translation[coordinate])
    }

    /// Compose this pose with `other`, returning `self * other`.
    ///
    /// Returns an error only if finite input translations overflow while the
    /// composed translation is evaluated.
    pub fn try_compose(self, other: Self) -> Result<Self, Pose3Error> {
        let rotated_translation = self.rotation.act(&other.translation);
        let translation = std::array::from_fn(|coordinate| {
            rotated_translation[coordinate] + self.translation[coordinate]
        });
        Self::try_new(self.rotation.compose(other.rotation), translation)
    }

    /// Return the inverse pose.
    ///
    /// Returns an error only if finite input values overflow while the inverse
    /// translation is evaluated.
    pub fn try_inverse(self) -> Result<Self, Pose3Error> {
        let rotation = self.rotation.inverse();
        let rotated_translation = rotation.act(&self.translation);
        let translation = rotated_translation.map(|value| -value);
        Self::try_new(rotation, translation)
    }
}

/// Exponential map from `so(3)` to `SO(3)`, with a small-angle series.
pub fn exp_so3(vector: &[f64; 3]) -> Mat3 {
    let theta_squared = squared_norm(vector);
    let theta = theta_squared.sqrt();
    if theta < SMALL_ANGLE {
        return matrix_polynomial(vector, 1.0, 1.0, 0.5);
    }
    // Keep the normalized-axis Rodrigues evaluation order: downstream fixed
    // point iterations rely on its deterministic rounding near convergence.
    let axis = vector.map(|value| value / theta);
    let (sine, cosine) = theta.sin_cos();
    matrix_polynomial(&axis, 1.0, sine, 1.0 - cosine)
}

/// Fallible `SO(3)` exponential for an untrusted rotation vector.
///
/// Returns `None` if the input or result is non-finite.
pub fn try_exp_so3(vector: &[f64; 3]) -> Option<Mat3> {
    if !finite_rotation_vector(vector) {
        return None;
    }
    finite_matrix(exp_so3(vector))
}

/// Logarithm from `SO(3)` to `so(3)` on the principal branch.
///
/// Near pi, `R-R^T` loses the rotation axis. In that branch the axis is
/// extracted from the explicitly symmetrized matrix and the small skew
/// component is used only to choose its otherwise ambiguous sign.
pub fn log_so3(rotation: &Mat3) -> [f64; 3] {
    let matrix = to_na(rotation);
    let cos_theta = ((matrix.trace() - 1.0) * 0.5).clamp(-1.0, 1.0);
    let theta = cos_theta.acos();
    let vee = Vector3::new(
        matrix[(2, 1)] - matrix[(1, 2)],
        matrix[(0, 2)] - matrix[(2, 0)],
        matrix[(1, 0)] - matrix[(0, 1)],
    );

    if theta < 1e-7 {
        let scale = 0.5 + theta * theta / 12.0;
        let vector = vee * scale;
        return [vector[0], vector[1], vector[2]];
    }

    if std::f64::consts::PI - theta < 1e-3 {
        // sym(R) = cos(theta) I + (1 - cos(theta)) axis axis^T.
        let symmetric = (matrix + matrix.transpose()) * 0.5;
        let q = (symmetric - Matrix3::identity() * cos_theta) / (1.0 - cos_theta);
        let mut major = 0usize;
        if q[(1, 1)] > q[(major, major)] {
            major = 1;
        }
        if q[(2, 2)] > q[(major, major)] {
            major = 2;
        }
        let mut axis = Vector3::zeros();
        axis[major] = q[(major, major)].max(0.0).sqrt();
        if axis[major] > 1e-8 {
            for coordinate in 0..3 {
                if coordinate != major {
                    axis[coordinate] = q[(major, coordinate)] / axis[major];
                }
            }
        } else {
            axis[major] = 1.0;
        }
        axis /= axis.norm().max(1e-15);
        if vee.dot(&axis) < 0.0 {
            axis = -axis;
        }
        let vector = axis * theta;
        return [vector[0], vector[1], vector[2]];
    }

    let vector = vee * (theta / (2.0 * theta.sin()));
    [vector[0], vector[1], vector[2]]
}

/// Left Jacobian of the `SO(3)` exponential.
///
/// It satisfies `Exp(w + dw) = Exp(J_l(w) dw) Exp(w) + O(||dw||^2)`.
pub fn so3_left_jacobian(vector: &[f64; 3]) -> Mat3 {
    let theta_squared = squared_norm(vector);
    let (a, b) = if theta_squared < JACOBIAN_SERIES_SQUARED {
        let theta_fourth = theta_squared * theta_squared;
        (
            0.5 - theta_squared / 24.0 + theta_fourth / 720.0,
            1.0 / 6.0 - theta_squared / 120.0 + theta_fourth / 5040.0,
        )
    } else {
        let theta = theta_squared.sqrt();
        (
            (1.0 - theta.cos()) / theta_squared,
            (theta - theta.sin()) / (theta_squared * theta),
        )
    };
    matrix_polynomial(vector, 1.0, a, b)
}

/// Inverse left Jacobian of the `SO(3)` exponential.
///
/// The half-angle expression avoids the removable `0/0` cancellation of the
/// common `(1 + cos(theta)) / sin(theta)` form near `theta = pi`.
pub fn so3_left_jacobian_inverse(vector: &[f64; 3]) -> Mat3 {
    let theta_squared = squared_norm(vector);
    let coefficient = if theta_squared < JACOBIAN_SERIES_SQUARED {
        let theta_fourth = theta_squared * theta_squared;
        1.0 / 12.0 + theta_squared / 720.0 + theta_fourth / 30_240.0
    } else {
        let theta = theta_squared.sqrt();
        let half_theta = 0.5 * theta;
        (1.0 - half_theta * (half_theta.cos() / half_theta.sin())) / theta_squared
    };
    matrix_polynomial(vector, 1.0, -0.5, coefficient)
}

fn validate_rotation_matrix(matrix: &Mat3) -> Result<(), Rotation3Error> {
    if matrix.iter().flatten().any(|value| !value.is_finite()) {
        return Err(Rotation3Error::NonFinite);
    }
    let matrix = to_na(matrix);
    let orthogonality_error = (matrix.transpose() * matrix - Matrix3::identity())
        .abs()
        .max();
    if !orthogonality_error.is_finite() {
        return Err(Rotation3Error::NonFinite);
    }
    if orthogonality_error > ROTATION_VALIDATION_TOLERANCE {
        return Err(Rotation3Error::NotOrthogonal);
    }
    let determinant = matrix.determinant();
    if !determinant.is_finite() {
        return Err(Rotation3Error::NonFinite);
    }
    if determinant <= 0.0 {
        return Err(Rotation3Error::ImproperDeterminant);
    }
    Ok(())
}

/// Two Newton–Schulz polar iterations suffice for the admitted near-SO(3) domain.
/// Its Gram defect has spectral norm below 3e-6. The iterations contract it
/// quadratically, reaching floating-point rounding without choosing an axis.
fn project_rotation_drift(matrix: Mat3) -> Mat3 {
    let mut rotation = to_na(&matrix);
    let defect = (rotation.transpose() * rotation - Matrix3::identity())
        .abs()
        .max();
    if defect <= ROTATION_PRESERVATION_TOLERANCE {
        return matrix;
    }
    for _ in 0..2 {
        let correction = Matrix3::identity() * 3.0 - rotation.transpose() * rotation;
        rotation = rotation * correction * 0.5;
    }
    from_na(&rotation)
}

fn squared_norm(vector: &[f64; 3]) -> f64 {
    vector.iter().map(|value| value * value).sum()
}

fn finite_rotation_vector(vector: &[f64; 3]) -> bool {
    vector.iter().all(|value| value.is_finite()) && squared_norm(vector).is_finite()
}

fn matrix_polynomial(vector: &[f64; 3], identity: f64, linear: f64, quadratic: f64) -> Mat3 {
    let skew = to_na(&skew(vector));
    from_na(&(Matrix3::identity() * identity + skew * linear + skew * skew * quadratic))
}

fn finite_matrix(matrix: Mat3) -> Option<Mat3> {
    matrix
        .iter()
        .flatten()
        .all(|value| value.is_finite())
        .then_some(matrix)
}

#[cfg(test)]
mod checked_rotation_tests {
    use super::*;

    #[test]
    fn numerical_rotation_preserves_proper_bits_and_removes_polar_drift() {
        let proper = exp_so3(&[0.2, -0.3, 0.4]);
        let bits = |matrix: Mat3| matrix.map(|row| row.map(f64::to_bits));
        assert_eq!(
            bits(Rotation3::try_from_matrix(proper).unwrap().into_matrix()),
            bits(proper)
        );
        let scale = 1.0 + 4.0e-7;
        let near = [[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, scale]];
        let rotation = Rotation3::try_from_matrix(near).unwrap();
        let acted = rotation.act(&[3.0, 4.0, 0.0]);
        let norm = (acted[0] * acted[0] + acted[1] * acted[1] + acted[2] * acted[2]).sqrt();
        assert!((norm - 5.0).abs() < 1e-12);
        assert!(Rotation3::try_from_matrix(rotation.compose(rotation).into_matrix()).is_ok());

        // R*S has polar factor R for any symmetric positive-definite S.
        // Off-diagonal S tests that correction does not select an axis as Gram-Schmidt would.
        let right_angle = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        let symmetric = [
            [1.0 + 2e-7, 1e-7, 0.0],
            [1e-7, 1.0 - 2e-7, 0.0],
            [0.0, 0.0, 1.0 + 3e-7],
        ];
        let recovered = Rotation3::try_from_matrix(mat_mul(&right_angle, &symmetric)).unwrap();
        for (actual, expected) in recovered
            .as_matrix()
            .iter()
            .flatten()
            .zip(right_angle.iter().flatten())
        {
            assert!((actual - expected).abs() < 1e-12);
        }
        assert!(Rotation3::try_from_matrix([[1e308; 3]; 3]).is_err());
        assert_eq!(
            Rotation3::try_from_matrix([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            Err(Rotation3Error::ImproperDeterminant)
        );
    }

    #[test]
    fn numerical_rotation_composition_is_closed_and_has_the_expected_group_action() {
        let z = Rotation3::try_from_matrix([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
            .unwrap();
        let x = Rotation3::try_from_matrix([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
            .unwrap();
        let point = [3.0, -4.0, 12.0];
        assert_eq!(z.compose(x).act(&point), z.act(&x.act(&point)));
        assert_eq!(z.inverse().act(&z.act(&point)), point);
        let scale = 1.0 + 2.0 * f64::EPSILON;
        let near = [[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, scale]];
        let mut accumulated = Rotation3::try_from_matrix(near).unwrap();
        assert_eq!(accumulated.into_matrix(), near);
        for _ in 0..32 {
            accumulated = accumulated.compose(accumulated);
            assert!(validate_rotation_matrix(accumulated.as_matrix()).is_ok());
            let acted = accumulated.act(&[3.0, 4.0, 0.0]);
            assert!((acted[0].hypot(acted[1]) - 5.0).abs() < 1e-12);
        }
    }
}
