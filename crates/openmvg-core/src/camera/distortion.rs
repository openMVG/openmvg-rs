// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Lens distortion models
//!
//! Lens distortion models commonly used in computer vision for camera calibration,
//! shared by the camera models that apply or remove distortion.
//!
//! ## Distortion types
//!
//! - Radial distortion: caused by lens curvature, affects points symmetrically based on distance from the center
//! - Tangential distortion: caused by lens misalignment, creates asymmetric distortion patterns
//!
//! ## Distortion API
//!
//! - Configuration: [`RadialDistortion`], [`TangentialDistortion`], and
//!   [`FisheyeDistortion`] enums hold the coefficients, re-exported
//!   from [`crate::camera`]
//! - Per-term evaluation: each enum has a low-level `apply()` method
//! - Application and removal: performed by the camera models that own a
//!   distortion configuration (Brown-Conrady and fisheye) through
//!   [`CameraModel`](crate::camera::CameraModel); the coordinate-level helpers in
//!   this module are crate-internal

/// Failure to invert a lens distortion mapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum DistortionError {
    /// The input or an intermediate value was not finite.
    #[error("distortion inversion encountered a non-finite value")]
    NonFinite,
    /// The local distortion Jacobian could not be inverted.
    #[error("distortion inversion encountered a singular Jacobian")]
    SingularJacobian,
    /// The requested point lies beyond the one-to-one branch of the model.
    #[error("coordinate lies outside the invertible distortion domain")]
    OutsideInvertibleDomain,
    /// The iterative inverse did not reach the requested accuracy.
    #[error("distortion inversion did not converge after {iterations} iterations")]
    DidNotConverge {
        /// Number of iterations attempted.
        iterations: usize,
    },
}

/// Radial distortion configuration with 0 to 3 coefficients.
///
/// Radial distortion models the symmetric distortion caused by lens curvature.
/// Higher-order terms can fit stronger distortion but need more calibration data.
///
/// The radial distortion factor is calculated as:
/// ```text
/// radial_factor = 1 + k1*r² + k2*r⁴ + k3*r⁶
/// ```
/// Where `r² = x² + y²` in normalized image coordinates.
///
/// # Coefficient interpretation
///
/// - Positive coefficients: pincushion distortion (points move away from the center)
/// - Negative coefficients: barrel distortion (points move toward the center)
/// - `k1`: primary radial term (largest effect)
/// - `k2`: secondary term (corrects the residual of the `k1` term)
/// - `k3`: tertiary term (for strong wide-angle distortion)
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
// Closed set for v0.1: adding a distortion model is a breaking change.
#[serde(rename_all = "snake_case")]
pub enum RadialDistortion {
    /// No radial distortion.
    ///
    /// Suitable for: Pinhole camera models, high-quality lenses with minimal distortion
    None,

    /// Simple radial distortion with only k1 coefficient.
    ///
    /// Suitable for: Basic distortion correction, simple lenses, initial calibration
    /// Formula: `radial_factor = 1 + k1*r²`
    K1(f64),

    /// Radial distortion with k1 and k2 coefficients.
    ///
    /// Suitable for: Most standard lenses, good balance of accuracy vs complexity
    /// Formula: `radial_factor = 1 + k1*r² + k2*r⁴`
    K1K2(f64, f64),

    /// Full radial distortion with k1, k2, and k3 coefficients.
    ///
    /// Suitable for: Wide-angle lenses, high-precision applications, extreme distortion
    /// Formula: `radial_factor = 1 + k1*r² + k2*r⁴ + k3*r⁶`
    K1K2K3(f64, f64, f64),
}

/// Tangential distortion configuration with 0 to 2 coefficients.
///
/// Tangential distortion models asymmetric distortion caused by lens misalignment
/// or manufacturing imperfections. It creates a "decentering" effect where the
/// optical center is not perfectly aligned with the sensor.
///
/// The tangential distortion offsets are calculated as:
/// ```text
/// dx_tangential = 2*p1*x*y + p2*(r² + 2*x²)
/// dy_tangential = p1*(r² + 2*y²) + 2*p2*x*y
/// ```
/// Where `r² = x² + y²` in normalized image coordinates.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
// Closed set for v0.1: adding a distortion model is a breaking change.
#[serde(rename_all = "snake_case")]
pub enum TangentialDistortion {
    /// No tangential distortion.
    ///
    /// Suitable for: Well-aligned lenses, when tangential effects are negligible
    None,

    /// Tangential distortion with only p1 coefficient.
    ///
    /// Suitable for: Minor lens misalignment, simplified models
    P1(f64),

    /// Full tangential distortion with p1 and p2 coefficients.
    ///
    /// Suitable for: Significant lens misalignment, high-precision applications
    P1P2(f64, f64),
}

/// Fisheye distortion configuration with 0 to 4 coefficients.
///
/// Fisheye distortion models the wide-angle lens distortion based on the angle θ
/// from the optical axis rather than radial distance. This is suitable for
/// equidistant wide-angle lenses represented by the OpenCV fisheye model.
///
/// The polynomial maps the incident angle to equidistant image radius:
/// ```text
/// θ_distorted = θ × (1 + k1×θ² + k2×θ⁴ + k3×θ⁶ + k4×θ⁸)
/// ```
///
/// # Coefficient interpretation
///
/// - Positive coefficients: typically increase the angle (pincushion-like effect)
/// - Negative coefficients: typically decrease the angle (barrel-like effect)
/// - `k1`: primary fisheye term (largest effect)
/// - `k2`: secondary term
/// - `k3`: tertiary term
/// - `k4`: quaternary term (for strong fisheye distortion)
///
/// # Fisheye and radial distortion
///
/// Unlike radial distortion (which uses r²), fisheye distortion uses θ² where θ = atan(r).
/// This makes it more suitable for wide-angle lenses where the relationship between
/// image coordinates and incident angle becomes highly non-linear.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
// Closed set for v0.1: adding a distortion model is a breaking change.
#[serde(rename_all = "snake_case")]
pub enum FisheyeDistortion {
    /// Ideal equidistant fisheye projection with no polynomial correction.
    None,

    /// Simple fisheye distortion with only k1 coefficient.
    ///
    /// Suitable for: Basic wide-angle correction, initial fisheye calibration
    /// Formula: `θ_distorted = θ × (1 + k1×θ²)`
    K1(f64),

    /// Fisheye distortion with k1 and k2 coefficients.
    ///
    /// Suitable for: Most fisheye lenses, good balance of accuracy vs complexity
    /// Formula: `θ_distorted = θ × (1 + k1×θ² + k2×θ⁴)`
    K1K2(f64, f64),

    /// Fisheye distortion with k1, k2, and k3 coefficients.
    ///
    /// Suitable for: High-precision fisheye modeling, extreme wide-angle lenses
    /// Formula: `θ_distorted = θ × (1 + k1×θ² + k2×θ⁴ + k3×θ⁶)`
    K1K2K3(f64, f64, f64),

    /// Full fisheye distortion with k1, k2, k3, and k4 coefficients.
    ///
    /// Suitable for: Ultra-wide fisheye lenses, highest precision applications
    /// Formula: `θ_distorted = θ × (1 + k1×θ² + k2×θ⁴ + k3×θ⁶ + k4×θ⁸)`
    K1K2K3K4(f64, f64, f64, f64),
}

impl RadialDistortion {
    /// Validate that every stored coefficient is finite.
    pub fn validate(&self) -> Result<(), DistortionError> {
        if self.k1().is_finite() && self.k2().is_finite() && self.k3().is_finite() {
            Ok(())
        } else {
            Err(DistortionError::NonFinite)
        }
    }

    /// Returns the k1 coefficient (0.0 if not present).
    pub fn k1(&self) -> f64 {
        match self {
            RadialDistortion::None => 0.0,
            RadialDistortion::K1(k1) => *k1,
            RadialDistortion::K1K2(k1, _) => *k1,
            RadialDistortion::K1K2K3(k1, _, _) => *k1,
        }
    }

    /// Returns the k2 coefficient (0.0 if not present).
    pub fn k2(&self) -> f64 {
        match self {
            RadialDistortion::None | RadialDistortion::K1(_) => 0.0,
            RadialDistortion::K1K2(_, k2) => *k2,
            RadialDistortion::K1K2K3(_, k2, _) => *k2,
        }
    }

    /// Returns the k3 coefficient (0.0 if not present).
    pub fn k3(&self) -> f64 {
        match self {
            RadialDistortion::None | RadialDistortion::K1(_) | RadialDistortion::K1K2(_, _) => 0.0,
            RadialDistortion::K1K2K3(_, _, k3) => *k3,
        }
    }

    /// Returns whether this represents any radial distortion.
    pub fn has_distortion(&self) -> bool {
        !matches!(self, RadialDistortion::None)
    }

    /// Returns the radial distortion factor for a squared normalized radius.
    ///
    /// `r_squared` is `x² + y²` in normalized image coordinates. Multiply the
    /// normalized coordinates by the result; it is 1.0 when there is no distortion.
    pub(crate) fn apply(&self, r_squared: f64) -> f64 {
        match self {
            RadialDistortion::None => 1.0,
            RadialDistortion::K1(k1) => 1.0 + k1 * r_squared,
            RadialDistortion::K1K2(k1, k2) => {
                let r4 = r_squared * r_squared;
                1.0 + k1 * r_squared + k2 * r4
            }
            RadialDistortion::K1K2K3(k1, k2, k3) => {
                let r4 = r_squared * r_squared;
                let r6 = r4 * r_squared;
                1.0 + k1 * r_squared + k2 * r4 + k3 * r6
            }
        }
    }
}

impl FisheyeDistortion {
    /// Validate that every stored coefficient is finite.
    pub fn validate(&self) -> Result<(), DistortionError> {
        if self.k1().is_finite()
            && self.k2().is_finite()
            && self.k3().is_finite()
            && self.k4().is_finite()
        {
            Ok(())
        } else {
            Err(DistortionError::NonFinite)
        }
    }

    /// Returns the k1 coefficient (0.0 if not present).
    pub fn k1(&self) -> f64 {
        match self {
            FisheyeDistortion::None => 0.0,
            FisheyeDistortion::K1(k1) => *k1,
            FisheyeDistortion::K1K2(k1, _) => *k1,
            FisheyeDistortion::K1K2K3(k1, _, _) => *k1,
            FisheyeDistortion::K1K2K3K4(k1, _, _, _) => *k1,
        }
    }

    /// Returns the k2 coefficient (0.0 if not present).
    pub fn k2(&self) -> f64 {
        match self {
            FisheyeDistortion::None | FisheyeDistortion::K1(_) => 0.0,
            FisheyeDistortion::K1K2(_, k2) => *k2,
            FisheyeDistortion::K1K2K3(_, k2, _) => *k2,
            FisheyeDistortion::K1K2K3K4(_, k2, _, _) => *k2,
        }
    }

    /// Returns the k3 coefficient (0.0 if not present).
    pub fn k3(&self) -> f64 {
        match self {
            FisheyeDistortion::None | FisheyeDistortion::K1(_) | FisheyeDistortion::K1K2(_, _) => {
                0.0
            }
            FisheyeDistortion::K1K2K3(_, _, k3) => *k3,
            FisheyeDistortion::K1K2K3K4(_, _, k3, _) => *k3,
        }
    }

    /// Returns the k4 coefficient (0.0 if not present).
    pub fn k4(&self) -> f64 {
        match self {
            FisheyeDistortion::None
            | FisheyeDistortion::K1(_)
            | FisheyeDistortion::K1K2(_, _)
            | FisheyeDistortion::K1K2K3(_, _, _) => 0.0,
            FisheyeDistortion::K1K2K3K4(_, _, _, k4) => *k4,
        }
    }

    /// Returns whether this represents any fisheye distortion.
    pub fn has_distortion(&self) -> bool {
        !matches!(self, FisheyeDistortion::None)
    }

    /// Applies fisheye distortion to the incident angle `theta`, in radians.
    ///
    /// `theta` is the angle from the optical axis, θ = atan(r) with r = √(x² + y²).
    /// The result is the distorted angle θ_distorted in radians, from which the
    /// distorted coordinates are computed.
    ///
    /// # Mathematical model
    ///
    /// The distortion factor is calculated as:
    /// ```text
    /// distortion_factor = 1 + k1×θ² + k2×θ⁴ + k3×θ⁶ + k4×θ⁸
    /// θ_distorted = θ × distortion_factor
    /// ```
    pub(crate) fn apply(&self, theta: f64) -> f64 {
        match self {
            FisheyeDistortion::None => theta,
            FisheyeDistortion::K1(k1) => {
                let theta2 = theta * theta;
                theta * (1.0 + k1 * theta2)
            }
            FisheyeDistortion::K1K2(k1, k2) => {
                let theta2 = theta * theta;
                let theta4 = theta2 * theta2;
                theta * (1.0 + k1 * theta2 + k2 * theta4)
            }
            FisheyeDistortion::K1K2K3(k1, k2, k3) => {
                let theta2 = theta * theta;
                let theta4 = theta2 * theta2;
                let theta6 = theta4 * theta2;
                theta * (1.0 + k1 * theta2 + k2 * theta4 + k3 * theta6)
            }
            FisheyeDistortion::K1K2K3K4(k1, k2, k3, k4) => {
                let theta2 = theta * theta;
                let theta4 = theta2 * theta2;
                let theta6 = theta4 * theta2;
                let theta8 = theta4 * theta4;
                theta * (1.0 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8)
            }
        }
    }

    /// Derivative of the distorted angle with respect to the incident angle.
    pub(crate) fn derivative(&self, theta: f64) -> f64 {
        let theta2 = theta * theta;
        let theta4 = theta2 * theta2;
        let theta6 = theta4 * theta2;
        let theta8 = theta4 * theta4;
        1.0 + 3.0 * self.k1() * theta2
            + 5.0 * self.k2() * theta4
            + 7.0 * self.k3() * theta6
            + 9.0 * self.k4() * theta8
    }

    pub(crate) fn invert_on_validated_interval(
        &self,
        theta_distorted: f64,
        upper: f64,
    ) -> Result<f64, DistortionError> {
        if !theta_distorted.is_finite() || !upper.is_finite() {
            return Err(DistortionError::NonFinite);
        }
        if theta_distorted < 0.0 || upper <= 0.0 {
            return Err(DistortionError::OutsideInvertibleDomain);
        }

        let mapped_upper = self.apply(upper);
        if !mapped_upper.is_finite() {
            return Err(DistortionError::NonFinite);
        }
        if theta_distorted > mapped_upper + 1e-12 {
            return Err(DistortionError::OutsideInvertibleDomain);
        }

        let mut lower = 0.0;
        let mut upper = upper;
        let mut theta = theta_distorted.min(upper);
        for _ in 0..48 {
            let residual = self.apply(theta) - theta_distorted;
            if residual.abs() <= 1e-13 {
                return Ok(theta);
            }
            if residual < 0.0 {
                lower = theta;
            } else {
                upper = theta;
            }

            let derivative = self.derivative(theta);
            let newton = theta - residual / derivative;
            theta =
                if derivative.is_finite() && derivative > 1e-15 && newton > lower && newton < upper
                {
                    newton
                } else {
                    0.5 * (lower + upper)
                };
        }
        let residual = self.apply(theta) - theta_distorted;
        if residual.abs() <= 1e-12 {
            Ok(theta)
        } else {
            Err(DistortionError::DidNotConverge { iterations: 48 })
        }
    }

    /// End of the principal positive-derivative branch, up to `max_theta`.
    ///
    /// With k3=k4=0, a quadratic bound in squared angle finds the first turn.
    /// Higher-order polynomials use sampled derivative checks; these do not
    /// certify the absence of narrow folds between samples.
    pub(crate) fn max_invertible_angle(&self, max_theta: f64) -> Result<f64, DistortionError> {
        if !max_theta.is_finite() {
            return Err(DistortionError::NonFinite);
        }
        if max_theta <= 0.0 {
            return Err(DistortionError::OutsideInvertibleDomain);
        }
        self.validate()?;
        if self.k3() == 0.0 && self.k4() == 0.0 {
            // The angular derivative has the same quadratic coefficients as
            // the two-coefficient Brown radial derivative, with q=theta².
            return Ok(first_radial_turn_squared(self.k1(), self.k2())
                .map_or(max_theta, |squared| squared.sqrt().min(max_theta)));
        }
        if self.k1() >= 0.0 && self.k2() >= 0.0 && self.k3() >= 0.0 && self.k4() >= 0.0 {
            return Ok(max_theta);
        }

        const SAMPLES: usize = 1024;
        let mut previous = 0.0;
        for index in 1..=SAMPLES {
            let theta = max_theta * index as f64 / SAMPLES as f64;
            if self.derivative(theta) <= 0.0 {
                let mut lower = previous;
                let mut upper = theta;
                for _ in 0..64 {
                    let middle = 0.5 * (lower + upper);
                    if self.derivative(middle) > 0.0 {
                        lower = middle;
                    } else {
                        upper = middle;
                    }
                }
                return Ok(lower);
            }
            previous = theta;
        }
        Ok(max_theta)
    }
}

impl TangentialDistortion {
    /// Validate that every stored coefficient is finite.
    pub fn validate(&self) -> Result<(), DistortionError> {
        if self.p1().is_finite() && self.p2().is_finite() {
            Ok(())
        } else {
            Err(DistortionError::NonFinite)
        }
    }

    /// Returns the p1 coefficient (0.0 if not present).
    pub fn p1(&self) -> f64 {
        match self {
            TangentialDistortion::None => 0.0,
            TangentialDistortion::P1(p1) => *p1,
            TangentialDistortion::P1P2(p1, _) => *p1,
        }
    }

    /// Returns the p2 coefficient (0.0 if not present).
    pub fn p2(&self) -> f64 {
        match self {
            TangentialDistortion::None | TangentialDistortion::P1(_) => 0.0,
            TangentialDistortion::P1P2(_, p2) => *p2,
        }
    }

    /// Returns whether this represents any tangential distortion.
    pub fn has_distortion(&self) -> bool {
        !matches!(self, TangentialDistortion::None)
    }

    /// Returns the tangential distortion offsets `(dx, dy)` at a normalized point.
    ///
    /// `x`, `y` are normalized image coordinates and `r_squared` is `x² + y²`.
    /// Add the offsets to the normalized coordinates.
    pub(crate) fn apply(&self, x: f64, y: f64, r_squared: f64) -> (f64, f64) {
        match self {
            TangentialDistortion::None => (0.0, 0.0),
            TangentialDistortion::P1(p1) => {
                let dx = 2.0 * p1 * x * y;
                let dy = p1 * (r_squared + 2.0 * y * y);
                (dx, dy)
            }
            TangentialDistortion::P1P2(p1, p2) => {
                let dx = 2.0 * p1 * x * y + p2 * (r_squared + 2.0 * x * x);
                let dy = p1 * (r_squared + 2.0 * y * y) + 2.0 * p2 * x * y;
                (dx, dy)
            }
        }
    }
}

/// Applies distortion to normalized image coordinates using the Brown-Conrady model.
///
/// This function combines the radial and tangential terms to transform
/// undistorted normalized coordinates to distorted ones. The result can be used with
/// camera intrinsic parameters to convert to pixel coordinates.
///
/// # Mathematical model
///
/// The distortion is applied as follows:
///
/// 1. Radial distortion: `radial_factor = 1 + k1*r² + k2*r⁴ + k3*r⁶` (depending on configuration)
/// 2. Tangential distortion:
///    - `dx_tangential = 2*p1*x*y + p2*(r² + 2*x²)`
///    - `dy_tangential = p1*(r² + 2*y²) + 2*p2*x*y`
/// 3. Combined result:
///    - `x_distorted = x * radial_factor + dx_tangential`
///    - `y_distorted = y * radial_factor + dy_tangential`
///
/// Where `r² = x² + y²` in normalized image coordinates. The inputs `x`, `y` are
/// normalized image coordinates, typically in [-1, 1].
pub(crate) fn apply_distortion(
    x: f64,
    y: f64,
    radial: &RadialDistortion,
    tangential: &TangentialDistortion,
) -> (f64, f64) {
    let r2 = x * x + y * y;

    // Apply radial distortion using the enum's apply method
    let radial_factor = radial.apply(r2);

    // Apply tangential distortion using the enum's apply method
    let (dx_tangential, dy_tangential) = tangential.apply(x, y, r2);

    // Combine radial and tangential distortions
    let x_distorted = x * radial_factor + dx_tangential;
    let y_distorted = y * radial_factor + dy_tangential;

    (x_distorted, y_distorted)
}

/// First finite positive zero of 1 + 3*k1*q + 5*k2*q² in the squared coordinate.
/// Used for Brown k1/k2 without tangential terms and fisheye k1/k2.
pub(crate) fn first_radial_turn_squared(k1: f64, k2: f64) -> Option<f64> {
    let scale = k1.abs().max(k2.abs()).max(1.0);
    let (a, b, c) = (5.0 * (k2 / scale), 3.0 * (k1 / scale), scale.recip());
    if a == 0.0 {
        let root = -c / b;
        return (b < 0.0 && root.is_finite() && root > 0.0).then_some(root);
    }
    let discriminant = b.mul_add(b, -4.0 * a * c);
    if discriminant < 0.0 || !discriminant.is_finite() {
        return None;
    }
    let q = -0.5 * (b + discriminant.sqrt().copysign(b));
    let roots = if q == 0.0 {
        [-b / (2.0 * a), f64::NAN]
    } else {
        [q / a, c / q]
    };
    roots
        .into_iter()
        .filter(|root| root.is_finite() && *root > 0.0)
        .min_by(f64::total_cmp)
}

/// Whether the straight path from the optical axis to `(x, y)` remains on a
/// locally one-to-one branch of the Brown-Conrady mapping. Coefficients must
/// already be validated. Radial k1/k2 uses the analytic turning bound and
/// endpoint conditioning. Higher-order/tangential paths use sampled checks.
pub(crate) fn is_validated_distortion_invertible_at(
    x: f64,
    y: f64,
    radial: &RadialDistortion,
    tangential: &TangentialDistortion,
) -> bool {
    if radial.k3() == 0.0 && tangential.p1() == 0.0 && tangential.p2() == 0.0 {
        let radius_squared = x.mul_add(x, y * y);
        if !radius_squared.is_finite()
            || first_radial_turn_squared(radial.k1(), radial.k2())
                .is_some_and(|bound| radius_squared >= bound)
        {
            return false;
        }
        return distortion_jacobian_determinant(x, y, radial, tangential)
            .is_some_and(|determinant| determinant > 1e-12);
    }
    (0..=32).all(|index| {
        let scale = index as f64 / 32.0;
        distortion_jacobian_determinant(x * scale, y * scale, radial, tangential)
            .is_some_and(|determinant| determinant > 1e-12)
    })
}

fn distortion_jacobian_determinant(
    x: f64,
    y: f64,
    radial: &RadialDistortion,
    tangential: &TangentialDistortion,
) -> Option<f64> {
    distortion_jacobian(x, y, radial, tangential).map(|(_, _, _, determinant)| determinant)
}

pub(crate) fn distortion_jacobian(
    x: f64,
    y: f64,
    radial: &RadialDistortion,
    tangential: &TangentialDistortion,
) -> Option<(f64, f64, f64, f64)> {
    let r2 = x.mul_add(x, y * y);
    let radial_factor = radial.apply(r2);
    let radial_derivative = radial.k1() + 2.0 * (radial.k2() * r2) + 3.0 * (radial.k3() * r2 * r2);
    let p1 = tangential.p1();
    let p2 = tangential.p2();

    let j11 = radial_factor + 2.0 * x * x * radial_derivative + 2.0 * p1 * y + 6.0 * p2 * x;
    let j12 = 2.0 * x * y * radial_derivative + 2.0 * p1 * x + 2.0 * p2 * y;
    let j22 = radial_factor + 2.0 * y * y * radial_derivative + 6.0 * p1 * y + 2.0 * p2 * x;
    let determinant = j11.mul_add(j22, -j12 * j12);
    (j11.is_finite() && j12.is_finite() && j22.is_finite() && determinant.is_finite()).then_some((
        j11,
        j12,
        j22,
        determinant,
    ))
}

pub(crate) fn remove_validated_distortion(
    x_distorted: f64,
    y_distorted: f64,
    radial: &RadialDistortion,
    tangential: &TangentialDistortion,
) -> Result<(f64, f64), DistortionError> {
    // Newton iterations before the inverse reports `DidNotConverge`.
    const MAX_ITERATIONS: usize = 30;
    // Step halvings tried per iteration before the line search stalls.
    const MAX_STEP_HALVINGS: usize = 16;
    // Squared residual, in normalized image coordinates, accepted as converged.
    const TOLERANCE_SQUARED: f64 = 1e-24;
    // Jacobian determinant at or below which a Newton step is treated as singular.
    const SINGULAR_DETERMINANT: f64 = 1e-15;

    if !x_distorted.is_finite() || !y_distorted.is_finite() {
        return Err(DistortionError::NonFinite);
    }
    if !radial.has_distortion() && !tangential.has_distortion() {
        return Ok((x_distorted, y_distorted));
    }

    let mut x = x_distorted;
    let mut y = y_distorted;

    for iteration in 1..=MAX_ITERATIONS {
        let (x_dist, y_dist) = apply_distortion(x, y, radial, tangential);
        let dx = x_distorted - x_dist;
        let dy = y_distorted - y_dist;
        let error_squared = dx.mul_add(dx, dy * dy);
        if error_squared <= TOLERANCE_SQUARED {
            return if is_validated_distortion_invertible_at(x, y, radial, tangential) {
                Ok((x, y))
            } else {
                Err(DistortionError::OutsideInvertibleDomain)
            };
        }

        let (j11, j12, j22, determinant) =
            distortion_jacobian(x, y, radial, tangential).ok_or(DistortionError::NonFinite)?;
        if determinant.abs() <= SINGULAR_DETERMINANT {
            return Err(DistortionError::SingularJacobian);
        }

        let step_x = (j22 * dx - j12 * dy) / determinant;
        let step_y = (j11 * dy - j12 * dx) / determinant;
        if !step_x.is_finite() || !step_y.is_finite() {
            return Err(DistortionError::NonFinite);
        }

        let mut damping = 1.0;
        let mut accepted = false;
        for _ in 0..MAX_STEP_HALVINGS {
            let candidate_x = x + damping * step_x;
            let candidate_y = y + damping * step_y;
            let (candidate_dist_x, candidate_dist_y) =
                apply_distortion(candidate_x, candidate_y, radial, tangential);
            let candidate_dx = x_distorted - candidate_dist_x;
            let candidate_dy = y_distorted - candidate_dist_y;
            let candidate_error = candidate_dx.mul_add(candidate_dx, candidate_dy * candidate_dy);
            if candidate_error.is_finite() && candidate_error < error_squared {
                x = candidate_x;
                y = candidate_y;
                accepted = true;
                break;
            }
            damping *= 0.5;
        }

        if !accepted {
            return Err(DistortionError::DidNotConverge {
                iterations: iteration,
            });
        }
    }

    Err(DistortionError::DidNotConverge {
        iterations: MAX_ITERATIONS,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stalled_inverse_reports_the_iteration_it_stopped_at() {
        // Barrel distortion k1 = -0.5 caps the distorted radius near 0.544, so a target of 1.0
        // has no preimage: the damped search stalls at the radius of maximum distortion.
        let result = remove_validated_distortion(
            1.0,
            0.0,
            &RadialDistortion::K1(-0.5),
            &TangentialDistortion::None,
        );
        assert!(
            matches!(result, Err(DistortionError::DidNotConverge { iterations }) if (1..30).contains(&iterations)),
            "{result:?}"
        );
    }

    #[test]
    fn test_radial_distortion_coefficients() {
        let none = RadialDistortion::None;
        assert_eq!(none.k1(), 0.0);
        assert_eq!(none.k2(), 0.0);
        assert_eq!(none.k3(), 0.0);
        assert!(!none.has_distortion());

        let k1 = RadialDistortion::K1(-0.1);
        assert_eq!(k1.k1(), -0.1);
        assert_eq!(k1.k2(), 0.0);
        assert_eq!(k1.k3(), 0.0);
        assert!(k1.has_distortion());

        let k1k2 = RadialDistortion::K1K2(-0.1, 0.01);
        assert_eq!(k1k2.k1(), -0.1);
        assert_eq!(k1k2.k2(), 0.01);
        assert_eq!(k1k2.k3(), 0.0);

        let k1k2k3 = RadialDistortion::K1K2K3(-0.1, 0.01, -0.001);
        assert_eq!(k1k2k3.k1(), -0.1);
        assert_eq!(k1k2k3.k2(), 0.01);
        assert_eq!(k1k2k3.k3(), -0.001);
    }

    #[test]
    fn fisheye_two_coefficient_domain_finds_narrow_turn() {
        use crate::camera::{CameraModel, FisheyeCamera, ProjectionError};
        use nalgebra::Vector3;
        let distortion =
            FisheyeDistortion::K1K2(-0.668_135_802_086_100_7, 0.200_882_434_393_896_44);
        // derivative = (1-theta²/0.99875²)*(1-theta²/0.99905²).
        // Its negative interval lies between the former 1024 sample locations.
        let bound = distortion
            .max_invertible_angle(std::f64::consts::PI)
            .unwrap();
        assert!((bound - 0.99875).abs() < 1e-9);
        assert!(distortion.derivative(0.9989) < 0.0);
        assert_eq!(distortion.max_invertible_angle(0.75).unwrap(), 0.75);
        let camera = FisheyeCamera::try_with_focal(300.0, 320.0, 240.0, distortion).unwrap();
        let outside = Vector3::new(1.1_f64.sin(), 0.0, 1.1_f64.cos());
        assert_eq!(
            camera.project(&outside),
            Err(ProjectionError::OutsideModelDomain)
        );
        let direction = Vector3::new(0.6_f64.sin(), 0.0, 0.6_f64.cos());
        let pixel = camera.project(&direction).unwrap();
        assert!((camera.unproject(&pixel).unwrap() - direction).norm() < 1e-10);
    }

    #[test]
    fn test_tangential_distortion_coefficients() {
        let none = TangentialDistortion::None;
        assert_eq!(none.p1(), 0.0);
        assert_eq!(none.p2(), 0.0);
        assert!(!none.has_distortion());

        let p1 = TangentialDistortion::P1(0.001);
        assert_eq!(p1.p1(), 0.001);
        assert_eq!(p1.p2(), 0.0);
        assert!(p1.has_distortion());

        let p1p2 = TangentialDistortion::P1P2(0.001, 0.002);
        assert_eq!(p1p2.p1(), 0.001);
        assert_eq!(p1p2.p2(), 0.002);
    }

    #[test]
    fn test_radial_distortion_apply() {
        let none = RadialDistortion::None;
        assert_eq!(none.apply(0.5), 1.0);

        let k1 = RadialDistortion::K1(-0.1);
        assert_eq!(k1.apply(0.5), 1.0 + (-0.1) * 0.5);

        let k1k2 = RadialDistortion::K1K2(-0.1, 0.01);
        let expected = 1.0 + (-0.1) * 0.5 + 0.01 * (0.5 * 0.5);
        assert!((k1k2.apply(0.5) - expected).abs() < 1e-6);
    }

    #[test]
    fn test_tangential_distortion_apply() {
        let none = TangentialDistortion::None;
        assert_eq!(none.apply(0.1, 0.2, 0.05), (0.0, 0.0));

        let p1 = TangentialDistortion::P1(0.001);
        let (dx, dy) = p1.apply(0.1, 0.2, 0.05);
        assert!((dx - 2.0 * 0.001 * 0.1 * 0.2).abs() < 1e-6);
        assert!((dy - 0.001 * (0.05 + 2.0 * 0.2 * 0.2)).abs() < 1e-6);
    }

    #[test]
    fn test_apply_distortion_no_distortion() {
        let radial = RadialDistortion::None;
        let tangential = TangentialDistortion::None;

        let (x_dist, y_dist) = apply_distortion(0.5, 0.3, &radial, &tangential);
        assert_eq!(x_dist, 0.5);
        assert_eq!(y_dist, 0.3);
    }

    #[test]
    fn test_apply_distortion_radial_only() {
        let radial = RadialDistortion::K1(-0.1);
        let tangential = TangentialDistortion::None;

        let (x_dist, y_dist) = apply_distortion(0.5, 0.3, &radial, &tangential);

        // Calculate expected result manually
        let r2 = 0.5 * 0.5 + 0.3 * 0.3;
        let radial_factor = 1.0 + (-0.1) * r2;
        let expected_x = 0.5 * radial_factor;
        let expected_y = 0.3 * radial_factor;

        assert!((x_dist - expected_x).abs() < 1e-6);
        assert!((y_dist - expected_y).abs() < 1e-6);
    }

    #[test]
    fn test_apply_distortion_tangential_only() {
        let radial = RadialDistortion::None;
        let tangential = TangentialDistortion::P1P2(0.001, 0.002);

        let (x_dist, y_dist) = apply_distortion(0.5, 0.3, &radial, &tangential);

        // Calculate expected result manually
        let r2 = 0.5 * 0.5 + 0.3 * 0.3;
        let dx_expected = 2.0 * 0.001 * 0.5 * 0.3 + 0.002 * (r2 + 2.0 * 0.5 * 0.5);
        let dy_expected = 0.001 * (r2 + 2.0 * 0.3 * 0.3) + 2.0 * 0.002 * 0.5 * 0.3;

        assert!((x_dist - (0.5 + dx_expected)).abs() < 1e-6);
        assert!((y_dist - (0.3 + dy_expected)).abs() < 1e-6);
    }

    #[test]
    fn test_remove_distortion_no_distortion() {
        let radial = RadialDistortion::None;
        let tangential = TangentialDistortion::None;

        let (x_undist, y_undist) =
            remove_validated_distortion(0.5, 0.3, &radial, &tangential).expect("identity inverse");
        assert_eq!(x_undist, 0.5);
        assert_eq!(y_undist, 0.3);
    }

    #[test]
    fn test_apply_remove_distortion_roundtrip() {
        let radial = RadialDistortion::K1K2(-0.1, 0.01);
        let tangential = TangentialDistortion::P1P2(0.001, 0.001);

        // Test several points
        let test_points = vec![
            (0.0, 0.0),  // Center point
            (0.5, 0.3),  // Off-center
            (-0.2, 0.4), // Different quadrant
            (0.1, -0.6), // Another quadrant
        ];

        for (x_orig, y_orig) in test_points {
            // Apply distortion
            let (x_dist, y_dist) = apply_distortion(x_orig, y_orig, &radial, &tangential);

            // Remove distortion
            let (x_recovered, y_recovered) =
                remove_validated_distortion(x_dist, y_dist, &radial, &tangential)
                    .expect("test calibration is invertible");

            // Should recover original coordinates within tolerance
            assert!(
                (x_recovered - x_orig).abs() < 1e-5,
                "X roundtrip failed: {} -> {} -> {}",
                x_orig,
                x_dist,
                x_recovered
            );
            assert!(
                (y_recovered - y_orig).abs() < 1e-5,
                "Y roundtrip failed: {} -> {} -> {}",
                y_orig,
                y_dist,
                y_recovered
            );
        }
    }

    #[test]
    fn newton_inverse_converges_for_monotone_high_order_radial_model() {
        let radial = RadialDistortion::K1K2K3(0.05, 0.01, 0.001);
        let tangential = TangentialDistortion::None;
        let distorted = apply_distortion(2.0, 0.0, &radial, &tangential);

        let recovered = remove_validated_distortion(distorted.0, distorted.1, &radial, &tangential)
            .expect("strictly monotone radial distortion must be invertible");

        assert!((recovered.0 - 2.0).abs() < 1e-10);
        assert!(recovered.1.abs() < 1e-10);
    }

    #[test]
    fn fallible_inverse_rejects_non_finite_input() {
        let result = remove_validated_distortion(
            f64::NAN,
            0.0,
            &RadialDistortion::K1(0.1),
            &TangentialDistortion::None,
        );
        assert_eq!(result, Err(DistortionError::NonFinite));
    }

    // ==== FISHEYE DISTORTION TESTS ====

    #[test]
    fn test_fisheye_distortion_coefficients() {
        let none = FisheyeDistortion::None;
        assert_eq!(none.k1(), 0.0);
        assert_eq!(none.k2(), 0.0);
        assert_eq!(none.k3(), 0.0);
        assert_eq!(none.k4(), 0.0);
        assert!(!none.has_distortion());

        let k1 = FisheyeDistortion::K1(-0.1);
        assert_eq!(k1.k1(), -0.1);
        assert_eq!(k1.k2(), 0.0);
        assert_eq!(k1.k3(), 0.0);
        assert_eq!(k1.k4(), 0.0);
        assert!(k1.has_distortion());

        let k1k2 = FisheyeDistortion::K1K2(-0.1, 0.01);
        assert_eq!(k1k2.k1(), -0.1);
        assert_eq!(k1k2.k2(), 0.01);
        assert_eq!(k1k2.k3(), 0.0);
        assert_eq!(k1k2.k4(), 0.0);

        let k1k2k3 = FisheyeDistortion::K1K2K3(-0.1, 0.01, -0.001);
        assert_eq!(k1k2k3.k1(), -0.1);
        assert_eq!(k1k2k3.k2(), 0.01);
        assert_eq!(k1k2k3.k3(), -0.001);
        assert_eq!(k1k2k3.k4(), 0.0);

        let k1k2k3k4 = FisheyeDistortion::K1K2K3K4(-0.1, 0.01, -0.001, 0.0001);
        assert_eq!(k1k2k3k4.k1(), -0.1);
        assert_eq!(k1k2k3k4.k2(), 0.01);
        assert_eq!(k1k2k3k4.k3(), -0.001);
        assert_eq!(k1k2k3k4.k4(), 0.0001);
    }

    #[test]
    fn test_fisheye_distortion_apply() {
        let none = FisheyeDistortion::None;
        let theta = 0.5;
        assert_eq!(none.apply(theta), theta);

        let k1 = FisheyeDistortion::K1(-0.1);
        let expected = theta * (1.0 + (-0.1) * theta * theta);
        assert!((k1.apply(theta) - expected).abs() < 1e-6);

        let k1k2 = FisheyeDistortion::K1K2(-0.1, 0.01);
        let theta2 = theta * theta;
        let theta4 = theta2 * theta2;
        let expected_k1k2 = theta * (1.0 + (-0.1) * theta2 + 0.01 * theta4);
        assert!((k1k2.apply(theta) - expected_k1k2).abs() < 1e-6);
    }
}
