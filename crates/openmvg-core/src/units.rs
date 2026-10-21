// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Thresholds and residuals that carry their unit in their type.
//!
//! A pixel threshold cannot reach a solver that measures radians: the unit is part of the type,
//! so the compiler rejects a value in the wrong unit. Every threshold is the plain, unsquared
//! bound; [`Threshold::squared`] gives its square for solvers that compare squared residuals.
//!
//! Each robust estimator of `openmvg-geometry` takes the threshold of its residual; the paths
//! below are relative to `openmvg_geometry`:
//!
//! | Threshold | Unit | Taken by |
//! |---|---|---|
//! | [`PixelThreshold`] | pixels | `homography::estimate_homography`, `fundamental::estimate_fundamental`, `pose::p4pf::estimate_absolute_pose_with_focal`, `triangulation::TriangulationConfig::with_max_pixel_residual` |
//! | [`NormalizedImageThreshold`] | pixels divided by the focal length | `essential::estimate_essential` |
//! | [`AngularThreshold`] | radians | `essential::estimate_essential_angular`, `pose::estimate_absolute_pose` (P3P) |
//! | [`WorldDistanceThreshold`] | the scene's own length unit | `alignment::estimate_rigid_alignment` |
//! | [`ScalarThreshold`] | none | a unitless residual of your own solver, through `robust::classic::ClassicConfig` |
//!
//! `estimate_essential` takes bearings, yet its threshold is in normalized image units: it
//! measures its Sampson residual, the first-order distance of a match to its epipolar
//! constraint, on the image plane at unit depth, where a pixel error is that error divided by
//! the focal length. [`Meters`] marks a length known to be metric; the estimators take
//! [`WorldUnits`] instead, because the scale of a scene reconstructed by Structure from Motion
//! (SfM) is arbitrary.
//!
//! A pixel threshold converts to the other image units with a focal length in pixels:
//!
//! ```rust
//! use openmvg_core::{AngularThreshold, NormalizedImageThreshold, PixelThreshold};
//!
//! let pixels = PixelThreshold::try_new(2.0)?;
//! let normalized = NormalizedImageThreshold::from_pixels(pixels, 1000.0)?;
//! assert!((normalized.value() - 0.002).abs() < 1e-15);
//! let angle = AngularThreshold::from_pixels(pixels, 1000.0)?;
//! assert!((angle.value() - 0.002_f64.atan()).abs() < 1e-15);
//! assert_eq!(AngularThreshold::from_degrees(0.5)?, AngularThreshold::HALF_DEGREE);
//! # Ok::<(), openmvg_core::ThresholdError>(())
//! ```

use std::marker::PhantomData;

use thiserror::Error;

/// Pixel-space quantity marker.
// Closed set: uninhabited unit marker with no variants, used only as a type parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pixels {}

/// Angular quantity marker; values are expressed in radians.
// Closed set: uninhabited unit marker with no variants, used only as a type parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Radians {}

/// Metric world-distance quantity marker; values are expressed in meters.
// Closed set: uninhabited unit marker with no variants, used only as a type parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Meters {}

/// World-distance quantity marker in the input scene's coordinate unit.
///
/// The coordinate scale may be arbitrary, as in monocular SfM. This marker
/// does not assert a metric scale in meters.
///
/// ```compile_fail
/// use openmvg_core::{Meters, Threshold, WorldDistanceThreshold};
/// let world = WorldDistanceThreshold::try_new(0.25).unwrap();
/// let _: Threshold<Meters> = world;
/// ```
// Closed set: uninhabited unit marker with no variants, used only as a type parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorldUnits {}

/// Dimensionless normalized-image quantity marker.
// Closed set: uninhabited unit marker with no variants, used only as a type parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormalizedImage {}

/// Dimensionless scalar quantity marker for generic mathematical residuals.
// Closed set: uninhabited unit marker with no variants, used only as a type parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unitless {}

/// Error returned when constructing or converting a geometric threshold.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum ThresholdError {
    /// The threshold is non-finite, non-positive, or too large to square.
    #[error("threshold in {unit} must be positive, finite, and safely squareable")]
    InvalidValue {
        /// Physical unit associated with the rejected threshold.
        unit: &'static str,
    },

    /// A pixel conversion received an invalid focal length.
    #[error("focal length must be finite and positive")]
    InvalidFocalLength,
}

/// Result type for threshold construction and conversion.
pub type ThresholdResult<T> = Result<T, ThresholdError>;

/// A validated, positive, unsquared threshold carrying its physical unit.
///
/// Build one with `try_new` in its unit, or with the conversions and constants of each unit.
/// The value must be finite and positive, and small enough that its square is finite.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct Threshold<Q> {
    value: f64,
    unit: PhantomData<fn() -> Q>,
}

impl<Q> serde::Serialize for Threshold<Q> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_f64(self.value)
    }
}

impl<Q> Threshold<Q> {
    const fn from_validated(value: f64) -> Self {
        Self {
            value,
            unit: PhantomData,
        }
    }

    /// Return the unsquared threshold in the unit carried by `Q`.
    #[must_use]
    pub const fn value(self) -> f64 {
        self.value
    }

    /// Return the squared threshold used by squared-residual scorers.
    #[must_use]
    pub fn squared(self) -> f64 {
        self.value * self.value
    }
}

macro_rules! threshold_constructor {
    ($unit:ty, $name:literal) => {
        impl Threshold<$unit> {
            /// Construct a threshold after validating its value.
            pub fn try_new(value: f64) -> ThresholdResult<Self> {
                validate_threshold(value, $name)?;
                Ok(Self::from_validated(value))
            }
        }
    };
}

macro_rules! threshold_deserialize {
    ($unit:ty) => {
        impl<'de> serde::Deserialize<'de> for Threshold<$unit> {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                let value = <f64 as serde::Deserialize>::deserialize(deserializer)?;
                Self::try_new(value).map_err(serde::de::Error::custom)
            }
        }
    };
}

threshold_constructor!(Pixels, "pixels");
threshold_constructor!(NormalizedImage, "normalized-image units");
threshold_constructor!(Meters, "meters");
threshold_constructor!(WorldUnits, "world-coordinate units");
threshold_constructor!(Unitless, "unitless residuals");
threshold_deserialize!(Pixels);
threshold_deserialize!(NormalizedImage);
threshold_deserialize!(Radians);
threshold_deserialize!(Meters);
threshold_deserialize!(WorldUnits);
threshold_deserialize!(Unitless);

impl Threshold<Radians> {
    /// One microradian.
    pub const ONE_MICRORADIAN: Self = Self::from_validated(1e-6);

    /// Construct an angular threshold from degrees, in `(0, 180]`.
    ///
    /// # Errors
    ///
    /// Returns [`ThresholdError::InvalidValue`] outside that range or for a non-finite value.
    pub fn from_degrees(value: f64) -> ThresholdResult<Self> {
        Self::try_new(value.to_radians())
    }

    /// Construct an angular threshold in radians, in `(0, π]`.
    ///
    /// # Errors
    ///
    /// Returns [`ThresholdError::InvalidValue`] outside that range or for a non-finite value.
    pub fn try_new(value: f64) -> ThresholdResult<Self> {
        if value.is_finite() && value > 0.0 && value <= std::f64::consts::PI {
            Ok(Self::from_validated(value))
        } else {
            Err(ThresholdError::InvalidValue { unit: "radians" })
        }
    }

    /// Convert a normalized-image threshold at the pinhole principal ray.
    pub(crate) fn from_normalized_image(
        normalized: NormalizedImageThreshold,
    ) -> ThresholdResult<Self> {
        Self::try_new(normalized.value().atan())
    }

    /// Convert a pixel threshold to the angle it spans at the principal point:
    /// `atan(pixels / focal_length_pixels)`.
    ///
    /// Away from the principal point the same pixel distance spans a slightly smaller angle.
    ///
    /// # Errors
    ///
    /// Returns [`ThresholdError::InvalidFocalLength`] unless the focal length is finite and
    /// positive.
    pub fn from_pixels(pixels: PixelThreshold, focal_length_pixels: f64) -> ThresholdResult<Self> {
        Self::from_normalized_image(NormalizedImageThreshold::from_pixels(
            pixels,
            focal_length_pixels,
        )?)
    }
}

impl Threshold<NormalizedImage> {
    /// Convert a pixel threshold to normalized image units: `pixels / focal_length_pixels`.
    ///
    /// # Errors
    ///
    /// Returns [`ThresholdError::InvalidFocalLength`] unless the focal length is finite and
    /// positive.
    pub fn from_pixels(pixels: PixelThreshold, focal_length_pixels: f64) -> ThresholdResult<Self> {
        validate_focal_length(focal_length_pixels)?;
        Self::try_new(pixels.value() / focal_length_pixels)
    }
}

impl Threshold<Pixels> {
    /// One-pixel image-space threshold.
    pub const ONE_PIXEL: Self = Self::from_validated(1.0);
    /// Two-pixel image-space threshold.
    pub const TWO_PIXELS: Self = Self::from_validated(2.0);
    /// Four-pixel image-space threshold.
    pub const FOUR_PIXELS: Self = Self::from_validated(4.0);
}

/// One pixel, the same as [`Threshold::ONE_PIXEL`].
impl Default for Threshold<Pixels> {
    fn default() -> Self {
        Self::from_validated(1.0)
    }
}

impl Threshold<Radians> {
    /// One quarter of a degree, stored in radians.
    pub const QUARTER_DEGREE: Self = Self::from_validated(std::f64::consts::PI / 720.0);
    /// One half of a degree, stored in radians.
    pub const HALF_DEGREE: Self = Self::from_validated(std::f64::consts::PI / 360.0);
    /// Two degrees, stored in radians.
    pub const TWO_DEGREES: Self = Self::from_validated(std::f64::consts::PI / 90.0);
    /// Fifteen degrees, stored in radians.
    pub const FIFTEEN_DEGREES: Self = Self::from_validated(std::f64::consts::PI / 12.0);
    /// Twenty degrees, stored in radians.
    pub const TWENTY_DEGREES: Self = Self::from_validated(std::f64::consts::PI / 9.0);
    /// Five milliradians, about 0.29°.
    pub const FIVE_MILLIRADIANS: Self = Self::from_validated(0.005);
}

/// A validated, finite, non-negative residual magnitude carrying its physical unit.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct Residual<Q> {
    value: f64,
    unit: PhantomData<fn() -> Q>,
}

impl<Q> serde::Serialize for Residual<Q> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_f64(self.value)
    }
}

impl<Q> Residual<Q> {
    /// Exact zero residual.
    pub const ZERO: Self = Self {
        value: 0.0,
        unit: PhantomData,
    };

    const fn from_validated(value: f64) -> Self {
        Self {
            value,
            unit: PhantomData,
        }
    }

    /// Return the residual magnitude in the unit carried by `Q`.
    #[must_use]
    pub const fn value(self) -> f64 {
        self.value
    }

    /// Return the squared residual magnitude.
    #[must_use]
    pub fn squared(self) -> f64 {
        self.value * self.value
    }
}

macro_rules! residual_constructor {
    ($unit:ty, $name:literal) => {
        impl Residual<$unit> {
            /// Construct a residual magnitude after validating its value.
            pub fn try_new(value: f64) -> Result<Self, ResidualError> {
                if value.is_finite() && value >= 0.0 && (value * value).is_finite() {
                    Ok(Self::from_validated(value))
                } else {
                    Err(ResidualError { unit: $name })
                }
            }
        }
    };
}

macro_rules! residual_deserialize {
    ($unit:ty) => {
        impl<'de> serde::Deserialize<'de> for Residual<$unit> {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                let value = <f64 as serde::Deserialize>::deserialize(deserializer)?;
                Self::try_new(value).map_err(serde::de::Error::custom)
            }
        }
    };
}

residual_constructor!(Pixels, "pixels");
residual_constructor!(NormalizedImage, "normalized-image units");
residual_constructor!(Meters, "meters");
residual_constructor!(WorldUnits, "world-coordinate units");
residual_constructor!(Unitless, "unitless residuals");
residual_deserialize!(Pixels);
residual_deserialize!(NormalizedImage);
residual_deserialize!(Radians);
residual_deserialize!(Meters);
residual_deserialize!(WorldUnits);
residual_deserialize!(Unitless);

impl Residual<Radians> {
    /// Construct an angular residual in radians, bounded to `[0, pi]`.
    pub fn try_new(value: f64) -> Result<Self, ResidualError> {
        if value.is_finite() && (0.0..=std::f64::consts::PI).contains(&value) {
            Ok(Self::from_validated(value))
        } else {
            Err(ResidualError { unit: "radians" })
        }
    }

    /// Construct an angular residual from degrees.
    pub fn from_degrees(value: f64) -> Result<Self, ResidualError> {
        Self::try_new(value.to_radians())
    }

    /// One quarter of a degree, stored in radians.
    pub const QUARTER_DEGREE: Self = Self::from_validated(std::f64::consts::PI / 720.0);
    /// One degree, stored in radians.
    pub const ONE_DEGREE: Self = Self::from_validated(std::f64::consts::PI / 180.0);
    /// Two degrees, stored in radians.
    pub const TWO_DEGREES: Self = Self::from_validated(std::f64::consts::PI / 90.0);
}

/// Error returned when constructing a geometric residual magnitude.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[error("residual in {unit} must be non-negative, finite, and safely squareable")]
pub struct ResidualError {
    /// Physical unit associated with the rejected residual.
    unit: &'static str,
}

/// Threshold measured in pixels.
///
/// ```
/// let threshold = openmvg_core::PixelThreshold::try_new(2.0)?;
/// # Ok::<(), openmvg_core::ThresholdError>(())
/// ```
pub type PixelThreshold = Threshold<Pixels>;
/// Threshold measured in normalized-image units: pixels divided by the focal length.
///
/// ```
/// use openmvg_core::{NormalizedImageThreshold, PixelThreshold};
///
/// let threshold = NormalizedImageThreshold::from_pixels(PixelThreshold::try_new(2.0)?, 900.0)?;
/// # Ok::<(), openmvg_core::ThresholdError>(())
/// ```
pub type NormalizedImageThreshold = Threshold<NormalizedImage>;
/// Threshold measured in radians.
///
/// ```
/// let threshold = openmvg_core::AngularThreshold::from_degrees(0.5)?;
/// # Ok::<(), openmvg_core::ThresholdError>(())
/// ```
pub type AngularThreshold = Threshold<Radians>;
/// Distance threshold in the input scene's world-coordinate unit.
///
/// ```
/// let threshold = openmvg_core::WorldDistanceThreshold::try_new(0.05)?;
/// # Ok::<(), openmvg_core::ThresholdError>(())
/// ```
pub type WorldDistanceThreshold = Threshold<WorldUnits>;
/// Threshold for a dimensionless scalar residual.
///
/// ```
/// let threshold = openmvg_core::ScalarThreshold::try_new(1e-3)?;
/// # Ok::<(), openmvg_core::ThresholdError>(())
/// ```
pub type ScalarThreshold = Threshold<Unitless>;

/// Residual magnitude measured in pixels.
pub type PixelResidual = Residual<Pixels>;
/// Residual magnitude measured in normalized-image units.
pub type NormalizedImageResidual = Residual<NormalizedImage>;
/// Residual magnitude measured in radians.
pub type AngularResidual = Residual<Radians>;
/// Distance residual magnitude in the input scene's world-coordinate unit.
pub type WorldDistanceResidual = Residual<WorldUnits>;
/// Dimensionless scalar residual magnitude.
pub type ScalarResidual = Residual<Unitless>;

fn validate_threshold(value: f64, unit: &'static str) -> ThresholdResult<()> {
    if value.is_finite() && value > 0.0 && (value * value).is_finite() {
        Ok(())
    } else {
        Err(ThresholdError::InvalidValue { unit })
    }
}

fn validate_focal_length(value: f64) -> ThresholdResult<()> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(ThresholdError::InvalidFocalLength)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_types_validate_without_erasing_values() {
        assert_eq!(PixelThreshold::try_new(2.0).unwrap().value(), 2.0);
        assert_eq!(
            NormalizedImageThreshold::try_new(0.002).unwrap().value(),
            0.002
        );
        assert_eq!(AngularThreshold::try_new(0.01).unwrap().value(), 0.01);
        assert_eq!(WorldDistanceThreshold::try_new(0.05).unwrap().value(), 0.05);
        assert_eq!(PixelResidual::try_new(0.0).unwrap().value(), 0.0);
        assert_eq!(AngularResidual::try_new(0.01).unwrap().value(), 0.01);
    }

    #[test]
    fn invalid_thresholds_and_residuals_are_rejected() {
        for value in [f64::NAN, f64::INFINITY, f64::MAX] {
            assert!(PixelThreshold::try_new(value).is_err());
            assert!(PixelResidual::try_new(value).is_err());
        }
        for value in [0.0, -1.0] {
            assert!(PixelThreshold::try_new(value).is_err());
        }
        assert!(PixelResidual::try_new(-1.0).is_err());
        assert!(AngularThreshold::try_new(std::f64::consts::PI + 1e-12).is_err());
    }

    #[test]
    fn pixel_conversions_preserve_the_existing_equations() {
        let pixels = PixelThreshold::FOUR_PIXELS;
        let normalized = NormalizedImageThreshold::from_pixels(pixels, 800.0).unwrap();
        let direct = AngularThreshold::from_pixels(pixels, 800.0).unwrap();
        let via_normalized = AngularThreshold::from_normalized_image(normalized).unwrap();

        assert_eq!(normalized.value(), 0.005);
        assert_eq!(direct, via_normalized);
        assert_eq!(direct.value(), 0.005_f64.atan());
    }

    #[test]
    fn world_units_preserve_bare_float_serialization() {
        let world = WorldDistanceThreshold::try_new(0.25).unwrap();
        let metric = Threshold::<Meters>::try_new(0.25).unwrap();
        let encoded = serde_json::to_string(&world).unwrap();
        assert_eq!(encoded, "0.25");
        assert_eq!(encoded, serde_json::to_string(&metric).unwrap());
        assert_eq!(
            serde_json::from_str::<WorldDistanceThreshold>(&encoded).unwrap(),
            world
        );

        let world = WorldDistanceResidual::try_new(0.25).unwrap();
        let metric = Residual::<Meters>::try_new(0.25).unwrap();
        let encoded = serde_json::to_string(&world).unwrap();
        assert_eq!(encoded, "0.25");
        assert_eq!(encoded, serde_json::to_string(&metric).unwrap());
        assert_eq!(
            serde_json::from_str::<WorldDistanceResidual>(&encoded).unwrap(),
            world
        );
    }

    #[test]
    fn serde_revalidates_unit_values() {
        let threshold = PixelThreshold::try_new(2.0).unwrap();
        let encoded = serde_json::to_string(&threshold).unwrap();
        assert_eq!(
            serde_json::from_str::<PixelThreshold>(&encoded).unwrap(),
            threshold
        );
        assert!(serde_json::from_str::<PixelThreshold>("0.0").is_err());

        let residual = AngularResidual::ONE_DEGREE;
        let encoded = serde_json::to_string(&residual).unwrap();
        assert_eq!(
            serde_json::from_str::<AngularResidual>(&encoded).unwrap(),
            residual
        );
        assert!(serde_json::from_str::<AngularResidual>("-1.0").is_err());
    }
}
