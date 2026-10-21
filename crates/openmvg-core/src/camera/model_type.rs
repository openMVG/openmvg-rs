// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Camera model type
//!
//! The name of a camera model, without its parameters.
//!
//! It says which model to estimate before the parameters are known, for example for each
//! camera of an `openmvg-sfm` image project.
//!
//! ## Usage
//!
//! ```rust
//! use openmvg_core::camera::CameraModelType;
//!
//! // Default is PinholeSingle (most common case)
//! let model = CameraModelType::default();
//! assert_eq!(model, CameraModelType::PinholeSingle);
//!
//! // Parse a canonical name (the same spelling `Display` and serde use)
//! let model: CameraModelType = "pinhole_radial3".parse().unwrap();
//! assert_eq!(model, CameraModelType::PinholeRadial3);
//!
//! // Display for serialization
//! assert_eq!(CameraModelType::PinholeBrown.to_string(), "pinhole_brown");
//! ```

use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

/// Which camera model to instantiate for each view.
///
/// Each variant is a [`Camera`](super::Camera) variant with a fixed set of parameters to
/// estimate; it names the model before the parameters are known. Not every consumer takes
/// every variant: an `openmvg-sfm` image project accepts `PinholeSingle`, `PinholeRadial1`,
/// `PinholeRadial2`, and `Spherical`, and EXIF-grouped image ingestion the three perspective
/// ones. Build the other models directly as a [`Camera`](super::Camera).
///
/// The distorted perspective models estimate one focal length `f`, so `fx = fy`. A camera built
/// directly, such as with [`Camera::try_brown_conrady`](super::Camera::try_brown_conrady), may
/// have separate `fx` and `fy`. The parameter counts are:
///
/// | Variant | Parameters | Description |
/// |---------|------------|-------------|
/// | `PinholeSingle` | 3 | f, cx, cy |
/// | `PinholeDouble` | 4 | fx, fy, cx, cy |
/// | `PinholeRadial1` | 4 | f, cx, cy, k1 |
/// | `PinholeRadial2` | 5 | f, cx, cy, k1, k2 |
/// | `PinholeRadial3` | 6 | f, cx, cy, k1, k2, k3 |
/// | `PinholeBrown` | 8 | f, cx, cy, k1, k2, k3, p1, p2 |
/// | `Fisheye` | 7 | f, cx, cy, k1, k2, k3, k4 |
/// | `Spherical` | 1 | width |
///
/// # Examples
///
/// ```rust
/// use openmvg_core::camera::CameraModelType;
///
/// let model = CameraModelType::PinholeRadial3;
/// assert_eq!(model.to_string(), "pinhole_radial3");
/// assert_eq!(model.parameter_count(), 6);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
// Closed set for v0.1: adding a camera model is a breaking change.
pub enum CameraModelType {
    /// Pinhole camera with single focal length (3 params: f, cx, cy).
    ///
    /// This is the simplest and most common model, assuming square pixels.
    /// Use this when the camera has equal horizontal and vertical focal lengths.
    #[default]
    PinholeSingle,

    /// Pinhole camera with separate fx/fy (4 params: fx, fy, cx, cy).
    ///
    /// Use this when the camera has non-square pixels or different
    /// horizontal and vertical focal lengths.
    PinholeDouble,

    /// Pinhole + single radial distortion coefficient (4 params: f, cx, cy, k1).
    ///
    /// Suitable for cameras with mild barrel or pincushion distortion.
    /// The distortion model is: r_distorted = r × (1 + k1 × r²)
    PinholeRadial1,

    /// Pinhole + two radial distortion coefficients (5 params: f, cx, cy, k1, k2).
    ///
    /// The distortion model is: r_distorted = r × (1 + k1×r² + k2×r⁴)
    PinholeRadial2,

    /// Pinhole + three radial distortion coefficients (6 params: f, cx, cy, k1, k2, k3).
    ///
    /// Suitable for most consumer cameras with moderate distortion.
    /// The distortion model is: r_distorted = r × (1 + k1×r² + k2×r⁴ + k3×r⁶)
    PinholeRadial3,

    /// Pinhole + full Brown-Conrady distortion (8 params: f, cx, cy, k1, k2, k3, p1, p2).
    ///
    /// Includes both radial (k1, k2, k3) and tangential (p1, p2) distortion.
    /// Use this for high-accuracy calibration or when tangential distortion
    /// is significant (e.g., due to lens decentering).
    PinholeBrown,

    /// Fisheye camera with theta-based distortion (7 params: f, cx, cy, k1, k2, k3, k4).
    ///
    /// Suitable for ultra-wide-angle lenses with field of view approaching
    /// or exceeding 180°. Uses angle-based distortion model.
    Fisheye,

    /// Spherical/equirectangular camera (1 param: width).
    ///
    /// For 360° panoramic cameras. Projects to/from equirectangular format
    /// with fixed 2:1 aspect ratio.
    Spherical,
}

impl CameraModelType {
    /// Returns the number of intrinsic parameters: focal length(s), principal point and
    /// distortion parameters.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use openmvg_core::camera::CameraModelType;
    ///
    /// assert_eq!(CameraModelType::PinholeSingle.parameter_count(), 3);
    /// assert_eq!(CameraModelType::PinholeRadial3.parameter_count(), 6);
    /// assert_eq!(CameraModelType::PinholeBrown.parameter_count(), 8);
    /// ```
    #[inline]
    pub const fn parameter_count(&self) -> usize {
        match self {
            CameraModelType::PinholeSingle => 3,  // f, cx, cy
            CameraModelType::PinholeDouble => 4,  // fx, fy, cx, cy
            CameraModelType::PinholeRadial1 => 4, // f, cx, cy, k1
            CameraModelType::PinholeRadial2 => 5, // f, cx, cy, k1, k2
            CameraModelType::PinholeRadial3 => 6, // f, cx, cy, k1, k2, k3
            CameraModelType::PinholeBrown => 8,   // f, cx, cy, k1, k2, k3, p1, p2
            CameraModelType::Fisheye => 7,        // f, cx, cy, k1, k2, k3, k4
            CameraModelType::Spherical => 1,      // width
        }
    }

    /// Returns the number of distortion parameters for this camera model.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use openmvg_core::camera::CameraModelType;
    ///
    /// assert_eq!(CameraModelType::PinholeSingle.distortion_parameter_count(), 0);
    /// assert_eq!(CameraModelType::PinholeRadial3.distortion_parameter_count(), 3);
    /// assert_eq!(CameraModelType::PinholeBrown.distortion_parameter_count(), 5);
    /// ```
    #[inline]
    pub const fn distortion_parameter_count(&self) -> usize {
        match self {
            CameraModelType::PinholeSingle => 0,
            CameraModelType::PinholeDouble => 0,
            CameraModelType::PinholeRadial1 => 1, // k1
            CameraModelType::PinholeRadial2 => 2, // k1, k2
            CameraModelType::PinholeRadial3 => 3, // k1, k2, k3
            CameraModelType::PinholeBrown => 5,   // k1, k2, k3, p1, p2
            CameraModelType::Fisheye => 4,        // k1, k2, k3, k4
            CameraModelType::Spherical => 0,
        }
    }

    /// Returns whether this camera model has distortion parameters.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use openmvg_core::camera::CameraModelType;
    ///
    /// assert!(!CameraModelType::PinholeSingle.has_distortion());
    /// assert!(CameraModelType::PinholeRadial3.has_distortion());
    /// ```
    #[inline]
    pub const fn has_distortion(&self) -> bool {
        self.distortion_parameter_count() > 0
    }

    /// Returns a human-readable description of the camera model.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use openmvg_core::camera::CameraModelType;
    ///
    /// assert_eq!(
    ///     CameraModelType::PinholeRadial3.description(),
    ///     "Pinhole + radial distortion (k1, k2, k3)"
    /// );
    /// ```
    #[inline]
    pub const fn description(&self) -> &'static str {
        match self {
            CameraModelType::PinholeSingle => "Pinhole (single focal length)",
            CameraModelType::PinholeDouble => "Pinhole (separate fx, fy)",
            CameraModelType::PinholeRadial1 => "Pinhole + radial distortion (k1)",
            CameraModelType::PinholeRadial2 => "Pinhole + radial distortion (k1, k2)",
            CameraModelType::PinholeRadial3 => "Pinhole + radial distortion (k1, k2, k3)",
            CameraModelType::PinholeBrown => "Pinhole + Brown-Conrady (k1, k2, k3, p1, p2)",
            CameraModelType::Fisheye => "Fisheye (theta-based distortion)",
            CameraModelType::Spherical => "Spherical (equirectangular)",
        }
    }

    /// Returns all available camera model types.
    ///
    /// Useful for CLI help text or configuration validation.
    #[inline]
    pub const fn all() -> &'static [CameraModelType] {
        &[
            CameraModelType::PinholeSingle,
            CameraModelType::PinholeDouble,
            CameraModelType::PinholeRadial1,
            CameraModelType::PinholeRadial2,
            CameraModelType::PinholeRadial3,
            CameraModelType::PinholeBrown,
            CameraModelType::Fisheye,
            CameraModelType::Spherical,
        ]
    }
}

impl CameraModelType {
    /// Canonical snake_case name, as printed by `Display` and accepted by `FromStr`.
    pub const fn as_str(self) -> &'static str {
        match self {
            CameraModelType::PinholeSingle => "pinhole_single",
            CameraModelType::PinholeDouble => "pinhole_double",
            CameraModelType::PinholeRadial1 => "pinhole_radial1",
            CameraModelType::PinholeRadial2 => "pinhole_radial2",
            CameraModelType::PinholeRadial3 => "pinhole_radial3",
            CameraModelType::PinholeBrown => "pinhole_brown",
            CameraModelType::Fisheye => "fisheye",
            CameraModelType::Spherical => "spherical",
        }
    }
}

impl fmt::Display for CameraModelType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Parses exactly the canonical names written by [`CameraModelType::as_str`],
/// `Display` and serde (for example `"pinhole_radial3"`); no other spelling is
/// accepted.
impl FromStr for CameraModelType {
    type Err = ParseCameraModelTypeError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::all()
            .iter()
            .copied()
            .find(|model| model.as_str() == s)
            .ok_or_else(|| ParseCameraModelTypeError {
                input: s.to_string(),
            })
    }
}

/// Error returned when parsing an invalid camera model type string.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct ParseCameraModelTypeError {
    /// The invalid input string.
    pub input: String,
}

impl fmt::Display for ParseCameraModelTypeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "unknown camera model type '{}', expected one of: {}",
            self.input,
            CameraModelType::all()
                .iter()
                .map(|m| m.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}

impl std::error::Error for ParseCameraModelTypeError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_is_complete_and_consistent() {
        let cases = [
            (CameraModelType::PinholeSingle, "pinhole_single", 3, 0),
            (CameraModelType::PinholeDouble, "pinhole_double", 4, 0),
            (CameraModelType::PinholeRadial1, "pinhole_radial1", 4, 1),
            (CameraModelType::PinholeRadial2, "pinhole_radial2", 5, 2),
            (CameraModelType::PinholeRadial3, "pinhole_radial3", 6, 3),
            (CameraModelType::PinholeBrown, "pinhole_brown", 8, 5),
            (CameraModelType::Fisheye, "fisheye", 7, 4),
            (CameraModelType::Spherical, "spherical", 1, 0),
        ];

        assert_eq!(CameraModelType::default(), CameraModelType::PinholeSingle);
        assert_eq!(CameraModelType::all().len(), cases.len());
        for ((model, name, parameters, distortion), listed) in
            cases.into_iter().zip(CameraModelType::all())
        {
            assert_eq!(model, *listed);
            assert_eq!(model.to_string(), name);
            assert_eq!(model.parameter_count(), parameters);
            assert_eq!(model.distortion_parameter_count(), distortion);
            assert_eq!(model.has_distortion(), distortion > 0);
            assert!(!model.description().is_empty());
        }
    }

    #[test]
    fn parses_only_canonical_names() {
        for model in CameraModelType::all() {
            assert_eq!(model.as_str().parse::<CameraModelType>(), Ok(*model));
        }
        // Aliases, other casings and COLMAP names (whose PINHOLE and
        // RADIAL denote different models) are rejected.
        for input in [
            "pinhole",
            "PINHOLE",
            "RADIAL",
            "simple",
            "brown",
            "360",
            "PINHOLE_SINGLE",
            "pinhole-single",
        ] {
            assert!(input.parse::<CameraModelType>().is_err(), "{input}");
        }
    }

    #[test]
    fn invalid_name_reports_the_input() {
        let result = "unknown_model".parse::<CameraModelType>();
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.input, "unknown_model");
        assert!(err.to_string().contains("unknown camera model type"));
    }

    #[test]
    fn serde_uses_canonical_names_and_roundtrips() {
        for model in CameraModelType::all() {
            let json = serde_json::to_string(model).unwrap();
            assert_eq!(json, format!("\"{model}\""));
            let parsed: CameraModelType = serde_json::from_str(&json).unwrap();
            assert_eq!(*model, parsed);
        }
    }
}
