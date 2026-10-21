// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Camera models as `cameras` rows (§5.1, §6), and back.
//!
//! ```text
//! Camera::Pinhole       <->  PINHOLE         fx fy cx cy
//! Camera::BrownConrady  <->  BROWN_CONRADY   fx fy cx cy, k1..k3 and p1 p2 as prefixes
//! Camera::Fisheye       <->  FISHEYE         fx fy cx cy, k1..k4 as a prefix
//! Camera::Spherical     <->  SPHERICAL       width = image width, height = width / 2
//! ```
//!
//! A distortion variant names exactly the coefficients it models, so a modeled coefficient whose
//! value is zero stays a value and an unmodeled one stays null (§6).

use openmvg_core::camera::{
    BrownConradyCamera, FisheyeCamera, FisheyeDistortion, PinholeCamera, RadialDistortion,
    SphericalCamera, TangentialDistortion,
};
use openmvg_sfm::{Camera, SfmError};

use super::columns::{Rows, TableBuilder};
use crate::error::{Error, Result};
use crate::format::table::Table;

/// One `cameras` row without its ID.
#[derive(Debug, Clone, Default)]
pub(super) struct CameraRow {
    model: &'static str,
    width: i32,
    height: i32,
    /// `fx`, `fy`, `cx`, `cy`.
    pinhole: Option<[f64; 4]>,
    /// `k1`, `k2`, `k3`, `k4`.
    k: [Option<f64>; 4],
    /// `p1`, `p2`.
    p: [Option<f64>; 2],
    pub(super) calibrated: Option<bool>,
}

fn pinhole_values(pinhole: &PinholeCamera) -> [f64; 4] {
    [
        pinhole.focal_x(),
        pinhole.focal_y(),
        pinhole.principal_x(),
        pinhole.principal_y(),
    ]
}

impl CameraRow {
    /// The row of `camera` for images of `width` x `height` pixels.
    pub(super) fn new(camera: &Camera, width: i32, height: i32, calibrated: bool) -> Result<Self> {
        let mut row = CameraRow {
            width,
            height,
            calibrated: Some(calibrated),
            ..CameraRow::default()
        };
        match camera {
            Camera::Pinhole(pinhole) => {
                row.model = "PINHOLE";
                row.pinhole = Some(pinhole_values(pinhole));
            }
            Camera::BrownConrady(camera) => {
                row.model = "BROWN_CONRADY";
                row.pinhole = Some(pinhole_values(camera.pinhole()));
                row.k = match *camera.radial_distortion() {
                    RadialDistortion::None => [None; 4],
                    RadialDistortion::K1(k1) => [Some(k1), None, None, None],
                    RadialDistortion::K1K2(k1, k2) => [Some(k1), Some(k2), None, None],
                    RadialDistortion::K1K2K3(k1, k2, k3) => [Some(k1), Some(k2), Some(k3), None],
                };
                row.p = match *camera.tangential_distortion() {
                    TangentialDistortion::None => [None; 2],
                    TangentialDistortion::P1(p1) => [Some(p1), None],
                    TangentialDistortion::P1P2(p1, p2) => [Some(p1), Some(p2)],
                };
            }
            Camera::Fisheye(camera) => {
                row.model = "FISHEYE";
                row.pinhole = Some(pinhole_values(camera.pinhole()));
                row.k = match *camera.fisheye_distortion() {
                    FisheyeDistortion::None => [None; 4],
                    FisheyeDistortion::K1(k1) => [Some(k1), None, None, None],
                    FisheyeDistortion::K1K2(k1, k2) => [Some(k1), Some(k2), None, None],
                    FisheyeDistortion::K1K2K3(k1, k2, k3) => [Some(k1), Some(k2), Some(k3), None],
                    FisheyeDistortion::K1K2K3K4(k1, k2, k3, k4) => {
                        [Some(k1), Some(k2), Some(k3), Some(k4)]
                    }
                };
            }
            Camera::Spherical(camera) => {
                row.model = "SPHERICAL";
                if i64::from(camera.width()) != i64::from(width) {
                    return Err(Error::Unsupported(format!(
                        "a spherical camera {} pixels wide is used by images {width} pixels wide",
                        camera.width()
                    )));
                }
            }
        }
        Ok(row)
    }
}

/// The `cameras` table of `rows`, each with its ID.
pub(super) fn cameras_table(rows: &[(i64, CameraRow)]) -> Result<Table> {
    let doubles = |value: &dyn Fn(&CameraRow) -> Option<f64>| -> Vec<Option<f64>> {
        rows.iter().map(|(_, row)| value(row)).collect()
    };
    let pinhole = |i: usize| doubles(&|row| row.pinhole.map(|values| values[i]));
    let k = |i: usize| doubles(&|row| row.k[i]);
    let p = |i: usize| doubles(&|row| row.p[i]);
    let models: Vec<Option<String>> = rows
        .iter()
        .map(|(_, row)| Some(row.model.to_owned()))
        .collect();
    TableBuilder::new("cameras", rows.len())
        .int64("id", rows.iter().map(|(id, _)| *id).collect())
        .nullable_string("model", &models)
        .int32("width", rows.iter().map(|(_, row)| row.width).collect())
        .int32("height", rows.iter().map(|(_, row)| row.height).collect())
        .nullable_double("fx", pinhole(0))
        .nullable_double("fy", pinhole(1))
        .nullable_double("cx", pinhole(2))
        .nullable_double("cy", pinhole(3))
        .nullable_double("k1", k(0))
        .nullable_double("k2", k(1))
        .nullable_double("k3", k(2))
        .nullable_double("k4", k(3))
        .nullable_double("p1", p(0))
        .nullable_double("p2", p(1))
        .nullable_boolean(
            "calibrated",
            rows.iter().map(|(_, row)| row.calibrated).collect(),
        )
        .build()
}

/// The camera model of `cameras` row `row`.
///
/// The row passed every check, so each value §6 requires is present, each it forbids is null,
/// and the coefficients form prefixes. A missing value would read as NaN or zero, which the
/// camera constructors reject instead of trusting.
pub(super) fn camera_from_row(cameras: &Rows<'_>, row: usize) -> Result<Camera> {
    let value = |column: &str| cameras.double(column, row);
    let required = |column: &str| value(column).unwrap_or(f64::NAN);
    let pinhole = || {
        PinholeCamera::try_new(
            required("fx"),
            required("fy"),
            required("cx"),
            required("cy"),
        )
    };
    let k = [value("k1"), value("k2"), value("k3"), value("k4")];
    let camera = match cameras.string("model", row) {
        Some("PINHOLE") => pinhole().map(Camera::Pinhole),
        Some("BROWN_CONRADY") => {
            let radial = match k {
                [Some(k1), Some(k2), Some(k3), _] => RadialDistortion::K1K2K3(k1, k2, k3),
                [Some(k1), Some(k2), None, _] => RadialDistortion::K1K2(k1, k2),
                [Some(k1), None, ..] => RadialDistortion::K1(k1),
                [None, ..] => RadialDistortion::None,
            };
            let tangential = match (value("p1"), value("p2")) {
                (Some(p1), Some(p2)) => TangentialDistortion::P1P2(p1, p2),
                (Some(p1), None) => TangentialDistortion::P1(p1),
                (None, _) => TangentialDistortion::None,
            };
            pinhole()
                .and_then(|pinhole| BrownConradyCamera::try_new(pinhole, radial, tangential))
                .map(Camera::BrownConrady)
        }
        Some("FISHEYE") => {
            let distortion = match k {
                [Some(k1), Some(k2), Some(k3), Some(k4)] => {
                    FisheyeDistortion::K1K2K3K4(k1, k2, k3, k4)
                }
                [Some(k1), Some(k2), Some(k3), None] => FisheyeDistortion::K1K2K3(k1, k2, k3),
                [Some(k1), Some(k2), None, _] => FisheyeDistortion::K1K2(k1, k2),
                [Some(k1), None, ..] => FisheyeDistortion::K1(k1),
                [None, ..] => FisheyeDistortion::None,
            };
            pinhole()
                .and_then(|pinhole| FisheyeCamera::try_new(pinhole, distortion))
                .map(Camera::Fisheye)
        }
        Some("SPHERICAL") => {
            let width = cameras.int("width", row).unwrap_or(0);
            SphericalCamera::try_new(u32::try_from(width).unwrap_or(0)).map(Camera::Spherical)
        }
        other => {
            return Err(Error::Unsupported(format!(
                "camera model {other:?} is unknown"
            )));
        }
    };
    camera.map_err(|error| Error::Sfm(SfmError::InvalidCamera(error)))
}
