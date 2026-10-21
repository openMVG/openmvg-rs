// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use openmvg_geometry::robust::threshold::{AngularThreshold, PixelThreshold};
use proptest::prelude::*;

#[test]
fn threshold_types_reject_invalid_values() {
    for value in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(PixelThreshold::try_new(value).is_err());
        assert!(AngularThreshold::try_new(value).is_err());
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    #[test]
    fn angular_threshold_is_monotonic_in_focal_length(
        pixels in 0.01_f64..20.0,
        focal in 100.0_f64..5_000.0,
        extra_focal in 1.0_f64..5_000.0,
    ) {
        let pixels = PixelThreshold::try_new(pixels).unwrap();
        let wide_angle = AngularThreshold::from_pixels(pixels, focal).unwrap();
        let narrow_angle = AngularThreshold::from_pixels(pixels, focal + extra_focal).unwrap();
        prop_assert!(narrow_angle.value() < wide_angle.value());
    }
}
