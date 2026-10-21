// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Cross-family conformance tests for the public solver failure contract.

use nalgebra::{Point2, Point3, UnitVector3, Vector3};
use openmvg_geometry::SolverError;
use openmvg_geometry::correspondence::{BearingCorrespondence, PixelCorrespondence};
use openmvg_geometry::{
    essential::FivePointSolver,
    fundamental::{EightPointSolver, SevenPointSolver},
    homography::{DltSolver, HomographyConfig},
    normalization::NormalizedCorrespondences,
    pose::BearingToPoint,
    pose::p3p::{self, P3pMethod},
};

#[test]
fn malformed_values_are_errors() {
    assert!(matches!(
        BearingCorrespondence::try_new(Vector3::new(f64::NAN, 0.0, 1.0), Vector3::z()),
        Err(SolverError::InvalidInput { .. })
    ));
    assert!(matches!(
        BearingCorrespondence::try_new(Vector3::z(), Vector3::new(0.0, 0.0, f64::INFINITY)),
        Err(SolverError::InvalidInput { .. })
    ));

    let finite_pixel = PixelCorrespondence::new(Point2::origin(), Point2::origin());
    let mut seven_point = [finite_pixel; 7];
    seven_point[0].point2.y = f64::INFINITY;
    assert!(matches!(
        SevenPointSolver::solve(&NormalizedCorrespondences::from_pixels(&seven_point)),
        Err(SolverError::InvalidInput { .. })
    ));
    let mut fundamental_eight_point = [finite_pixel; 8];
    fundamental_eight_point[0].point1.y = f64::NAN;
    assert!(matches!(
        EightPointSolver::solve(&NormalizedCorrespondences::from_pixels(
            &fundamental_eight_point
        )),
        Err(SolverError::InvalidInput { .. })
    ));
    let mut homography = [finite_pixel; 4];
    homography[0].point1.x = f64::NAN;
    assert!(matches!(
        DltSolver::solve(
            &NormalizedCorrespondences::from_pixels(&homography),
            &HomographyConfig::default()
        ),
        Err(SolverError::InvalidInput { .. })
    ));

    let invalid_unit = UnitVector3::new_unchecked(Vector3::new(f64::NAN, 0.0, 1.0));
    assert!(matches!(
        BearingToPoint::try_new(Point3::origin(), invalid_unit),
        Err(SolverError::InvalidInput { .. })
    ));
}

/// Minimal-solver degeneracy classification: valid but degenerate measurements return no model
/// rather than fabricated geometry; degeneracy remains distinct from invalid-input errors.
#[test]
fn valid_degenerate_geometry_returns_no_model() {
    let repeated_bearing = BearingCorrespondence::try_new(Vector3::z(), Vector3::z()).unwrap();
    assert!(
        FivePointSolver::solve(&[repeated_bearing; 5])
            .unwrap()
            .is_empty()
    );
    #[cfg(feature = "unstable-test-support")]
    assert!(
        openmvg_geometry::essential::test_support::EightPointSolver::solve(&[repeated_bearing; 8])
            .unwrap()
            .is_empty()
    );

    let repeated_pixel = PixelCorrespondence::new(Point2::origin(), Point2::origin());
    assert!(
        SevenPointSolver::solve(&NormalizedCorrespondences::from_pixels(
            &[repeated_pixel; 7]
        ))
        .unwrap()
        .is_empty()
    );
    assert!(
        EightPointSolver::solve(&NormalizedCorrespondences::from_pixels(
            &[repeated_pixel; 8]
        ))
        .unwrap()
        .is_empty()
    );
    assert!(
        DltSolver::solve(
            &NormalizedCorrespondences::from_pixels(&[repeated_pixel; 4]),
            &HomographyConfig::default()
        )
        .unwrap()
        .is_empty()
    );

    let bearing = UnitVector3::new_normalize(Vector3::z());
    assert!(p3p::solve([Point3::origin(); 3], [bearing; 3], P3pMethod::default()).is_empty());
}
