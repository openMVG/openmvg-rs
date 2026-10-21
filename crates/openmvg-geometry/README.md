# openmvg-geometry

Geometric solvers and robust estimation for OpenMVG-rs: two-view models, camera pose,
triangulation, alignment and averaging.

```toml
[dependencies]
openmvg-geometry = "0.1"
```

Recover a homography from 50 matches, 5 of them wrong:

```rust
use nalgebra::{Matrix3, Point2};
use openmvg_core::PixelThreshold;
use openmvg_geometry::correspondence::PixelCorrespondence;
use openmvg_geometry::homography::estimate_homography;
use openmvg_geometry::robust::RobustOptions;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // A known homography maps a grid in the first image to the second image.
    let truth = Matrix3::new(1.1, 0.02, 15.0, -0.03, 0.95, -8.0, 1e-4, -2e-4, 1.0);
    let mut matches: Vec<_> = (0..45)
        .map(|i| {
            let p = Point2::new(20.0 * f64::from(i % 9), 30.0 * f64::from(i / 9));
            let q = truth * p.to_homogeneous();
            PixelCorrespondence::new(p, Point2::new(q.x / q.z, q.y / q.z))
        })
        .collect();
    // Five wrong matches, far from where the homography sends their points.
    matches.extend((0..5).map(|i| {
        let p = Point2::new(15.0 + 30.0 * f64::from(i), 100.0);
        PixelCorrespondence::new(p, Point2::new(p.x + 80.0, p.y - 60.0))
    }));

    let threshold = PixelThreshold::try_new(1.0)?;
    let estimate = estimate_homography(&matches, threshold, &RobustOptions::default())?;
    assert_eq!(estimate.inliers(), (0..45).collect::<Vec<_>>());
    Ok(())
}
```

The [crate documentation](https://docs.rs/openmvg-geometry) gives the learning path through
the solvers, and each solver's page explains its method with a diagram, its failure modes and
its references.

## License

Licensed under either of Apache License, Version 2.0 or the MIT license, at your option. See
`LICENSE-APACHE` and `LICENSE-MIT`.
