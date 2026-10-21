# openmvg-ba

Bundle adjustment (BA) for OpenMVG-rs: refine camera poses, optional intrinsics and 3D points
together, so that every point reprojects as close as possible to the pixels where it was
observed.

```toml
[dependencies]
openmvg-ba = "0.1"
```

Two fixed cameras see nine points; refine the points from a start 5 cm off on each axis:

```rust
use nalgebra::{Point3, Rotation3, Vector3};
use openmvg_ba::{
    BundleAdjuster, BundleIntrinsic, BundleObservation, BundlePoint, BundlePose, BundleProblem,
    IntrinsicVariables, NativeBundleAdjuster, PoseVariables, SolveConfig,
};
use openmvg_core::camera::{Camera, CameraModel};
use openmvg_core::model::{CameraIntrinsics, WorldToCameraPose};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let camera = Camera::try_pinhole(800.0, 800.0, 320.0, 240.0)?;
    let intrinsic = CameraIntrinsics::try_new(camera.clone(), 640, 480)?.with_fixed(true);
    let mut poses = Vec::new();
    for x in [0.0, 1.0] {
        let center = Point3::new(x, 0.0, 0.0);
        poses.push(WorldToCameraPose::from_rotation_center(Rotation3::identity(), center)?);
    }
    let points: Vec<_> =
        (0..9).map(|i| Point3::new(f64::from(i % 3) - 0.5, f64::from(i / 3) - 1.0, 5.0)).collect();

    // One observation per camera and point: (pose index, intrinsic index, point index, pixel).
    let mut observations = Vec::new();
    for (pose_index, pose) in poses.iter().enumerate() {
        for (point_index, point) in points.iter().enumerate() {
            let pixel = camera.project(&pose.project_to_camera(point).coords)?;
            observations.push(BundleObservation::try_new(pose_index, 0, point_index, pixel)?);
        }
    }

    let offset = Vector3::new(0.05, -0.05, 0.05);
    let start = points.iter().map(|point| BundlePoint::try_new(point + offset));
    let problem = BundleProblem::builder(
        poses.into_iter().map(|pose| BundlePose::new(pose, PoseVariables::FIXED)).collect(),
        vec![BundleIntrinsic::try_new(intrinsic, IntrinsicVariables::FIXED)?],
        start.collect::<Result<_, _>>()?,
        observations,
    )
    .build()?;
    let solution = NativeBundleAdjuster::new().solve(&problem, SolveConfig::default())?;
    assert!(solution.report().final_rmse() < 1e-6);
    Ok(())
}
```

The [crate documentation](https://docs.rs/openmvg-ba) explains the gauge, the robust losses,
how the native adjuster solves a problem and which problems it supports, and how to read the
report before using a result.

## License

Licensed under either of Apache License, Version 2.0 or the MIT license, at your option. See
`LICENSE-APACHE` and `LICENSE-MIT`.
