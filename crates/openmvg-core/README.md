# openmvg-core

Camera models, poses, identifiers, typed units and the seed protocol that every OpenMVG-rs
crate shares.

```toml
[dependencies]
nalgebra = "0.35"
openmvg-core = "0.1"
```

Project a world point into a camera, then recover the ray that sees it:

```rust
use nalgebra::{Point3, Rotation3};
use openmvg_core::camera::{Camera, CameraModel};
use openmvg_core::WorldToCameraPose;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // A pinhole camera: focal length 800 px, principal point at pixel (320, 240).
    let camera = Camera::try_pinhole(800.0, 800.0, 320.0, 240.0)?;
    // A camera at the world origin that looks along +Z.
    let pose = WorldToCameraPose::from_rotation_center(Rotation3::identity(), Point3::origin())?;

    let point = Point3::new(0.5, -0.25, 4.0);
    let pixel = camera.project(&pose.project_to_camera(&point).coords)?;
    assert!((pixel.x - 420.0).abs() < 1e-9 && (pixel.y - 190.0).abs() < 1e-9);

    // The bearing is the unit ray from the camera center through that pixel.
    let bearing = camera.bearing(&pixel)?;
    assert!((bearing.into_inner() - point.coords.normalize()).norm() < 1e-12);
    Ok(())
}
```

The [crate documentation](https://docs.rs/openmvg-core) states the conventions every crate
shares (pixel origin, camera frame, pose direction), and gives the learning path through the
modules.

## License

Licensed under either of Apache License, Version 2.0 or the MIT license, at your option. See
`LICENSE-APACHE` and `LICENSE-MIT`.
