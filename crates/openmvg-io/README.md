# openmvg-io

Share multi-view datasets and reconstructions in the OMVG community format, check that they
conform, and export a reconstruction to COLMAP. Pure Rust; the only OpenMVG crate that brings
Arrow and Parquet.

```toml
[dependencies]
openmvg-io = "0.1"
# The example below also builds the reconstruction with these; add
# `features = ["image-io"]` to openmvg-sfm to reconstruct a folder of images first.
openmvg-sfm = "0.1"
nalgebra = "0.35"
```

Store a two-view reconstruction, check the files, and read it back:

```rust
use nalgebra::{Point2, Point3, Rotation3};
use openmvg_io::{ReadOptions, WriteOptions, check, read_reconstruction, write_reconstruction};
use openmvg_sfm::model::ImageMeta;
use openmvg_sfm::{
    Camera, Dataset, IntrinsicId, PoseId, Reconstruction, ReconstructionParts, SfmEngine,
    ViewId, WorldToCameraPose,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Two views share one pinhole camera; each view has its own pose.
    let dataset = Dataset::builder()
        .image_meta(["left", "right"].map(|name| ImageMeta::external_id(name, 640, 480)))
        .camera_models([Camera::try_pinhole(500.0, 500.0, 320.0, 240.0)?])
        .shared_intrinsic(IntrinsicId::new(0))
        .build()?;
    let pose = |x| {
        WorldToCameraPose::from_rotation_center(Rotation3::identity(), Point3::new(x, 0.0, 0.0))
    };
    let (cameras, views) = (dataset.camera_models().to_vec(), dataset.views().to_vec());
    // One point, seen in both views at the pixels it projects to.
    let parts = ReconstructionParts::new(7, SfmEngine::Incremental, cameras, views)
        .with_pose(PoseId::new(0), pose(0.0)?)
        .with_pose(PoseId::new(1), pose(1.0)?)
        .with_point(Point3::new(0.5, 0.0, 5.0), vec![
            (ViewId::new(0), Point2::new(370.0, 240.0)),
            (ViewId::new(1), Point2::new(270.0, 240.0)),
        ]);
    let reconstruction = Reconstruction::try_from_parts(parts)?;

    let directory = std::env::temp_dir().join("openmvg-io-example.omvg");
    let _ = std::fs::remove_dir_all(&directory);
    write_reconstruction(&reconstruction, &directory, &WriteOptions::new())?;
    assert!(check(&directory).is_empty());
    let stored = read_reconstruction(&directory, &ReadOptions::new())?;
    assert_eq!(stored.reconstruction.registered_view_count(), 2);
    std::fs::remove_dir_all(&directory)?;
    Ok(())
}
```

Install the `omvg` command and check a dataset from the command line:

```text
$ cargo install openmvg-io
$ omvg validate scene.omvg
conforms
```

The [crate documentation](https://docs.rs/openmvg-io) explains what each function carries, what
a round trip keeps, and how to replay a stored view graph. The format itself is specified in
`docs/format/OMVG_FORMAT_SPEC.md` in the [repository](https://github.com/openMVG/openmvg-rs).

## License

Licensed under either of Apache License, Version 2.0 or the MIT license, at your option. See
`LICENSE-APACHE` and `LICENSE-MIT`.
