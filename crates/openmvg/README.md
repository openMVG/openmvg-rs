# openmvg

Multiple View Geometry and Structure from Motion (SfM) in Rust, behind one dependency: cameras,
features, geometric solvers, the Incremental and Global reconstruction engines, and the OMVG
community format, under `openmvg::core`, `features`, `geometry`, `sfm` and `io`.

```toml
[dependencies]
openmvg = { version = "0.1", features = ["sfm-image-io", "io"] }
```

Reconstruct a folder of images, then save the points and cameras as a PLY point cloud and as
a COLMAP model. The example reads the Sceaux Castle images
(`git clone https://github.com/openMVG/ImageDataset_SceauxCastle`); pass your own folder the
same way:

```rust,no_run
use openmvg::io::write_colmap;
use openmvg::sfm::{Dataset, Incremental};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dataset = Dataset::from_dir("ImageDataset_SceauxCastle/images")?;
    let output = Incremental::default().run(&dataset)?;
    let reconstruction = output.reconstruction();
    println!("{} views registered", reconstruction.registered_view_count());
    reconstruction.write_ply("scene.ply")?;
    write_colmap(reconstruction, "colmap")?;
    Ok(())
}
```

The [crate documentation](https://docs.rs/openmvg) maps each goal to its module, shows how to
choose the front end and the engine and how the crates fit together, defines the terms, and
gives the learning path through the library.

## License

Licensed under either of Apache License, Version 2.0 or the MIT license, at your option. See
`LICENSE-APACHE` and `LICENSE-MIT`.
