# openmvg-sfm

Structure from Motion (SfM) for OpenMVG-rs: turn a set of overlapping photographs into camera
poses, camera models and a sparse cloud of 3D points, with an Incremental and a Global engine.

```toml
[dependencies]
openmvg-sfm = { version = "0.1", features = ["image-io"] }
```

`image-io` reads folders of JPEG, PNG and TIFF images; add `console-progress` for terminal
progress bars.

Reconstruct a folder of images and save the points:

```rust,no_run
use openmvg_sfm::{Dataset, Frontend, Incremental, ViewGraph};

fn main() -> openmvg_sfm::Result<()> {
    let dataset = Dataset::from_dir("images")?;
    let graph = ViewGraph::build(&dataset, &Frontend::default())?;
    let output = Incremental::default().run_prepared(&graph)?;

    let registered = output.reconstruction().registered_view_count();
    println!("{registered} of {} views registered", dataset.views().len());
    output.reconstruction().write_ply("sparse.ply")?;
    Ok(())
}
```

The [crate documentation](https://docs.rs/openmvg-sfm) explains how each engine works and when
to choose it, how to read the run report, and how to save the prepared view graph so the front
end runs once. The `openmvg-sfm` command of the `openmvg-cli` crate runs the same pipeline from
the command line.

## License

Licensed under either of Apache License, Version 2.0 or the MIT license, at your option. See
`LICENSE-APACHE` and `LICENSE-MIT`.
