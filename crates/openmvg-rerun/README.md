Not part of the v0.1 public API.

# OpenMVG Rerun adapter

`openmvg-rerun` records coarse Structure from Motion (SfM) checkpoints as Rerun
`.rrd` files. It is an optional leaf crate: no other OpenMVG-rs crate depends
on the Rerun SDK.

The default blueprint contains:

- a camera-model-safe 3D reconstruction view;
- an auto-laid-out initial verified view graph with live unused, tentative, and committed camera
  status;
- a separate auto-laid-out current reconstruction graph containing registered cameras and active
  registered-induced edges;
- landmark growth and registered-camera coverage;
- bearing, pixel-reprojection, and pose-graph quality curves;
- a pipeline event log, including tentative and rolled-back states.

The initial graph keeps its sampled verified topology fixed and colors nodes gray, amber, or green
as cameras become unused, tentative, or committed. Current reconstruction nodes are colored by
active-landmark incidence. Reconstructed camera-pair edges are colored by shared active landmarks.
Incidence colors use a colorblind-safe Viridis scale normalized by the maximum support available
from the immutable source tracks, so colors remain stable while reconstruction grows.

Camera and graph growth remain visible at every registration checkpoint. By default, the landmark
cloud and residual summaries are also refreshed at every checkpoint. Applications that prefer
smaller recordings can select a larger `GeometryCapture::Every` stride; global, rollback,
selection, and terminal boundaries always retain full detail.

## Use

Create a recorder, pass it as the `SfmEventSink` to an observed engine entry
point, and finish it explicitly:

```rust,no_run
use openmvg_rerun::{RerunRecorder, RerunRecorderConfig};

let mut config = RerunRecorderConfig::default();
config.max_landmarks = 50_000;
let mut recorder = RerunRecorder::save("reconstruction.rrd", config)?;

// Pass `&mut recorder` to the observed engine entry point.

let report = recorder.finish();
assert!(report.first_error.is_none());
# Ok::<(), openmvg_rerun::RerunRecorderError>(())
```

The crate is excluded from the root workspace: it has its own `[workspace]` and
`Cargo.lock`, so the Rerun SDK never enters the root lockfile. Build, test, and record an image directory with:

```text
pixi run cargo test --manifest-path crates/openmvg-rerun/Cargo.toml
pixi run cargo run --release --manifest-path crates/openmvg-rerun/Cargo.toml \
  --example record_sfm -- /path/to/images reconstruction.rrd
```

Open recordings with Rerun 0.36.3:

```text
rerun reconstruction.rrd
```

For long-term storage, Rerun can perform an additional offline compaction pass:

```text
rerun rrd optimize reconstruction.rrd -o reconstruction.compact.rrd
```

Recording does not run this pass, because it reads and writes the whole file a second time.

## Design guarantees

- Visualization failures never affect reconstruction routing.
- Rendered landmark and edge snapshots are deterministically sampled to configured limits.
- Rerun dependencies remain outside default OpenMVG dependency graphs.
- Camera poses use camera-frame axes for every model, avoiding an implied perspective frustum for
  fisheye and spherical cameras.
- Coordinates are recentered before `f32` conversion, with the exact `f64`
  origin recorded as metadata.

The crate is for repository and application use and is not published on
crates.io.
