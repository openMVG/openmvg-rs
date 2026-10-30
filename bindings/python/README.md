# openmvg-rs

![OpenMVG-rs, Structure from Motion in Rust: an illustration of photographs of Mont-Saint-Michel taken from many viewpoints, each linked by rays to the island they show](https://raw.githubusercontent.com/openMVG/openmvg-rs/HEAD/docs/images/hero.jpg)

Turn a folder of overlapping photographs into camera poses and a sparse 3D point cloud with
Structure from Motion (SfM). `openmvg-rs` is the Python package of
[OpenMVG-rs](https://github.com/openMVG/openmvg-rs).

## Installation

```bash
pip install openmvg-rs
```

Or, in a [uv](https://docs.astral.sh/uv/) project, `uv add openmvg-rs`. It needs CPython 3.11 or
newer, and wheels cover Linux, macOS (Apple silicon) and Windows; see Platforms and building from
source below for the details.

To try a release candidate when a final release also exists, ask for pre-releases:
`pip install --pre openmvg-rs` (with uv: `--prerelease allow`).

## Quick start

Download the 11 Sceaux Castle photographs:

```bash
git clone https://github.com/openMVG/ImageDataset_SceauxCastle
```

Save this script as `sceaux.py` next to the downloaded folder, and run it with `python sceaux.py`
(or, with uv and no project, `uv run --with openmvg-rs sceaux.py`):

```python
import openmvg_rs

scene = openmvg_rs.reconstruct("ImageDataset_SceauxCastle/images")
print(scene)  # <openmvg_rs.Reconstruction engine='incremental' registered=11/11 points=...>
scene.write_ply("sceaux.ply")
```

Open `sceaux.ply` in a point-cloud viewer such as MeshLab or CloudCompare. The downloaded folder
holds `K.txt`, the camera matrix of the photographs, so the run keeps that camera fixed.

### Writing the results

```python
scene.write_colmap("sceaux")  # COLMAP text model in sceaux/sparse/0/
scene.write_ply("sceaux/points.ply")  # Binary PLY: white points and green camera centers
scene.write_ply("sceaux/points_ascii.ply", format=openmvg_rs.PlyFormat.Ascii)
scene.save("sceaux/scene.omvgrs")  # read back with openmvg_rs.load_reconstruction
```

`scene.save` writes a binary file that only OpenMVG-rs reads back, with a release that writes
the same schema version. It is not an exchange format: share results as COLMAP or PLY.

### Using the results in Python

The points and the registered views are also NumPy values, ready for your own code or for a
library such as Open3D:

```python
points = scene.points  # (n, 3) array of 3D points, in world coordinates
for view in scene.views:  # the registered images, in the order of registered_images
    print(view.image.name, view.pose.center)  # where the photograph was taken
    R, t = view.pose.rotation, view.pose.translation  # X_cam = R @ X_world + t
    K = view.camera.matrix  # 3x3 camera matrix, None for a spherical camera
```

Each `View` holds the image path, a `Pose` and a `Camera`. A `Camera` names its `model`
(`"PINHOLE"`, `"BROWN_CONRADY"`, `"FISHEYE"` or `"SPHERICAL"`), the image `width` and `height`, and
its `parameters` by name: `fx`, `fy`, `cx`, `cy`, and the distortion coefficients it models (`k1` to
`k4`, `p1`, `p2`), as in the OMVG format's `cameras` table. Views that share a camera share one
`Camera` object. Pixel coordinates put the center of the top-left pixel at `(0, 0)`; COLMAP puts it
at `(0.5, 0.5)`, so `write_colmap` adds 0.5 to `cx` and `cy`. All arrays are read-only copies.

## `reconstruct`

```python
reconstruct(image_dir, /, *, engine="incremental", extraction_method="alp", seed=42,
            intrinsics=None, camera_model="pinhole-radial1", share_intrinsics=False,
            progress=True, max_workers=None) -> Reconstruction
```

- `image_dir` holds the images directly or in an `images/` subfolder. Images are JPEG, PNG or
  TIFF; convert HEIC photographs first.
- A `K.txt` file (a 3x3 camera matrix) in `image_dir` or next to the images, or the file passed
  as `intrinsics`, gives every image the same fixed intrinsics. Without it, the intrinsics are
  initialized from EXIF metadata and refined during the run; `Reconstruction.intrinsics_mode`
  reports which case applied.
- `camera_model` is the model of the cameras estimated from EXIF metadata: `"pinhole"`,
  `"pinhole-radial1"` (the default) or `"pinhole-radial2"`, whose radial distortion starts unknown
  and is estimated, or `"spherical"` for 2:1 equirectangular images. A `K.txt` calibration keeps
  its own pinhole camera. The names are those of `openmvg-sfm --camera-model`.
- An image without EXIF camera make and model gets a camera of its own. `share_intrinsics=True`
  gives such images, when they have the same size, one shared camera whose focal length and
  distortion are estimated from all of them, as `openmvg-sfm --share-intrinsics` does. Frames
  extracted from one video need it.
- `engine="incremental"` adds views one at a time; `engine="global"` estimates all their poses
  at once from the pairwise geometry.
- `extraction_method="alp"`, the default, detects features with ALP (Analytic Laplacian
  Polynomial), the project's own detector; `"sift"` uses SIFT. Both describe the features the same
  way, and the choice is independent of `engine`.
- The same inputs and `seed` give the same result, whatever the number of threads.
- `progress=True` draws progress bars on stderr when it is a terminal. Jupyter and other
  notebooks do not give stderr a terminal, so no bars appear there.
- `max_workers` caps the compute worker threads of the call; `None` uses all cores.

The GIL is released while the reconstruction runs, so other Python threads keep running. Ctrl-C
stops the run promptly and raises `KeyboardInterrupt` when `reconstruct` is called from the main
thread; from other threads Python does not deliver signals and the run completes. Processes
forked after a run (multiprocessing's `fork` start method) can call `reconstruct` as well.

### Getting good results

- Give a `K.txt` when you know the camera. Its intrinsics stay fixed, instead of being
  initialized from EXIF metadata and estimated during the run.
- For frames extracted from a video, or other images from one camera without EXIF, pass
  `share_intrinsics=True`, and `camera_model="pinhole-radial2"` when the lens distorts visibly.
  Without it every frame gets its own focal length, and the poses drift.
- Start with `engine="incremental"`, which tries to register every image. `engine="global"`
  keeps only the images whose poses pass its consistency checks, and raises
  `NoReconstructionError` when none do; `unregistered_images` lists the images a run left out.
- Keep `seed` fixed to repeat a run exactly, and change it to see how stable a result is.
- Photographs that overlap more, with some change of viewpoint between neighbors, register more
  reliably than distant or nearly identical ones.

## `Reconstruction`

An immutable result with `engine`, `seed`, `intrinsics_mode`, `num_images`,
`num_registered_images`, `num_points`, the `registered_images` and `unregistered_images` path
tuples, and the `points` array and `views` tuple described above. Each input image is one view;
a registered view is one the engine gave a pose.
`write_colmap(directory)` and `write_ply(path)` create missing directories and
replace an earlier export.

`save(path)` writes the reconstruction to a binary file (its parent directory must exist) and
`openmvg_rs.load_reconstruction(path)` reads it back. The file is checksummed and versioned; a
file that is not a reconstruction, was written by an incompatible version, or is corrupt raises
`OpenMVGError`. It records the image paths, cameras, poses, points, seed, `engine` and
`intrinsics_mode`. Pickling uses the same bytes.

## Geometry estimators

Robust estimators fit two-view geometry or a camera pose to NumPy arrays of correspondences.
Here `x1` and `x2` hold matched pixel coordinates in two images, and `points3d` the 3D points
seen at `x2`:

```python
import numpy as np
import openmvg_rs

K = np.array([[2759.48, 0, 1520.69], [0, 2764.16, 1006.81], [0, 0, 1]])
f = openmvg_rs.estimate_fundamental(x1, x2)  # x1, x2: (n, 2) pixels
h = openmvg_rs.estimate_homography(x1, x2)
e = openmvg_rs.estimate_essential(x1, x2, K)  # camera_matrix2=None uses K for both views
a = openmvg_rs.estimate_absolute_pose(points3d, x2, K)  # points3d: (n, 3)

print(e.pose.rotation, e.pose.translation, e.inlier_mask)  # X2 = R X1 + s t, |t| = 1
print(a.pose.rotation, a.pose.translation, a.pose.center)  # X_cam = R X_world + t
```

- Every estimator takes keyword-only `threshold` (pixels, default 4) and `seed` (default 42);
  the same inputs and seed give the same result.
- Camera matrices are `[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]`. `estimate_essential` and
  `estimate_absolute_pose` (P3P, then refinement on the inliers) convert the pixel threshold
  with the smallest focal length.
- The results (`FundamentalEstimate`, `HomographyEstimate`, `EssentialEstimate`,
  `AbsolutePoseEstimate`) are frozen dataclasses of read-only arrays: the model and a boolean
  `inlier_mask`. The two pose estimators return the same `Pose` type as `Reconstruction.views`.
- The input arrays are converted to `float64` and copied into the library's types, whatever
  their memory layout, and the estimation runs with the GIL released.

## Errors

All library errors derive from `openmvg_rs.OpenMVGError`:

| Exception | Raised when |
|---|---|
| `DatasetError` | the folder has no usable images or the calibration is malformed |
| `ReconstructionError` | matching or reconstruction fails |
| `NoReconstructionError` | (a `ReconstructionError`) no view could be registered |
| `ExportError` | a reconstruction cannot be written |
| `EstimationError` | a geometry estimator finds no model (too few or inconsistent correspondences) |

A missing image folder, image, calibration or reconstruction file raises `FileNotFoundError`,
and a read or write the operating system refuses raises `PermissionError`. Invalid arguments
raise `TypeError` or `ValueError`.

## Platforms and building from source

Wheels cover CPython 3.11 and newer on Linux (x86-64 and aarch64, glibc 2.17 or later), macOS
(Apple silicon) and Windows (x86-64). On x86-64 they need a CPU with SSE4.2 and POPCNT (the
x86-64-v2 level, every x86-64 CPU since about 2009). On other platforms, and on free-threaded
CPython, pip builds the source distribution instead, which needs a
[Rust toolchain](https://rustup.rs/) 1.97 or newer.

To build from a checkout of the repository, with [pixi](https://pixi.sh):

```bash
pixi run -e python py-build   # build and install into the pixi Python environment
pixi run -e python py-test    # run the contract tests
```

## License

Licensed under either of the
[Apache License, Version 2.0](https://github.com/openMVG/openmvg-rs/blob/HEAD/LICENSE-APACHE) or
the [MIT license](https://github.com/openMVG/openmvg-rs/blob/HEAD/LICENSE-MIT) at your option.
Wheels and the source distribution also ship `THIRD_PARTY_LICENSES.txt`, the licenses of the
compiled-in Rust dependencies.
