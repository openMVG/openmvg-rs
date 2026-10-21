# OpenMVG-rs

Rust implementations of Multiple View Geometry and Structure from Motion (SfM)
algorithms. The v0.1 tree keeps a small public API; non-release research and
dataset runners are maintained outside this source tree.

## Workspace structure

```text
crates/
├── openmvg-core/          # Cameras, IDs, graph/kernel data, shared contracts
├── openmvg-geometry/      # Problem-oriented geometry and robust estimation
├── openmvg-features/      # Detection, description, matching, tracks
├── openmvg-sfm/           # Incremental and Global SfM orchestration
├── openmvg-ba/            # Bundle adjustment and nonlinear numerical primitives
├── openmvg-io/            # OMVG community format: manifest plus Parquet tables
├── openmvg/               # Umbrella crate
├── openmvg-cli/           # `openmvg-sfm` command; `openmvg-sfm-configure` TUI behind `tui`
└── openmvg-rerun/         # Optional visualization integration (unpublished, own workspace: --manifest-path)

bindings/python/           # Python application API
```

Ordinary root Cargo commands cover the supported v0.1 source crates via
`default-members`. Use `--workspace` to include the Python bindings; the Rerun adapter builds only through
`--manifest-path crates/openmvg-rerun/Cargo.toml`.

## Supported crates

| Crate | Purpose |
|---|---|
| `openmvg` | Umbrella crate re-exporting `core`, `geometry`, `features`, `sfm`, and `io`; recommended entry point |
| `openmvg-core` | Camera models, IDs, graphs, poses, typed units, and deterministic seeds |
| `openmvg-geometry` | F/E/H, pose, registration, triangulation, averaging, and robust estimation |
| `openmvg-features` | Feature detection, description, matching, and tracking |
| `openmvg-sfm` | Transactional Incremental and Global reconstruction engines |
| `openmvg-ba` | Checked bundle adjustment and its nonlinear numerical primitives |
| `openmvg-io` | Reader, writer, and validator of the draft OMVG community format |
| `openmvg-cli` | The `openmvg-sfm` command over the stable SfM facade; the `openmvg-sfm-configure` terminal app behind the non-default `tui` feature |

All workspace packages use version `0.1.0`. Serialized v0.1 reports, ViewGraph
files, and Reconstruction files use schema version `1`.

## Stable architecture

### `openmvg-core`

- `camera/` — the `Camera` enum over pinhole, Brown-Conrady, fisheye, and spherical models
- `graph/` — connectivity, components, spanning trees, block-cut topology, and dense node indexing
- `model/` — SfM IDs, poses, intrinsics, relative measurements, and the estimation error
- `units` and `determinism` — typed residual units and the versioned seed/RNG protocol

### `openmvg-geometry`

- `fundamental/`, `essential/`, `homography/` — models, solvers, robust facades, and refinement
- `pose/`, `alignment/` — P3P (`P3pMethod`: fused default, LambdaTwist, Ding, Ke, Kneip,
  Nakano, Wu), resection, bearing-pose refinement, and the single
  closed-form Kabsch/Umeyama (rigid or Sim(3), weighted) alignment fit
- `triangulation/` — angular, bearing, midpoint, DLT, and track consensus
- `averaging/` — rotation/translation averaging, view-graph conditioning, and positioning
- `robust/` — shared private GenRANSAC machinery, the compact educational RANSAC/MSAC API, and
  the public `Solver`/`MultiSolutionSolver` traits that plug external solvers into
  `ClassicConfig::estimate`
- `correspondence` and normalization — concrete `f64` solver inputs and shared policies

Each family owns its model, free-function solver surface, residual policy,
typed GenRANSAC facade, refinement, tests, and provenance. Open each solver module
with an algorithm diagram in its module docs.

### `openmvg-features`

- `extraction/` — ALP and SIFT/RootSIFT detectors and descriptors, and the
  `F32Descriptor`/`U8Descriptor` storage types
- `matching/` — brute-force and HNSW matching, ratio filtering, and distance
  metrics (L2, Hamming)
- `model` — checked keypoint, descriptor, feature, and match values
- `tracking/` — deterministic track construction and consistency checks

`openmvg-features` has no image decoder and no SIFT feature gate: SIFT compiles in every
profile. Image decoding is the `openmvg-sfm` `image-io` opt-in; supported image formats are
JPEG, PNG, and TIFF; HEIC is outside the v0.1 dependency budget.

### `openmvg-sfm`

The normal application path is:

```text
Dataset::builder()
        |
        v
      ViewGraph
        |
        +-> Incremental::run_prepared(...)
        `-> Global::run_prepared(...)
```

- `Dataset`, `ViewGraph`, and `Reconstruction` are immutable phase boundaries.
- `Incremental` and `Global` are the two engine entry points and part of the stable v0.1 API.
- Mutable orchestration is available only through the unstable `unstable-internals`
  feature used by in-tree validation tooling; engine-backend fault-injection seams, diagnostics,
  and determinism helpers live under `testing` behind `unstable-test-support`.
- The separate `observer` feature exposes behavior-neutral reconstruction events for the Rerun
  adapter; progress reporting through `SfmProgressSink` needs no feature.
- Execution is deterministic for identical inputs and seeds.
- Mutating engine runs are transactional; errors restore the input workspace.
- Every engine run includes bundle adjustment, using the native `openmvg-ba`
  implementation through the backend-neutral SfM contract.
- `image-io`, `console-progress`, and `unstable-test-support` are explicit opt-ins.
- Features named `unstable-*` carry no semver guarantee.

Source layout:

- `workflow/` — immutable `Dataset`, `ViewGraph`, and `Reconstruction` values, the
  `Incremental`/`Global`/`Runner` engines, and engine routing
- `workspace/` — the mutable SfM workspace behind those values: builder, sparse store, views,
  pair graph, quality metrics, and validation
- `pipelines/` — Incremental and Global engine implementations, including Global graph preparation
- `ops/` — reusable stages shared by both engines (front end, averaging, reconstruction)
- `backends/` and `adapters/` — the stage traits (one file per stage) and their concrete feature,
  matching (brute force, HNSW), verification, and triangulation implementations
- `ba/` — bundle adjustment configuration, the backend contract, and the native adjuster bridge
- `image/` — image discovery and decoding (`decode`), EXIF focal lookup, and workspace
  preparation (`prepare`) (`image-io` only)
- `checkpoint/` — versioned binary ViewGraph and Reconstruction files; PLY export is
  `Reconstruction::write_ply`, and the OMVG format and COLMAP export live in `openmvg-io`
- `progress` — `SfmProgressSink` and, with `console-progress`, terminal progress bars
- `unstable` — expert orchestration, policies, reports, backends, and operations
  (`unstable-internals` only)
- `testing` — fixtures, diagnostics, determinism helpers, and fault-injection seams
  (`unstable-test-support` only)

### `openmvg-io`

- Reads, writes, and validates the OMVG community format specified in
  `docs/format/OMVG_FORMAT_SPEC.md`: a `manifest.json` plus one Parquet file per table, with
  per-column content hashes. The format is a draft; breaking changes are allowed until 1.0.
- It is the only supported crate that depends on Arrow and Parquet (pure Rust, Snappy only),
  all behind its default `omvg` feature; without it the crate is the COLMAP text export
  (`write_colmap`) alone, which is what the Python bindings use. `openmvg-sfm` keeps the internal
  binary files (postcard) and the PLY export.
- `format/` is the format core: manifest, schema, table files, hashes, and every rule, shared by
  the reader, the writer, and `check`; the conformance vectors in `docs/format/testdata` run as
  its unit tests.
- `scripts/check_architecture.py` enforces these boundaries on the resolved dependency trees:
  Arrow, Parquet, and `getrandom` only in trees that contain `openmvg-io`, no SQLite, HDF5, or
  ZIP, and no native `*-sys` crate under default features.

## Umbrella features

`openmvg` defaults to `features`, `faer` and `parallel`, favoring maximum native
performance while preserving deterministic output ordering.

| Feature | Purpose |
|---|---|
| `features` | Feature algorithms (default) |
| `faer` | Sparse averaging and positioning in `openmvg-geometry` (default); `sfm` always uses `faer` for averaging and bundle adjustment |
| `parallel` | Native multi-threading for every enabled crate (default) |
| `sfm` | SfM facade with native `openmvg-ba` bundle adjustment (`openmvg::sfm` re-exports `openmvg-sfm`) |
| `sfm-image-io` | Image discovery, decoding, and EXIF for SfM |
| `sfm-observer` | Behavior-neutral SfM progress events (`openmvg::sfm::events`) |
| `io` | OMVG community format (`openmvg::io`); the only feature that adds Arrow and Parquet |
| `full` | `features`, `faer`, `parallel`, `sfm-image-io`, `sfm-observer`, and `io` |

## Portability claims

- `openmvg-features` supports `no_std + alloc` through `libm` (with default features disabled).
- Do not make a workspace-wide `no_std` claim.
- The library stack supports serial `wasm32-unknown-unknown` builds with default
  features disabled. Native Rayon execution is not a browser-threading claim.
- `wasm-bindgen-rayon` is not part of the supported v0.1 surface.

## Development commands

Always run Cargo through `pixi run cargo`, never bare `cargo`: Pixi pins the Rust toolchain and
provides the native dependencies, so plain `cargo` can build with a different toolchain or fail
on missing libraries.

```bash
# Supported v0.1 source crates
pixi run cargo check
pixi run cargo test
pixi run cargo clippy --all-targets --all-features -- -D warnings

# Include every workspace package
pixi run cargo check --workspace
pixi run cargo test --workspace

# Feature crate's explicit no_std + alloc configuration
pixi run cargo check -p openmvg-features \
  --no-default-features --features libm

# Serial WASM SfM library
pixi run cargo check --target wasm32-unknown-unknown -p openmvg-sfm \
  --no-default-features

# Serial WASM umbrella configuration
pixi run cargo check --target wasm32-unknown-unknown -p openmvg \
  --no-default-features --features sfm

# Release build and feature benchmarks
pixi run cargo build --release
pixi run cargo bench -p openmvg-features

# Documentation checks and the self-checking examples (synthetic; then on Sceaux Castle)
pixi run algorithm-docs
pixi run crate-docs
pixi run examples
pixi run examples-real
```

The SfM validation protocols, their ETH3D, Strecha, and BlendedMVS records and the corpus
harness live in the maintainers' evaluation workspace, outside this repository. Ask for a
corpus run after numerical changes to cameras, robust estimation, averaging, registration,
triangulation, bundle adjustment, or pipeline policy; in-repository tests do not replace it.

`openmvg-cli` is the command-line release: `openmvg-sfm` reconstructs an image folder using only
the stable `openmvg-sfm` and `openmvg-io` API, and `openmvg-sfm-configure` (feature `tui`) groups a
folder's images into cameras and writes the project file for `openmvg-sfm --project`
(`ImageProject` in `openmvg-sfm`). Only the `tui` feature may resolve ratatui;
`scripts/check_architecture.py` enforces it.

```bash
pixi run openmvg-sfm path/to/images
pixi run sfm-configure path/to/images
pixi run cargo test -p openmvg-cli --features tui
```

## Contribution rules

- Apply KISS and YAGNI to the v0.1 public API; non-release research is maintained
  separately from the supported source and public compatibility surface.
- Prefer the shared `Camera`, IDs, threshold units, and backend contracts over
  crate-local duplicate vocabulary.
- Tests must protect a contract, numerical invariant, regression, degeneracy,
  deterministic seed, or platform boundary. Timing belongs in benchmarks, not
  unit tests.
- Reject non-finite input and invalid backend output at public boundaries.
- Keep library code free of unsafe Rust and panic-based control flow. `scripts/check_architecture.py`
  requires `#![forbid(unsafe_code)]` and a deny of `clippy::unwrap_used`, `clippy::expect_used`
  and `clippy::panic` (at least outside tests) in every supported crate and binary root.
- Open every algorithm's teaching docs with a diagram, where docs.rs shows them, and list the
  file in `scripts/check_algorithm_docs.py`; `docs/DOCUMENTATION.md` defines the layout.
- Learn the public API from each crate root's learning path and its examples. Docs and
  examples use the stable API only, never `unstable::`, and follow the teaching voice and the
  example rules of `docs/STYLE.md`.
- Record provenance and compatible licensing before importing an algorithm,
  generated solver, coefficient table, or reference implementation.
- Do not run `cargo publish`. Release preparation in this repository is limited
  to local checks and package-content verification unless the user explicitly
  authorizes publication.

## Project documentation

- `PHILOSOPHY.md` — project principles
- `DEVELOPER.md` — contributor setup and conventions
- `docs/ARCHITECTURE.md` — current architecture
- `docs/STYLE.md` — writing style for docs, rustdoc, messages, and commits
- `docs/DOCUMENTATION.md` — where each kind of fact lives, the module-docs diagram rule, and its checks
- `docs/REPRODUCIBILITY.md` — determinism and RNG protocol contract
