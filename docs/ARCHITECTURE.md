# OpenMVG-rs architecture

This document describes the v0.1 tree of OpenMVG-rs, a Multiple View Geometry and Structure
from Motion (SfM) library.

## Workspace structure

```text
crates/
├── openmvg-core/                   cameras, IDs, graph and operator contracts
├── openmvg-geometry/               geometry families and robust estimation
├── openmvg-features/               detection, description, matching, tracks
├── openmvg-sfm/                    Incremental and Global SfM orchestration
├── openmvg-ba/                     bundle adjustment (BA) and nonlinear numerics
├── openmvg-io/                     OMVG community format (manifest + Parquet tables)
├── openmvg/                        umbrella crate
├── openmvg-cli/                    `openmvg-sfm` command and, with `tui`, `openmvg-sfm-configure`
└── openmvg-rerun/                  optional event visualization (unpublished, own workspace)
```

## Dependency graph

```text
openmvg
├── openmvg-core
├── openmvg-geometry ──────> openmvg-core, openmvg-ba numerics
├── openmvg-features ──────> openmvg-core (feature `std`)
├── [sfm] openmvg-sfm ─────> core, geometry, features, openmvg-ba
└── [io] openmvg-io ───────> openmvg-core, openmvg-sfm, parquet (arrow, snap)

openmvg-cli ───────────────> openmvg-core, openmvg-sfm, openmvg-io, [tui] ratatui
openmvg-rerun ─────────────> openmvg-core, openmvg-sfm
Python bindings ────────────> supported crates only
```

`openmvg-geometry` uses only `openmvg-ba`'s numerics: the jets, manifold maps and refinement
loop always, and with `faer` the shared sparse Cholesky, so its `faer` feature turns on
`openmvg-ba/faer`, which also compiles the native BA kernels. `openmvg-sfm` lowers its
backend-neutral contract into the checked native `openmvg-ba` contract. `openmvg-rerun` is an
optional leaf excluded from the root workspace (own `[workspace]` and lockfile); supported crates never depend on the Rerun SDK, and the root lockfile never contains it.
`openmvg-io` is the only supported crate that depends on Arrow and Parquet; they reach the
umbrella only through its `io` feature, and `openmvg-cli`, which always writes OMVG output, through
its direct dependency. The ratatui terminal stack appears only under the `tui` feature of
`openmvg-cli`. `postcard`, used by the internal binary files, appears
only in `openmvg-sfm`, and no supported crate's default tree contains a C-backed `*-sys` crate.
`scripts/check_architecture.py` checks these boundaries on the resolved dependency trees.

The default workspace members are `openmvg-core`, `openmvg-geometry`, `openmvg-features`,
`openmvg-ba`, `openmvg-sfm`, `openmvg-io`, the `openmvg` umbrella, and `openmvg-cli`.
`--workspace` additionally includes the Python bindings; the Rerun adapter builds via
`--manifest-path crates/openmvg-rerun/Cargo.toml`.

## Crate responsibilities

| Crate | Responsibility | Release posture |
|---|---|---|
| `openmvg-core` | Camera models, graph algorithms, shared values (`model`), IDs, units, and the estimation error | Stable foundation |
| `openmvg-geometry` | F/E/H, pose, alignment, triangulation, averaging, positioning, typed robust estimation | Stable numerical library |
| `openmvg-features` | Feature, descriptor, match, and track values; detection, description, matching, tracking | Stable feature library |
| `openmvg-sfm` | Dataset/ViewGraph/Reconstruction values, Incremental/Global engines, reports, events, optional I/O and fixtures | Stable workflow library (the v0.1 SfM product) |
| `openmvg-ba` | Checked BA contract, native kernels, and low-level nonlinear primitives | Stable numerical library |
| `openmvg-io` | OMVG community format (manifest, Parquet tables, content hashes, and the SfM mapping) and COLMAP text export | Draft format, version `0.1-draft` |
| `openmvg` | Umbrella crate re-exporting `core`, `geometry`, `features`, `sfm`, and `io` | Recommended entry point |
| `openmvg-cli` | The `openmvg-sfm` command over the stable SfM facade, and the `openmvg-sfm-configure` terminal app behind `tui` | Published binaries, source install |
| Python bindings and Rerun | Application API and optional visualization | Unpublished Rust packages |

`openmvg-ba` owns both the checked BA path and the low-level nonlinear primitives
used by in-tree geometric refinement; its `numerics` module is their supported, documented
surface. There is no standalone optimization crate. With `faer`, `numerics` also holds the one
deterministic sparse Cholesky (`numerics::sparse`, hidden and without a semver guarantee) that
the BA kernels, rotation averaging, and global positioning share.

`openmvg-features` owns the checked `Keypoint`, `DescriptorSet`, `ExtractedFeatures`,
`FeatureSet`, `FeatureMatch`, `MatchSet`, and typed track vocabulary. SfM reuses those values and
adds only workflow context such as the image pair attached to a match set and stable track IDs.
Track observations store `(ViewId, FeatureId)` identity; pixel geometry lives once in each view's
`FeatureSet`.

## Geometry and estimation

Each geometry problem family owns concrete `f64` minimal-solver types, typed GenRANSAC facades,
refinement, tests, and provenance. `robust::classic` is the compact educational
RANSAC/MSAC baseline; external solvers plug into it through the public `robust::Solver` and
`robust::MultiSolutionSolver` traits. The GenRANSAC loop behind the family estimators is
crate-private. The supported correspondence families distinguish pixel and bearing inputs.

Solver modules open with an algorithm diagram in their module docs.
Threshold units, degeneracy, non-finite input, and deterministic sampling are explicit contracts.

## Structure from Motion

```text
Dataset::builder()
        |
        v
extract -> match -> verify -> build tracks
        |
        +-> Incremental::run_prepared(...)
        `-> Global::run_prepared(...)
```

The private `SfmWorkspace` owns image metadata, shared intrinsics, features, matches, verified
geometry, tracks, poses, landmarks, and reports. Every `ViewRecord` maps one `ViewId` to an
`IntrinsicId` and a `PoseId`; multiple views may share either block. The plain records that
`Dataset`, `ViewGraph`, and `Reconstruction` return (`ViewRecord`, `ImageMeta`, `PoseRecord`,
`Landmark`, `Landmarks`, `ReconstructionComponent`, ...) live in the stable `model` module.

The v0.1 persistence boundary is the immutable `ViewGraph`: it stores verified front-end state
only. Poses, landmarks, global averages, components, and registration failures are reconstruction
state and are never accepted by the ViewGraph writer. The accepted `Reconstruction` has its own
immutable binary file (`checkpoint::write_reconstruction`). Mid-engine resume is outside v0.1. Report
JSON, including the engine configuration a `RunReport` embeds, is versioned by its
`schema_version` field, and the ViewGraph and Reconstruction files by the schema version in their
header, not by the crate's semver; version 1 is the v0.1 layout.

The umbrella's `Incremental` and `Global` engines select the default geometric and bundle-
adjustment implementations automatically; `openmvg-sfm` also exposes an explicit BA contract
(`ba::BundleAdjuster`). Mutable workspace access and three `Runner` entry points,
`Runner::for_config`, `Runner::run_workspace`, and `Runner::run_in_place`, require the unstable
`unstable-internals` feature, which exists for in-tree tooling. The rest of `Runner` is stable,
including `progress`, `bundle_adjuster`, and `observer` (with the `observer` feature). Replacing an engine's
registration, averaging, or triangulation algorithms (`testing::seams::{IncrementalBackends,
GlobalBackends, PoseRegistrar}`) is a test-only fault-injection seam behind `unstable-test-support`.
Those mutating runs are transactional: failure restores the caller's workspace.

`SfmEventSink` observes deterministic orchestration boundaries through a borrowed, curated
`SfmSnapshot`; it does not expose `SfmWorkspace` through the stable observer surface and cannot
steer control flow. `openmvg_rerun::RerunRecorder` is the optional visualization sink; observer
errors remain telemetry and never alter reconstruction. Progress is a separate, write-only
`SfmProgressSink`, available without the `observer` feature: every front-end stage reports item
counts, extraction, matching, and verification from their parallel loops through `&self`, so it
never shares the sequential event sink and a run can drive progress bars and a Rerun recording
together. The engines report their main reconstruction stages to the same sink at sequential
stage boundaries (initial pair, view registration, rotation and translation averaging, triangulation,
bundle adjustment, self-calibration); a rejected Global candidate abandons its open stage and the
next candidate repeats the stages. The library keeps no wall clock; callers time stages through
the sink.

The public workflow state is `Dataset`, `ViewGraph`, and `Reconstruction`, with exactly two
engines, Incremental and Global; this workflow is part of the stable v0.1 API.

A dataset records whether its camera models are a known calibration or initial estimates; the
`ViewGraph` and its checkpoint file carry that provenance, radial pre-calibration skips calibrated
datasets, and both engines keep calibrated intrinsics
fixed while refining estimates. The `ViewGraph` and its checkpoint file also record the
`ExtractionMethod` (SIFT or ALP) the `Frontend` extracted with.

## Feature boundaries

The main capability flags:

| Crate | Feature | Meaning |
|---|---|---|
| `openmvg` | `features` | Feature APIs; default |
| `openmvg` | `faer` | Sparse averaging and positioning in `openmvg-geometry`; default (`sfm` always enables `faer` in geometry and BA) |
| `openmvg` | `parallel` | Native multi-threading for every enabled crate; default |
| `openmvg` | `sfm` | Curated SfM facade with automatic BA composition |
| `openmvg` | `sfm-image-io`, `sfm-observer` | Independent workflow capabilities |
| `openmvg` | `io` | OMVG community format as `openmvg::io`; implies `sfm`, adds Arrow and Parquet |
| `openmvg-features` | `std`, `parallel` | Default native profile |
| `openmvg-features` | `libm` | Explicit alloc-only feature profile |
| `openmvg-sfm` | `parallel` | Native Rayon execution; default |
| `openmvg-sfm` | `image-io` | Filesystem image discovery, decoding, and EXIF metadata |
| `openmvg-sfm` | `observer` | Behavior-neutral reconstruction events |
| `openmvg-sfm` | `console-progress` | Terminal progress bars; native only |
| `openmvg-sfm` | `unstable-internals` | Mutable orchestration, backends, and policies for in-tree tooling (no semver) |
| `openmvg-sfm` | `unstable-test-support` | Test-only fixtures, diagnostics, and fault-injection seams; implies `unstable-internals` (no semver) |
| `openmvg-geometry` | `faer` | Sparse rotation averaging and global positioning on the sparse Cholesky of `openmvg-ba`; implies `openmvg-ba/faer` |
| `openmvg-geometry` | `parallel` | Optional Rayon for large averaging/filtering loops; serial fallbacks give identical results |
| `openmvg-geometry` | `unstable-test-support` | Hidden hooks for the crate's own integration tests (no semver) |
| `openmvg-ba` | `faer` | Sparse native bundle adjuster and the shared sparse Cholesky; default |
| `openmvg-ba` | `parallel` | Native multi-threading in the sparse BA kernels; implies `faer` |
| `openmvg-ba` | `unstable-internals` | In-tree validation kernels (no semver) |

Features named `unstable-*` are the no-semver boundary: their items may change or disappear in
any release, they are never enabled by a default or umbrella feature, and stable APIs never
require them. They exist for in-tree tooling, tests, and research previews; the
`unstable-test-support` features are for tests only. Everything reachable without these
features, including the Incremental and Global SfM workflows, is the supported v0.1 API.

Library crates accept explicit seeds and do not expose a browser-entropy feature.

The umbrella root exports modules rather than duplicating leaf symbols: `openmvg::core`,
`openmvg::geometry`, optional `openmvg::features`, optional `openmvg::sfm`, and optional
`openmvg::io`. Direct leaf-crate
dependencies remain the smaller choice for applications using only one layer or expert backend
composition.

## Portability

- Native Rust is the primary supported profile.
- `openmvg-features` has an explicit `no_std + alloc` profile. This is not a workspace-wide
  `no_std` claim.
- Core numerical crates and serial SfM are checked for `wasm32-unknown-unknown` without default
  features.
- Rayon on native targets is not a browser-threading claim; `wasm-bindgen-rayon` is not part of
  the v0.1 contract.
- Every stronger portability statement requires a matching CI check and, for numerical paths,
  a runtime smoke.

## Validation

```bash
pixi run cargo fmt --all -- --check
pixi run cargo check --workspace --all-targets
pixi run cargo test
pixi run cargo clippy --all-targets --all-features -- -D warnings
pixi run cargo check --target wasm32-unknown-unknown -p openmvg-sfm \
  --no-default-features
pixi run cargo check --target wasm32-unknown-unknown -p openmvg-features \
  --no-default-features --features libm
```

Release qualification runs both engines on a fixed ETH3D, Strecha, and BlendedMVS corpus
outside this repository. A qualification record binds inputs, configuration, metric
definitions, RNG receipts and toolchain, and accepted new evidence receives a separate dated
baseline.
