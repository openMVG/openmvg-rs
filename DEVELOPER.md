# Developer guide

## Prerequisites

This project uses [Pixi](https://pixi.sh) for environment and task management.

Install Pixi:
```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

## Commands

| Command | Description |
|---------|-------------|
| `pixi run build` | Build the supported crates |
| `pixi run cargo check --all-targets` | Check supported crate libraries, tests and examples |
| `pixi run test` | Run the supported crates' default tests |
| `pixi run cargo clippy --all-targets --all-features -- -D warnings` | Lint the supported crates and their opt-in configurations |
| `pixi run fmt` | Format code with rustfmt |
| `pixi run doc` | Generate documentation |
| `pixi run clean` | Clean build artifacts |
| `pixi run architecture` | Check dependency and publication boundaries |
| `pixi run deny` | Check dependency licenses against `deny.toml` |
| `pixi run cargo <args>` | Run arbitrary cargo commands |

For the public rustdoc profile, run:

```bash
RUSTDOCFLAGS="-D warnings" pixi run cargo doc --no-deps --features openmvg/full
```

Portable code generation is the default. For a local, non-distributable
Linux benchmark build, native CPU tuning is explicit:

```bash
RUSTFLAGS="-C link-arg=-fuse-ld=mold -C target-cpu=native" pixi run cargo build --release
```

## Running specific tests

```bash
# Run tests matching a name pattern
pixi run cargo test test_name

# Run tests in a specific module
pixi run cargo test module_name::

# Run a single test with full output
pixi run cargo test test_name -- --nocapture

# Run only library tests (no integration tests)
pixi run cargo test --lib

# Run tests sequentially (useful for debugging)
pixi run cargo test -- --test-threads=1
```

### Ignored tests

`#[ignore]` is reserved for checks that need external data; the ignore reason names
what the test needs. Public numerical regressions use self-contained fixtures. Every other test runs in the normal suite: one-off performance
probes and development studies are deleted once they have served their purpose
rather than kept as ignored tests.

## Quick start

```bash
# Clone and enter the project
cd openmvg-rs

# Build
pixi run build

# Run tests
pixi run test

# Format before committing
pixi run fmt
```

## ALP table maintenance

The ALP generators use Julia 1.12.6 as a separate maintainer prerequisite. Their
[setup and check commands](crates/openmvg-features/scripts/alp/README.md) verify the generated
coefficients and cross-language reference cases.

## Adding a geometry solver

A new public solver in `openmvg-geometry` includes:

1. a concrete `f64` solver type with an explicit sample shape and residual units;
2. teaching docs that open with an algorithm diagram matching the implementation, listed in
   `scripts/check_algorithm_docs.py`;
3. degeneracy and non-finite-input tests;
4. a numerical invariant or property test, and a case in the cross-family failure-contract
   test (`tests/solver_failure_contract.rs`);
5. a typed robust facade when the problem uses consensus estimation;
6. an allocation test only when allocation behavior is part of the contract.

Each problem directory owns its model, minimal and non-minimal solvers, residual policy, robust
facade, refinement, tests and references; the shared robust-estimation machinery stays private
to those family APIs. Before submitting:

```sh
pixi run cargo test -p openmvg-geometry --all-features
pixi run cargo clippy -p openmvg-geometry --all-features --all-targets -- -D warnings
```

## Maintaining bundle adjustment

The checked BA contract is covered by synthetic numerical regressions in `openmvg-ba`:

```sh
pixi run --locked cargo test --locked -p openmvg-ba --test native_capabilities
pixi run --locked cargo test --locked -p openmvg-ba \
  --features parallel,unstable-internals --test sparse_bundle_adjustment
pixi run --locked cargo test --locked -p openmvg-ba \
  --features parallel,unstable-internals --test sparse_self_calibration
```

They exercise camera models, variable masks, controls, gauges, shared poses and intrinsics, and
transactional failure; the kernel regressions compare the dense, direct sparse and iterative
paths on the same fixtures. They do not certify memory or accuracy on external datasets.

- Nielsen gain-ratio Levenberg-Marquardt is the only nonlinear controller. A new public
  controller must first beat it on the seeded and BAL acceptance gates without an objective or
  success-rate regression.
- A core audit is complete when the stable feature matrix, the full sparse and parallel tests,
  the allocation guard, Clippy, rustdoc and the statistical acceptance suite pass. Repeating it
  without a concrete failing contract should change no code; new camera models or solver
  experiments are separate projects.
- A comparison on an external dataset records its source and input fingerprints, camera and
  pixel conventions, fixed parameters, intrinsics grouping, residual units, noise and loss,
  gauge, stopping conditions, memory policy, thread count and toolchain, and compares like
  objectives only. BAL's centered radial camera must be mapped explicitly to the model under
  test.

## Continuous integration

Pull requests and pushes to `develop` or `develop_*` run `Tests`,
`WASM, no_std & MSRV`, and `Python bindings`. Rust builds and tests select the eight
supported default-members, with explicit integration-test features and a separate
stable Clippy profile. Python uses CPython 3.14 for routine checks. `Windows files` tests
the crates that read and write files on Windows, then runs `openmvg-sfm` on Sceaux Castle and
validates its OMVG output.

A manual `ci.yml` run also executes extended qualification: additional feature
profiles, Python Rust bindings, optimized geometry tests, Rerun, docs.rs
profiles, package verification, Python 3.11 and free-threaded Python 3.14.
Run it before release. Corpus-level qualification on the ETH3D, Strecha and BlendedMVS
corpus runs outside this repository, with the maintainers' evaluation tools. Builds use the
committed lockfiles.

## Python bindings

`bindings/python` builds the `openmvg-rs` Python package with maturin. Its tasks
run in the `python` (CPython 3.14), `python-311`, and `python-ft` (free-threaded
3.14t) environments:

| Command | Description |
|---------|-------------|
| `pixi run -e python py-build` | Build the extension and install it in the environment |
| `pixi run -e python py-test` | Build, then run the pytest contract tests |
| `pixi run -e python py-lint` | ruff, and `mypy --strict` on the package and the native stub |
| `pixi run -e python py-stubtest` | Check `_native.pyi` against the built module |
| `pixi run -e python py-notices` | Generate `THIRD_PARTY_LICENSES.txt` with cargo-about |

The Sceaux Castle tests and `pixi run examples-real` get the images from `scripts/sceaux.py`,
which uses `$OPENMVG_DATASETS/ImageDataset_SceauxCastle-master` when present and otherwise
downloads the files once, verified against `scripts/sceaux.sha256`. Release wheels and the sdist are built by the manual
`wheels.yml` workflow. Its `publish` input uploads them to TestPyPI or PyPI once every build and
test has passed, through Trusted Publishing from the `testpypi` or `pypi` environment; the
default, `none`, publishes nothing. `pyproject.toml` selects the `wheel` Cargo profile
(fat LTO, one codegen unit) for `maturin build` and for pip builds from the sdist; the
`--release` of `maturin develop` overrides it.

## WebAssembly profiles

The library uses deterministic seeds and supports these serial
`wasm32-unknown-unknown` configurations:

```bash
pixi run cargo check --target wasm32-unknown-unknown -p openmvg-core \
  --no-default-features
pixi run cargo check --target wasm32-unknown-unknown -p openmvg-geometry \
  --no-default-features --features faer
pixi run cargo check --target wasm32-unknown-unknown -p openmvg-ba \
  --no-default-features
pixi run cargo check --target wasm32-unknown-unknown -p openmvg-sfm \
  --no-default-features
pixi run cargo check --target wasm32-unknown-unknown -p openmvg-io
```

OpenMVG-rs libraries do not acquire browser or host entropy. Applications that
want a nondeterministic run generate a seed outside OpenMVG-rs and pass the
resolved seed into the library. Library algorithms derive full-width child seeds from `RunSeed` and use the versioned ChaCha12 protocol;
random work is keyed by stable IDs rather than worker scheduling.

The descriptor layer also has an explicit `no_std + alloc` profile. This is
separate from ordinary WASM builds, which may still use `std`:

```bash
pixi run cargo check -p openmvg-features \
  --no-default-features --features libm
pixi run cargo check --target wasm32-unknown-unknown -p openmvg-features \
  --no-default-features --features libm
```

The umbrella crate enables its performance-oriented native features by default.
Portable WASM consumers must disable native defaults explicitly. They can omit
`faer` entirely:

```bash
pixi run cargo check --target wasm32-unknown-unknown \
  -p openmvg --no-default-features
```

Or they can enable the serial `faer` sparse-linear-algebra backend:

```bash
pixi run cargo check --target wasm32-unknown-unknown \
  -p openmvg --no-default-features --features faer
```

The workspace `wasm-release` profile (`opt-level = "z"`, LTO, one codegen unit, stripped) builds
size-optimized wasm32 artifacts from this workspace; CI builds the serial SfM umbrella with it.
Cargo ignores the profiles of dependencies, so a downstream application applies its own.

```bash
pixi run cargo build --profile wasm-release --target wasm32-unknown-unknown \
  -p openmvg --no-default-features --features sfm
```

The `parallel` feature is for native builds only: `wasm32-unknown-unknown` has
no threads, so WASM consumers leave it off.

Ordinary root commands target the supported v0.1 source-release crates.
Use `--workspace` to also check the Python Rust bindings.
The Rerun adapter is excluded
from the root workspace so its `re_sdk` dependency tree stays out of
`Cargo.lock`; it has its own workspace and lockfile:

```bash
pixi run cargo test --manifest-path crates/openmvg-rerun/Cargo.toml
pixi run -e msrv msrv-cargo check --manifest-path crates/openmvg-rerun/Cargo.toml --all-targets
```

`openmvg-cli` holds the `openmvg-sfm` command and, behind its `tui` feature, the
`openmvg-sfm-configure` terminal app that writes the project files `openmvg-sfm --project` reads.
The ratatui stack is in `Cargo.lock` but resolves only with `tui`:

```bash
pixi run cargo test -p openmvg-cli --features tui
pixi run openmvg-sfm path/to/images
pixi run sfm-configure path/to/images
```

Passing `None` as a matcher, robust-estimator, or synthetic-scene seed selects
a stable library seed. Pass an explicit seed when an experiment needs its own
reproducibility contract. See the project-wide
[reproducibility contract](docs/REPRODUCIBILITY.md) for the supported replay
scope, receipt requirements, and RNG protocol migration rules.

## crates.io release order

All publishable path dependencies carry an explicit registry version. Publish
in dependency layers, allowing crates.io's index to update between layers:

1. `openmvg-core`
2. `openmvg-features`, `openmvg-ba`
3. `openmvg-geometry`
4. `openmvg-sfm`
5. `openmvg-io`
6. `openmvg`
7. `openmvg-cli`

The Python Rust bindings and Rerun adapter are marked `publish = false`.
Manual extended qualification packages the eight public crates in one
`cargo package -p ...` call, which builds each crate from its `.crate` and
resolves the unpublished OpenMVG dependencies from the packaged siblings.
`pixi run deny` checks that every dependency of the published crates is
licensed under a reviewed license (`deny.toml`).
