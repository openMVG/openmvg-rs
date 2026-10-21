# Contributing

Thank you for helping improve OpenMVG-rs.

## Development environment

The repository uses Pixi to pin the Rust toolchain and native dependencies.
Run Cargo through Pixi from the repository root:

```bash
pixi install
pixi run cargo check --all-targets
```

The [developer guide](DEVELOPER.md#commands) owns the build, test, formatting, Clippy and
rustdoc commands. Ordinary root commands select the seven supported crates; `--workspace`
also includes the Python Rust bindings. Use the guide's explicit
[serial WASM and no_std configurations](DEVELOPER.md#webassembly-profiles) for portability.

Maintainers regenerating ALP tables need the separate
[Julia 1.12.6 setup](crates/openmvg-features/scripts/alp/README.md).

## Changes

- Keep public numerical types valid by construction when they carry geometric
  invariants. Prefer typed errors to panics or sentinel values.
- Preserve deterministic behavior for fixed labeled inputs and seeds.
- Add focused regression tests for correctness fixes and numerical edge cases.
- Document public APIs and any feature, platform, or algorithmic limitations.
- Keep benchmark and experimental code out of foundational crate APIs.
- Run formatting, tests, strict Clippy, and rustdoc for every crate you change.
- Follow the [writing style](docs/STYLE.md) for documentation, messages, and commits, and the
  [documentation strategy](docs/DOCUMENTATION.md): every algorithm module opens with a diagram.

Please keep commits focused and explain externally visible API or numerical
behavior changes in the pull request description.

## Licensing

Contributions are licensed inbound = outbound: unless explicitly stated
otherwise, every contribution is accepted under the repository's dual
MIT OR Apache-2.0 license, the same terms under which the project is
distributed.
