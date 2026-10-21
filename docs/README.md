# Documentation

## Start here

- [Quick start and examples](../README.md#quick-start)
- [Python application API](../bindings/python/README.md)
- [Contributor setup and commands](../DEVELOPER.md)

## Crates

Crate READMEs introduce the entry points and features. Rustdoc describes API contracts;
algorithm module docs explain the mathematics, coordinate frames, units and failure modes.

| Crate | Guide |
|---|---|
| `openmvg` | [Application entry point](../crates/openmvg/README.md) |
| `openmvg-core` | [Cameras and shared model types](../crates/openmvg-core/README.md) |
| `openmvg-geometry` | [Geometry and estimation](../crates/openmvg-geometry/README.md) |
| `openmvg-features` | [Extraction, matching and tracks](../crates/openmvg-features/README.md) |
| `openmvg-sfm` | [Reconstruction engines and export](../crates/openmvg-sfm/README.md) |
| `openmvg-ba` | [Bundle adjustment](../crates/openmvg-ba/README.md) |
| `openmvg-io` | [Dataset and reconstruction interchange](../crates/openmvg-io/README.md) |
| `openmvg-cli` | [The `openmvg-sfm` command and project configurator](../crates/openmvg-cli/README.md) |

Build local rustdoc with `pixi run cargo doc --no-deps --features openmvg/full`.

## Contracts and community format

- [Architecture and feature boundaries](ARCHITECTURE.md)
- [Reproducibility and random streams](REPRODUCIBILITY.md)
- [OMVG format specification, 0.1-draft](format/OMVG_FORMAT_SPEC.md)
- [Independent format reference](format/reference/omvg.py) and
  [conformance vectors](format/testdata)

## Contributing documentation

- [Project philosophy](../PHILOSOPHY.md)
- [Writing and naming conventions](STYLE.md)
- [Documentation ownership and checks](DOCUMENTATION.md)
- [Release changes](../CHANGELOG.md) and [compatibility notes](../MIGRATION.md)
