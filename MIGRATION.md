# OpenMVG-rs 0.1: initial release

0.1 is the first published release of the OpenMVG-rs crates, so there is no earlier public API
to migrate from. This note states the compatibility rules that start with it. For the list of
what the release contains, read [CHANGELOG.md](CHANGELOG.md).

## Versioning

- The seven publishable crates (`openmvg-core`, `openmvg-geometry`, `openmvg-features`,
  `openmvg-ba`, `openmvg-sfm`, `openmvg-io` and the `openmvg` umbrella) are released together with one version.
  Their intra-workspace dependencies are pinned exactly, so upgrade them together.
- Before 1.0, a minor release (0.1 → 0.2) can break the API; patch releases do not.
- Removed or renamed items get no deprecated aliases. The compiler points to every site to change,
  and the changelog lists each break.

## What semver covers

Semver covers the supported public API in both default and stable opt-in configurations,
including image decoding, observer events and OMVG interchange. APIs enabled by features
named `unstable-*` have no semver guarantee and can change in any release:

- `openmvg_sfm::unstable` and the unstable members of stable types (feature `unstable-internals`);
- `openmvg_sfm::testing` (feature `unstable-test-support`, which enables `unstable-internals`);
- the `openmvg-ba` oracle kernels (feature `unstable-internals`);

Unpublished applications and visualization adapters, and externally maintained research,
benchmark and validation tools, have no public release compatibility commitment.

The OMVG community format remains `0.1-draft`; its evolution rules are stated in the
[format specification](docs/format/OMVG_FORMAT_SPEC.md). The Rust API and the draft storage
format have separate compatibility rules.

## Conventions you can rely on

- Mathematical meaning is held stricter than syntax: the residual sign and units, the gauge, the
  local chart, noise whitening, the robust loss, the covariance interpretation and the
  transactional behavior of a solve never change silently, even in a minor release.
- Config types have private fields, by-value `with_*` setters and getters named after the field.
  Their serde field names are part of the schema.
- Fallible constructors are named `try_new` or `try_<what>` (`PinholeCamera::try_with_focal`);
  `new` never fails.
- Most public enums and records are `#[non_exhaustive]`: match them with a `_ =>` arm and build
  records through their constructors.
- Each public item has one path per crate. `openmvg-core` also re-exports its vocabulary types
  (cameras, IDs, units, `RunSeed`) at the crate root.
- Randomized APIs take a caller-supplied seed; the same inputs, seed and build give the same output.
