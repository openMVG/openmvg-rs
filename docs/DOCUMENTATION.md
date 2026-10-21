# Documentation strategy

This page says where each kind of fact lives and which checks keep it there.
[STYLE.md](STYLE.md) covers how to write the prose.

## Where things go

| Surface | Holds | Does not hold |
|---|---|---|
| Item docs (`///`) | One summary sentence, then what a caller needs: preconditions, units, frames, errors, determinism. | The method, history, benchmarks. |
| Module docs (`//!`) | The educational layer: the problem, the algorithm diagram, the math, trade-offs, references. | A restatement of the item docs. |
| Crate root docs (`lib.rs`) | How to use the crate: what it is for, where to start, one runnable example that checks its result, the learning path, the feature table. The `openmvg-core` root also holds the conventions that every crate shares. | Per-module detail. |
| Crate `README.md` | A storefront: what the crate is, the install line, one example compiled as a doc test, and a link to the learning path on docs.rs. The `openmvg-cli` README is the programs' user guide and stays complete. | Anything the root docs lack; the root docs are the source. |
| `examples/` | Tutorials: one idea per file, synthetic data first, checking its own result ([STYLE.md](STYLE.md#examples)). | Benchmarks and dataset runners. |
| `figures/` | SVG figures embedded in rustdoc, for spatial ideas ([STYLE.md](STYLE.md#figures)). | Facts the prose around them lacks. |
| `README.md` (repository) | What the project is, a quickstart, links. | Per-crate API detail. |
| `docs/README.md` | Navigation to crate guides, contracts, the format and contributor instructions. | Copies of those guides. |
| `docs/ARCHITECTURE.md` | How the crates fit together, and the dependency rules. | Per-module lists that rustdoc already has. |
| `AGENTS.md`, `DEVELOPER.md` | Contributor and agent instructions, commands. | Facts about the library API. |
| `CHANGELOG.md`, `MIGRATION.md` | What changed between versions. | Current-state description. |
| `docs/format/` | The normative OMVG specification, independent reference and conformance vectors. | Internal format implementation plans. |

Each fact has one home and everything else links to it. A quantitative claim in rustdoc names
a retained evidence record, test, or available benchmark binary with matching source and inputs.

Collector commands, corpus configuration, operational reports and corpus records stay with the
external benchmark workspace that runs them. The format specification, reference and vectors
remain together; changes to one must pass both Rust and independent Python conformance.

## The educational layer

Every algorithm has teaching docs that open with a diagram. An algorithm module is the module
that owns one public algorithm: a solver, a robust estimator, a detector, a matcher, an
optimizer, or an averaging method. An index module that only lists its children, such as
`fundamental`, is not one.

The teaching docs stay in the algorithm's file and must reach docs.rs, where readers find
them:

- In a public module, they are the module docs (`//!`).
- When the module is private and one public item runs the algorithm, they are that item's docs
  (`///`): `AlpExtractor` carries the ALP method, `FivePointSolver` the five-point solver.
- When several methods sit behind one public entry, such as the `P3pMethod` variants, each
  method's module is public and holds only its docs; its solver stays `pub(crate)`, so no item
  gains a second path.
- Implementation modules behind `unstable-internals` or inside the engines keep private module
  docs for contributors; `scripts/check_algorithm_docs.py` lists them as `INTERNAL`.

The teaching docs contain, in this order:

1. A one-line summary and the reference (paper, book chapter, or original implementation).
2. A fenced `text` diagram from the input through each step to the output, with the sizes and
   units that matter: sample size, number of solutions, residual unit.
3. The problem statement and the method, with the formulas that explain why it works.
4. Failure modes: degenerate configurations, the checks that reject them, and the error
   returned.
5. Where it sits among its alternatives: when to use it and when to prefer another.

The diagram follows the existing solver modules, for example
`crates/openmvg-geometry/src/pose/p3p/kneip.rs`:

````text
//! ```text
//! input ──► step one ──► step two ──► output
//! ```
````

A new algorithm, or a rewrite of one, adds its teaching docs in the same change. A port of
external code records its provenance in those docs before it merges.

## Checks

| Check | Command | Covers |
|---|---|---|
| Rustdoc warnings, broken intra-doc links | `pixi run --locked cargo doc --locked --no-deps --features openmvg/full` with `RUSTDOCFLAGS=-D warnings` | Supported public profile; manual CI also checks each docs.rs profile |
| Missing item docs | `#![deny(missing_docs)]` | Supported crates |
| Algorithm diagrams, and that teaching docs render | `pixi run algorithm-docs` | The modules listed in `scripts/check_algorithm_docs.py`, and any file that implements `Solver` or `MultiSolutionSolver` |
| Dependency and architecture boundaries | `pixi run architecture` | Resolved dependency trees |
| Crate roots, examples and figures | `pixi run crate-docs` | The crate roots and examples listed in `scripts/check_crate_docs.py`, and every SVG under `crates/*/figures` |

Adding an algorithm module means adding the file that holds its teaching docs to `REQUIRED` in
`scripts/check_algorithm_docs.py`, or to `INTERNAL` for an implementation module. A module that
cannot have its diagram yet goes in `PENDING`, and one whose docs do not render yet goes in
`PENDING_RENDER`; each leaves its list as soon as it meets the rule.

Crate roots and examples follow the same pattern in `scripts/check_crate_docs.py`: one that does
not meet the rules yet stays in `PENDING_ROOTS` or `PENDING_EXAMPLES` until it does.
