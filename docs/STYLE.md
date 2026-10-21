# Writing style

This guide covers every piece of prose in the repository: READMEs, `docs/`, rustdoc, Python
docstrings and stubs, the CHANGELOG, error and log messages, and commit messages. The goal is
for each surface to read as if one careful author wrote it.

## Voice

- Write plainly and specifically. Say what the code does, under which conditions, and what it
  returns or rejects. Leave out whatever doesn't help the reader use or understand the code.
- **Item docs are short.** A function, type or field doc gives one summary sentence, then only
  what a caller needs: preconditions, units, frame conventions, the error conditions, and
  determinism if relevant. Don't repeat the signature.
- **Module docs teach.** A module-level doc (`//!`) is where the method gets explained: the
  problem, the math, the algorithm's steps, the trade-offs between variants, and references.
  This is the educational layer described in [PHILOSOPHY.md](../PHILOSOPHY.md). When the
  module is private, these docs go on the public item that runs the algorithm, so they reach
  docs.rs ([DOCUMENTATION.md](DOCUMENTATION.md#the-educational-layer)).
- Describe the current state only. History belongs in the CHANGELOG and in git. Rustdoc and
  guides never say "now", "new", "no longer", "previously", "was renamed" or "used to".
- Every quantitative claim (speed, accuracy, size, coverage) names the evidence: a test, a
  benchmark binary, or a committed report. A claim without evidence gets dropped.
- Never mention internal process: waves, batches, plans, review rounds, agents, scratch paths,
  machine paths, or private datasets. Public benchmark datasets (ETH3D, BlendedMVS, Strecha)
  are fine wherever a claim cites them.

## Teaching

The documentation teaches in three layers. A reader can start at any of them, so each layer
reads on its own:

- The crate root teaches how to use the crate: what it is for, where to start, one example,
  and how to get good results.
- Module docs teach how an algorithm works: the problem, the method, its failure modes, and
  when to choose it over its alternatives.
- Item docs state the contract that a caller or an implementer relies on.

In every layer:

- Lead with what the reader wants to do, then show how. Edge cases, invariants and error lists
  come after the explanation.
- Define a term the first time a page uses it, or link to the page that defines it. Prefer the
  plain word when there is one.
- Make the first example on a page the recommended path: the robust estimator before the raw
  solver, the default preset before the expert settings.
- Keep advice on getting good results qualitative: which settings matter, and what to check
  when a result is poor. Speed and accuracy numbers stay out of the docs.
- A page is done when a reader new to the project can follow it to a working result. Ask one
  to try, and fix the places where they get stuck.

## Words to avoid

| Avoid | Write instead |
|---|---|
| comprehensive, powerful, seamless, cutting-edge, blazing fast, state-of-the-art (uncited) | the concrete property, or nothing |
| leverage, utilize, facilitate | use, help |
| simply, just, easily, obviously, of course | delete |
| "it's worth noting", "note that", "importantly" | state the fact |
| ensures / guarantees (unless it truly is an invariant) | checks, rejects, returns |
| "production" (as in "production policy/solver/path") | default, or the recommended configuration |
| validated (with no pointer to the evidence) | the benchmark or test that does the validating |
| deliberately, intentionally | give the reason, or delete |
| guarded (as in "guarded solver/path") | name the check: "with a cheirality check", "falls back to …" |
| robust, as a general compliment | only in its technical sense: robust estimation, robust loss |
| canonical, as emphasis | only for a unique normal form: canonical pair order, canonical pose |

Things that read as generated text, to avoid everywhere:

- triplets of adjectives;
- "not just X, but Y";
- chains of em-dashes;
- bulleted lists whose items each start with a bold label for no reason;
- a closing summary sentence that repeats the paragraph;
- emoji.

## Terms

| Term | Rule |
|---|---|
| OpenMVG-rs | The project name in prose. The copyright lines keep their holder, "Pierre Moulon and OpenMVG Rust contributors". Crate names go in code format: `openmvg-sfm`. The Python distribution is `openmvg-rs`, imported as `openmvg_rs`. The repository is `openMVG/openmvg-rs`. |
| SfM | Write "Structure from Motion (SfM)" at the first use in each document or crate root, then "SfM". Never write SFM, Structure-from-Motion or structure-from-motion in prose. |
| Incremental, Global | Engine names are capitalized when they refer to the engine (`SfmEngine::Incremental`, "the Global engine"). Lowercase is for the general technique ("incremental SfM"). Python: `reconstruct(engine="incremental")`. |
| bundle adjustment (BA) | Expand at first use per document. |
| view, image, camera | A **view** is one image in a dataset, identified by `ViewId`. An **image** is the pixel data or file. A **camera** is the intrinsic model (`Camera`, `IntrinsicId`) that views share. A view's position and orientation is its **pose**. |
| pose | World-to-camera by default: `X_camera = R X_world + t`. Name the direction whenever it isn't the default (`camera-to-world`, center `C = -Rᵀ t`). |
| reprojection error, reprojection RMSE | The pixel distance between a landmark's projection and one observation, evaluated per observation. **Reprojection RMSE** is the root mean square of those errors over the observations supporting active landmarks. A **per-landmark mean error** averages a landmark's observations first (`error_px` in OMVG, `landmark_reprojection_error_stats`); say "per-landmark" whenever you use it, and never call its root mean square the reprojection RMSE. |
| BA objective | The configured sum of residual costs, including weights, robust losses and any prior or constraint terms. Reprojection RMSE is a separate reported quality metric. When the objective contains only unweighted squared pixel residuals over a fixed observation set, minimizing it also minimizes reprojection RMSE. |
| bearing | A unit-norm ray in the camera frame. Never use "bearing vector" and "ray" interchangeably within one document. |
| track, landmark, point | A **track** is the set of feature observations of one scene point across views. A **landmark** is a triangulated track in the reconstruction. Use "point" for exported geometry (point cloud, `point_count`). |
| ViewGraph | One word, code-formatted when referring to the type (`ViewGraph`). |
| seed | An RNG seed (`seed: u64`). The two views that start an Incremental reconstruction form the **initial pair** (identifiers that already say `seed_pair` stay). |
| gate | A threshold test that admits or rejects an item (parallax gate, residual gate). In prose, say what is being tested. |
| certified core | Internal to the Global engine. In public docs, describe it ("the largest subgraph whose rotations and translations pass the consistency checks") or leave it out. |
| inlier threshold | Always give the unit: pixels, radians, or the solver's residual unit (`Solver::Unit`). |
| WASM | Written `WASM`; the target is written `wasm32-unknown-unknown`. |
| Acronyms | Upper case in prose: RANSAC, MSAC, P3P, DLT, SIFT, COLMAP, PLY, JSON, EXIF, RNG. Identifiers follow Rust case rules (`P3pMethod`). |

## Spelling and format

- American English: initialize, normalize, behavior, color, license (noun and verb), modeling,
  center.
- Headings use sentence case ("Robust estimation", not "Robust Estimation").
- Code identifiers, paths, feature flags and command lines go in code format.
- Math is plain Unicode in module docs (`Rᵀ`, `‖x‖`, `∈ SO(3)`), with identifiers in code
  format whenever the text means the variable in the code.
- Units come after numbers with a space: `2.0 px`, `0.5°`, `30 MB`.

## Naming

- Identifiers use the glossary terms above. A function that takes a `ViewId` says `view`
  (`observing_view`, `view_intrinsics`); `image` is for pixel data, files and `ImageMeta`.
  `seed` is only an RNG seed; the starting views are the `initial_pair`.
- American spelling in identifiers too: `Meters`, `Canceled`, `center`, `neighbor`.
- Acronyms are one word in identifiers (`P3pMethod`, `HnswMatcher`); spell out uncommon ones
  (`InverseDepthWeightedMidpoint`, not `Idw…`). Don't abbreviate `index`, `iteration`,
  `direction` or `parameter`.
- Counts end in `_count` (`point_count`, `parameter_count`); no `num_` prefix, except
  domain terms from the literature (`num_octaves`). Python may use `num_` (PEP 8 idiom).
  Budgets keep `max_*` (`max_iterations`); the count actually used is `iteration_count`.
- Booleans read as predicates: `is_*`/`has_*` getters; a setter is `with_<field>(bool)`.
  Pick the positive form once per concept (`is_fixed`, not `is_variable` next to it).
- Setters are `with_<field>`, getters are `<field>`; the serde key is `<field>`.
- Errors are `<Thing>Error`; parse errors are `Parse<Thing>Error`. Variants don't repeat
  `Error` (`Numerical`, `InvalidConfig`), and one condition has one variant name across
  crates.
- Units live in the type (`PixelThreshold`, `AngularThreshold`). When a plain `f64` must
  carry a unit, suffix it: `_px`, `_deg`, `_rad`, `_m`, `_bytes`.
- Serde: structs use the Rust field name; enums use `rename_all = "snake_case"`.
- Fallible constructors are `try_new` or `try_<what>`; `new` never fails.
- Feature flags are kebab-case; unstable or test-only flags start with `unstable-`. An
  optional-dependency feature takes the dependency's name (`faer`), and a feature keeps one
  name in every crate (`parallel`, `libm`); umbrella forwarding features take the target
  crate's prefix (`sfm-observer`).
- Python names follow PEP 8 and the Rust concept; they may differ from the Rust name only
  where the Python value differs (`inlier_mask` holds booleans, `inliers` holds indices).
- No codenames in stable names: name the task (`TranslationAveraging`, `RobustOptions`) and put
  the algorithm (L1-IRLS, GenRANSAC) in the doc. A literature name the user picks among
  siblings is fine (`P3pMethod::LambdaTwist`, `FivePointSolver`, `EdgeCutError`).
- One name per stage: event phases, progress stages, reports and docs share the stage noun
  (`BundleAdjustment`, `TranslationAveraging`, `RotationAveraging`, `Registration`,
  `Triangulation`, `SelfCalibration`). Only events add an engine prefix (`Global*`).
- Name a module after what a user looks for (`extraction`, `alignment`, `view_graph`). No
  `pub mod` repeats its parent (`matching::matchers`); re-export leaf types one level up.
- A stable identifier longer than about 30 characters needs a reason; let the enclosing type
  carry the scope (`GlobalAveragingConfig::min_edge_isotropy`).
- Config switches are plain imperatives (`retriangulate`, `release_matching_data`); state
  predicates are `is_*`/`has_*` (`has_reconstruction`).
- Serde names are API: renaming a field or variant reachable from `RunReport` is a schema
  change. Batch such renames and bump `schema_version` once.

### Vocabulary

One word per concept, in identifiers, Python names, docs, CLI flags and serde names.

| Concept | Use | Not |
|---|---|---|
| A dataset image's slot | `view` (`ViewId`) | `image` for anything holding a `ViewId` |
| Intrinsic model shared by views | `camera`, `intrinsic` | `calibration` for the value |
| Start of Incremental SfM | `initial_pair` | `seed` |
| Adding a view to a reconstruction | `register`, `registration` | for a 3D–3D fit |
| 3D–3D rigid or similarity fit | `align`, `alignment` | `registration` |
| Verified pairwise graph | `view graph` (`ViewGraph`) | `viewing graph`, `pair graph` |
| Detect plus describe | `extraction`, `extract` | `description` for the whole step |
| Descriptor search over all candidates | `brute_force` | `exhaustive` (all *image pairs* only), `exact` |
| Parameter held constant in optimization | `fixed` (`is_fixed`, `with_fixed`) | `locked`, `frozen`, `constant` |
| Intrinsics known from the user | `calibrated` | `fixed` for provenance |
| Ends of an unordered pair | `first`, `second` | `a`/`b`, `left`/`right` |
| Ends of a directed measurement | `view_i`, `view_j` | — |
| Matcher roles | `query`, `target` | `train` |
| Bundle adjustment | `bundle_adjust` (fields), `BundleAdjustment` (types, stages) | `ba` outside crate and module names; `refinement` |
| Local polish of one model | `refine`, `Refinement` | for bundle adjustment |
| Freeing intermediate data | `release` | `drop`, `discard` |
| Rejected input | `reject` (gates), `drop` (cleanup) | — |
| Iterations and rounds | `iteration_count`, `round_count`; budgets `max_iterations` | `iters`, `n_rounds` |
| Focal length, principal point | `focal_x`, `focal_y`, `principal_x`, `principal_y` | `fx`/`cx` in identifiers |
| Plain checked record module | `model` | `kernel`, `types` |
| Error of a solver or estimator | `EstimationError` | `OperatorError` |

### Verbs

| Verb | Meaning | Example |
|---|---|---|
| `estimate_*` | Robust (RANSAC/MSAC) estimation from correspondences | `estimate_fundamental` |
| `solve` | Minimal or linear solver, no outlier handling | `Solver::solve` |
| `fit_*` | Closed-form least-squares fit on clean data | `fit_alignment` |
| `refine_*` | Local nonlinear polish of an existing estimate | `refine_bearing_pose` |
| `average_*` | Rotation or translation averaging over a graph | `average_rotations` |
| `triangulate_*` | Points from posed observations | `triangulate` |
| `filter_*` | Return the kept subset | `filter_by_scale_dispersion` |
| `audit_*` | Inspect without changing; return diagnostics | `audit_parallel_rigidity` |
| `build` | Assemble a value from parts | `Frontend::build` |
| `run*` | Execute an engine | `Incremental::run_prepared` |

### Type suffixes

| Suffix | Use for |
|---|---|
| `*Config` | Stored engine or solver configuration (`Solver::Config`, `IncrementalSfmConfig`) |
| `*Options` | Per-call arguments of a free function (`RobustOptions`) |
| `*Policy` | Enum selecting a configured behavior (`GaugeScalePolicy`) |
| `*Method` | Enum selecting an algorithm (`P3pMethod`) |
| `*Preset` | Named bundle of several knobs (`SiftFrontendPreset`), not one scalar knob |
| `*Estimate` | Value returned by an estimator |
| `*Report` | Statistics or diagnostics of a step or run (`RunReport`, `TrackBuildReport`) |
| `*Verdict` | Enum of a per-item accept/reject decision (`VerificationVerdict`, `RegistrationVerdict`) |
| `*Status` | Enum of a stage's completion state or a per-item state (`GlobalSelfCalibrationStatus`, `ViewVerdictStatus`) |
| `*Reason` | Enum explaining a rejection (`RegistrationFailureReason`) |
| `*Diagnostics` | Optional extra detail attached to an estimate or report |
| `*Error` | Error enum; variants drop the suffix |
| `*Solver` | Minimal or linear solver struct, not a robust estimator |
| `*Builder` | Builder with `build()` |

Not used: `*Result` (reserved for `std::result::Result` aliases), `*Outcome` (except
run-level outcome enums: `SfmOutcome` and its `GlobalOutcome` payload), `*Strategy`, `*Mode`,
`*Info`, and `*Data` for new types.

## Rustdoc

- Every summary line is one sentence that ends with a period. Use the imperative for functions
  ("Estimate the …", "Return …"). Use a noun phrase for types and fields ("Camera intrinsics
  shared by …").
- Sections use standard rustdoc headings: `# Errors`, `# Panics`, `# Examples`, `# References`,
  and `# Algorithm` in module docs. Don't use `# Arguments` or `# Returns` when the
  signature and summary already say the same thing. Keep them only when they add units,
  frames, or constraints.
- Link items with intra-doc links (``[`estimate_absolute_pose`]``) instead of backticks only.
- Diagrams belong in teaching docs: module docs, or the docs of the public item that carries
  an algorithm's teaching docs. Other item docs have none. Use one when it shows data flow or
  geometry that prose would obscure. Box diagrams use sentence-case labels (`Step 1: Algebraic
  roots`, not `STEP 1: ALGEBRAIC ROOTS`), plus `Input:` / `Output:` / `Complexity:` lines.
- References list author(s), title, venue and year. Add a DOI or arXiv ID when there is one.
- A doc example (`# Examples`) is a short call that runs and asserts its result. When the
  input takes more than a few lines to build, link to the example in `examples/` that builds it
  instead of hiding the setup behind `#` lines. A README example writes out its `fn main`,
  because GitHub shows hidden lines.

### Figures

- A figure is a static SVG embedded in rustdoc. Use one only where the idea is spatial and easy
  to get wrong in words: frames, pixel centers, which crate owns which step. Algorithm flow
  stays a fenced `text` diagram.
- Every fact a figure shows is also in the prose around it, so a reader who cannot see the
  figure, or a tool that reads only text, loses nothing. Equations stay in the prose.
- Draw text and lines in `currentColor`, so the figure follows rustdoc's light, dark and ayu
  themes. Use at most two accent colors, and never color alone to carry meaning.
- The SVG has a `<title>` and a `<desc>`, prefixes its ids with a short figure tag, stays under
  15 KB, and contains no `<script>`, `<style>`, `<foreignObject>` or external reference.
- Embed it with `#![doc = include_str!("../figures/<name>.svg")]` between blank doc lines,
  never in an item's summary paragraph. Without the blank line before, rustdoc breaks the page;
  without the one after, the next paragraph loses its Markdown.

## Examples

An example in a crate's `examples/` directory teaches one idea in one file that a reader can
copy:

- It opens with a `//!` header: a one-line summary, then `You will learn:`, `Data:`, `Run:`,
  `Expected:` and `Next:` lines.
- It builds its own input when it can, a synthetic scene with a fixed seed, runs the library,
  asserts that the result is right within a stated tolerance, and prints a short summary. An
  example that teaches reporting failures without panicking checks its result the same way and
  exits with `ExitCode::FAILURE` instead. Results can differ in the last bits across platforms,
  so it asserts tolerances, never exact counts, and `Expected:` gives ranges.
- It uses the supported API only, with the features named in its `required-features`, and
  never `unstable::`.
- It stays under about 120 lines, comments included, and uses no helper module. Setup that the
  reader needs to understand belongs in the file.
- A real-image example takes the image folder as its argument. `pixi run examples-real`
  fetches the Sceaux Castle images and runs these examples.

## Python

- Docstrings follow the same voice. The summary line is imperative and ends with a period.
  Parameters are documented with Google-style `Args:` / `Returns:` / `Raises:` sections, as the
  package already does.
- Names match the Rust concepts: `engine`, `AbsolutePoseEstimate`, `threshold`, `seed`.

## Error and log messages

- Messages start lowercase and have no trailing period, following Rust convention
  (`#[error("invalid focal length {value}: must be finite and positive")]`).
- Say what was wrong and include the offending value or identifier. Don't add a "please" or an
  "error:" prefix, and don't speculate about causes.
- Log lines describe what happened and give counts (`registered 21 of 24 views`), not the
  program's feelings about it.

## CHANGELOG

- Follow [Keep a Changelog](https://keepachangelog.com): `Added`, `Changed`, `Removed`,
  `Fixed`. A release starts with a short summary paragraph.
- Each entry is one sentence, or two at most, written for a user of the previous release: what
  changed and what they should do about it. Implementation detail belongs in the commit.

## Commit messages

- A title in the imperative, prefixed with the area (`geometry: name the robust absolute-pose
  entry point after its model`), then one paragraph that says why.
