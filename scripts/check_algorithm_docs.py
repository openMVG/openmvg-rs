#!/usr/bin/env python3
"""Check that every algorithm's teaching docs have a diagram and reach docs.rs.

An algorithm module is the module that owns one public algorithm: a solver, an estimator, a
detector, a matcher, an optimizer. Its teaching docs must contain a fenced ```text block: the
diagram from input through the steps to output (see docs/DOCUMENTATION.md). Index modules that
only list their children are not algorithm modules.

REQUIRED lists the files that hold a user-facing algorithm's teaching docs. Those docs must
render: either they are the `//!` docs of a public module, or they sit in the `///` docs of a
public item that the crate exports. INTERNAL lists implementation modules whose docs serve
contributors and may stay private. PENDING holds modules without a diagram, and PENDING_RENDER
holds REQUIRED files whose docs do not render yet; both only shrink, because the check fails as
soon as an entry meets the rule. Any non-test file that implements `Solver` or
`MultiSolutionSolver` must be listed, so a new solver cannot skip the rule.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CRATES = ROOT / "crates"

REQUIRED = (
    "openmvg-ba/src/native/mod.rs",
    "openmvg-ba/src/numerics/refine/mod.rs",
    "openmvg-features/src/extraction/alp/mod.rs",
    "openmvg-features/src/extraction/sift/detector.rs",
    "openmvg-features/src/matching/matchers/brute_force.rs",
    "openmvg-features/src/matching/matchers/hnsw/mod.rs",
    "openmvg-features/src/tracking/mod.rs",
    "openmvg-geometry/src/alignment/mod.rs",
    "openmvg-geometry/src/alignment/robust.rs",
    "openmvg-geometry/src/averaging/filtering/graph_connectivity.rs",
    "openmvg-geometry/src/averaging/filtering/parallel_rigidity.rs",
    "openmvg-geometry/src/averaging/global_positioning/solver.rs",
    "openmvg-geometry/src/averaging/rotation/mod.rs",
    "openmvg-geometry/src/averaging/translation/mod.rs",
    "openmvg-geometry/src/averaging/view_graph/adaptive/mod.rs",
    "openmvg-geometry/src/averaging/view_graph/conditioning.rs",
    "openmvg-geometry/src/averaging/view_graph/strongest_edges.rs",
    "openmvg-geometry/src/essential/eight_point.rs",
    "openmvg-geometry/src/essential/five_point.rs",
    "openmvg-geometry/src/essential/mod.rs",
    "openmvg-geometry/src/essential/pose_decomposition.rs",
    "openmvg-geometry/src/essential/robust.rs",
    "openmvg-geometry/src/fundamental/eight_point.rs",
    "openmvg-geometry/src/fundamental/robust.rs",
    "openmvg-geometry/src/fundamental/seven_point.rs",
    "openmvg-geometry/src/homography/aca.rs",
    "openmvg-geometry/src/homography/dlt.rs",
    "openmvg-geometry/src/homography/refinement.rs",
    "openmvg-geometry/src/homography/robust.rs",
    "openmvg-geometry/src/pose/p3p/ding.rs",
    "openmvg-geometry/src/pose/p3p/fused.rs",
    "openmvg-geometry/src/pose/p3p/ke.rs",
    "openmvg-geometry/src/pose/p3p/kneip.rs",
    "openmvg-geometry/src/pose/p3p/lambdatwist.rs",
    "openmvg-geometry/src/pose/p3p/mod.rs",
    "openmvg-geometry/src/pose/p3p/nakano.rs",
    "openmvg-geometry/src/pose/p3p/robust.rs",
    "openmvg-geometry/src/pose/p3p/wu.rs",
    "openmvg-geometry/src/pose/p4pf/mod.rs",
    "openmvg-geometry/src/pose/resection.rs",
    "openmvg-geometry/src/pose/uncalibrated_resection.rs",
    "openmvg-geometry/src/robust/classic/mod.rs",
    "openmvg-geometry/src/triangulation/angular.rs",
    "openmvg-geometry/src/triangulation/bearing.rs",
    "openmvg-geometry/src/triangulation/dlt.rs",
    "openmvg-geometry/src/triangulation/midpoint.rs",
    "openmvg-geometry/src/triangulation/tracks/triangulator.rs",
)

# Implementation detail behind `unstable-internals` or the engines: contributor documentation,
# which may stay private.
INTERNAL = (
    "openmvg-ba/src/kernel/fixed_rotation/mod.rs",
    "openmvg-ba/src/kernel/multi_view/mod.rs",
    "openmvg-ba/src/kernel/multi_view/self_calibration/mod.rs",
    "openmvg-ba/src/kernel/multi_view/triangulation.rs",
    "openmvg-ba/src/kernel/two_view/mod.rs",
    "openmvg-ba/src/numerics/sparse.rs",
    "openmvg-geometry/src/robust/genransac/mod.rs",
    "openmvg-sfm/src/ops/averaging/rigid_core.rs",
    "openmvg-sfm/src/ops/radial_calibration.rs",
    "openmvg-sfm/src/ops/track_scale.rs",
    "openmvg-sfm/src/pipelines/global/mod.rs",
    "openmvg-sfm/src/pipelines/incremental/mod.rs",
)

PENDING: tuple[str, ...] = ()

# Teaching docs that docs.rs does not show yet: the module is private and no public item carries
# the diagram.
PENDING_RENDER: tuple[str, ...] = ()

SOLVER_IMPL = re.compile(
    r"(?m)^impl(?:<[^>]*>)?\s+(?:crate::solver_traits::)?(?:Multi(?:Solution)?)?Solver\s+for\b"
)
EXCLUDED_SUFFIXES = ("tests.rs", "test_support.rs")
# The trait definitions themselves, whose docs show a sample implementation.
EXEMPT = ("openmvg-geometry/src/solver_traits.rs",)


DIAGRAM = "```text"
PUBLIC_ITEM = re.compile(r"^\s*pub (?:const |async |unsafe )*(?:struct|enum|fn|trait|type) (\w+)")
MOD_DECL = r"(?m)^\s*(pub(?:\([a-z]+\))? )?mod {name};"
IMPL = re.compile(r"^impl(?:<[^>]*>)?\s+(?:[\w:<>, ]+\s+for\s+)?(\w+)")
PUB_USE = re.compile(r"pub use [^;]*;", re.DOTALL)


def has_diagram(path: Path) -> bool:
    return any(
        line.strip() in (f"//! {DIAGRAM}", f"/// {DIAGRAM}")
        for line in path.read_text().splitlines()
    )


def module_is_public(path: Path) -> bool:
    """Whether every `mod` declaration from the crate root down to `path` is `pub`."""
    while path.name != "lib.rs":
        name = path.parent.name if path.name == "mod.rs" else path.stem
        directory = path.parent.parent if path.name == "mod.rs" else path.parent
        parents = [directory / "mod.rs", directory.with_suffix(".rs"), directory / "lib.rs"]
        for parent in parents:
            if not parent.exists():
                continue
            declaration = re.search(MOD_DECL.format(name=re.escape(name)), parent.read_text())
            if declaration is not None:
                if declaration.group(1) != "pub ":
                    return False
                path = parent
                break
        else:
            return False
    return True


def exported_names(crate: str) -> str:
    """Every `pub use` statement of the crate, joined."""
    sources = (CRATES / crate / "src").glob("**/*.rs")
    return "\n".join(match for source in sources for match in PUB_USE.findall(source.read_text()))


def renders(path: Path, crate_exports: str) -> bool:
    """Whether the teaching diagram in `path` reaches docs.rs.

    While the module docs hold the diagram, the module must be public. Once the docs move to
    the item that runs the algorithm, that item must be public and exported.
    """
    lines = path.read_text().splitlines()
    public = module_is_public(path)
    if f"//! {DIAGRAM}" in (line.strip() for line in lines):
        return public
    for index, line in enumerate(lines):
        if line.strip() != f"/// {DIAGRAM}":
            continue
        rest = (
            later for later in lines[index + 1 :] if not later.strip().startswith(("///", "#["))
        )
        line = next(rest, "")
        item = PUBLIC_ITEM.match(line)
        if item is None:
            continue
        name = item.group(1)
        if line[0].isspace():
            # A method renders on its type's page.
            owner = next(
                (IMPL.match(above) for above in reversed(lines[:index]) if IMPL.match(above)), None
            )
            name = owner.group(1) if owner else name
        if public or re.search(rf"\b{name}\b", crate_exports) or defined_publicly(path, name):
            return True
    return False


def defined_publicly(path: Path, name: str) -> bool:
    """Whether the crate defines `name` as a public type in a public module."""
    crate_src = CRATES / path.relative_to(CRATES).parts[0] / "src"
    definition = re.compile(rf"(?m)^pub (?:struct|enum|trait|type) {re.escape(name)}\b")
    return any(
        definition.search(source.read_text()) and module_is_public(source)
        for source in crate_src.glob("**/*.rs")
    )


def implements_solver(path: Path) -> bool:
    text = path.read_text()
    test_module = text.find("mod tests")
    match = SOLVER_IMPL.search(text)
    return match is not None and (test_module < 0 or match.start() < test_module)


def main() -> int:
    errors: list[str] = []
    registered = set(REQUIRED) | set(INTERNAL) | set(PENDING)
    exports: dict[str, str] = {}
    for relative in (*REQUIRED, *INTERNAL):
        path = CRATES / relative
        if not path.exists():
            errors.append(f"{relative}: listed but missing")
        elif not has_diagram(path):
            errors.append(f"{relative}: teaching docs have no ```text diagram")
    for relative in REQUIRED:
        path = CRATES / relative
        if not path.exists():
            continue
        crate = relative.split("/")[0]
        exports.setdefault(crate, exported_names(crate))
        shown = renders(path, exports[crate])
        if relative in PENDING_RENDER and shown:
            errors.append(f"{relative}: its docs now render; remove it from PENDING_RENDER")
        elif relative not in PENDING_RENDER and not shown:
            errors.append(
                f"{relative}: its diagram does not reach docs.rs; put it on a public module or "
                "on the public item that runs the algorithm"
            )
    errors += [
        f"{entry}: in PENDING_RENDER but not in REQUIRED"
        for entry in set(PENDING_RENDER) - set(REQUIRED)
    ]
    for relative in PENDING:
        path = CRATES / relative
        if not path.exists():
            errors.append(f"{relative}: listed in PENDING but missing")
        elif has_diagram(path):
            errors.append(f"{relative}: now has a diagram; move it from PENDING to REQUIRED")
    for path in sorted(CRATES.glob("*/src/**/*.rs")):
        relative = path.relative_to(CRATES).as_posix()
        if relative.endswith(EXCLUDED_SUFFIXES) or relative in EXEMPT or relative in registered:
            continue
        if implements_solver(path):
            errors.append(f"{relative}: implements a solver trait but is not listed")
    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    print(
        f"algorithm modules: {len(REQUIRED)} user-facing ({len(PENDING_RENDER)} not rendered yet), "
        f"{len(INTERNAL)} internal, {len(PENDING)} without a diagram"
    )
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
