#!/usr/bin/env python3
"""Check architecture and dependency-tree boundaries."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import tomllib

ROOT = Path(__file__).resolve().parent.parent
# Runtime reads of the process environment, however `env` is imported; `env!` and `option_env!`
# are compile-time and allowed.
PROCESS_ENVIRONMENT_READ = re.compile(r"\benv::vars?(?:_os)?\s*\(")
# Every crate root and binary root of a supported package carries this attribute.
FORBID_UNSAFE = "#![forbid(unsafe_code)]"
# Every crate root and binary root of a supported package denies these lints, at least outside
# tests, so its code returns errors instead of panicking.
DENIED_PANIC_LINTS = ("clippy::unwrap_used", "clippy::expect_used", "clippy::panic")
# The unpublished packages that sit on top of the supported crates and may depend on them only:
# the Python bindings (a workspace member) and the Rerun adapter (its own workspace).
UNPUBLISHED_CLIENTS = {
    "openmvg-rs-python": Path("bindings/python"),
    "openmvg-rerun": Path("crates/openmvg-rerun"),
}
RANDOM_IMPLEMENTATION_PACKAGES = frozenset(
    {"getrandom", "rand", "rand_chacha", "rand_core", "rand_distr", "rand_pcg"}
)
# The OMVG community-format crate is the only supported crate allowed Arrow and Parquet, through
# its default `omvg` feature; without it the crate is the COLMAP export alone. Parquet
# also seeds its hash tables from OS entropy through `getrandom` on native targets, so trees that
# contain this crate are the only ones allowed `getrandom`; no stored value depends on that seed.
OMVG_PACKAGE = "openmvg-io"
# The umbrella feature that is the only way into the OMVG crate.
OMVG_FEATURE = "io"
# The command-line crate always writes OMVG output, so every tree of it admits the OMVG crate.
CLI_PACKAGE = "openmvg-cli"
# The terminal-UI stack of the `openmvg-sfm-configure` app: only the `tui` feature of the
# command-line crate may resolve it.
TUI_FEATURE = "tui"
TUI_PACKAGES = frozenset({"crossterm", "ratatui", "ratatui-core", "ratatui-image", "termwiz"})
# The internal binary files are the only users of postcard.
POSTCARD_PACKAGE = "openmvg-sfm"
# Storage formats no supported crate uses: OMVG has no ZIP form, SQLite and HDF5 are out of scope.
FORBIDDEN_FORMAT_PACKAGES = frozenset({"hdf5", "hdf5-sys", "libsqlite3-sys", "rusqlite", "zip"})

# This records the current, supported v0.1 graph. Target-architecture changes
# update the relevant entry in the same commit that changes a dependency edge.
ALLOWED_SUPPORTED_EDGES: dict[str, frozenset[str]] = {
    "openmvg-core": frozenset(),
    "openmvg-features": frozenset({"openmvg-core"}),
    "openmvg-ba": frozenset({"openmvg-core"}),
    "openmvg-geometry": frozenset({"openmvg-ba", "openmvg-core"}),
    "openmvg-sfm": frozenset(
        {
            "openmvg-core",
            "openmvg-ba",
            "openmvg-features",
            "openmvg-geometry",
        }
    ),
    "openmvg-io": frozenset({"openmvg-core", "openmvg-sfm"}),
    "openmvg": frozenset(
        {
            "openmvg-core",
            "openmvg-features",
            "openmvg-geometry",
            "openmvg-io",
            "openmvg-sfm",
        }
    ),
    "openmvg-cli": frozenset({"openmvg-core", "openmvg-io", "openmvg-sfm"}),
}


def load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as source:
        return tomllib.load(source)


def normal_dependencies(manifest: dict[str, Any]) -> dict[str, Any]:
    dependencies = dict(manifest.get("dependencies", {}))
    for target in manifest.get("target", {}).values():
        dependencies.update(target.get("dependencies", {}))
    return dependencies


def package_name(dependency_name: str, specification: Any) -> str:
    if isinstance(specification, dict):
        return specification.get("package", dependency_name)
    return dependency_name


def is_columnar(package: str) -> bool:
    return package == "parquet" or package.startswith("arrow")


def enables_feature(features: dict[str, list[str]], feature: str, target: str) -> bool:
    """Whether enabling `feature` enables the package's own feature `target`."""
    pending, seen = [feature], set()
    while pending:
        current = pending.pop()
        if current == target:
            return True
        if current not in seen:
            seen.add(current)
            pending.extend(item for item in features.get(current, []) if item in features)
    return False


def tree_selections(name: str, manifest: dict[str, Any]) -> list[tuple[str, list[str], bool]]:
    """Feature selections the dependency contracts cover, and whether each admits Arrow and
    Parquet."""
    features = manifest.get("features", {})
    selections = [
        (
            "all features",
            ["--all-features"],
            name in (OMVG_PACKAGE, CLI_PACKAGE) or OMVG_FEATURE in features,
        )
    ]
    if name == OMVG_PACKAGE:
        selections.append(("no default features", ["--no-default-features"], False))
    if OMVG_FEATURE in features:
        without = sorted(
            feature for feature in features if not enables_feature(features, feature, OMVG_FEATURE)
        )
        selections.append(
            (f"all features except {OMVG_FEATURE}", ["--features", ",".join(without)], False)
        )
    return selections


def resolved_packages(package: str, arguments: list[str], failures: list[str]) -> set[str] | None:
    """Package names in the resolved normal dependency tree of `package`."""
    command = [
        "pixi",
        "run",
        "cargo",
        "tree",
        "--locked",
        "--quiet",
        "-p",
        package,
        "-e",
        "normal",
        "--prefix",
        "none",
        "--format",
        "{p}",
        *arguments,
    ]
    result = subprocess.run(command, cwd=ROOT, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        failures.append(
            f"cargo tree failed for {package} {' '.join(arguments)}: {result.stderr.strip()}"
        )
        return None
    return {line.split()[0] for line in result.stdout.splitlines() if line.strip()}


def check_dependency_trees(
    packages: dict[str, tuple[Path, dict[str, Any]]], failures: list[str]
) -> None:
    for name in sorted(ALLOWED_SUPPORTED_EDGES):
        manifest = packages[name][1]
        for label, arguments, admits_columnar in tree_selections(name, manifest):
            tree = resolved_packages(name, [*arguments, "--target", "all"], failures)
            if tree is None:
                continue
            if not admits_columnar:
                if name != OMVG_PACKAGE and OMVG_PACKAGE in tree:
                    failures.append(f"{name} ({label}) resolves {OMVG_PACKAGE}")
                columnar = sorted(filter(is_columnar, tree))
                if columnar:
                    failures.append(
                        f"{name} ({label}) resolves Arrow or Parquet: " + ", ".join(columnar)
                    )
                if "getrandom" in tree:
                    failures.append(f"{name} ({label}) resolves getrandom")
            if name != CLI_PACKAGE:
                terminal = sorted(tree & TUI_PACKAGES)
                if terminal:
                    failures.append(
                        f"{name} ({label}) resolves the terminal-UI stack: " + ", ".join(terminal)
                    )
            formats = sorted(tree & FORBIDDEN_FORMAT_PACKAGES)
            if formats:
                failures.append(
                    f"{name} ({label}) resolves an unsupported storage format: "
                    + ", ".join(formats)
                )

        # Host target only: platform bindings such as `windows-sys` or `js-sys` compile no C.
        default_tree = resolved_packages(name, [], failures)
        if default_tree is not None:
            terminal = sorted(default_tree & TUI_PACKAGES)
            if terminal:
                failures.append(
                    f"{name} default features resolve the terminal-UI stack: "
                    + ", ".join(terminal)
                    + f" (only its `{TUI_FEATURE}` feature may)"
                )
            native = sorted(package for package in default_tree if package.endswith("-sys"))
            if native:
                failures.append(
                    f"{name} default features resolve a native *-sys crate: " + ", ".join(native)
                )


def crate_roots(member_path: Path, manifest: dict[str, Any]) -> set[Path]:
    """The library and binary roots of a package, as Cargo resolves them."""
    source_root = ROOT / member_path / "src"
    roots = {source_root / "lib.rs", source_root / "main.rs"}
    roots |= set(source_root.glob("bin/*.rs")) | set(source_root.glob("bin/*/main.rs"))
    roots |= {
        ROOT / member_path / target["path"]
        for target in manifest.get("bin", [])
        if "path" in target
    }
    return {root for root in roots if root.is_file()}


def check_unsafe_forbidden(
    member_path: Path, manifest: dict[str, Any], failures: list[str]
) -> None:
    for root in sorted(crate_roots(member_path, manifest)):
        if FORBID_UNSAFE not in root.read_text(encoding="utf-8"):
            failures.append(f"crate root does not forbid unsafe code: {root.relative_to(ROOT)}")


def inner_attributes(text: str) -> list[str]:
    """The `#![...]` attributes of a source file, each with its whole bracketed body."""
    attributes = []
    start = text.find("#![")
    while start != -1:
        depth = 0
        for end in range(start + 2, len(text)):
            if text[end] == "[":
                depth += 1
            elif text[end] == "]":
                depth -= 1
                if depth == 0:
                    attributes.append(text[start : end + 1])
                    break
        else:
            break
        start = text.find("#![", end + 1)
    return attributes


def check_panic_lints_denied(
    member_path: Path, manifest: dict[str, Any], failures: list[str]
) -> None:
    for root in sorted(crate_roots(member_path, manifest)):
        attributes = inner_attributes(root.read_text(encoding="utf-8"))
        denied = " ".join(attribute for attribute in attributes if "deny(" in attribute)
        missing = [lint for lint in DENIED_PANIC_LINTS if lint not in denied]
        if missing:
            failures.append(
                f"crate root does not deny {', '.join(missing)}: {root.relative_to(ROOT)}"
            )


def check_unpublished_clients(
    packages: dict[str, tuple[Path, dict[str, Any]]],
    supported: set[str],
    failures: list[str],
) -> None:
    """The bindings and the Rerun adapter may depend on the supported crates only."""
    workspace_names = set(packages) | set(UNPUBLISHED_CLIENTS)
    for name, member_path in UNPUBLISHED_CLIENTS.items():
        manifest_path = ROOT / member_path / "Cargo.toml"
        if not manifest_path.is_file():
            failures.append(f"unpublished client has no manifest: {member_path}")
            continue
        manifest = load_toml(manifest_path)
        internal = {
            package_name(dependency_name, specification)
            for dependency_name, specification in normal_dependencies(manifest).items()
            if package_name(dependency_name, specification) in workspace_names
            or package_name(dependency_name, specification).startswith("openmvg")
        }
        forbidden = internal - supported
        if forbidden:
            failures.append(
                f"{name} depends on unsupported packages: " + ", ".join(sorted(forbidden))
            )


def main() -> int:
    workspace_manifest = load_toml(ROOT / "Cargo.toml")
    workspace = workspace_manifest["workspace"]
    member_paths = [Path(path) for path in workspace["members"]]
    default_paths = {Path(path) for path in workspace["default-members"]}

    failures: list[str] = []
    packages: dict[str, tuple[Path, dict[str, Any]]] = {}
    path_to_name: dict[Path, str] = {}
    for member_path in member_paths:
        manifest_path = ROOT / member_path / "Cargo.toml"
        if not manifest_path.is_file():
            failures.append(f"workspace member has no manifest: {member_path}")
            continue
        manifest = load_toml(manifest_path)
        name = manifest["package"]["name"]
        if name in packages:
            failures.append(f"duplicate workspace package name: {name}")
            continue
        packages[name] = (member_path, manifest)
        path_to_name[member_path] = name

    unknown_defaults = default_paths - set(member_paths)
    if unknown_defaults:
        failures.append(
            "default-members are not workspace members: "
            + ", ".join(sorted(map(str, unknown_defaults)))
        )

    default_names = {path_to_name[path] for path in default_paths if path in path_to_name}
    expected_supported = set(ALLOWED_SUPPORTED_EDGES)
    if default_names != expected_supported:
        failures.append(
            "default-members must match the supported package set: "
            f"missing={sorted(expected_supported - default_names)}, "
            f"unexpected={sorted(default_names - expected_supported)}"
        )

    workspace_names = set(packages)
    for name, (member_path, manifest) in packages.items():
        is_supported = name in expected_supported
        is_publishable = manifest["package"].get("publish", True) is not False
        if not is_supported and is_publishable:
            failures.append(f"non-default package must set publish = false: {member_path}")

        internal_dependencies = {
            package_name(dependency_name, specification)
            for dependency_name, specification in normal_dependencies(manifest).items()
            if package_name(dependency_name, specification) in workspace_names
        }
        if is_supported:
            forbidden = internal_dependencies - ALLOWED_SUPPORTED_EDGES[name]
            if forbidden:
                failures.append(
                    f"{name} has forbidden normal workspace dependencies: "
                    + ", ".join(sorted(forbidden))
                )

            random_dependencies = {
                package_name(dependency_name, specification)
                for dependency_name, specification in normal_dependencies(manifest).items()
                if package_name(dependency_name, specification) in RANDOM_IMPLEMENTATION_PACKAGES
            }
            allowed_random_dependencies = (
                {"rand", "rand_chacha", "rand_core"} if name == "openmvg-core" else set()
            )
            forbidden_random = random_dependencies - allowed_random_dependencies
            if forbidden_random:
                failures.append(
                    f"{name} bypasses openmvg-core's deterministic RNG protocol: "
                    + ", ".join(sorted(forbidden_random))
                )

            direct_names = {
                package_name(dependency_name, specification)
                for dependency_name, specification in normal_dependencies(manifest).items()
            }
            if "postcard" in direct_names and name != POSTCARD_PACKAGE:
                failures.append(f"{name} depends on postcard, allowed only in {POSTCARD_PACKAGE}")

            if name == "openmvg-core":
                rand_specification = normal_dependencies(manifest).get("rand")
                if rand_specification is not None and (
                    not isinstance(rand_specification, dict)
                    or not rand_specification.get("optional", False)
                ):
                    failures.append(
                        "openmvg-core may use rand only as an optional test-support dependency"
                    )

        if is_supported:
            source_root = ROOT / member_path / "src"
            for source_path in sorted(source_root.rglob("*.rs")):
                if "tests" in source_path.relative_to(source_root).parts:
                    continue
                source = source_path.read_text(encoding="utf-8")
                if PROCESS_ENVIRONMENT_READ.search(source):
                    failures.append(
                        "supported library reads process environment directly: "
                        f"{source_path.relative_to(ROOT)}"
                    )

        if is_supported:
            check_unsafe_forbidden(member_path, manifest, failures)
            check_panic_lints_denied(member_path, manifest, failures)

    check_unpublished_clients(packages, expected_supported, failures)

    if not failures:
        check_dependency_trees(packages, failures)

    if failures:
        print("architecture check failed:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    print(
        "architecture check passed: "
        f"{len(default_names)} supported and "
        f"{len(packages) - len(default_names)} unpublished packages"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
