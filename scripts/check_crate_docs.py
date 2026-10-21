#!/usr/bin/env python3
"""Check the teaching layer of the documentation: crate roots, examples and figures.

The rules are in docs/STYLE.md. A crate root in REQUIRED_ROOTS has a runnable doc example that
asserts its result, a "Learning path" section, and compiles its README as a doc test. An
example in REQUIRED_EXAMPLES opens with the header lines, asserts its result, and is named in
its crate root. Every SVG under crates/*/figures follows the figure rules and is embedded
between blank doc lines. Every Rust block of the repository README is a copy of a block in a
crate README, which its crate compiles as a doc test.

PENDING_ROOTS and PENDING_EXAMPLES hold what does not meet the rules yet. An entry leaves PENDING
when it does, and the check fails if a PENDING entry already passes, so the lists only shrink.
Every library crate root and every example must be in one of the lists.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CRATES = ROOT / "crates"

# The command-line programs document themselves in their README; Rerun is unpublished.
EXEMPT_CRATES = ("openmvg-cli", "openmvg-rerun")

REQUIRED_ROOTS = (
    "openmvg",
    "openmvg-ba",
    "openmvg-core",
    "openmvg-features",
    "openmvg-geometry",
    "openmvg-io",
    "openmvg-sfm",
)
PENDING_ROOTS: tuple[str, ...] = ()

REQUIRED_EXAMPLES = (
    "openmvg-ba/examples/native_bundle_adjustment.rs",
    "openmvg-ba/examples/robust_loss_and_gauge.rs",
    "openmvg-core/examples/camera_projection.rs",
    "openmvg-features/examples/detect_and_compare.rs",
    "openmvg-features/examples/match_and_filter.rs",
    "openmvg-geometry/examples/relative_pose.rs",
    "openmvg-geometry/examples/robust_homography.rs",
    "openmvg-geometry/examples/triangulate_and_resect.rs",
    "openmvg-io/examples/write_check_read.rs",
    "openmvg-sfm/examples/engines_and_reports.rs",
    "openmvg-sfm/examples/progress_and_cancellation.rs",
    "openmvg/examples/sfm.rs",
    "openmvg/examples/two_view_from_scratch.rs",
)
PENDING_EXAMPLES: tuple[str, ...] = ()

EXAMPLE_HEADER = ("You will learn:", "Data:", "Run:", "Expected:", "Next:")
FIGURE_MAX_BYTES = 15_000
FIGURE_FORBIDDEN = ("<script", "<style", "<foreignObject", "href=")
FENCE = re.compile(r"^```(\S*)")
NOT_RUN = ("text", "ignore", "no_run", "compile_fail", "toml", "bash", "json", "sh", "console")
SVG_ID = re.compile(r'\bid="([^"]+)"')
SVG_INCLUDE = re.compile(r'include_str!\("([^"]+\.svg)"\)')


def doc_lines(path: Path, marker: str) -> list[str]:
    """The text of the `marker` doc comments in `path`, without the marker."""
    lines = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith(marker):
            lines.append(stripped[len(marker) :].removeprefix(" "))
    return lines


def has_asserting_example(docs: list[str]) -> bool:
    """Whether the docs hold a code block that rustdoc runs and that asserts."""
    tag: str | None = None
    asserts = False
    for line in docs:
        fence = FENCE.match(line)
        if fence is None:
            asserts = asserts or (tag is not None and "assert" in line)
        elif tag is None:
            tag, asserts = fence.group(1), False
        else:
            if not any(word in tag for word in NOT_RUN) and asserts:
                return True
            tag = None
    return False


def root_errors(crate: str) -> list[str]:
    lib = CRATES / crate / "src" / "lib.rs"
    docs = doc_lines(lib, "//!")
    errors = []
    if not has_asserting_example(docs):
        errors.append("root docs have no runnable example that asserts its result")
    if "## Learning path" not in docs:
        errors.append("root docs have no '## Learning path' section")
    if 'include_str!("../README.md")' not in lib.read_text():
        errors.append("the README is not compiled as a doc test")
    return errors


def example_errors(relative: str) -> list[str]:
    path = CRATES / relative
    header = doc_lines(path, "//!")
    errors = [
        f"header has no '{key}' line"
        for key in EXAMPLE_HEADER
        if not any(line.startswith(key) for line in header)
    ]
    code = [line for line in path.read_text().splitlines() if not line.lstrip().startswith("//")]
    # An example that reports failures without panicking checks its result through its exit code.
    if not any("assert" in line or "ExitCode::FAILURE" in line for line in code):
        errors.append("the example does not assert its result")
    crate = relative.split("/")[0]
    if path.stem not in "\n".join(doc_lines(CRATES / crate / "src" / "lib.rs", "//!")):
        errors.append(f"the example is not named in the {crate} root docs")
    return errors


def figure_errors(path: Path, embedded: set[Path]) -> list[str]:
    text = path.read_text()
    lines = text.splitlines()
    errors = []
    if len(text.encode()) > FIGURE_MAX_BYTES:
        errors.append(f"larger than {FIGURE_MAX_BYTES} bytes")
    if not lines or not lines[0].startswith("<svg"):
        errors.append("the first line is not the <svg> tag")
    if "<title" not in text or "<desc" not in text:
        errors.append("missing <title> or <desc>")
    errors += [f"contains {word}" for word in FIGURE_FORBIDDEN if word in text]
    if any(not line.strip() for line in lines):
        errors.append("contains a blank line, which ends the Markdown HTML block")
    if any(line.startswith("    ") for line in lines):
        errors.append("a line starts with four spaces, which Markdown reads as code")
    ids = SVG_ID.findall(text)
    prefix = ids[0].split("-")[0] + "-" if ids else ""
    if any(not identifier.startswith(prefix) or prefix == "-" for identifier in ids):
        errors.append("ids do not share one '<tag>-' prefix")
    if path.resolve() not in embedded:
        errors.append("not embedded in any rustdoc")
    return errors


def embedding_errors(source: Path, embedded: set[Path]) -> list[str]:
    """Check every SVG embedding in `source` and record which figures it embeds."""
    lines = source.read_text().splitlines()
    errors = []
    for index, line in enumerate(lines):
        include = SVG_INCLUDE.search(line)
        if include is None:
            continue
        embedded.add((source.parent / include.group(1)).resolve())
        before = lines[index - 1].strip() if index > 0 else ""
        after = lines[index + 1].strip() if index + 1 < len(lines) else ""
        if before not in ("//!", "///") or after not in ("//!", "///"):
            errors.append(f"line {index + 1}: embed the figure between blank doc lines")
    return errors


def repository_readme_errors() -> list[str]:
    """Rust blocks of the repository README that no doc-tested crate README contains."""
    crate_readmes = "\n".join(path.read_text() for path in CRATES.glob("*/README.md"))
    blocks = re.findall(r"```rust[^\n]*\n(.*?)```", (ROOT / "README.md").read_text(), re.DOTALL)
    return [
        "README.md: a Rust example is not a copy of a crate README example, so nothing compiles it"
        for block in blocks
        if block not in crate_readmes
    ]


def check_listed(
    name: str, found: set[str], required: tuple[str, ...], pending: tuple[str, ...]
) -> list[str]:
    listed = set(required) | set(pending)
    errors = [f"{entry}: not in REQUIRED_{name} or PENDING_{name}" for entry in found - listed]
    errors += [f"{entry}: listed in {name.lower()} but missing" for entry in listed - found]
    return errors


def main() -> int:
    errors: list[str] = []
    crates = {
        path.parent.parent.name
        for path in CRATES.glob("*/src/lib.rs")
        if path.parent.parent.name not in EXEMPT_CRATES
    }
    examples = {
        path.relative_to(CRATES).as_posix()
        for path in CRATES.glob("*/examples/*.rs")
        if path.relative_to(CRATES).parts[0] not in EXEMPT_CRATES
    }
    errors += check_listed("ROOTS", crates, REQUIRED_ROOTS, PENDING_ROOTS)
    errors += check_listed("EXAMPLES", examples, REQUIRED_EXAMPLES, PENDING_EXAMPLES)

    for crate in sorted(crates & set(REQUIRED_ROOTS)):
        errors += [f"{crate}: {error}" for error in root_errors(crate)]
    for crate in sorted(crates & set(PENDING_ROOTS)):
        if not root_errors(crate):
            errors.append(f"{crate}: now meets the rules; move it to REQUIRED_ROOTS")
    for example in sorted(examples & set(REQUIRED_EXAMPLES)):
        errors += [f"{example}: {error}" for error in example_errors(example)]
    for example in sorted(examples & set(PENDING_EXAMPLES)):
        if not example_errors(example):
            errors.append(f"{example}: now meets the rules; move it to REQUIRED_EXAMPLES")

    errors += repository_readme_errors()

    embedded: set[Path] = set()
    for source in sorted(CRATES.glob("*/src/**/*.rs")):
        relative = source.relative_to(CRATES).as_posix()
        errors += [f"{relative}: {error}" for error in embedding_errors(source, embedded)]
    figures = sorted(CRATES.glob("*/figures/*.svg"))
    for figure in figures:
        relative = figure.relative_to(CRATES).as_posix()
        errors += [f"{relative}: {error}" for error in figure_errors(figure, embedded)]

    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    print(
        f"crate docs: {len(REQUIRED_ROOTS)} roots and {len(REQUIRED_EXAMPLES)} examples "
        f"checked, {len(PENDING_ROOTS)} roots and {len(PENDING_EXAMPLES)} examples pending, "
        f"{len(figures)} figures"
    )
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
