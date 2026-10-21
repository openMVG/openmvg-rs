#!/usr/bin/env python3
"""Build and run the crate examples, which check their own results.

Each example in `crates/<crate>/examples/` opens with a `//!` header (docs/STYLE.md#examples).
An example whose `Data:` line names Sceaux Castle takes the image folder as its argument and
runs only with `--real`, which fetches the verified images through `scripts/sceaux.py`; the
others build their own synthetic input and run by default. Required features come from the
crate's `[[example]]` entries. The run stops at the first failing example.

    python -I scripts/run_examples.py          # synthetic examples, as CI runs them
    python -I scripts/run_examples.py --real   # the Sceaux Castle examples
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parent.parent
CRATES = ROOT / "crates"
# Rerun is a separate, unpublished workspace.
EXEMPT_CRATES = ("openmvg-rerun",)


def header(path: Path) -> list[str]:
    lines = []
    for line in path.read_text().splitlines():
        if not line.startswith("//!"):
            if lines:
                break
            continue
        lines.append(line[3:].strip())
    return lines


def needs_sceaux(path: Path) -> bool:
    return any(line.startswith("Data:") and "Sceaux" in line for line in header(path))


def required_features(crate: Path, name: str) -> list[str]:
    manifest = tomllib.loads((crate / "Cargo.toml").read_text())
    for example in manifest.get("example", []):
        if example.get("name") == name:
            return list(example.get("required-features", []))
    return []


def sceaux_images() -> Path:
    spec = importlib.util.spec_from_file_location("sceaux", ROOT / "scripts" / "sceaux.py")
    if spec is None or spec.loader is None:
        raise ImportError("cannot load scripts/sceaux.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    images: Path = module.fetch()
    return images


def main(arguments: list[str]) -> int:
    real = "--real" in arguments
    examples = [
        path
        for path in sorted(CRATES.glob("*/examples/*.rs"))
        if path.parts[-3] not in EXEMPT_CRATES and needs_sceaux(path) == real
    ]
    extra = [str(sceaux_images())] if real else []
    for path in examples:
        crate = path.parent.parent
        command = ["cargo", "run", "--locked", "--release", "--quiet", "-p", crate.name]
        command += ["--example", path.stem]
        features = required_features(crate, path.stem)
        if features:
            command += ["--features", ",".join(features)]
        print(f"==> {crate.name} {path.stem}", flush=True)
        if subprocess.run([*command, "--", *extra], cwd=ROOT, check=False).returncode != 0:
            print(f"error: example {crate.name}/{path.stem} failed", file=sys.stderr)
            return 1
    print(f"{len(examples)} examples passed")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
