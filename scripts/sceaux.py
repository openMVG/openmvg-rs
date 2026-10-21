#!/usr/bin/env python3
"""Fetch the Sceaux Castle images, verified against sceaux.sha256.

Sceaux Castle, 11 photographs and their K.txt camera matrix, is the real-image scene of the
examples and the Python tests. The files come from a pinned commit of
https://github.com/openMVG/ImageDataset_SceauxCastle. The directory
`$OPENMVG_DATASETS/ImageDataset_SceauxCastle-master` is used when present; otherwise the files
are downloaded once to `$OPENMVG_TEST_CACHE` (default `~/.cache/openmvg-rs-tests`).

Without arguments, print the images directory. With a command, run it with the images
directory appended as its last argument:

    python -I scripts/sceaux.py cargo run --release -p openmvg --example sfm --
"""

from __future__ import annotations

import hashlib
import http.client
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

COMMIT = "fde7f5faba555e3c54700477c304488613346a19"
URL = f"https://raw.githubusercontent.com/openMVG/ImageDataset_SceauxCastle/{COMMIT}"
MANIFEST = Path(__file__).with_name("sceaux.sha256")


class ChecksumError(Exception):
    """A dataset file does not match the committed SHA-256 manifest."""


def manifest() -> dict[str, str]:
    """Map each dataset file, relative to the dataset root, to its SHA-256 digest."""
    entries = {}
    for line in MANIFEST.read_text().splitlines():
        if line and not line.startswith("#"):
            digest, name = line.split()
            entries[name] = digest
    return entries


def download(root: Path, files: dict[str, str]) -> None:
    """Download each missing file under `root`."""
    for name in files:
        target = root / name
        if target.is_file():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(target.suffix + ".part")
        with urllib.request.urlopen(f"{URL}/{name}", timeout=60) as response:
            partial.write_bytes(response.read())
        partial.replace(target)


def fetch() -> Path:
    """Return the verified images directory, downloading the files first when needed.

    Raises `OSError` or `http.client.HTTPException` when a file cannot be read or downloaded,
    and `ChecksumError` when a file does not match the manifest.
    """
    files = manifest()
    datasets = os.environ.get("OPENMVG_DATASETS")
    local = Path(datasets, "ImageDataset_SceauxCastle-master") if datasets else None
    if local is not None and local.is_dir():
        root = local
    else:
        cache = os.environ.get("OPENMVG_TEST_CACHE", Path.home() / ".cache" / "openmvg-rs-tests")
        root = Path(cache, f"ImageDataset_SceauxCastle-{COMMIT[:12]}")
        download(root, files)
    for name, digest in files.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ChecksumError(f"{root / name} does not match {MANIFEST.name}")
    return root / "images"


def main(command: list[str]) -> int:
    try:
        images = fetch()
    except (ChecksumError, OSError, http.client.HTTPException) as error:
        print(f"error: cannot provide the Sceaux images: {error}", file=sys.stderr)
        return 1
    if not command:
        print(images)
        return 0
    return subprocess.run([*command, str(images)], check=False).returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
