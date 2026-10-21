# SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Deterministic renderer of a small synthetic scene for the contract tests.

A textured cube stands on a textured ground plane; the cameras look at it from an arc of a
circle. The images are ray-cast with NumPy and saved as lossless PNG, so a given seed always
produces the same files.
"""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

WIDTH, HEIGHT, FOCAL = 640, 480, 560.0
VIEWS = 8


def k_matrix(focal: float = FOCAL) -> str:
    return f"{focal} 0 {WIDTH / 2}\n0 {focal} {HEIGHT / 2}\n0 0 1\n"


def texture(rng: np.random.Generator, size: int = 512) -> NDArray[np.uint8]:
    """Multi-scale smooth noise, rich in blob-like features at every scale."""
    acc = np.zeros((size, size))
    for cells in (6, 12, 24, 48, 96):
        grid = Image.fromarray(rng.random((cells, cells)).astype(np.float32), mode="F")
        acc += np.asarray(grid.resize((size, size), Image.Resampling.BICUBIC)) / cells**0.3
    acc -= acc.min()
    acc /= acc.max()
    return (acc * 255).astype(np.uint8)


def _sample(tex: NDArray[np.uint8], u: NDArray[np.float64], v: NDArray[np.float64]) -> NDArray:
    n = tex.shape[0] - 1
    return tex[np.clip((v * n).astype(int), 0, n), np.clip((u * n).astype(int), 0, n)]


def _render(
    center: NDArray[np.float64], faces: list[NDArray[np.uint8]], ground: NDArray[np.uint8]
) -> Image.Image:
    forward = -center / np.linalg.norm(center)
    right = np.cross(forward, (0.0, 1.0, 0.0))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rows, cols = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float64)
    x = (cols + 0.5 - WIDTH / 2) / FOCAL
    y = (rows + 0.5 - HEIGHT / 2) / FOCAL
    rays = x[..., None] * right + y[..., None] * down + forward

    image = np.full((HEIGHT, WIDTH), 128.0)
    depth = np.full((HEIGHT, WIDTH), np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        # Ground plane y = -1, |x|, |z| <= 6.
        t = (-1.0 - center[1]) / rays[..., 1]
        point = center + t[..., None] * rays
        hit = (t > 0) & (np.abs(point[..., 0]) <= 6) & (np.abs(point[..., 2]) <= 6)
        image[hit] = _sample(ground, (point[..., 0][hit] + 6) / 12, (point[..., 2][hit] + 6) / 12)
        depth[hit] = t[hit]
        # Cube [-1, 1]^3, intersected slab by slab.
        t0 = (-1.0 - center) / rays
        t1 = (1.0 - center) / rays
    near = np.minimum(t0, t1).max(-1)
    far = np.maximum(t0, t1).min(-1)
    hit = (far >= near) & (near > 0) & (near < depth)
    point = center + near[..., None] * rays
    axis = np.argmin(np.abs(np.abs(point) - 1.0), -1)
    for a in range(3):
        u_axis, v_axis = (i for i in range(3) if i != a)
        for side, sign in enumerate((-1, 1)):
            face = hit & (axis == a) & (np.sign(point[..., a]) == sign)
            image[face] = _sample(
                faces[2 * a + side],
                (point[..., u_axis][face] + 1) / 2,
                (point[..., v_axis][face] + 1) / 2,
            )
    return Image.fromarray(image.astype(np.uint8)).convert("RGB")


def render_scene(directory: Path, *, seed: int = 7) -> list[Path]:
    """Write ``VIEWS`` overlapping views and their ``K.txt`` into ``directory``."""
    rng = np.random.default_rng(seed)
    faces = [texture(rng) for _ in range(6)]
    ground = texture(rng, 1024)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for view in range(VIEWS):
        angle = np.radians(30 + 12 * view)
        center = np.array([5 * np.sin(angle), 1.5, -5 * np.cos(angle)])
        path = directory / f"view_{view:02d}.png"
        _render(center, faces, ground).save(path)
        paths.append(path)
    (directory / "K.txt").write_text(k_matrix())
    return paths


def render_unrelated(path: Path, *, seed: int) -> None:
    """Write an image of plain texture that overlaps nothing in the scene."""
    tex = texture(np.random.default_rng(seed), WIDTH)
    Image.fromarray(tex[:HEIGHT]).convert("RGB").save(path)
