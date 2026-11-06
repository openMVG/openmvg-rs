# SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Reconstruction behavior on the hermetic synthetic scene."""

import multiprocessing
import os
import re
import shutil
import signal
import struct
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from conftest import colmap_bytes
from openmvg_rs import (
    DatasetError,
    ExportError,
    NoReconstructionError,
    PlyFormat,
    ReconstructionError,
    reconstruct,
)
from PIL import Image
from synthetic import FOCAL, HEIGHT, VIEWS, WIDTH, k_matrix, render_unrelated

ENGINES = ("incremental", "global")


def camera_focals(directory: Path) -> set[str]:
    lines = (directory / "sparse" / "0" / "cameras.txt").read_text().splitlines()
    return {line.split()[4] for line in lines if line and not line.startswith("#")}


@pytest.mark.parametrize("engine", ENGINES)
def test_registers_every_view(rendered: Path, engine: str) -> None:
    result = reconstruct(rendered, engine=engine, progress=False)
    assert result.engine == engine
    assert result.num_images == VIEWS
    assert result.num_registered_images == VIEWS
    assert result.num_points > 300
    assert result.registered_images == tuple(sorted(rendered.glob("*.png")))
    assert result.unregistered_images == ()
    assert result.intrinsics_mode == "fixed"


@pytest.mark.parametrize("engine", ENGINES)
def test_same_seed_gives_the_same_reconstruction(
    rendered: Path, tmp_path: Path, engine: str
) -> None:
    first = reconstruct(rendered, engine=engine, progress=False)
    second = reconstruct(rendered, engine=engine, progress=False)
    assert repr(first) == repr(second)
    assert first.registered_images == second.registered_images
    first.write_colmap(tmp_path / "first")
    second.write_colmap(tmp_path / "second")
    assert colmap_bytes(tmp_path / "first") == colmap_bytes(tmp_path / "second")


def test_points_and_views_match_the_colmap_export(rendered: Path, tmp_path: Path) -> None:
    """The arrays are the exported model: same points, and poses that reproject the observations."""
    result = reconstruct(rendered, progress=False)
    result.write_colmap(tmp_path)
    model = tmp_path / "sparse" / "0"

    def rows(name: str) -> list[list[str]]:
        lines = (model / name).read_text().splitlines()
        return [line.split() for line in lines if not line.startswith("#")]

    points = result.points
    assert points.shape == (result.num_points, 3)
    assert points.dtype == np.float64
    with pytest.raises(ValueError, match="read-only"):
        points[0, 0] = 0
    exported = rows("points3D.txt")
    np.testing.assert_array_equal(points, [[float(v) for v in row[1:4]] for row in exported])
    row_of_point = {int(row[0]): index for index, row in enumerate(exported)}

    views = result.views
    assert tuple(view.image for view in views) == result.registered_images
    camera = views[0].camera
    assert all(view.camera is camera for view in views)
    assert camera.model == "PINHOLE"
    assert (camera.width, camera.height) == (WIDTH, HEIGHT)
    assert dict(camera.parameters) == {"fx": FOCAL, "fy": FOCAL, "cx": WIDTH / 2, "cy": HEIGHT / 2}
    assert camera.matrix is not None
    lines = rows("images.txt")
    for view, header, observations in zip(views, lines[0::2], lines[1::2], strict=True):
        rotation, translation = view.pose.rotation, view.pose.translation
        np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(translation, [float(v) for v in header[5:8]], atol=1e-9)
        np.testing.assert_allclose(view.pose.center, -rotation.T @ translation)
        observed = np.array(observations, dtype=float).reshape(-1, 3)
        world = points[[row_of_point[int(i)] for i in observed[:, 2]]]
        projected = (world @ rotation.T + translation) @ camera.matrix.T
        pixels = projected[:, :2] / projected[:, 2:]
        # COLMAP puts the center of the top-left pixel at (0.5, 0.5), the library at (0, 0).
        error = np.linalg.norm(pixels - (observed[:, :2] - 0.5), axis=1)
        assert np.median(error) < 1.0


def test_images_subfolder_layout(rendered: Path, tmp_path: Path) -> None:
    shutil.copytree(rendered, tmp_path / "root" / "images")
    nested = reconstruct(tmp_path / "root", progress=False)
    flat = reconstruct(str(rendered), progress=False)
    assert nested.registered_images == tuple(sorted((tmp_path / "root" / "images").glob("*.png")))
    nested.write_colmap(tmp_path / "nested")
    flat.write_colmap(tmp_path / "flat")
    assert colmap_bytes(tmp_path / "nested") == colmap_bytes(tmp_path / "flat")


def test_without_calibration_the_intrinsics_are_estimated(scene: Path) -> None:
    (scene / "K.txt").unlink()
    result = reconstruct(scene, progress=False)
    assert result.intrinsics_mode == "estimated"
    assert result.num_registered_images == VIEWS


def test_images_without_exif_share_one_camera_on_request(scene: Path) -> None:
    (scene / "K.txt").unlink()
    separate = reconstruct(scene, progress=False)
    assert len({id(view.camera) for view in separate.views}) == separate.num_registered_images

    shared = reconstruct(
        scene, share_intrinsics=True, camera_model="pinhole-radial2", progress=False
    )
    assert shared.intrinsics_mode == "estimated"
    assert shared.num_registered_images == VIEWS
    camera = shared.views[0].camera
    assert all(view.camera is camera for view in shared.views)
    assert camera.model == "BROWN_CONRADY"
    assert {"k1", "k2"} <= dict(camera.parameters).keys()


def test_k_file_takes_precedence_over_shared_intrinsics(scene: Path, tmp_path: Path) -> None:
    result = reconstruct(
        scene, share_intrinsics=True, camera_model="pinhole-radial2", progress=False
    )
    assert result.intrinsics_mode == "fixed"
    result.write_colmap(tmp_path / "out")
    assert camera_focals(tmp_path / "out") == {f"{FOCAL:g}"}


def test_k_file_takes_precedence_over_exif(scene: Path, tmp_path: Path) -> None:
    exif = Image.Exif()
    exif.get_ifd(0x8769)[0xA405] = 50  # FocalLengthIn35mmFilm, far from the K.txt focal
    for png in scene.glob("*.png"):
        with Image.open(png) as image:
            image.save(png.with_suffix(".jpg"), quality=95, exif=exif)
        png.unlink()
    with_k = reconstruct(scene, progress=False)
    assert with_k.intrinsics_mode == "fixed"
    with_k.write_colmap(tmp_path / "with_k")
    assert camera_focals(tmp_path / "with_k") == {f"{FOCAL:g}"}

    (scene / "K.txt").unlink()
    exif_only = reconstruct(scene, progress=False)
    assert exif_only.intrinsics_mode == "estimated"


def test_explicit_intrinsics_take_precedence(scene: Path, tmp_path: Path) -> None:
    explicit = tmp_path / "calibration.txt"
    explicit.write_text(k_matrix(570.0))
    result = reconstruct(scene, intrinsics=explicit, progress=False)
    assert result.intrinsics_mode == "fixed"
    result.write_colmap(tmp_path / "out")
    assert camera_focals(tmp_path / "out") == {"570"}

    (scene / "K.txt").unlink()
    assert reconstruct(scene, intrinsics=str(explicit), progress=False).intrinsics_mode == "fixed"


@pytest.mark.parametrize("engine", ENGINES)
def test_unrelated_image_is_reported_unregistered(scene: Path, engine: str) -> None:
    render_unrelated(scene / "zz_unrelated.png", seed=5)
    result = reconstruct(scene, engine=engine, progress=False)
    assert result.num_images == VIEWS + 1
    assert result.num_registered_images == VIEWS
    assert result.unregistered_images == (scene / "zz_unrelated.png",)


@pytest.mark.parametrize("engine", ENGINES)
def test_two_view_reconstruction_registers_both_images(scene: Path, engine: str) -> None:
    for png in sorted(scene.glob("*.png"))[2:]:
        png.unlink()
    result = reconstruct(scene, engine=engine, progress=False)
    assert result.num_registered_images == 2
    assert result.num_points > 0


def test_global_without_a_core_raises_no_reconstruction(scene: Path) -> None:
    for png in sorted(scene.glob("*.png"))[2:]:
        png.unlink()
    # Nearly parallel bearings under this wrong calibration cannot establish a metric core.
    (scene / "K.txt").write_text(k_matrix(1_000_000.0))
    with pytest.raises(NoReconstructionError):
        reconstruct(scene, engine="global", progress=False)


def test_images_that_never_overlap_raise_reconstruction_error(tmp_path: Path) -> None:
    for seed in range(3):
        render_unrelated(tmp_path / f"unrelated_{seed}.png", seed=100 + seed)
    with pytest.raises(ReconstructionError):
        reconstruct(tmp_path, progress=False)


def test_missing_folder(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError) as caught:
        reconstruct(tmp_path / "missing", progress=False)
    assert caught.value.filename == str(tmp_path / "missing")


def test_missing_explicit_intrinsics(rendered: Path, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError) as caught:
        reconstruct(rendered, intrinsics=tmp_path / "K.txt", progress=False)
    assert caught.value.filename == str(tmp_path / "K.txt")


def test_folder_without_images(tmp_path: Path) -> None:
    with pytest.raises(DatasetError):
        reconstruct(tmp_path, progress=False)


def test_single_image(scene: Path) -> None:
    for png in sorted(scene.glob("*.png"))[1:]:
        png.unlink()
    with pytest.raises(DatasetError):
        reconstruct(scene, progress=False)


def test_malformed_k_file(scene: Path) -> None:
    (scene / "K.txt").write_text("1 2 3\n")
    with pytest.raises(DatasetError, match=r"K\.txt"):
        reconstruct(scene, progress=False)


def test_writers(rendered: Path, tmp_path: Path) -> None:
    result = reconstruct(rendered, progress=False)
    out = tmp_path / "a" / "b"
    result.write_colmap(out)
    (out / "sparse" / "0" / "notes.txt").write_text("kept")
    first = colmap_bytes(out)
    result.write_colmap(str(out))
    assert colmap_bytes(out) == first
    assert (out / "sparse" / "0" / "notes.txt").read_text() == "kept"
    images = first["images.txt"].decode()
    assert f"# Number of images: {VIEWS}" in images

    ply = tmp_path / "c" / "points.ply"
    result.write_ply(ply)
    data = ply.read_bytes()
    assert data.startswith(b"ply\nformat binary_little_endian 1.0\n")
    assert f"element vertex {result.num_points + result.num_registered_images}\n".encode() in data
    assert b"property uchar red\nproperty uchar green\nproperty uchar blue\n" in data
    _, payload = data.split(b"end_header\n", 1)
    vertices = list(struct.iter_unpack("<fffBBB", payload))
    assert len(vertices) == result.num_points + result.num_registered_images
    assert sum(vertex[3:] == (0, 255, 0) for vertex in vertices) == result.num_registered_images
    assert sum(vertex[3:] == (255, 255, 255) for vertex in vertices) == result.num_points
    result.write_ply(str(ply))
    assert ply.read_bytes() == data
    result.write_ply(ply, format=PlyFormat.Binary)
    assert ply.read_bytes() == data
    result.write_ply(ply, format=PlyFormat.Ascii)
    ascii_data = ply.read_bytes()
    assert ascii_data.startswith(b"ply\nformat ascii 1.0\n")
    _, ascii_payload = ascii_data.split(b"end_header\n", 1)
    ascii_vertices = [line.split() for line in ascii_payload.splitlines()]
    assert len(ascii_vertices) == len(vertices)
    for text, binary in zip(ascii_vertices, vertices, strict=True):
        assert struct.pack("<fff", *(float(value) for value in text[:3])) == struct.pack(
            "<fff", *binary[:3]
        )
        assert tuple(int(value) for value in text[3:]) == binary[3:]
    with pytest.raises(TypeError):
        result.write_ply(ply, format="ascii")  # type: ignore[arg-type]
    assert ply.read_bytes() == ascii_data


def test_writer_errors(rendered: Path, tmp_path: Path) -> None:
    result = reconstruct(rendered, progress=False)
    blocker = tmp_path / "file"
    blocker.write_text("")
    with pytest.raises(ExportError, match=re.escape(str(blocker))):
        result.write_colmap(blocker)
    with pytest.raises(ExportError, match=re.escape(str(blocker / "points.ply"))):
        result.write_ply(blocker / "points.ply")
    with pytest.raises(TypeError):
        result.write_ply(b"points.ply")  # type: ignore[arg-type]


@pytest.mark.skipif(
    sys.platform == "win32" or os.geteuid() == 0, reason="needs POSIX permissions as non-root"
)
def test_permission_denied_raises_permission_error(rendered: Path, tmp_path: Path) -> None:
    result = reconstruct(rendered, progress=False)
    locked = tmp_path / "locked"
    locked.mkdir(mode=0o500)
    try:
        with pytest.raises(PermissionError) as caught:
            result.write_ply(locked / "points.ply")
        assert caught.value.filename == str(locked / "points.ply")
        with pytest.raises(PermissionError) as caught:
            result.write_colmap(locked / "model")
        assert caught.value.filename == str(locked / "model")
    finally:
        locked.chmod(0o700)


@pytest.mark.skipif(
    sys.platform == "win32" or os.geteuid() == 0, reason="needs POSIX permissions as non-root"
)
def test_unreadable_input_raises_permission_error(scene: Path) -> None:
    calibration = scene / "K.txt"
    calibration.chmod(0o000)
    with pytest.raises(PermissionError) as caught:
        reconstruct(scene, progress=False)
    assert caught.value.filename == str(calibration)
    calibration.chmod(0o600)

    image = min(scene.glob("*.png"))
    image.chmod(0o000)
    with pytest.raises(PermissionError) as caught:
        reconstruct(scene, progress=False)
    assert caught.value.filename == str(image)


def test_one_worker_gives_the_same_reconstruction(rendered: Path, tmp_path: Path) -> None:
    reconstruct(rendered, progress=False, max_workers=1).write_colmap(tmp_path / "one")
    reconstruct(rendered, progress=False).write_colmap(tmp_path / "all")
    assert colmap_bytes(tmp_path / "one") == colmap_bytes(tmp_path / "all")


def test_progress_draws_nothing_when_stderr_is_not_a_terminal(
    rendered: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    reconstruct(rendered, progress=True)
    captured = capfd.readouterr()
    assert captured.err == ""
    assert captured.out == ""


def test_gil_is_released_during_the_run(rendered: Path) -> None:
    ticks = 0
    done = threading.Event()

    def tick() -> None:
        nonlocal ticks
        while not done.is_set():
            ticks += 1
            time.sleep(0.005)

    ticker = threading.Thread(target=tick)
    ticker.start()
    start = time.perf_counter()
    try:
        reconstruct(rendered, progress=False, max_workers=1)
    finally:
        elapsed = time.perf_counter() - start
        done.set()
        ticker.join()
    # With the GIL held, the ticker cannot run during the call: 0 to 2 ticks, however long it
    # lasts. Released, it ticks about every 5 ms, or every 15.6 ms on a coarse timer such as
    # Windows'. One tick per 100 ms of the call keeps a wide margin on any machine speed.
    required = max(3, int(elapsed / 0.1))
    assert ticks >= required, f"{ticks} ticks during a {elapsed:.2f} s run, {required} required"


def test_concurrent_calls(rendered: Path, tmp_path: Path) -> None:
    expected = {engine: reconstruct(rendered, engine=engine, progress=False) for engine in ENGINES}
    for engine, result in expected.items():
        result.write_colmap(tmp_path / engine)

    def run(index: int, engine: str) -> None:
        result = reconstruct(rendered, engine=engine, progress=False, max_workers=2)
        assert repr(result) == repr(expected[engine])
        result.write_colmap(tmp_path / f"run{index}")
        assert colmap_bytes(tmp_path / f"run{index}") == colmap_bytes(tmp_path / engine)

    calls = list(enumerate(ENGINES * 2))
    with ThreadPoolExecutor(max_workers=len(calls)) as pool:
        futures = [pool.submit(run, index, engine) for index, engine in calls]
    for future in futures:
        future.result()


def registered_count(image_dir: str) -> int:
    return reconstruct(image_dir, progress=False).num_registered_images


@pytest.mark.skipif(sys.platform != "linux", reason="the fork start method is a Linux default")
def test_process_forked_after_a_run(rendered: Path) -> None:
    # A child forked after a run inherits a global thread pool whose threads did not survive the
    # fork; without a private pool it would wait for them forever.
    reconstruct(rendered, progress=False)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "This process .* is multi-threaded", DeprecationWarning)
        with multiprocessing.get_context("fork").Pool(1) as pool:
            assert pool.apply_async(registered_count, (str(rendered),)).get(timeout=120) == VIEWS


@pytest.mark.skipif(not hasattr(signal, "setitimer"), reason="needs signal.setitimer")
def test_keyboard_interrupt_cancels_the_run(rendered: Path) -> None:
    if threading.current_thread() is not threading.main_thread():
        pytest.skip("signals are delivered to the main thread only")
    # One worker keeps the run (about 1.5 s natively) well past the alarm. The bound scales with
    # the uninterrupted run so that slow machines and CPU emulation keep the same contract.
    start = time.monotonic()
    reconstruct(rendered, progress=False, max_workers=1)
    full_run = time.monotonic() - start
    delay = 0.2
    previous = signal.signal(signal.SIGALRM, lambda *_: signal.raise_signal(signal.SIGINT))
    try:
        signal.setitimer(signal.ITIMER_REAL, delay)
        start = time.monotonic()
        with pytest.raises(KeyboardInterrupt):
            reconstruct(rendered, progress=False, max_workers=1)
        elapsed = time.monotonic() - start
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    assert elapsed < delay + max(1.0, full_run / 4)
    assert reconstruct(rendered, progress=False).num_registered_images == VIEWS
