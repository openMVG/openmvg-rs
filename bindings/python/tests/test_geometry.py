# SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
# SPDX-License-Identifier: MIT OR Apache-2.0

"""The robust geometry estimators on a synthetic two-view scene with outliers."""

import copy
import dataclasses
import pickle
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from openmvg_rs import (
    EstimationError,
    estimate_absolute_pose,
    estimate_essential,
    estimate_fundamental,
    estimate_homography,
)

K = np.array([[800.0, 0.0, 320.0], [0.0, 780.0, 240.0], [0.0, 0.0, 1.0]])
COUNT, OUTLIERS = 200, 40


def rotation(axis: tuple[float, float, float], degrees: float) -> NDArray[np.float64]:
    a = np.asarray(axis) / np.linalg.norm(axis)
    skew = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    angle = np.radians(degrees)
    return np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * skew @ skew


R = rotation((0.2, 1.0, 0.1), 12.0)
T = np.array([-1.0, 0.1, 0.2])


def project(points: NDArray[np.float64], r: NDArray[np.float64], t: NDArray[np.float64]) -> Any:
    camera = points @ r.T + t
    pixels = camera @ K.T
    return pixels[:, :2] / pixels[:, 2:]


def corrupt(rng: np.random.Generator, pixels: NDArray[np.float64]) -> NDArray[np.float64]:
    noisy = pixels + rng.normal(0, 0.3, pixels.shape)
    noisy[:OUTLIERS] = rng.uniform((0, 0), (640, 480), (OUTLIERS, 2))
    return noisy


def scene(planar: bool = False) -> tuple[NDArray[np.float64], NDArray[np.float64], Any]:
    rng = np.random.default_rng(7)
    world = np.column_stack(
        [rng.uniform(-2, 2, COUNT), rng.uniform(-1.5, 1.5, COUNT), rng.uniform(4, 8, COUNT)]
    )
    if planar:
        world[:, 2] = 6.0 + 0.3 * world[:, 0]
    x1 = project(world, np.eye(3), np.zeros(3))
    x2 = corrupt(rng, project(world, R, T))
    return world, x1, x2


def angle_degrees(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    cosine = (np.trace(a.T @ b) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(cosine, -1, 1))))


def outlier_recall(mask: NDArray[np.bool_]) -> float:
    return float(np.mean(mask[OUTLIERS:]))


def test_fundamental_recovers_the_epipolar_geometry() -> None:
    _, x1, x2 = scene()
    estimate = estimate_fundamental(x1, x2)
    assert estimate.matrix.shape == (3, 3)
    # A random point can fall within the threshold of its epipolar line.
    assert estimate.inlier_mask[:OUTLIERS].sum() <= 2
    assert outlier_recall(estimate.inlier_mask) > 0.95
    h1 = np.column_stack([x1, np.ones(COUNT)])[estimate.inlier_mask]
    h2 = np.column_stack([x2, np.ones(COUNT)])[estimate.inlier_mask]
    lines = h1 @ estimate.matrix.T
    distance = np.abs(np.sum(h2 * lines, axis=1)) / np.hypot(lines[:, 0], lines[:, 1])
    assert np.median(distance) < 1.0


def test_homography_maps_a_plane() -> None:
    _, x1, x2 = scene(planar=True)
    estimate = estimate_homography(x1, x2)
    assert not estimate.inlier_mask[:OUTLIERS].any()
    assert outlier_recall(estimate.inlier_mask) > 0.95
    mapped = np.column_stack([x1, np.ones(COUNT)]) @ estimate.matrix.T
    error = np.linalg.norm(mapped[:, :2] / mapped[:, 2:] - x2, axis=1)[estimate.inlier_mask]
    assert np.median(error) < 1.0


def test_essential_recovers_the_relative_pose() -> None:
    _, x1, x2 = scene()
    estimate = estimate_essential(x1, x2, K)
    assert angle_degrees(estimate.pose.rotation, R) < 0.2
    direction = T / np.linalg.norm(T)
    assert np.degrees(np.arccos(np.clip(estimate.pose.translation @ direction, -1, 1))) < 2.0
    assert np.linalg.norm(estimate.pose.translation) == pytest.approx(1.0)
    assert outlier_recall(estimate.inlier_mask) > 0.95
    same = estimate_essential(x1, x2, K, K)
    np.testing.assert_array_equal(same.matrix, estimate.matrix)


def test_absolute_pose_recovers_the_camera_pose() -> None:
    world, _, x2 = scene()
    estimate = estimate_absolute_pose(world, x2, K)
    assert angle_degrees(estimate.pose.rotation, R) < 0.1
    np.testing.assert_allclose(estimate.pose.translation, T, atol=0.02)
    np.testing.assert_allclose(estimate.pose.center, -R.T @ T, atol=0.05)
    assert not estimate.inlier_mask[:OUTLIERS].any()
    assert outlier_recall(estimate.inlier_mask) > 0.95


def test_the_same_seed_gives_the_same_result() -> None:
    _, x1, x2 = scene()
    first, second = estimate_fundamental(x1, x2, seed=3), estimate_fundamental(x1, x2, seed=3)
    np.testing.assert_array_equal(first.matrix, second.matrix)
    np.testing.assert_array_equal(first.inlier_mask, second.inlier_mask)


def test_array_layout_and_type_do_not_change_the_result() -> None:
    _, x1, x2 = scene()
    reference = estimate_fundamental(x1, x2)
    strided = np.zeros((COUNT, 4))
    strided[:, ::2] = x1
    for points in (np.asfortranarray(x1), strided[:, ::2], x1.tolist()):
        result = estimate_fundamental(points, x2)
        np.testing.assert_array_equal(result.matrix, reference.matrix)


def test_results_are_immutable_and_pickle() -> None:
    world, _, x2 = scene()
    estimate = estimate_absolute_pose(world, x2, K)
    with pytest.raises(dataclasses.FrozenInstanceError):
        estimate.pose = estimate.pose  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        estimate.pose.rotation = np.eye(3)  # type: ignore[misc]
    pose = estimate.pose
    for array in (pose.rotation, pose.translation, pose.center, estimate.inlier_mask):
        with pytest.raises(ValueError, match="read-only"):
            array[0] = 0
    assert type(estimate).__module__ == "openmvg_rs"
    restored = pickle.loads(pickle.dumps(estimate))
    np.testing.assert_array_equal(restored.pose.rotation, estimate.pose.rotation)
    assert estimate.inlier_mask.dtype == np.bool_
    kept = np.count_nonzero(estimate.inlier_mask)
    assert repr(estimate) == f"<openmvg_rs.AbsolutePoseEstimate inliers={kept}/{COUNT}>"


def test_threads_estimate_concurrently() -> None:
    _, x1, x2 = scene()
    reference = estimate_essential(x1, x2, K)
    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(lambda _: estimate_essential(x1, x2, K), range(8)))
    for result in results:
        np.testing.assert_array_equal(result.matrix, reference.matrix)


def test_too_few_correspondences() -> None:
    _, x1, x2 = scene()
    with pytest.raises(EstimationError):
        estimate_fundamental(x1[:3], x2[:3])
    with pytest.raises(EstimationError):
        estimate_essential(x1[:0], x2[:0], K)


SKEWED = K.copy()
SKEWED[0, 1] = 1.0


@pytest.mark.parametrize(
    ("call", "error", "message"),
    [
        (lambda x1, x2: estimate_fundamental(x1[:, :1], x2), ValueError, "shape"),
        (lambda x1, x2: estimate_fundamental(x1, x2[1:]), ValueError, "differ in length"),
        (lambda x1, x2: estimate_homography(np.full_like(x1, np.nan), x2), ValueError, "finite"),
        (lambda x1, x2: estimate_essential(x1, x2, SKEWED), ValueError, "zero skew"),
        (lambda x1, x2: estimate_essential(x1, x2, K[:2]), ValueError, "shape"),
        (lambda x1, x2: estimate_essential(x1, x2, -K), ValueError, "zero skew"),
        (lambda x1, x2: estimate_fundamental(x1, x2, threshold=0), ValueError, "threshold"),
        (lambda x1, x2: estimate_fundamental(x1, x2, seed=-1), ValueError, "seed"),
        (lambda x1, x2: estimate_fundamental(x1, x2, seed=True), TypeError, "seed"),
        (lambda x1, x2: estimate_absolute_pose(x1, x2, K), ValueError, r"shape \(n, 3\)"),
    ],
)
def test_invalid_arguments(call: Any, error: type[Exception], message: str) -> None:
    _, x1, x2 = scene()
    with pytest.raises(error, match=message):
        call(x1, x2)


@pytest.mark.parametrize("kind", ["array", "object", "scalar"])
def test_complex_inputs_never_reach_native_estimation(
    monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    import openmvg_rs._geometry as geometry

    def unexpected_native(*args: Any) -> Any:
        pytest.fail("complex input reached native estimation")

    for name in (
        "_estimate_fundamental",
        "_estimate_homography",
        "_estimate_essential",
        "_estimate_absolute_pose",
    ):
        monkeypatch.setattr(geometry, name, unexpected_native)
    pixels = np.zeros((8, 2))
    world = np.zeros((8, 3))
    camera = np.eye(3)

    def complex_input(value: NDArray[np.float64]) -> Any:
        if kind == "scalar":
            return np.complex128(complex(1.0, float("nan")))
        if kind == "object":
            result = value.astype(object)
            result.flat[0] = np.complex128(complex(1.0, float("nan")))
            return result
        result = value.astype(np.complex128)
        result.flat[0] = complex(1.0, float("nan"))
        return result

    bad_pixels, bad_world, bad_camera = map(complex_input, (pixels, world, camera))
    calls = [
        lambda: estimate_fundamental(bad_pixels, pixels),
        lambda: estimate_homography(pixels, bad_pixels),
        lambda: estimate_essential(pixels, pixels, bad_camera),
        lambda: estimate_essential(pixels, pixels, camera, bad_camera),
        lambda: estimate_absolute_pose(bad_world, pixels, camera),
        lambda: estimate_absolute_pose(world, bad_pixels, camera),
        lambda: estimate_absolute_pose(world, pixels, bad_camera),
    ]
    for call in calls:
        with pytest.raises(TypeError, match="only real values"):
            call()


@pytest.mark.parametrize("kind", ["fundamental", "homography", "essential", "absolute_pose"])
def test_all_estimate_states_keep_arrays_readonly_after_pickle_and_deepcopy(kind: str) -> None:
    import openmvg_rs

    matrix = np.eye(3)
    vector = np.array([1.0, 0.0, 0.0])
    mask = np.array([True, False])
    for array in (matrix, vector, mask):
        array.flags.writeable = False
    if kind == "fundamental":
        estimate: Any = openmvg_rs.FundamentalEstimate(matrix, mask)
    elif kind == "homography":
        estimate = openmvg_rs.HomographyEstimate(matrix, mask)
    elif kind == "essential":
        estimate = openmvg_rs.EssentialEstimate(matrix, openmvg_rs.Pose(matrix, vector), mask)
    else:
        estimate = openmvg_rs.AbsolutePoseEstimate(openmvg_rs.Pose(matrix, vector), mask)
    for restored in (pickle.loads(pickle.dumps(estimate)), copy.deepcopy(estimate)):
        arrays = [getattr(restored, member.name) for member in dataclasses.fields(restored)]
        if hasattr(restored, "pose"):
            arrays = [a for a in arrays if isinstance(a, np.ndarray)]
            arrays += [restored.pose.rotation, restored.pose.translation]
        for array in arrays:
            assert not array.flags.writeable
            with pytest.raises(ValueError, match="read-only"):
                array.flat[0] = 0


def test_pose_checks_and_copies_its_input() -> None:
    import openmvg_rs

    rotation, translation = np.eye(3), np.array([1.0, 2.0, 3.0])
    pose = openmvg_rs.Pose(rotation, translation)
    rotation[0, 0] = 5.0
    assert pose.rotation[0, 0] == 1.0
    assert not pose.rotation.flags.writeable
    np.testing.assert_array_equal(pose.center, [-1.0, -2.0, -3.0])
    with pytest.raises(ValueError, match=r"shape \(3, 3\)"):
        openmvg_rs.Pose(np.eye(4), translation)
    with pytest.raises(ValueError, match="finite"):
        openmvg_rs.Pose(np.eye(3), [0.0, np.nan, 0.0])
    with pytest.raises(TypeError, match="only real values"):
        openmvg_rs.Pose(np.eye(3) * 1j, translation)
