"""
test_image_processing.py

contains unit tests for the image_processing code

Ben Folkard
10/10/2025 & 13/10/2025-15/10/2025
"""
import image_processing as ip_new
import image_processing_old as ip_old
from test_image_processing import FakeVideo

import cv2
import pytest
import time
import numpy as np


@pytest.fixture
def sample_frames():
    # Simulate 10 frames of size 128x128
    rng = np.random.default_rng(0)
    return rng.random((10, 128, 128)).astype(np.float32)


def time_func(func, args, repeat=5):
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        func(*args)
        times.append(time.perf_counter() - start)
    return float(np.mean(times))


@pytest.mark.parametrize("func_name", [
    "_find_background",
    "compute_static_mask",
    "load_video_frames",
    "compute_background",
    "apply_static_suppression",
    "filter_frames_by_optical_flow",
    "find_damaged_pixel_heatmap",
])
def test_equivalence_and_benchmark(func_name, sample_frames, benchmark, tmp_path, monkeypatch):
    func_old = getattr(ip_old, func_name)
    func_new = getattr(ip_new, func_name)

    have_funcs_run = False

    # Preparing arguments for each function
    match func_name:
        case "_find_background":
            args = (sample_frames,)
        case "compute_static_mask":
            masks = sample_frames > 0.5  # dummy boolean masks
            brightness_threshold = 0.5,
            static_threshold = 3,
            min_valid_frames = 5

            args = (
                sample_frames,
                masks,
                brightness_threshold,
                static_threshold,
                min_valid_frames,
            )
        case "load_video_frames":
            args = (tmp_path / "out.mp4",)
            dummy_frame = np.full((10, 10, 3), 128, dtype=np.uint8)
            dummy_frames = [dummy_frame for _ in range(10)]
            monkeypatch.setattr(cv2, "VideoCapture", lambda fn: FakeVideo(dummy_frames))

        case "compute_background":
            index = 2
            radius = 1
            args = (sample_frames, index, radius)
        case "apply_static_suppression":
            masks = [np.array([[True, False], [True, True]]), None]
            persistent = np.array([[True, False], [False, False]])
            static_mask = np.array([[False, True], [False, False]])

            args = (
                masks,
                persistent,
                static_mask,
            )

            final_masks_old, counts_old = func_old(*args)
            final_masks_new, counts_new = func_new(*args)

            assert len(final_masks_old) == len(final_masks_new)
            for m_old, m_new in zip(final_masks_old, final_masks_new):
                if m_old is None or m_new is None:
                    assert m_old is m_new
                else:
                    assert np.array_equal(m_old, m_new)

            assert np.allclose(counts_old, counts_new, atol=1e-6, equal_nan=True)

            have_funcs_run = True

        case "filter_frames_by_optical_flow":
            frames = np.array([
                [[[10, 20], [30, 40]]],
                [[[11, 21], [31, 41]]],
                [[[12, 22], [32, 42]]],
                [[[13, 23], [33, 43]]],
            ], dtype=np.uint8)
            pixel_counts = np.array([1, 2, 3, 4])
            optical_flows = np.array([0.0, 0.5, 3.0, 1.0])
            damaged_pixel_masks = np.array([
                [[[False, True], [False, False]]],
                [[[False, True], [True, False]]],
                [[[True, True], [False, True]]],
                [[[True, True], [True, True]]],
            ], dtype=np.bool_)
            threshold = 2

            args = (
                frames,
                pixel_counts,
                optical_flows,
                damaged_pixel_masks,
                threshold,
            )

            output_frames_old, output_counts_old, output_masks_old, \
                output_flows_old = func_old(*args)
            output_frames_new, output_counts_new, output_masks_new, \
                output_flows_new = func_new(*args)

            assert np.array_equal(output_frames_old, output_frames_new)
            assert np.array_equal(output_counts_old, output_counts_new)
            assert np.array_equal(output_masks_old, output_masks_new)
            assert np.array_equal(output_flows_old, output_flows_new)

            have_funcs_run = True
        case "find_damaged_pixel_heatmap":
            frames = [np.array([[50, 200], [60, 70]], dtype=np.uint8) for _ in range(20)]
            damaged_pixel_masks = [np.array([[True, False], [False, False]]) for _ in range(20)]
            brightness_threshold = 170
            args = (
                frames,
                damaged_pixel_masks,
                brightness_threshold
            )
        case _:
            pytest.skip(f"No test data defined for {func_name}")

    if not have_funcs_run:
        result_old = func_old(*args)
        result_new = func_new(*args)
        assert np.allclose(result_old, result_new, atol=1e-6), f"{func_name} differs"

    t_old = time_func(func_old, args)
    t_new = time_func(func_new, args)

    speedup = t_old / t_new if t_new > 0 else np.nan
    print(f"\n{func_name}: old={t_old:.6f}s, new={t_new:.6f}s, speedup={speedup:.2f}x faster")

    # feed one result into benchmark so pytest-benchmark still logs timing
    benchmark(lambda: func_new(*args))
