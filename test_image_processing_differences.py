"""
test_image_processing.py

contains unit tests for the image_processing code

Ben Folkard
10/10/2025 & 13/10/2025-15/10/2025
"""
import image_processing as ip_new
import image_processing_old as ip_old
from test_image_processing import FakeVideo, DummySettings

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
    # "_find_background",
    "compute_static_mask",
    "load_video_frames",
    # "compute_background",
    "apply_static_suppression",
    "filter_frames_by_optical_flow",
    "find_damaged_pixel_heatmap",
    "compute_cluster_stats",
    "_get_final_count",
    "find_bright_area_estimates",
    "compute_optical_flow_metric",
    "get_damaged_pixel_mask",
    "filter_damaged_pixel_clusters",
    "detect_damaged_pixels",
    "filter_consecutive_pixels",
])
def test_equivalence_and_benchmark(func_name, sample_frames, benchmark, tmp_path, monkeypatch):
    func_old = getattr(ip_old, func_name)
    func_new = getattr(ip_new, func_name)

    has_func_run = False

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

            has_func_run = True

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

            has_func_run = True
        case "find_damaged_pixel_heatmap":
            frames = [np.array([[50, 200], [60, 70]], dtype=np.uint8) for _ in range(20)]
            damaged_pixel_masks = [np.array([[True, False], [False, False]]) for _ in range(20)]
            brightness_threshold = 170
            args = (
                frames,
                damaged_pixel_masks,
                brightness_threshold
            )
        case "compute_cluster_stats":
            # frames, masks, flows, settings
            frames = [np.zeros((2, 2)) for _ in range(3)]
            masks = [np.ones((2, 2), bool) for _ in range(3)]
            flows = [0.0, 6.0, 0.0]

            args = (frames, masks, flows, DummySettings)

            monkeypatch.setattr(ip_old, 'filter_damaged_pixel_clusters',
                                lambda frame, mask, count, size, brightness:
                                (None, 42, 7, 123))
            cluster_count_old, cluster_size_old, cluster_brightness_old = func_old(*args)

            monkeypatch.setattr(ip_new, 'filter_damaged_pixel_clusters',
                                lambda frame, mask, count, size, brightness:
                                (None, 42, 7, 123))
            cluster_count_new, cluster_size_new, cluster_brightness_new = func_new(*args)

            assert np.array_equal(cluster_count_old, cluster_count_new)
            assert np.array_equal(cluster_size_old, cluster_size_new)
            assert np.array_equal(cluster_brightness_old, cluster_brightness_new)

            has_func_run = True
        case "_get_final_count":
            masks = [np.array([[1, 1], [0, 1]], bool), None]
            estimates = [10, np.nan]

            args = (masks, estimates)
        case "find_bright_area_estimates":
            frames = [np.array([[150, 200], [50, 80]]), np.array([[10, 20], [30, 40]])]
            masks = [np.array([[False, False], [False, False]]), None]
            brightness_threshold = 170

            args = (frames, masks, brightness_threshold)
        case "compute_optical_flow_metric":
            frame0 = np.zeros((10, 10), dtype=float)
            frame1 = np.zeros_like(frame0)

            # add a single bright pixel in each frame
            frame0[3, 3] = 255
            frame1[7, 7] = 255

            args = ([frame0, frame1],)
        case "get_damaged_pixel_mask":
            frame = np.zeros((2, 2), dtype=float)
            frame[0, 0] = 200.0
            background = np.zeros((2, 2), dtype=float)
            background[0, 0] = 255.0
            args = (frame, 2, 2, background)

            mask_uint8_old, thresholds_old = func_old(*args)
            mask_uint8_new, thresholds_new = func_new(*args)

            assert np.array_equal(mask_uint8_old, mask_uint8_new)
            assert np.array_equal(thresholds_old, thresholds_new)

            has_func_run = True
        case "detect_damaged_pixels":
            frames = [np.zeros((1, 1)), np.ones((1, 1))]
            plot = False

            class Settings:
                flow_threshold = 5.0
                sliding_window_radius = 1
                brightness_threshold = 128
                static_threshold = 50
                consecutive_threshold = 2
                max_cluster_size = 10
                number_of_plots = 1

            def set_monkeypatches(monkeypatch, ip, version="old"):
                monkeypatch.setattr(ip, '_prepare_settings', lambda params: Settings)
                monkeypatch.setattr(ip, 'compute_optical_flow_metric', lambda fs:
                                    [0.0, 10.0])

                # raw masks: first frame processed, second skipped
                if version == "old":
                    monkeypatch.setattr(ip, 'compute_background', lambda frames, i, radius:
                                        np.zeros((1, 1)))
                else:
                    monkeypatch.setattr(ip, 'compute_background', lambda frames, radius:
                                        np.array([np.zeros((1, 1)), np.zeros((1, 1))]))
                monkeypatch.setattr(ip, '_raw_damaged_mask', lambda frames, brightness:
                                    np.array([[True]]))
                monkeypatch.setattr(ip, 'remove_bright_regions', lambda background,
                                    bright_threshold, raw_mask, max_cluster_size:
                                    np.array([[False]]))

                monkeypatch.setattr(ip, 'compute_persistent_mask', lambda masks,
                                    consecutive_threshold: np.array([[False]]))
                monkeypatch.setattr(ip, 'filter_consecutive_pixels', lambda masks,
                                    persistent: ([np.array([[False]]), None], [0, np.nan]))
                monkeypatch.setattr(ip, 'compute_static_mask', lambda frames, background,
                                    brightness_threshold, static_threshold,
                                    min_valid_frames: np.array([[False]]))
                monkeypatch.setattr(ip, 'apply_static_suppression', lambda masks,
                                    persistent, static_mask: ([np.array([[False]]), None],
                                                              np.array([0.0, np.nan])))
                monkeypatch.setattr(ip, 'find_bright_area_estimates', lambda frames, mask,
                                    brightness_threshold: [1.0, np.nan])
                monkeypatch.setattr(ip, '_get_final_count', lambda frames,
                                    bright_estimates: np.array([1.0, np.nan]))
                monkeypatch.setattr(ip, 'compute_cluster_stats', lambda frames, masks,
                                    flows, settings: (np.array([0.0, 0.0]),
                                                      np.array([0.0, 0.0]),
                                                      np.array([0.0, 0.0])))

                called = {}
                monkeypatch.setattr(ip, '_generate_plots', lambda *args, **kwargs:
                                    called.setdefault('plot', True))

                return monkeypatch

            monkeypatch = set_monkeypatches(monkeypatch, ip_old, version="old")
            monkeypatch = set_monkeypatches(monkeypatch, ip_new, version="new")

            args = (frames, plot)
        case "filter_consecutive_pixels":
            masks = [
                np.array([[True, False], [True, True]]),
                np.array([[True, True], [False, True]])
            ]
            persistent = np.array([[True, False], [False, True]])
            args = (masks, persistent)
            filtered_old, counts_old = func_old(*args)
            filtered_new, counts_new = func_old(*args)
            assert np.array_equal(filtered_old, filtered_new)
            assert np.array_equal(counts_old, counts_new)

            masks = [None, np.array([[True]])]
            persistent = np.array([[False]])
            args = (masks, persistent)
            filtered_old, counts_old = func_old(*args)
            filtered_new, counts_new = func_old(*args)
            for old, new in zip(filtered_old, filtered_new):
                if old is None or new is None:
                    assert old is new
                else:
                    assert np.array_equal(old, new)
            assert np.allclose(counts_old, counts_new, equal_nan=True)

            has_func_run = True
        case "filter_damaged_pixel_clusters":
            # clusters separated sufficiently to not be joined by closing
            frame = np.arange(16).reshape((4, 4)).astype(float)
            damaged_mask = np.zeros((4, 4), dtype=np.uint8)

            # cluster A at (0,0),(0,1)
            damaged_mask[0, 0] = 1
            damaged_mask[0, 1] = 1

            # cluster B at (3,2),(3,3)
            damaged_mask[3, 2] = 1
            damaged_mask[3, 3] = 1
            damaged_mask[2, 3] = 1

            min_cluster_size = 2
            max_cluster_size = 2
            min_circularity = 0.0

            args = (
                frame,
                damaged_mask,
                min_cluster_size,
                max_cluster_size,
                min_circularity
            )

            cleaned_mask_old, count_old, _, _ = func_old(*args)
            cleaned_mask_new, count_new, _, _ = func_new(*args)

            assert np.array_equal(cleaned_mask_old, cleaned_mask_new)
            assert np.array_equal(count_old, count_new)

            has_func_run = True
        case _:
            pytest.skip(f"No test data defined for {func_name}")

    if not has_func_run:
        result_old = func_old(*args)
        result_new = func_new(*args)
        if func_name == "compute_background":
            print(f"compute_background:\n{result_old = }\n{result_new = }")
        assert np.allclose(result_old, result_new, atol=1e-5, rtol=1e-4, equal_nan=True), f"{func_name} differs"
    t_old = time_func(func_old, args)
    t_new = time_func(func_new, args)

    speedup = t_old / t_new if t_new > 0 else np.nan
    print(f"\n{func_name}: old={t_old:.6f}s, new={t_new:.6f}s, speedup={speedup:.2f}x faster")

    # feed one result into benchmark so pytest-benchmark still logs timing
    benchmark(lambda: func_new(*args))
