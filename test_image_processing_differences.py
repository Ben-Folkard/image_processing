"""
test_image_processing.py

contains unit tests for the image_processing code

Ben Folkard
10/10/2025 & 13/10/2025
"""

import image_processing as ip_new
import image_prcoessing_old as ip_old
# import test_image_processing as tip

import pytest
import numpy as np


@pytest.fixture
def sample_frames():
    # Simulate 10 frames of size 128x128
    rng = np.random.default_rng(0)
    return rng.random((10, 128, 128)).astype(np.float32)


@pytest.mark.parametrize("func_name", [
    "_find_background",
    "compute_static_mask",
    "detect_features",
])
def test_equivalence_and_benchmark(func_name, sample_frames, benchmark):
    """
    Compare old vs new implementations for correctness and speed.
    """

    func_old = getattr(ip_old, func_name)
    func_new = getattr(ip_new, func_name)

    result_old = func_old(sample_frames)
    result_new = func_new(sample_frames)

    assert np.allclose(
        result_old, result_new, atol=1e-6
    ), f"{func_name} results differ!"

    time_old = benchmark.pedantic(lambda: func_old(sample_frames), iterations=3, rounds=5)
    time_new = benchmark.pedantic(lambda: func_new(sample_frames), iterations=3, rounds=5)

    speedup = time_old / time_new if time_new > 0 else np.nan
    print(f"\n{'-'*70}")
    print(f"{func_name}")
    print(f"Old: {time_old:.6f}s | New: {time_new:.6f}s | Speedup: {speedup:.2f}x faster")
    print(f"{'-'*70}\n")

    # assert time_new <= time_old * 1.05, f"{func_name} got slower!"  # allow 5% tolerance
