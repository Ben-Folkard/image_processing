"""
test_image_processing.py

contains unit tests for the image_processing code

ella beck
24/06/2025
"""

import os
import cv2
import pytest
import requests
import numpy as np
# import time
import image_processing as ip  # _new
import imageio.v3 as iio
# import image_processing_old as ip_old
# ips = (ip_old, ip_new)

"""
def benchmark(func, args, repeat=5):
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        func(*args)
        times.append(time.perf_counter() - start)
    return float(np.mean(times))
"""


class FakeResponse:
    "provides a fake status code"

    def __init__(self, status_code, content=b""):
        self.status_code = status_code
        self.content = content


class FakeVideo:
    "fakes a video which can pass through the image processing functions"

    def __init__(self, frames, start=0, fail_after=None):
        self.frames = frames
        self.idx = start
        self.total_frames = len(frames)
        self.fail_after = fail_after if fail_after is not None else self.total_frames

    def isOpened(self):
        return self.idx < self.fail_after and self.idx < self.total_frames

    def read(self):
        if not self.isOpened():
            return False, None
        frame = self.frames[self.idx]
        self.idx += 1
        return True, frame

    def set(self, prop, val):
        if prop == cv2.CAP_PROP_POS_FRAMES:
            self.idx = val

    def release(self):
        pass

    def get(self, prop_id):
        if prop_id == cv2.CAP_PROP_FRAME_COUNT:
            return self.total_frames
        elif prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return self.frames[0].shape[0]
        elif prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            # Handle both grayscale (2D) and color (3D)
            return self.frames[0].shape[1]
        else:
            return 0.0


class DummySettings:
    consecutive_threshold = 2
    brightness_threshold = 170
    flow_threshold = 5
    static_threshold = 50
    min_cluster_size = 2
    max_cluster_size = 10
    min_circularity = 0.3
    sliding_window_radius = 3
    number_of_plots = 20
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


@pytest.fixture(autouse=True)
def patch_requests_get(monkeypatch):
    # default response is success with dummy content
    monkeypatch.setattr(requests, 'get', lambda url: FakeResponse(200, b""))


@pytest.fixture
def temp_file(tmp_path):
    return tmp_path / "test_video.mp4"


def test_download_success(tmp_path, monkeypatch, capsys):
    # patch requests.get to return success and video frames
    fake_response = FakeResponse(200, b"video-content")
    monkeypatch.setattr(requests, 'get', lambda url: fake_response)

    filename = tmp_path / "out.mp4"

    result = ip.download_video_from_url("http://example.com/video.mp4",
                                        str(filename))

    # check return value
    assert result == str(filename)
    # file created with correct content
    with open(filename, 'rb') as f:
        assert f.read() == b"video-content"
    captured = capsys.readouterr()
    assert f"Downloaded video as {filename}" in captured.out


def test_download_failure(tmp_path, monkeypatch, capsys):
    fake_error = FakeResponse(404)
    monkeypatch.setattr(requests, 'get', lambda url: fake_error)

    filename = tmp_path / "out.mp4"
    result = ip.download_video_from_url("http://example.com/video.mp4",
                                        str(filename))

    assert result == str(filename)
    # file value does not exist
    assert not os.path.exists(str(filename))
    captured = capsys.readouterr()
    assert "Failed to download video" in captured.out


def test_load_all_frames(monkeypatch, tmp_path):
    # create dummy colour/greyscale frames
    dummy_frame = np.full((10, 10, 3), 128, dtype=np.uint8)
    dummy_frames = [dummy_frame for _ in range(10)]

    monkeypatch.setattr(iio, 'imiter', lambda fn:  # Need to fix
                        FakeVideo(dummy_frames))

    # should convert image to 2d grayscale
    frames = ip.load_video_frames("dummy.mp4")
    assert all(frame.ndim == 2 for frame in frames)
    assert len(frames) == len(dummy_frames)


def test_load_selected_frames(monkeypatch):
    "tests that load_video_frames can load in user defined ranges"
    # create dummy frames
    dummy_frame = np.full((10, 10, 3), 128, dtype=np.uint8)
    dummy_frames = [dummy_frame for _ in range(10)]

    def make_cap(funtion):
        return FakeVideo(dummy_frames, fail_after=10)

    monkeypatch.setattr(cv2, 'VideoCapture', make_cap)

    # load frames 2, 3, 4
    frames = ip.load_video_frames('dummy_frames', frames_start=2,
                                  frames_end=5)

    assert len(frames) == 3


def test_get_video_frames_from_url(monkeypatch):
    called = {}

    def fake_download(url, filename):
        called['dl'] = (url, filename)
        return filename

    def fake_load(filename, start, end):
        called['ld'] = (filename, start, end)
        return ['frame1', 'frame2']

    monkeypatch.setattr(ip, 'download_video_from_url', fake_download)
    monkeypatch.setattr(ip, 'load_video_frames', fake_load)

    result = ip.get_video_frames_from_url(
        'http://example.com/video',
        local_filename='my.avi',
        frames_start=1,
        frames_end=3
    )

    assert result == ['frame1', 'frame2']
    assert called['dl'] == ('http://example.com/video', 'my.avi')
    assert called['ld'] == ('my.avi', 1, 3)


def test_prepare_settings_default():
    # test that default settings are loaded correctly
    Settings = ip._prepare_settings(None)

    assert Settings.consecutive_threshold == 2
    assert Settings.brightness_threshold == 170
    assert Settings.sliding_window_radius == 3
    assert Settings.number_of_plots == 20


def test_prepare_settings_override():
    params = {
        'brightness_threshold': 200,
        'min_cluster_size': 10,
    }

    Settings = ip._prepare_settings(params)
    assert Settings.brightness_threshold == 200
    assert Settings.min_cluster_size == 10
    assert Settings.consecutive_threshold == 2


"""
def test_find_background_basic():
    # create 3 frames of shape (2, 2)

    # brightness values:
    # f0 = [[1, 5], [5, 1]]
    # f1 = [[2, 6], [6, 2]]
    # f2 = [[200, 200], [200, 200]]

    frames = np.array([
        [[1, 5], [5, 1]],
        [[2, 6], [6, 2]],
        [[200, 200], [200, 200]],
        ], dtype=float)

    # check that the bright frame is properly excluded
    bg = ip._find_background(frames)
    expected00 = (1+2) / 2
    assert pytest.approx(bg[0, 0]) == expected00

    expected01 = (5+6) / 2
    assert pytest.approx(bg[0, 1]) == expected01


def test_find_background_bright():
    # tests the implementation of find_background when all frames must be
    # excluded

    frames = np.full((3, 2, 2), 255)
    bg = ip._find_background(frames)

    assert np.all(bg == 255)
"""


"""
# Old
def test_compute_background_calls(monkeypatch):
    # tests that the correct neighbour frames are sliced before calling
    # find_background

    frames = [np.full((4, 4), i, dtype=float) for i in range(5)]
    captures = {}

    def fake_find(frame_stack):
        captures['arg'] = frame_stack
        return np.zeros((4, 4))

    monkeypatch.setattr(ip, '_find_background', fake_find)

    bg = ip.compute_background(frames, index=2, radius=1)
    assert 'arg' in captures
    # stack should have frames[1] and frames[3]
    assert captures['arg'].shape == (2, 4, 4)
    assert (captures['arg'][0] == frames[1]).all()
    assert (captures['arg'][1] == frames[3]).all()
    assert np.all(bg == 0)
    print('tests passed')
"""


def test_compute_background():

    frames = np.array([
        [[1, 5], [5, 1]],
        [[2, 6], [6, 2]],
        [[200, 200], [200, 200]],
        [[7, 5], [3, 8]],
        [[3, 1], [4, 5]],
        ], dtype=float)

    bg = ip.compute_background(frames, radius=1)

    expected_output = np.array([
                                [
                                 [1.5, 5.5],
                                 [5.5, 1.5],
                                ],

                                [
                                 [1.5, 5.5],
                                 [5.5, 1.5],
                                ],

                                [
                                 [2., 5.5],
                                 [4.5, 2.],
                                ],

                                [
                                 [70., 68.666664],
                                 [69., 71.]],

                                [
                                 [5., 3.],
                                 [3., 6.5],
                                ]
                               ], dtype=np.float32)

    assert np.allclose(bg, expected_output)


def test_raw_damaged_mask(monkeypatch):
    # set up dummy video
    frame = np.array([[10, 20], [30, 40]], dtype=float)
    background = np.array([[0, 0], [0, 0]], dtype=float)

    fake_mask = np.array([[1, 0], [0, 1]], dtype=int)
    monkeypatch.setattr(ip, 'get_damaged_pixel_mask', lambda frame, background: (fake_mask, None))

    mask = ip._raw_damaged_mask(frame, background)
    assert mask.dtype == bool

    expected = np.array([[True, False], [False, True]])
    assert np.array_equal(mask, expected)


def test_compute_persistent_mask_simple():
    mask1 = np.array([[True, False], [True, True]])
    mask2 = np.array([[True, True], [False, True]])

    persistent = ip.compute_persistent_mask([mask1, mask2],
                                            consecutive_threshold=2)
    expected = np.array([[True, False], [False, True]])

    assert persistent.dtype == bool
    assert np.array_equal(persistent, expected)


def test_compute_persistent_mask_with_reset():
    mask1 = np.array([[True, True], [False, True]])
    # mask2 = None
    mask2 = np.array([[False, False], [False, False]])
    mask3 = np.array([[True, True], [True, False]])

    persistent = ip.compute_persistent_mask([mask1, mask2, mask3],
                                            consecutive_threshold=2)
    expected = np.array([[False, False], [False, False]])

    assert np.array_equal(persistent, expected)


def test_filter_consecutive_pixels_basic():
    masks = [
        np.array([[True, False], [True, True]]),
        np.array([[True, True], [False, True]])
    ]

    persistent = np.array([[True, False], [False, True]])

    filtered, counts = ip.filter_consecutive_pixels(masks, persistent, return_counts=True)

    # assert isinstance(filtered, list) and isinstance(counts, list)
    assert filtered[0].shape == masks[0].shape
    assert counts[0] == 1
    assert counts[1] == 1


"""
def test_filter_consecutive_pixels_with_none():
    masks = [None, np.array([[True]])]
    persistent = np.array([[False]])

    filtered, counts = ip.filter_consecutive_pixels(masks, persistent, return_counts=True)

    assert filtered[0] is None
    assert np.isnan(counts[0])
    assert filtered[1].dtype == bool
    assert counts[1] == 1
"""


def test_compute_cluster_stats_flow_skips(monkeypatch):
    frames = [np.zeros((2, 2)) for _ in range(3)]
    masks = [np.ones((2, 2), bool) for _ in range(3)]

    flows = [0.0, 6.0, 0.0]

    monkeypatch.setattr(ip, 'filter_damaged_pixel_clusters',
                        lambda frame, mask, min_cluster_size, max_cluster_size, min_circularity, kernel: (None, 42, 7, 123))

    cluster_count, cluster_size, cluster_brightness = \
        ip.compute_cluster_stats(frames, masks, flows, DummySettings)

    # check that the second frame was skipped
    assert cluster_count.tolist() == [42, 0, 42]
    assert cluster_size.tolist() == [7, 0, 7]
    assert cluster_brightness.tolist() == [123, 0, 123]


def test_compute_cluster_stats_mask_skips(monkeypatch):
    frames = [np.zeros((2, 2)) for _ in range(2)]
    masks = [np.zeros((2, 2), bool), np.ones((2, 2), bool)]
    flows = [0.0, 0.0]

    monkeypatch.setattr(ip, 'filter_damaged_pixel_clusters',
                        lambda frame, mask, min_cluster_size, max_cluster_size, min_circularity, kernel: (None, 5, 2, 50))

    cluster_count, cluster_size, cluster_brightness = \
        ip.compute_cluster_stats(frames, masks, flows, DummySettings)

    assert cluster_count.tolist() == [0, 5]
    assert cluster_size.tolist() == [0, 2]
    assert cluster_brightness.tolist() == [0, 50]


def test_get_final_count_basic():
    masks = [np.array([[1, 1], [0, 1]], bool), np.zeros((2, 2), bool)]
    estimates = [10, np.nan]
    out = ip._get_final_count(masks, estimates)
    print(out)

    assert pytest.approx(out[0]) == 13.0
    assert pytest.approx(out[1]) == 0
    # assert np.isnan(out[1])


def test_generate_plot_calls(monkeypatch):
    frames = np.zeros((4, 1, 1))
    masks = np.ones((4, 1, 1), bool)
    counts = np.array([1, 2, 3, 4])
    flows = np.array([0.0, 1.0, 2.0, 3.0])
    settings = DummySettings()
    settings.flow_threshold = 2.0
    settings.number_of_plots = 2

    called = {'visualisation': [], 'plot_counts': 0, 'heatmap_input': None}
    monkeypatch.setattr(ip, 'visualise_damaged_pixels',
                        lambda frame, i, mask, count, show_plots, save_plots, output_folder: called['visualisation'].append(i))
    monkeypatch.setattr(ip, 'plot_damaged_pixels',
                        lambda counts, show_plots, save_plots, output_folder:
                            called.__setitem__('plot_counts', called['plot_counts']+1))
    monkeypatch.setattr(ip, 'find_damaged_pixel_heatmap',
                        lambda frame, mask, brightness_threshold: np.array([[9]]))
    monkeypatch.setattr(ip, 'plot_heatmap',
                        lambda heatmap, show_plots, save_plots, output_folder: called.__setitem__('heatmap_input', heatmap))

    ip._generate_plots(frames, masks, counts, flows, settings)

    # only frames [0, 1] should be plotted
    assert called['visualisation'] == [0, 1]
    assert called['plot_counts'] == 1
    assert np.array_equal(called['heatmap_input'], np.array([[9]]))


def test_compute_static_mask_simple():
    # generate 3 2x2 frames
    frames = np.array([[[10, 200], [200, 10]],
                      [[10, 200], [200, 10]],
                      [[10, 50], [50, 10]]])

    masks = np.array([frames[0] > 170, frames[1] > 170, frames[2] > 170])
    out = ip.compute_static_mask(frames, masks, brightness_threshold=180,
                                 static_threshold=50, min_valid_frames=1)

    # pixel (0,1): masked twice in 3 valid frames; 67% > 50% so true
    expected = np.array([[False, True], [True, False]])
    assert np.array_equal(out, expected)


def test_apply_static_suppression_basic():
    mask1 = np.array([[True, False], [True, True]])
    # mask2 = None
    mask2 = np.zeros((2, 2), bool)
    masks = [mask1, mask2]

    persistent = np.array([[True, False], [False, False]])
    static_mask = np.array([[False, True], [False, False]])

    final_masks, counts = ip.apply_static_suppression(masks, persistent, static_mask, calc_counts=True)
    expected = mask1 & ~(persistent | static_mask)

    assert np.array_equal(expected, final_masks[0])
    assert counts[0] == int(expected.sum())
    # assert final_masks[1] is None
    # assert np.isnan(counts[1])
    assert np.array_equal(final_masks[1], np.zeros((2, 2), bool))
    assert counts[1] == 0


def test_detect_damaged_pixels_pipeline(monkeypatch):
    frames = [np.zeros((1, 1)), np.ones((1, 1))]

    class Settings:
        flow_threshold = 5.0
        sliding_window_radius = 1
        brightness_threshold = 128
        static_threshold = 50
        consecutive_threshold = 2
        max_cluster_size = 10
        number_of_plots = 1

    monkeypatch.setattr(ip, '_prepare_settings', lambda params: Settings)
    monkeypatch.setattr(ip, 'compute_optical_flow_metric', lambda fs:
                        [0.0, 10.0])

    # raw masks: first frame processed, second skipped
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

    # without plot
    results = ip.detect_damaged_pixels(frames, plot=False)

    assert isinstance(results, tuple) and len(results) == 4
    assert 'plot' not in called

    result2 = ip.detect_damaged_pixels(frames, plot=True)

    assert 'plot' in called
    total_counts, cluster_count, cluster_size, cluster_brightness = result2

    assert np.allclose(total_counts, [1.0, np.nan], equal_nan=True)
    assert np.array_equal(cluster_count, np.array([0.0, 0.0]))


def test_get_damaged_pixel_mask_simple():
    frame = np.zeros((3, 3), dtype=float)
    frame[1, 1] = 100
    background = np.zeros((3, 3), dtype=float)

    mask_uint8, thresholds = ip.get_damaged_pixel_mask(frame, background)

    assert thresholds.shape == (3, 3)
    assert np.all(thresholds == 30)

    expected_mask = np.zeros((3, 3), dtype=np.uint8)
    expected_mask[1, 1] = 1
    assert np.array_equal(mask_uint8, expected_mask)


def test_get_damaged_pixel_mask_background_scaling():
    frame = np.zeros((2, 2), dtype=float)
    frame[0, 0] = 200.0
    background = np.zeros((2, 2), dtype=float)
    background[0, 0] = 255.0

    mask_uint8, thresholds = ip.get_damaged_pixel_mask(frame, background)
    assert thresholds[0, 0] == pytest.approx(255.0)
    assert mask_uint8[0, 0] == 0


def test_filter_damaged_pixel_clusters():
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

    settings = DummySettings()
    settings.min_cluster_size = 2
    settings.max_cluster_size = 2
    settings.min_circularity = 0.0

    cleaned_mask, count, _, _ = ip.filter_damaged_pixel_clusters(
        frame,
        damaged_mask,
        settings.min_cluster_size,
        settings.max_cluster_size,
        settings.min_circularity,
        settings.kernel,
    )

    # two clusters survive => cleaned_mask has 4 True entries
    assert count == 1
    assert np.sum(cleaned_mask) == 2


def test_remove_bright_regions_negative():
    "frame with no bright regions"
    background = np.array([[50, 60], [70, 80]], dtype=float)

    bright_threshold = 100

    # add a single damaged pixel
    damaged_pixel_mask = np.zeros((2, 2), bool)
    damaged_pixel_mask[1, 1] = True

    cleaned_mask = ip.remove_bright_regions(background, bright_threshold,
                                            damaged_pixel_mask, 2)

    assert np.array_equal(cleaned_mask, damaged_pixel_mask)


def test_remove_bright_regions_positive():
    "frame with bright regions"
    background = np.array([[200, 200, 10],
                           [200, 200, 10],
                           [10, 10, 10]])

    bright_threshold = 100

    damaged_pixel_mask = np.zeros((3, 3), bool)
    cleaned_mask = ip.remove_bright_regions(background, bright_threshold,
                                            damaged_pixel_mask, 2)

    assert np.array_equal(damaged_pixel_mask, cleaned_mask)


def test_estimate_damaged_pixels_in_bright_areas():
    "nonzero damaged pixel density"
    frames = np.array([
                       [[10, 10], [200, 200]],
                       [[255, 255], [255, 255]],
                      ], dtype=np.float32)

    damaged_pixel_masks = np.array([
                                    [[True, False], [False, False]],
                                    [[False, False], [False, False]],
                                   ], dtype=np.bool_)

    output = ip.estimate_damaged_pixels_in_bright_areas(
                                                        frames,
                                                        damaged_pixel_masks,
                                                        brightness_threshold=100,
                                                        )

    assert output.shape == (2,)
    # density of frame 1 should be ((1 damaged / 1 valid) * 2 bright) = 2
    assert output[0] == pytest.approx(2.0)
    assert np.isnan(output[1])


def test_estimate_damaged_pixels_in_bright_areas_fail():
    "zero low brightness area"
    frames = [np.array([[200, 200], [200, 200]], float)]
    damaged_pixel_masks = [np.array([[False, True], [False, False]], float)]

    output = ip.estimate_damaged_pixels_in_bright_areas(
        frames,
        damaged_pixel_masks,
        brightness_threshold=100
    )

    assert output.shape == (1,)
    assert np.isnan(output[0])


def test_find_bright_area_estimates():
    frames = np.array([
                       [[150, 200], [50, 80]],
                       [[10, 20], [30, 40]],
                      ])
    # masks = [np.array([[False, False], [False, False]]), None]
    masks = np.zeros((2, 2, 2), bool)

    output = ip.find_bright_area_estimates(frames, masks,
                                           brightness_threshold=170)

    assert output.shape == (2,)

    # frame 1 has 1 bright pixel and zero damaged
    assert output[0] == pytest.approx(0)
    assert np.isnan(output[1])


def test_compute_optical_flow_metric_identical_frames():
    # initialise identical dark frames
    frame = np.zeros((8, 8), dtype=np.uint8)
    optical_flows = ip.compute_optical_flow_metric([frame, frame.copy(),
                                                   frame.copy()])

    assert optical_flows[0] == pytest.approx(0.0)
    assert np.allclose(optical_flows, [0.0, 0.0, 0.0])


def test_compute_optical_flow_metric_different_frames():
    frame0 = np.zeros((10, 10), dtype=float)
    frame1 = np.zeros_like(frame0)

    # add a single bright pixel in each frame
    frame0[3, 3] = 255
    frame1[7, 7] = 255

    optical_flows = ip.compute_optical_flow_metric([frame0, frame1])
    assert optical_flows.shape == (2,)
    assert optical_flows[0] == pytest.approx(0.0)
    assert optical_flows[1] > 0.0


def test_filter_frames_by_optical_flow():
    frames = [
        [np.array([[10, 20], [30, 40]])],
        [np.array([[11, 21], [31, 41]])],
        [np.array([[12, 22], [32, 42]])],
        [np.array([[13, 23], [33, 43]])],
    ]

    damaged_pixel_masks = [
        [np.array([[False, True], [False, False]])],
        [np.array([[False, True], [True, False]])],
        [np.array([[True, True], [False, True]])],
        [np.array([[True, True], [True, True]])],
    ]

    damaged_pixel_counts = np.array([1, 2, 3, 4])
    optical_flows = [0.0, 0.5, 3.0, 1.0]

    output_frames, output_counts, output_masks, output_flows = \
        ip.filter_frames_by_optical_flow(frames, damaged_pixel_counts,
                                         optical_flows, damaged_pixel_masks, 2)

    expected_frames = [
        [np.array([[10, 20], [30, 40]])],
        [np.array([[11, 21], [31, 41]])],
        [np.array([[13, 23], [33, 43]])],
    ]

    expected_masks = [
        [np.array([[False, True], [False, False]])],
        [np.array([[False, True], [True, False]])],
        [np.array([[True, True], [True, True]])],
    ]

    expected_counts = np.array([1, 2, 4])
    expected_flows = [0.0, 0.5, 1.0]

    assert np.array_equal(output_frames, expected_frames)
    assert np.array_equal(output_counts, expected_counts)
    assert np.array_equal(output_masks, expected_masks)
    assert np.array_equal(output_flows, expected_flows)


def test_find_damaged_pixel_heatmap():
    frames = [np.array([[50, 200], [60, 70]], dtype=np.uint8) for
              _ in range(20)]

    damaged_pixel_masks = [np.array([[True, False], [False, False]]) for
                           _ in range(20)]

    result = ip.find_damaged_pixel_heatmap(frames, damaged_pixel_masks,
                                           brightness_threshold=170)

    assert result[0, 0] == pytest.approx(100.0)
    assert result[0, 1] == 0.0
    assert result[1, 0] == 0.0
    assert result[1, 1] == 0.0
