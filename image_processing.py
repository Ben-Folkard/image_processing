"""
image_processing_optimisation.py

detects gamma radiation damaged pixels from camera footage
for use on scarf

ella beck
18/06/25

Ben Folkard
07/10/25-09/10/25
* my comments
I should go through here and time each thing to see where the bottlenecks are
"""


# importing libraries
import cv2
import imageio.v3 as iio
import os
import numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor
# from joblib import Parallel, delayed
# from multiprocessing import cpu_count, get_context
# import gc
# import psutil

try:
    import matplotlib
    # Disables GUI rendering on SCARF
    if os.environ.get("DISPLAY", "") == "":
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None
    mpatches = None

DTYPE_IMAGE = np.uint8
DTYPE_COMPUTE = np.float32
DTYPE_MASK = np.bool_


def download_video_from_url(url, filename):
    """
    downloads video to be processed, requires url and filename as strings
    """

    response = requests.get(url)

    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded video as {filename}")

    else:
        print("Failed to download video")  # *No way of detecting based on output*

    return filename


"""
loads in video frames as greyscale arrays with brightness values ranging
from 0 to 255 can load in specific chunk of frames from given video
(given frame start and end values as integers)
requires video filename as string
"""
"""
def load_video_frames(filename, frames_start=None, frames_end=None, grayscale=True):
    cap = cv2.VideoCapture(filename)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {filename}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    start = frames_start or 0
    end = frames_end or total_frames
    n_frames = max(0, end - start)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = np.empty((n_frames, height, width), dtype=np.uint8)

    i = 0
    while i < n_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if grayscale and len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames[i] = frame
        i += 1

    cap.release()

    return frames[:i]
"""


"""
def load_video_frames(filename, frames_start=None, frames_end=None, grayscale=True):
    cap = cv2.VideoCapture(filename)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video file: {filename}")
        return np.empty((0, 0, 0), dtype=DTYPE_IMAGE)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start = frames_start or 0
    end = frames_end if frames_end is not None else total_frames
    start = max(0, min(start, total_frames))
    end = max(start, min(end, total_frames))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []

    for i in range(start, end):
        ret, frame = cap.read()
        if not ret:
            break
        if grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame)

    cap.release()

    if not frames:
        print(f"[WARNING] No frames loaded from {filename} between {start}-{end}")
        return np.empty((0, 0, 0), dtype=DTYPE_IMAGE)

    # Guarantee uniform shape (some codecs yield slightly uneven edges)
    frame_shapes = {f.shape for f in frames}
    if len(frame_shapes) > 1:
        h, w = frames[0].shape
        frames = [cv2.resize(f, (w, h)) for f in frames]

    return np.stack(frames).astype(DTYPE_IMAGE)
"""

"""
# If the gpu version of openCV is installed, the following will probably be faster:
def load_video_frames(filename, frames_start=None, frames_end=None):
    reader = cv2.cudacodec.createVideoReader(filename)
    frames = []
    while True:
        ret, gpu_frame = reader.nextFrame()
        if not ret:
            break
        gray = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray.download())
    return np.stack(frames)
"""


def load_video_frames(filename, frames_start=None, frames_end=None, grayscale=True):
    """
    Streams video frames lazily using imageio.v3.imiter().
    Only the requested range is read into memory.
    """
    frames = []

    for i, frame in enumerate(iio.imiter(filename)):
        if frames_start is not None and i < frames_start:
            continue
        if frames_end is not None and i >= frames_end:
            break
        if grayscale and frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame)

    return np.asarray(frames, dtype=DTYPE_IMAGE)


def get_video_frames_from_url(
        url,
        local_filename='temp_video.avi',
        frames_start=None,
        frames_end=None
        ):
    # takes video url and loads frames in directly

    download_video_from_url(url, local_filename)

    return load_video_frames(local_filename, frames_start, frames_end)

# helper functions


def _prepare_settings(params):
    """
    allow user to change default thresholds

    consecutive threshold adjusts how many frames a pixel is bright
        consecutively before being disregarded as damaged
    brightness_threshold should be cut off at the brightness point where tests
        start failing
    min and max cluster size adjusts how big a pixel cluster should be before
        being disregarded
    flow_threshold should be adjusted depending on how similar the frames are
        expected to be
    """
    defaults = {
        'consecutive_threshold': 2,
        'brightness_threshold': 170,
        'flow_threshold': 2.0,
        'static_threshold': 50,
        'min_cluster_size': 5,
        'max_cluster_size': 20,
        'min_circularity': 0.1,
        'sliding_window_radius': 3,
        'number_of_plots': 20,
        'kernel': cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    }
    if params:
        defaults.update(params)
    return type('S', (), defaults)


def compute_background(frames, radius, pixel_std_coeff=1.0):
    """
    estimate background brightness for each pixel of a given frame, based on
    the mean brightness of that pixel in the frames in a sliding window
    (providing the pixel is undamaged in those frames), implementing a
    rolling buffer
    """
    frames = frames.astype(DTYPE_COMPUTE)
    frames_sq = frames ** 2
    n, h, w = frames.shape
    window = 2 * radius + 1

    rolling_sum = np.zeros((h, w), DTYPE_COMPUTE)
    rolling_sq_sum = np.zeros((h, w), DTYPE_COMPUTE)
    backgrounds = np.zeros_like(frames, DTYPE_COMPUTE)

    # Pre-fill the first window manually
    for i in range(min(window, n)-1):
        rolling_sum += frames[i]
        rolling_sq_sum += frames_sq[i]

    for i in range(n):
        # Determine actual window boundaries
        start = max(0, i - radius)
        end = min(n, i + radius + 1)
        num_neighbours = end - start - 1

        # Updating rolling window:
        # (The ends move if required)
        if end < n:
            rolling_sum += frames[end]
            rolling_sq_sum += frames_sq[end]
        if start > 0:
            rolling_sum -= frames[start - 1]
            rolling_sq_sum -= frames_sq[start - 1]

        # (Only neighbours are computed)
        central_frame = frames[i]
        central_frame_sq = frames_sq[i]
        rolling_sum -= central_frame
        rolling_sq_sum -= central_frame_sq

        # Compute mean/std using current sums
        mean = rolling_sum / num_neighbours
        std = np.sqrt(np.maximum(rolling_sq_sum / num_neighbours - mean**2, 0.0))

        thr = mean + pixel_std_coeff * std
        window_frames = frames[start:end]
        masked = np.where(window_frames <= thr, window_frames, np.nan)
        bg = np.nanmean(masked, axis=0)
        backgrounds[i] = np.nan_to_num(bg, nan=mean)

        # Re-include the central frame to become a future neighbour
        rolling_sum += central_frame
        rolling_sq_sum += central_frame_sq
    return backgrounds


def _raw_damaged_mask(frame, background):
    """
    returns first pass mask of potentially damaged pixels
    """
    mask, _ = get_damaged_pixel_mask(frame, background)

    return mask.astype(bool)


def compute_persistent_mask(masks, consecutive_threshold):
    """
    flags pixels which have been flagged as damaged for multiple
    consecutive frames
    """
    height, width = masks[0].shape
    current_status = np.zeros((height, width), np.int32)
    longest_flag = np.zeros_like(current_status)

    for mask in masks:
        current_status[mask] += 1
        current_status[~mask] = 0

        longest_flag = np.maximum(longest_flag, current_status)

    return longest_flag >= consecutive_threshold


def filter_consecutive_pixels(masks, persistent, return_counts=False):
    """
    filters out pixels which have been flagged as damaged for multiple
    consecutive frames
    """
    filtered_masks = np.logical_and(masks, ~persistent)

    if return_counts:
        counts = filtered_masks.sum(axis=(1, 2)).astype(DTYPE_COMPUTE)

        return filtered_masks, counts

    return filtered_masks


"""
# Going to have to monitor, as shows significant slow down for small tests
def compute_cluster_stats(frames, masks, flows, settings, n_jobs=-1):
    def process_single(frame, mask, flow):
        if np.all(mask is False) or flow > settings.flow_threshold:
            return 0.0, 0.0, 0.0
        _, count, size, bright = filter_damaged_pixel_clusters(
            frame,
            mask,
            settings.min_cluster_size,
            settings.max_cluster_size,
            settings.min_circularity,
            settings.kernel,
        )
        return count, size, bright

    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(process_single)(frame, mask, flow)
        for frame, mask, flow in zip(frames, masks, flows)
    )

    # returning cluster_count, cluster_size, cluster_brightness
    return np.array(results).T
"""


def compute_cluster_stats(frames, masks, flows, settings):
    """
    computes the number of clusters for each frame, as well as the mean
    cluster size and brightness
    """
    n = len(frames)
    cluster_count = np.zeros(n, float)
    cluster_size = np.zeros(n, float)
    cluster_brightness = np.zeros(n, float)

    for i, (frame, mask) in enumerate(zip(frames, masks)):
        if (not mask.any()) or (flows[i] > settings.flow_threshold):
            continue
        clean_uint8 = mask.astype(np.uint8)
        _, count, avg_size, avg_brightness = filter_damaged_pixel_clusters(
            frame,
            clean_uint8,
            settings.min_cluster_size,
            settings.max_cluster_size,
            settings.min_circularity,
            settings.kernel,
        )

        cluster_count[i], cluster_size[i], cluster_brightness[i] = \
            count, avg_size, avg_brightness

    return cluster_count, cluster_size, cluster_brightness


def _get_final_count(masks, bright_estimates):
    """
    gets final damaged pixel count, adding in the bright area estimates to
    the raw dark area count
    """
    # counts = base noise + estimated radiation induced noise
    # Possible bug: kept in the original possible bug where
    # base still contains nans whilst estimate doesn't
    counts = np.array([
       np.sum(mask) +
       (estimate if not np.isnan(estimate) else 0)
       for mask, estimate in zip(masks, bright_estimates)
    ], dtype=DTYPE_COMPUTE)

    return counts


def _generate_plots(
        frames,
        masks,
        counts,
        flows,
        settings,
        show_plots=True,
        save_plots=False,
        output_folder="results",
        ):
    """
    generates plots for user
    """
    survivors = [i for i, flow in enumerate(flows) if flow <=
                 settings.flow_threshold]
    for i in survivors[:settings.number_of_plots]:
        visualise_damaged_pixels(
                                 frames[i],
                                 i,
                                 masks[i],
                                 counts[i],
                                 show_plots=show_plots,
                                 save_plots=save_plots,
                                 output_folder=output_folder,
                                )

    plot_damaged_pixels(
                        counts,
                        show_plots=show_plots,
                        save_plots=save_plots,
                        output_folder=output_folder,
                        )

    heatmap = find_damaged_pixel_heatmap(
        frames,
        masks.astype(DTYPE_IMAGE),
        settings.brightness_threshold,
    )
    plot_heatmap(
                 heatmap,
                 show_plots=show_plots,
                 save_plots=save_plots,
                 output_folder=output_folder,
                 )


def compute_static_mask(
        frames,
        masks,
        brightness_threshold,
        static_threshold,
        min_valid_frames
        ):
    """
    builds a percentage heatmap of damage frequency
    returns a boolean mask of pixels damaged for a
        percentage > static_threshold
    """
    frame_stack = np.asarray(frames, dtype=DTYPE_IMAGE)
    mask_stack = masks.astype(DTYPE_IMAGE)

    heatmap = mask_stack.sum(axis=0)
    bright = (frame_stack > brightness_threshold) & (~mask_stack.astype(bool))
    valid_counts = (~bright).sum(axis=0)

    with np.errstate(invalid='ignore'):
        percentage_map = np.where(
            valid_counts > min_valid_frames,
            (heatmap / valid_counts) * 100,
            0.0,
        )
    return percentage_map > static_threshold


def apply_static_suppression(masks, persistent, static_mask, calc_counts=False):
    bad = np.logical_or(persistent, static_mask)
    final_masks = np.logical_and(masks, ~bad)
    if calc_counts:
        counts = np.zeros(final_masks.shape[0], dtype=DTYPE_COMPUTE)
        for i, m in enumerate(final_masks):
            counts[i] = m.sum()
        return final_masks, counts
    else:
        return final_masks


def detect_damaged_pixels(
        frames,
        settings,
        plot=False,
        show_plots=False,
        save_plots=False,
        output_folder="results",
        ):
    """
    main code for detecting damaged pixels
    requires video frames as greyscale arrays of brightness values

    - converts frames to arrays
    - filters by optical flow
    - computes background
    - computes raw and cleaned damaged pixel masks
    - aggregates statistics
    - optional plotting
    """

    frames = np.asarray(frames)

    # optical flow screening
    optical_flows = compute_optical_flow_metric(frames)

    # find damaged pixel mask for each frame
    raw_masks = np.zeros_like(frames, dtype=DTYPE_MASK)

    backgrounds = compute_background(
                                     frames,
                                     radius=settings.sliding_window_radius,
    )

    for i, frame in enumerate(frames):
        if optical_flows[i] <= settings.flow_threshold:
            raw_mask = _raw_damaged_mask(frame, backgrounds[i])
            bright_filtered = remove_bright_regions(
                backgrounds[i],
                settings.brightness_threshold,
                raw_mask,
                settings.max_cluster_size
            )
            raw_masks[i] = bright_filtered

    # filter pixels marked as damaged for too many consecutive frames
    persistent_pixels = compute_persistent_mask(
        raw_masks,
        settings.consecutive_threshold
        )

    clean_masks = filter_consecutive_pixels(raw_masks, persistent_pixels)

    # initial heatmap calculation and static hotspot suppression
    static_mask = compute_static_mask(
        frames,
        clean_masks,
        settings.brightness_threshold,
        settings.static_threshold,
        10
    )

    final_masks = apply_static_suppression(
        clean_masks,
        persistent_pixels,
        static_mask
    )

    # find estimated number of damaged pixels in bright areas
    bright_area_estimates = find_bright_area_estimates(
        np.stack(frames, axis=0).astype(DTYPE_COMPUTE),
        final_masks,
        settings.brightness_threshold
        )

    total_counts = _get_final_count(final_masks, bright_area_estimates)

    # cluster_stats
    cluster_counts, avg_sizes, avg_brightnesses = compute_cluster_stats(
        frames,
        final_masks,
        optical_flows,
        settings
    )

    # create plots
    if plot:
        _generate_plots(
                        frames,
                        final_masks,
                        total_counts,
                        optical_flows,
                        settings,
                        show_plots=show_plots,
                        save_plots=save_plots,
                        output_folder=output_folder,
                        )

    return total_counts, cluster_counts, avg_sizes, avg_brightnesses


def get_damaged_pixel_mask(frame, background):
    # condition 1: pixel brightness should exceed background by a
    #   threshold scaled with background brightness
    threshold = np.maximum(30, 30 + (background / 255) * (255 - 30))
    damaged = frame > threshold

    # condition 2: pixel's brightness should exceed mean of its
    #   neighbours in a 30x30 kernel
    kernel_size = (30, 30)
    local_mean = cv2.blur(frame.astype(DTYPE_COMPUTE), kernel_size)

    damaged &= frame > local_mean

    return damaged.astype(DTYPE_IMAGE), threshold


def filter_damaged_pixel_clusters(
                                  frame,
                                  damaged_pixel_mask,
                                  min_cluster_size,
                                  max_cluster_size,
                                  min_circularity,
                                  kernel,
                                  circularity_size_threshold=10,
                                  ):
    """
    filters large groups of damaged pixels from the mask
    prevents bright noise such as reflections or glare being misidentified as
    damaged pixels
    """

    # close gaps (test)  # Dilation followed by erosion to get rid of noise by bluring
    closed_mask = cv2.morphologyEx(
                                   damaged_pixel_mask.astype(DTYPE_IMAGE),
                                   cv2.MORPH_CLOSE,
                                   kernel,
    )

    # isolate groups of damaged pixels
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed_mask, connectivity=8)

    # prepare outputs
    cleaned_mask = np.zeros_like(damaged_pixel_mask, dtype=DTYPE_MASK)
    areas = stats[1:, cv2.CC_STAT_AREA]

    valid_labels = np.flatnonzero((areas >= min_cluster_size) & (areas <= max_cluster_size)) + 1
    if len(valid_labels) == 0:
        return cleaned_mask, 0, 0.0, float('nan')

    # perimeters = np.zeros(num_labels, DTYPE_COMPUTE)
    circularities = np.zeros(num_labels, DTYPE_COMPUTE)

    # Only compute contours for labels above threshold
    large_labels = [label for label in valid_labels if stats[label, cv2.CC_STAT_AREA] >= circularity_size_threshold]
    if large_labels:
        mask_tmp = np.zeros_like(closed_mask, dtype=DTYPE_IMAGE)
        for label in large_labels:
            mask_tmp[:] = 0
            mask_tmp[labels == label] = 255
            contours, _ = cv2.findContours(mask_tmp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if contours:
                per = cv2.arcLength(contours[0], True)
                if per > 0:
                    # perimeters[label] = per
                    circularities[label] = 4 * np.pi * (stats[label, cv2.CC_STAT_AREA] / (per ** 2))

    kept_labels = []
    for label in valid_labels:
        if (
            stats[label, cv2.CC_STAT_AREA] >= circularity_size_threshold
            and circularities[label] < min_circularity
        ):
            continue
        kept_labels.append(label)
        cleaned_mask[labels == label] = True

    if not kept_labels:
        return cleaned_mask, 0, 0.0, float("nan")

    # Vectorized brightness aggregation
    # (converting labels to 32-bit to avoid memory blowup)
    flat_labels = labels.ravel()
    flat_frame = frame.ravel().astype(DTYPE_COMPUTE)
    label_vals = np.bincount(flat_labels, weights=flat_frame)
    label_areas = np.bincount(flat_labels)

    used = np.array(kept_labels)
    areas = label_areas[used]
    brightness_sums = label_vals[used]

    cluster_count = len(kept_labels)
    avg_cluster_size = float(np.mean(areas))
    avg_cluster_brightness = float(np.sum(brightness_sums) / np.sum(areas))

    return cleaned_mask, cluster_count, avg_cluster_size, avg_cluster_brightness


def remove_bright_regions(
                          background,
                          brightness_threshold,
                          filtered_damaged_pixels,
                          max_cluster_size
                          ):
    """
    removes damaged pixels from the mask if they exist in bright areas
    avoids inaccuracies due to the code's capabilities of operating
    in low contrast/bright background
    """

    bright_background_mask = (background > brightness_threshold).astype(
        DTYPE_IMAGE)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        bright_background_mask, connectivity=8)

    # create a mask for large bright regions
    remove = np.zeros_like(bright_background_mask, dtype=DTYPE_MASK)

    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= max_cluster_size:
            remove[labels == label] = True

    damaged_pixel_mask_uint8 = filtered_damaged_pixels.astype(DTYPE_IMAGE)
    num_labels2, labels2, stats2, _ = cv2.connectedComponentsWithStats(
        damaged_pixel_mask_uint8, connectivity=8)

    cleaned = np.zeros_like(filtered_damaged_pixels, dtype=DTYPE_MASK)
    for lbl in range(1, num_labels2):
        comp = (labels2 == lbl)
        if np.any(comp & remove):
            continue
        cleaned[comp] = True

    return cleaned


def estimate_damaged_pixels_in_bright_areas(
                                            frames,
                                            damaged_pixel_masks,
                                            brightness_threshold=170,
                                            ):
    """
    Estimates the number of damaged pixels present in bright areas or areas of low contrast,
    (where the code would otherwise fail to pick up damaged_pixels)
    """

    # identify low and high brightness regions, excluding existing
    #   damaged pixels
    frames = np.asarray(frames, dtype=DTYPE_COMPUTE)
    damaged_pixel_masks = np.asarray(damaged_pixel_masks, dtype=DTYPE_MASK)

    inv_damaged_pixel_masks = ~damaged_pixel_masks
    low_brightness_masks = np.logical_and((frames < brightness_threshold), inv_damaged_pixel_masks)
    low_brightness_areas = low_brightness_masks.sum(axis=(1, 2))
    high_brightness_masks = np.logical_and((frames >= brightness_threshold), inv_damaged_pixel_masks)
    high_brightness_areas = high_brightness_masks.sum(axis=(1, 2))

    # density of damaged pixels in low-brightness areas
    damaged_pixel_density = np.divide(
                                      damaged_pixel_masks.sum(axis=(1, 2)),
                                      low_brightness_areas,
                                      out=np.full_like(low_brightness_areas, np.nan, dtype=DTYPE_COMPUTE),
                                      where=low_brightness_areas != 0,
                                      )

    # estimate damaged pixels in high-brightness areas
    estimated_high_brightness_damaged_pixels = damaged_pixel_density * high_brightness_areas

    return estimated_high_brightness_damaged_pixels


def find_bright_area_estimates(
    frames,
    damaged_pixel_masks,
    brightness_threshold
):
    """
    finds estimated number of damaged pixels in bright areas using
        estimate_damaged_pixels_in_bright_areas()
    """

    bright_area_estimates = np.full(len(frames), np.nan, dtype=DTYPE_COMPUTE)

    estimate = estimate_damaged_pixels_in_bright_areas(frames, damaged_pixel_masks)

    for i, (frame, mask) in enumerate(zip(frames, damaged_pixel_masks)):
        high_brightness_mask = (frame > brightness_threshold) & ~mask

        if high_brightness_mask.sum() > 0:
            bright_area_estimates[i] = estimate[i]
        else:
            bright_area_estimates[i] = np.nan

    return bright_area_estimates


# There are performance gains to be made if the gpu version (CUDA version) of cv2 is installed
# Even though this is via tests is quite a lot slower, I think in reality this should be faster
def compute_optical_flow_metric(frames):
    optical_flows = np.zeros(len(frames))

    def flow_between(i):
        prev = frames[i - 1].astype(DTYPE_COMPUTE)
        curr = frames[i].astype(DTYPE_COMPUTE)
        flow = cv2.calcOpticalFlowFarneback(prev,
                                            curr,
                                            None,
                                            pyr_scale=0.5,
                                            levels=3,
                                            winsize=15,
                                            iterations=3,
                                            poly_n=5,
                                            poly_sigma=2,
                                            flags=0)
        return np.mean(cv2.cartToPolar(flow[..., 0], flow[..., 1])[0])

    with ThreadPoolExecutor() as ex:
        for i, val in enumerate(ex.map(flow_between, range(1, len(frames)))):
            optical_flows[i + 1] = val
    return optical_flows


def filter_frames_by_optical_flow(
        frames,
        pixel_counts,
        optical_flows,
        damaged_pixel_masks,
        threshold,
        removed_displayed=False):
    """
    filters out frames whose optical flow exceeds the given threshold.
    frames with optical flow above the threshold are removed entirely from the
    frames list, and their corresponding pixel counts and masks are discarded.
    """
    frames = np.asarray(frames)
    pixel_counts = np.asarray(pixel_counts)
    optical_flows = np.asarray(optical_flows)
    damaged_pixel_masks = np.asarray(damaged_pixel_masks)

    valid = optical_flows <= threshold
    filtered_frames = frames[valid]
    filtered_counts = pixel_counts[valid]
    filtered_masks = damaged_pixel_masks[valid]
    filtered_flows = optical_flows[valid]

    # *Added option to not print as it was just cluttering up and slowing down the output*
    if removed_displayed:
        removed_counter = len(frames)-len(filtered_frames)
        print(f"Removed {removed_counter} frames due to high optical flow")

    return filtered_frames, filtered_counts, filtered_masks, filtered_flows


def find_damaged_pixel_heatmap(
        frames,
        damaged_pixel_masks,
        brightness_threshold
        ):
    """
    produces heatmap of damaged pixel occurrences
    can be used to verify uniformity of damaged pixels (unless frames contain
        a lot of bright noise, which will be excluded on the heatmap)
    """
    MIN_VALID_FRAMES = 10

    frame_stack = np.asarray(frames)
    mask_stack = np.asarray(damaged_pixel_masks)

    heatmap = mask_stack.sum(axis=0)

    bright_stack = (frame_stack > brightness_threshold) & (~mask_stack)
    valid_counts = (~bright_stack).sum(axis=0)

    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(
            valid_counts > MIN_VALID_FRAMES,
            heatmap / valid_counts * 100.0,
            0.0,
        )
    return result


def visualise_damaged_pixels(
        frame,
        frame_index,
        cluster_mask,
        cluster_count,
        bright_threshold=170,
        show_plots=True,
        save_plots=False,
        output_folder="results",
        ):
    """
    plots two versions of a given frame side by side, the second frame
        highlighting detected damaged pixels

    plots detected damaged pixels in red
    plots bright areas (where the code has estimated the damaged pixel count)
        in green
    """

    if not HAS_MATPLOTLIB:
        print("matplotlib not available - skipping"
              "damaged pixel visualisation")

    else:
        bright_areas = frame > bright_threshold
        vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)  # Already did this in load_video_frames

        # overlay clusters in red
        cluster_overlay = np.zeros_like(vis)
        cluster_overlay[cluster_mask] = (0, 165, 255)
        vis = cv2.addWeighted(vis, 0.8, cluster_overlay, 1.0, 0)
        # *addWeighted(source1, alpha, source2, beta, gamma[, dst[, dtype]]) *
        # * Image= alpha * image1 + beta * image2 + γ, create an image by blending

        # overlay bright areas in green for reference
        bright_overlay = np.zeros_like(vis)
        bright_overlay[bright_areas] = (255, 0, 0)
        vis = cv2.addWeighted(vis, 0.8, bright_overlay, 1.0, 0)

        plt.figure(figsize=(14, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(frame, cmap='gray', vmin=0, vmax=255)
        plt.title(f"original frame {frame_index}")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        plt.title(f"clusters: {cluster_count}")
        plt.axis('off')

        damaged_pixel_patch = mpatches.Patch(color='orange',
                                             label='Damaged Pixels')
        bright_background_patch = mpatches.Patch(color='royalblue',
                                                 label='Bright Background '
                                                 'Areas')

        plt.legend(handles=[damaged_pixel_patch, bright_background_patch],
                   loc='upper left', fontsize='small', frameon=True)

        if show_plots:
            plt.show()

        if save_plots:
            plt.tight_layout()
            filename = os.path.join(output_folder, f"visualise_damaged_pixels_frame_{frame_index}.png")
            plt.savefig(filename, dpi=300)
            plt.close()


def plot_heatmap(
                 heatmap,
                 title="Damaged Pixel Heatmap",
                 show_plots=True,
                 save_plots=False,
                 output_folder="results",
                 ):
    """
    plots heatmap showing damaged pixel distribution over every frame
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available - skipping heatmap plot")

    else:

        plt.figure(figsize=(15, 10))
        plt.imshow(heatmap, cmap='viridis', interpolation='nearest')
        plt.colorbar(label="Percentage of frames (%)")
        plt.title(title)
        if show_plots:
            plt.show()
        if save_plots:
            plt.tight_layout()
            filename = os.path.join(output_folder, f"{title.lower().replace(' ', '_')}.png")
            plt.savefig(filename, dpi=300)
            plt.close()


def plot_damaged_pixels(
                        damaged_pixel_counts,
                        show_plots=True,
                        save_plots=False,
                        output_folder="results",
                        ):
    """
    plots the count of damaged pixels across frames
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available - skipping damaged pixel output graph")

    else:
        plt.figure(figsize=(10, 5))
        plt.plot(damaged_pixel_counts, label='Damaged Pixels Count',
                 color='blue')
        plt.xlabel('Frame Number')
        plt.ylabel('Number of Damaged Pixels')
        plt.title('Damaged Pixels Detected Over Time')
        plt.legend()

        if show_plots:
            plt.show()

        if save_plots:
            plt.tight_layout()
            filename = os.path.join(output_folder, "plot_damged_pixels.png")
            plt.savefig(filename, dpi=300)
            plt.close()


# (changed since last fully run)
def chunked_nanmean(array, step):
    if len(array) == 0:
        return [np.nan]

    # Trim to multiple of step
    n_full = (len(array) // step) * step

    # Compute means on all full chunks
    reshaped = array[:n_full].reshape(-1, step)
    means = np.nanmean(reshaped, axis=1)

    # Could include left over stuff
    """
    remainder = len(array) % step
    if remainder > 0:
        last_chunk_mean = np.nanmean(array[-remainder:])
        means = np.append(means, last_chunk_mean)
    """

    return means.tolist()


"""
def process_chunk(args):
    filename, start, end, plot, show_plots, save_plots, output_folder, params = args

    frames = load_video_frames(filename, frames_start=start, frames_end=end)
    if frames.size == 0:
        print(f"[WARNING] Skipping empty chunk ({start}-{end})")
        return (np.array([]), np.array([]), np.array([]), np.array([]))

    results = detect_damaged_pixels(
        frames,
        plot=plot,
        show_plots=show_plots,
        save_plots=save_plots,
        output_folder=output_folder,
        params=params,
    )
    return results
"""


def main(
    video_filename: str,
    average_time: float = 1.0,
    max_chunks: int | None = None,
    step_size: int = 1000,
    plot: bool = False,
    show_plots: bool = False,
    save_plots: bool = False,
    output_folder: str = "results",
    params: dict | None = None
):
    """
    Processes a video in chunks, computes damaged‐pixel statistics,
    and returns per‐window averages for counts, clusters, sizes, brightness,
    and times.

    - video_filename: path to the AVI file
    - average_time: how many seconds to average over in the final summaries
    - max_chunks: if not None, only process that many chunks (for quick tests)
    - step_size: The number of steps each chunk is split into
    - plot: True or False, determines whether the program plots
    - show_plots: True or False, determines whether the program visually shows plots
    - save_plots: True or False, determines whether the program saves plots
    - output_folder: Location of where saved plots are saved
    """
    settings = _prepare_settings(params)

    # Getting total frame count and FPS without loading the whole video
    cap = cv2.VideoCapture(video_filename)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    FPS = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    # break into chunks for easier parsing
    monolith_frames_list = np.arange(0, num_frames, step_size)
    if monolith_frames_list[-1] != num_frames:
        monolith_frames_list = np.concatenate([monolith_frames_list, [num_frames]])

    # apply user defined limit of how many video chunks to process
    if max_chunks is not None:
        monolith_frames_list = monolith_frames_list[:max_chunks+1]

    """
    chunk_args = [
        (
            video_filename,
            start,
            end,
            plot,
            show_plots,
            save_plots,
            output_folder,
            params,
        )
        for start, end in zip(monolith_frames_list[:-1], monolith_frames_list[1:])
    ]

    num_workers = min(len(chunk_args), cpu_count())
    print(f"Using {num_workers} worker processes")

    ctx = get_context("spawn")
    with ctx.Pool(processes=num_workers) as pool:
        try:
            results = pool.map(process_chunk, chunk_args)
        finally:
            pool.close()
            pool.join()
            gc.collect()

    # unpack & flatten results
    counts, clusters, sizes, brightness = zip(*results)
    counts = np.concatenate(counts)
    clusters = np.concatenate(clusters)
    sizes = np.concatenate(sizes)
    brightness = np.concatenate(brightness)

    # how many frames per averaging window
    step = int(round(FPS * average_time))
    """

    # storage for each chunk’s raw results
    frames_count = []
    all_clusters = []
    all_sizes = []
    all_brightness = []

    # how many frames per averaging window
    step = int(round(FPS * average_time))

    # loop over video chunks and used damaged pixel detector
    for idx in range(len(monolith_frames_list)-1):
        start = monolith_frames_list[idx]
        end = monolith_frames_list[idx + 1]
        print(f"processing chunk {idx}: frames {start}–{end}")
        chunk = load_video_frames(video_filename,
                                  frames_start=start,
                                  frames_end=end)
        counts, clusters, sizes, brightness = detect_damaged_pixels(
                                                                    chunk,
                                                                    settings,
                                                                    plot=plot,
                                                                    show_plots=show_plots,
                                                                    save_plots=save_plots,
                                                                    output_folder=output_folder
                                                                   )

        frames_count.append(counts)
        all_clusters.append(clusters)
        all_sizes.append(sizes)
        all_brightness.append(brightness)

    # flatten results
    counts = np.concatenate(frames_count)
    clusters = np.concatenate(all_clusters)
    sizes = np.concatenate(all_sizes)
    brightness = np.concatenate(all_brightness)

    averages_counts = chunked_nanmean(counts, step)

    # get time interval midpoints
    n_windows = len(averages_counts)
    times = ((np.arange(n_windows) * step) + step / 2) / FPS

    return {
        "averages_counts": averages_counts,
        "averages_clusters": chunked_nanmean(clusters, step),
        "averages_size": chunked_nanmean(sizes, step),
        "averages_brightness": chunked_nanmean(brightness, step),
        "times": times
    }


if __name__ == "__main__":
    # Default number of threads is only 2 (so upping it significantly)
    num_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))
    num_threads_used = num_threads // 2
    cv2.setNumThreads(num_threads_used)
    print(f"Using {num_threads_used} threads")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-vf", "--video_filename", help="Path to the AVI file")
    parser.add_argument("-at", "--average_time", help="How many seconds to average over in the final summaries")
    parser.add_argument("-mc", "--max_chunks", help="If not None, only process that many chunks (for quick tests)")
    parser.add_argument("-ss", "--step_size", help="The number of steps each chunk is split into")
    parser.add_argument("-p", "--plot", help="True or False, determines whether the program plots")
    parser.add_argument("-shp", "--show_plots", help="True or False, determines whether the program visually shows plots")
    parser.add_argument("-svp", "--save_plots", help="True or False, determines whether the program saves plots")
    parser.add_argument("-of", "--output_folder", help="Location of where saved plots are saved")
    parser.add_argument("-ct", "--consecutive_threshold")
    parser.add_argument("-bt", "--brightness_threshold")
    parser.add_argument("-ft", "--flow_threshold")
    parser.add_argument("-st", "--static_threshold")
    parser.add_argument("-mncs", "--min_cluster_size")
    parser.add_argument("-mxcs", "--max_cluster_size")
    parser.add_argument("-mnc", "--min_circularity")
    parser.add_argument("-swr", "--sliding_window_radius")
    parser.add_argument("-np", "--number_of_plots")
    args = parser.parse_args()

    VIDEO_FILENAME = args.video_filename if args.video_filename else "11_01_H_170726081325.avi"
    average_time = float(args.average_time) if args.average_time else 1.0
    if args.max_chunks:
        if args.max_chunks.lower() == "none":
            max_chunks = None
        else:
            max_chunks = int(args.max_chunks)
    else:
        max_chunks = 2
    step_size = int(args.step_size) if args.step_size else 1000
    plot = (args.plot.lower() == "true") if args.plot else False
    show_plots = (args.show_plots.lower() == "true") if args.show_plots else plot
    save_plots = (args.save_plots.lower() == "true") if args.save_plots else False
    output_folder = args.output_folder if args.output_folder else "results"

    params = {
        'consecutive_threshold': int(args.consecutive_threshold) if args.consecutive_threshold else 2,
        'brightness_threshold': int(args.brightness_threshold) if args.brightness_threshold else 170,
        'flow_threshold': float(args.flow_threshold) if args.flow_threshold else 2.0,
        'static_threshold': int(args.static_threshold) if args.static_threshold else 50,
        'min_cluster_size': int(args.min_cluster_size) if args.min_cluster_size else 5,
        'max_cluster_size': int(args.max_cluster_size) if args.max_cluster_size else 20,
        'min_circularity': float(args.min_circularity) if args.min_circularity else 0.1,
        'sliding_window_radius': int(args.sliding_window_radius) if args.sliding_window_radius else 3,
        'number_of_plots': int(args.number_of_plots) if args.number_of_plots else 20,
    }

    # for a quick test on only 2 chunks:
    results = main(
                   VIDEO_FILENAME,
                   average_time=average_time,
                   max_chunks=max_chunks,
                   step_size=step_size,
                   plot=plot,
                   show_plots=show_plots,
                   save_plots=save_plots,
                   output_folder=output_folder,
                   params=params,
    )

    print("counts:", results["averages_counts"])
    print("clusters:", results["averages_clusters"])
    print("sizes:", results["averages_size"])
    print("brightness:", results["averages_brightness"])
    print("times:", results["times"])

"""
# (old version)
def load_video_frames(filename, frames_start=None, frames_end=None, grayscale=True):
    # loads in video frames as greyscale arrays with brightness values ranging
    # from 0 to 255 can load in specific chunk of frames from given video
    # (given frame start and end values as integers)
    # requires video filename as string

    cap = cv2.VideoCapture(filename)
    frames = []

    if frames_start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frames_start)

    frame_idx = frames_start or 0

    while cap.isOpened():
        if frames_end and frame_idx >= frames_end:
            break

        ret, frame = cap.read()
        if not ret:
            break

        if grayscale and len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame)

        frame_idx += 1

    cap.release()
    return frames
"""


"""
def load_video_frames(filename, frames_start=None, frames_end=None, grayscale=True):
    try:
        # Read all frames at once (lazy streaming, not full memory load)
        frames = iio.imread(filename, plugin="ffmpeg", format="gray" if grayscale else None)
    except Exception as e:
        raise IOError(f"Failed to read video '{filename}': {e}")

    # Convert to NumPy array (uint8)
    frames = np.asarray(frames, dtype=DTYPE_IMAGE)

    total_frames = frames.shape[0]
    start = frames_start or 0
    end = frames_end or total_frames

    # Slice requested frame range
    frames = frames[start:end]

    print(f"Loaded {len(frames)} frames from {filename} (shape: {frames.shape})")

    return frames
"""

"""
# CPU version (for unknown reasons ended up being slower)
def load_video_frames(filename, frames_start=None, frames_end=None, grayscale=True):
    cap = cv2.VideoCapture(filename)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {filename}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    start = frames_start or 0
    end = frames_end or total_frames
    n_frames = max(0, end - start)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = np.empty((n_frames, height, width), dtype=DTYPE_IMAGE)

    i = 0
    while i < n_frames:
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"Warning: failed to read frame {i}")
            break
        if grayscale and len(frame.shape) == 3:
            try:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            except Exception as e:
                print(f"Frame {start + i} conversion failed: {e}")
                continue
        frames[i] = frame
        i += 1

    cap.release()

    return frames[:i]
"""

"""
def load_video_frames(filename, frames_start=None, frames_end=None, grayscale=True):
    frames = []
    reader = iio.imiter(filename)

    for i, frame in enumerate(iio.imiter(filename)):
        if frames_start is not None and i < frames_start:
            continue
        if frames_end is not None and i >= frames_end:
            break
        if grayscale and frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame)
    try:
        for i, frame in enumerate(reader):
            if frames_start is not None and i < frames_start:
                continue
            if frames_end is not None and i >= frames_end:
                break
            if grayscale and frame.ndim == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(frame)
    except Exception as e:
        print(f"[WARNING] Frame loading interrupted: {e}")
    finally:
        reader.close()

        # Kill lingering FFmpeg child processes
        current = psutil.Process()
        for child in current.children(recursive=True):
            if "ffmpeg" in child.name().lower():
                try:
                    child.kill()
                except Exception:
                    pass

    return np.asarray(frames, dtype=DTYPE_IMAGE)
"""
