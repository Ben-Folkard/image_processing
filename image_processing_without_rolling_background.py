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
import os
import numpy as np
import requests
from numba import prange, njit
from concurrent.futures import ThreadPoolExecutor
from joblib import Parallel, delayed
from multiprocessing import Pool, cpu_count

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None
    mpatches = None


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
# If the gpu version is installed:
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


def get_video_frames_from_url(
        url,
        local_filename='temp_video.avi',
        frames_start=None,
        frames_end=None
        ):
    """
    takes video url and loads frames in directly
    """

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
    }
    if params:
        defaults.update(params)
    return type('S', (), defaults)


def compute_background(frames, index, radius):
    """
    estimate background brightness for each pixel of a given frame, based on
    the mean brightness of that pixel in the frames in a sliding window
    (providing the pixel is undamaged in those frames)
    """
    start = max(0, index - radius)
    end = min(len(frames), index + radius + 1)

    neighbours = np.asarray(frames[start:end], dtype=np.float32)
    center_idx = index - start

    mask = np.ones(len(neighbours), dtype=bool)
    mask[center_idx] = False

    neighbours = neighbours[mask]

    return _find_background(neighbours)


@njit(parallel=True, fastmath=True)
def _find_background(frames, pixel_std_coeff=1.0):
    n, h, w = frames.shape
    pixel_means = np.zeros((h, w), np.float32)
    pixel_stds = np.zeros((h, w), np.float32)
    bg = np.zeros((h, w), np.float32)

    # Compute mean
    for i in prange(h):
        for j in range(w):
            s = 0.0
            for k in range(n):
                s += frames[k, i, j]
            pixel_means[i, j] = s / n

    # Compute std
    for i in prange(h):
        for j in range(w):
            s = 0.0
            for k in range(n):
                diff = frames[k, i, j] - pixel_means[i, j]
                s += diff * diff
            pixel_stds[i, j] = (s / n) ** 0.5

    # Compute background excluding outliers
    for i in prange(h):
        for j in range(w):
            thr = pixel_means[i, j] + pixel_std_coeff * pixel_stds[i, j]
            s = 0.0
            c = 0
            for k in range(n):
                val = frames[k, i, j]
                if val <= thr:
                    s += val
                    c += 1
            if c > 0:
                bg[i, j] = s / c
            else:
                s_all = 0.0
                for k in range(n):
                    s_all += frames[k, i, j]
                bg[i, j] = s_all / n

    return bg


def _raw_damaged_mask(frame, background):
    """
    returns first pass mask of potentially damaged pixels
    """
    # *Seems a little reduntant returning threasholds if you're not going to use it*
    mask, _ = get_damaged_pixel_mask(
        frame,
        frame.shape[0],
        frame.shape[1],
        background
    )

    return mask.astype(bool)


def compute_persistent_mask(masks, consecutive_threshold):
    """
    flags pixels which have been flagged as damaged for multiple
    consecutive frames
    """
    height, width = masks[0].shape
    current_status = np.zeros((height, width), np.int32)
    longest_flag = np.zeros_like(current_status)

    for i in prange(len(masks)):
        mask = masks[i]
        if mask is None:
            current_status[:] = 0
        else:
            current_status[mask] += 1
            current_status[~mask] = 0

        longest_flag = np.maximum(longest_flag, current_status)

    return longest_flag >= consecutive_threshold


def filter_consecutive_pixels(masks, persistent):
    """
    filters out pixels which have been flagged as damaged for multiple
    consecutive frames
    """

    filtered_masks = []
    counts = []

    for i, m in enumerate(masks):
        if m is None:
            filtered_masks.append(None)
            counts.append(np.nan)
        else:
            filtered = m & ~persistent
            filtered_masks.append(filtered)
            counts.append(int(filtered.sum()))

    return filtered_masks, counts


"""
# Until general code restructure, this will have to remain like this
def filter_consecutive_pixels(masks, persistent):
    filtered_masks = np.full_like(masks, None, dtype=object)
    counts = np.full(len(masks), np.nan, dtype=np.float32)

    inv_persistent = ~persistent

    for i, mask in enumerate(masks):
        if mask is not None:
            filtered = np.logical_and(mask, inv_persistent)
            filtered_masks[i] = filtered
            counts[i] = filtered.sum()

    return filtered_masks, counts
"""


# Going to have to monitor, as shows significant slow down for small tests
def compute_cluster_stats(frames, masks, flows, settings, n_jobs=-1):
    def process_single(frame, mask, flow):
        if mask is None or flow > settings.flow_threshold:
            return 0.0, 0.0, 0.0
        _, count, size, bright = filter_damaged_pixel_clusters(
            frame, mask, settings.min_cluster_size,
            settings.max_cluster_size, settings.min_circularity
        )
        return count, size, bright

    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(process_single)(frame, mask, flow)
        for frame, mask, flow in zip(frames, masks, flows)
    )

    # returning cluster_count, cluster_size, cluster_brightness
    return np.array(results).T


def _get_final_count(masks, bright_estimates):
    """
    gets final damaged pixel count, adding in the bright area estimates to
    the raw dark area count
    """
    # counts = base noise + estimated radiation induced noise
    # Possible bug: kept in the original possible bug where
    # base still contains nans whilst estimate doesn't
    counts = np.array([
       (np.sum(mask) if mask is not None else np.nan) +
       (estimate if not np.isnan(estimate) else 0)
       for mask, estimate in zip(masks, bright_estimates)
    ], dtype=float)

    return counts


"""
# (old version)
def _get_final_count(masks, bright_estimates):
    counts = []

    for mask, estimate in zip(masks, bright_estimates):
        base = int(mask.sum()) if mask is not None else np.nan
        counts.append(base + (estimate if not np.isnan(estimate) else 0))

    return np.array(counts, dtype=float)
"""


def _generate_plots(
        frames,
        masks,
        counts,
        flows,
        settings):
    """
    generates plots for user
    """
    survivors = [i for i, flow in enumerate(flows) if flow <=
                 settings.flow_threshold]
    for i in survivors[:settings.number_of_plots]:
        visualise_damaged_pixels(frames[i], masks[i], i)

    plot_damaged_pixels(counts)
    heatmap = find_damaged_pixel_heatmap(
        frames,
        [m.astype(np.uint8) for m in masks if m is not None]
    )
    plot_heatmap(heatmap)


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
    mask_shape = frames[0].shape
    filled_masks = [(mask if mask is not None else np.zeros(mask_shape, dtype=bool)) for mask in masks]

    mask_stack = np.stack([mask.astype(np.uint8) for mask in filled_masks], axis=0)
    frame_stack = np.stack(frames, axis=0)

    heatmap = mask_stack.sum(axis=0)
    bright = (frame_stack > brightness_threshold) & \
        (~mask_stack.astype(bool))
    valid_counts = (~bright).sum(axis=0)

    percentage_map = np.zeros_like(heatmap, dtype=float)
    good = valid_counts > min_valid_frames
    percentage_map[good] = (heatmap[good] / valid_counts[good]) * 100

    return percentage_map > static_threshold


def apply_static_suppression(masks, persistent, static_mask):
    bad = persistent | static_mask
    final_masks = []
    counts = []

    for m in masks:
        if m is None:
            final_masks.append(None)
            counts.append(np.nan)
        else:
            final_mask = m & ~bad
            final_masks.append(final_mask)
            counts.append(int(final_mask.sum()))
    return final_masks, np.array(counts, dtype=float)


""" (Also somehow slower)
def apply_static_suppression(masks, persistent, static_mask):
    bad = persistent | static_mask
    final_masks = []
    counts = np.full(len(masks), np.nan, dtype=np.float64)

    for i, mask in enumerate(masks):
        if mask is None:
            final_masks.append(None)
        else:
            final_mask = mask & ~bad
            final_masks.append(final_mask)
            counts[i] = int(final_mask.sum())
    return final_masks, counts
"""

""" (Ended up being slower)
def apply_static_suppression(masks, persistent, static_mask):
    bad = np.logical_or(persistent, static_mask)

    # Separate valid masks
    valid_indices = [i for i, m in enumerate(masks) if m is not None]
    if not valid_indices:
        return masks, np.full(len(masks), np.nan)

    # Stack only valid ones to work in batch
    stacked = np.stack([masks[i] for i in valid_indices])
    suppressed = np.logical_and(stacked, ~bad)  # vectorized suppression

    # Replace back into original list
    final_masks = list(masks)
    for idx, m in zip(valid_indices, suppressed):
        final_masks[idx] = m

    # Compute counts (vectorized where possible)
    counts = np.full(len(masks), np.nan)
    counts[valid_indices] = suppressed.sum(axis=(1, 2))

    return final_masks, counts
"""


def detect_damaged_pixels(
        frames,
        plot=False,
        params=None
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
    # unpack and prepare inputs
    settings = _prepare_settings(params)
    frames = np.asarray(frames)

    # optical flow screening
    optical_flows = compute_optical_flow_metric(frames)

    # find damaged pixel mask for each frame
    raw_masks = np.full(len(frames), None, dtype=object)

    for i, frame in enumerate(frames):
        if optical_flows[i] <= settings.flow_threshold:
            background = compute_background(
                                     frames,
                                     i,
                                     settings.sliding_window_radius,
            )
            raw_mask = _raw_damaged_mask(frame, background)
            bright_filtered = remove_bright_regions(
                background,
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
    # *Seems a waste to return counts*
    clean_masks, _ = filter_consecutive_pixels(raw_masks,
                                               persistent_pixels)

    # initial heatmap calculation and static hotspot suppression
    static_mask = compute_static_mask(
        frames,
        clean_masks,
        settings.brightness_threshold,
        settings.static_threshold,
        10
    )

    final_masks, total_counts = apply_static_suppression(
        clean_masks,
        persistent_pixels,
        static_mask
    )

    # find estimated number of damaged pixels in bright areas
    bright_area_estimates = find_bright_area_estimates(
        np.stack(frames, axis=0).astype(np.float64),
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
            settings)

    return total_counts, cluster_counts, avg_sizes, avg_brightnesses


def get_damaged_pixel_mask(frame, height, width, background):
    # condition 1: pixel brightness should exceed background by a
    #   threshold scaled with background brightness
    threshold = np.maximum(30, 30 + (background / 255) * (255 - 30))
    damaged = frame > threshold

    # condition 2: pixel's brightness should exceed mean of its
    #   neighbours in a 30x30 kernel
    kernel_size = (30, 30)
    local_mean = cv2.blur(frame.astype(np.float64), kernel_size)

    damaged &= frame > local_mean

    return damaged.astype(np.uint8), threshold


def filter_damaged_pixel_clusters(
        frame,
        damaged_pixel_mask,
        min_cluster_size,
        max_cluster_size,
        min_circularity,
        circularity_size_threshold=10):
    """
    filters large groups of damaged pixels from the mask
    prevents bright noise such as reflections or glare being misidentified as
    damaged pixels
    """

    # close gaps (test)  # Dilation followed by erosion to get rid of noise by bluring
    closed_mask = cv2.morphologyEx(
                                   damaged_pixel_mask.astype(np.uint8),
                                   cv2.MORPH_CLOSE,
                                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )

    # isolate groups of damaged pixels
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed_mask, connectivity=8)

    # prepare outputs
    cleaned_mask = np.zeros_like(damaged_pixel_mask, dtype=bool)
    areas = stats[1:, cv2.CC_STAT_AREA]

    valid_labels = np.flatnonzero((areas >= min_cluster_size) & (areas <= max_cluster_size)) + 1
    if len(valid_labels) == 0:
        return cleaned_mask, 0, 0.0, float('nan')

    perimeters = np.zeros(num_labels, np.float32)
    circularities = np.zeros(num_labels, np.float32)

    # Only compute contours for labels above threshold
    large_labels = [label for label in valid_labels if stats[label, cv2.CC_STAT_AREA] >= circularity_size_threshold]
    if large_labels:
        mask_tmp = np.zeros_like(closed_mask, dtype=np.uint8)
        for label in large_labels:
            mask_tmp[:] = 0
            mask_tmp[labels == label] = 255
            contours, _ = cv2.findContours(mask_tmp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if contours:
                per = cv2.arcLength(contours[0], True)
                if per > 0:
                    perimeters[label] = per
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
    flat_frame = frame.ravel().astype(np.float32)
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
        np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        bright_background_mask, connectivity=8)

    # create a mask for large bright regions
    remove = np.zeros_like(bright_background_mask, dtype=np.bool_)

    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= max_cluster_size:
            remove[labels == label] = True

    damaged_pixel_mask_uint8 = filtered_damaged_pixels.astype(np.uint8)
    num_labels2, labels2, stats2, _ = cv2.connectedComponentsWithStats(
        damaged_pixel_mask_uint8, connectivity=8)

    cleaned = np.zeros_like(filtered_damaged_pixels, dtype=bool)
    for lbl in range(1, num_labels2):
        comp = (labels2 == lbl)
        if np.any(comp & remove):
            continue
        cleaned[comp] = True

    return cleaned


def estimate_damaged_pixels_in_bright_areas(
    frames,
    damaged_pixel_masks,
    brightness_threshold=170
):

    """
    estimates the number of damaged pixels present in bright areas or areas of
        low contrast
    provides estimates for the correct damaged pixel count where my code would
        otherwise fail
    """

    num_frames = len(frames)
    frame_shape = frames[0].shape
    estimated_damaged_pixel_counts = np.full(num_frames, np.nan, dtype=np.float64)  # could just define as a np.empty

    # preprocess masks
    processed_masks = np.zeros((num_frames, *frame_shape), dtype=np.bool_)

    for i, mask in enumerate(damaged_pixel_masks):
        if mask is not None:
            processed_masks[i] = mask

    for i in range(len(frames)):
        frame = frames[i]
        mask = processed_masks[i]

        # identify low and high brightness regions, excluding existing
        #   damaged pixels
        low_brightness_mask = (frame < brightness_threshold) & ~mask
        high_brightness_mask = (frame >= brightness_threshold) & ~mask

        # calculate areas
        low_brightness_area = np.sum(low_brightness_mask)  # low_brightness_mask.sum() exists
        high_brightness_area = np.sum(high_brightness_mask)  # high_brightness_mask.sum() exists

        if low_brightness_area > 0:
            # density of damaged pixels in low-brightness areas
            damaged_pixel_density = np.sum(mask) / low_brightness_area

            # estimate damaged pixels in high-brightness areas
            estimated_high_brightness_damaged_pixels = round(
                damaged_pixel_density * high_brightness_area)

        else:
            estimated_high_brightness_damaged_pixels = np.nan

        estimated_damaged_pixel_counts[i] = \
            estimated_high_brightness_damaged_pixels

    return estimated_damaged_pixel_counts  # is actually returning estimated_high_brightness_damaged_pixels


def find_bright_area_estimates(
    frames,
    damaged_pixel_masks,
    brightness_threshold
):
    """
    finds estimated number of damaged pixels in bright areas using
        estimate_damaged_pixels_in_bright_areas()
    """

    bright_area_estimates = np.full(len(frames), np.nan, dtype=np.float64)

    estimate = estimate_damaged_pixels_in_bright_areas(frames, damaged_pixel_masks)

    for i, (frame, mask) in enumerate(zip(frames, damaged_pixel_masks)):
        if mask is not None:
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
        prev = frames[i - 1].astype(np.float32)
        curr = frames[i].astype(np.float32)
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
        bright_threshold=170
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

        plt.show()


def plot_heatmap(heatmap, title="Damaged Pixel Heatmap"):
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
        plt.show()


def plot_damaged_pixels(damaged_pixel_counts):
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
        plt.show()


def chunked_nanmean(array, step):
    return [np.nanmean(array[i:i+step]) for i in range(0, len(array), step)]


def process_chunk(args):
    """Worker function for parallel processing of video chunks"""
    filename, start, end = args
    chunk = load_video_frames(filename, frames_start=start, frames_end=end)
    return detect_damaged_pixels(chunk, plot=False)


def main(
    video_filename: str,
    average_time: float = 1.0,
    max_chunks: int | None = None,
    STEP_SIZE: int = 1000
):
    """
    Processes a video in chunks, computes damaged‐pixel statistics,
    and returns per‐window averages for counts, clusters, sizes, brightness,
    and times.

    - video_filename: path to the AVI file
    - average_time: how many seconds to average over in the final summaries
    - max_chunks: if not None, only process that many chunks (for quick tests)
    - STEP_SIZE: The number of steps each chunk is split into
    """
    # open video
    cap = cv2.VideoCapture(video_filename)
    NUM_FRAMES = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    FPS = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    # break into chunks for easier parsing
    monolith_frames_list = np.arange(0, NUM_FRAMES, STEP_SIZE)
    if monolith_frames_list[-1] != NUM_FRAMES:
        monolith_frames_list = np.concatenate([monolith_frames_list, [NUM_FRAMES]])

    # apply user defined limit of how many video chunks to process
    if max_chunks is not None:
        monolith_frames_list = monolith_frames_list[:max_chunks+1]

    chunk_args = [
        (video_filename, monolith_frames_list[i], monolith_frames_list[i + 1])
        for i in range(len(monolith_frames_list)-1)
    ]

    num_workers = min(len(chunk_args), cpu_count())
    print(f"Using {num_workers} worker processes")

    with Pool(processes=num_workers) as pool:
        results = pool.map(process_chunk, chunk_args)

    # unpack & flatten results
    counts, clusters, sizes, brightness = zip(*results)
    counts = np.concatenate(counts)
    clusters = np.concatenate(clusters)
    sizes = np.concatenate(sizes)
    brightness = np.concatenate(brightness)

    # how many frames per averaging window
    step = int(round(FPS * average_time))

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
    STEP_SIZE = int(args.step_size) if args.step_size else 1000

    # for a quick test on only 2 chunks:
    results = main(
                   VIDEO_FILENAME,
                   average_time=average_time,
                   max_chunks=max_chunks,
                   STEP_SIZE=STEP_SIZE,
    )

    print("counts:", results["averages_counts"])
    print("clusters:", results["averages_clusters"])
    print("sizes:", results["averages_size"])
    print("brightness:", results["averages_brightness"])
    print("times:", results["times"])
