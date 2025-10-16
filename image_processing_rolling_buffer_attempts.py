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
import numpy as np
import requests
from collections import deque
# from numba import njit, prange
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None
    mpatches = None


# including functions

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


# Developing on the rolling buffer idea,
# though might try parallesliation options first before integrating this
class BackgroundEstimator:
    def __init__(self, frames, buffer_size=30):
        self.buffer_size = buffer_size
        self.frames = deque(frames[1:buffer_size])
        self.sum = None
        self.sumsq = None

    def load(self, pixel_std_coeff=1):
        frames = np.stack(self.frames)

        self.sum = frames.sum(axis=0)
        self.sumsq = (frames**2).sum(axis=0)

        pixel_means = frames.mean(axis=0)
        pixel_std = frames.std(axis=0)

        valid = frames <= pixel_means + pixel_std_coeff * pixel_std

        # find background, excluding unusually bright pixels
        masked = np.where(valid, frames, np.nan)
        bg = np.nanmean(masked, axis=0)

        if np.isnan(bg).any():
            bg = np.nan_to_num(bg, nan=frames.mean(axis=0))

        return bg

    def update(self, frame, pixel_std_coeff=1):
        f = frame.astype(np.float32)
        self.frames.append(f)

        self.sum += f
        self.sumsq += f ** 2

        if len(self.frames) > self.buffer_size:
            old = self.frames.popleft()
            self.sum -= old
            self.sumsq -= old ** 2

        n = len(self.frames)
        mean = self.sum / n
        std = np.sqrt(self.sumsq / n - mean ** 2)

        valid = f <= mean + pixel_std_coeff * std
        bg = np.mean(np.stack(self.frames)[valid], axis=0)
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
    current_status = np.zeros((height, width), int)
    longest_flag = np.zeros_like(current_status)

    for m in masks:
        if m is None:
            current_status[:] = 0
        else:
            current_status[m] += 1
            current_status[~m] = 0

        longest_flag = np.maximum(longest_flag, current_status)

    return longest_flag >= consecutive_threshold


def filter_consecutive_pixels(masks, persistent):
    """
    filters out pixels which have been flagged as damaged for multiple
    consecutive frames
    """

    # Realistically should initialize these as numpy arrays
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


def compute_cluster_stats(frames, masks, flows, settings):
    """
    computes the number of clusters for each frame, as well as the mean
    cluster size and brightness
    """
    n = len(frames)
    cluster_count = np.zeros(n, float)
    cluster_size = np.zeros(n, float)
    cluster_brightness = np.zeros(n, float)

    # Vectorise: Need to alter filter_damaged_pixel_clusters
    for i, (frame, mask, flow) in enumerate(zip(frames, masks, flows)):
        if mask is not None and flow <= settings.flow_threshold:
            _, cluster_count[i], cluster_size[i], cluster_brightness[i] = filter_damaged_pixel_clusters(
                frame,
                mask.astype(np.uint8),
                settings.min_cluster_size,
                settings.max_cluster_size,
                settings.min_circularity
            )
            # *Feels like a waste to return clean_mask

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
    frames = np.asarray(frames)
    masks = np.asarray(masks)

    heatmap = masks.astype(np.uint8).sum(axis=0)
    bright = (frames > brightness_threshold) & (~masks)
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
    raw_masks = []  # *Better just to initially define this as a numpy array*
    # *Optimizing this loop will bring massive performance gains*

    bg_estimator = BackgroundEstimator(frames, settings.sliding_window_radius)

    for i, frame in enumerate(frames):

        if optical_flows[i] > settings.flow_threshold:
            raw_masks.append(None)
            continue

        if i == 0:
            background = bg_estimator.load()
        else:
            background = bg_estimator.update(frame)

        raw_mask = _raw_damaged_mask(frame, background)
        bright_filtered = remove_bright_regions(
            background,
            settings.brightness_threshold,
            raw_mask,
            settings.max_cluster_size
        )
        raw_masks.append(bright_filtered)

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


# @njit(parallel=True)
def get_damaged_pixel_mask(frame, height, width, background):
    """
    finds damaged pixels for a given frame
    takes background brightness as input, should be an array of brightness
    values corresponding to each pixel in the frame
    """

    damaged_pixels = np.zeros_like(frame, dtype=np.bool_)
    thresholds = np.empty((height, width), dtype=np.float64)

    # *Feels like this loop could be vectorised*
    # *Though actually running it in parallel might also be a good idea*
    for row in range(height):
        for col in range(width):

            # condition 1: pixel brightness should exceed background by a
            #   threshold scaled with background brightness
            threshold = max(30, 30 + (background[row, col] / 255) * (255 - 30))
            thresholds[row, col] = threshold

            if frame[row, col] > threshold:
                # condition 2: pixel's brightness should exceed mean of its
                #   neighbours in a 30x30 kernel
                kernel = frame[max(row - 10, 0): min(row + 20, height),
                               max(col - 10, 0): min(col + 20, width)]
                kernel_mean = np.mean(kernel)

                if frame[row, col] > (1 * kernel_mean):
                    damaged_pixels[row, col] = True

    damaged_pixels_uint8 = damaged_pixels.astype(np.uint8)

    return damaged_pixels_uint8, thresholds


"""
*Should just do something like this (Vectorised version, should be a lot faster)*
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
"""


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
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed_mask = cv2.morphologyEx(damaged_pixel_mask.astype(np.uint8),
                                   cv2.MORPH_CLOSE, kernel)

    # isolate groups of damaged pixels
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        closed_mask,
        connectivity=8)

    # prepare outputs
    cleaned_mask = np.zeros_like(damaged_pixel_mask, dtype=bool)
    # *alternative:
    # *areas = np.full(num_labels, np.nan)
    # *brightness_sums = np.full(num_labels, np.nan)*
    areas = []
    brightness_sums = []

    # filters clusters of damaged pixels if the area is too large
    # *Potential BUG: Doesn't check the first label*
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]

        if area < min_cluster_size or area > max_cluster_size:
            continue

        # rule out non circular clusters
        if area >= circularity_size_threshold:
            comp_mask = (labels == label).astype(np.uint8)
            contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)
            if not contours:
                continue
            perimeter = cv2.arcLength(contours[0], True)
            if perimeter <= 0:
                continue
            circularity = 4 * np.pi * (area / (perimeter ** 2))
            if circularity < min_circularity:
                continue

        cleaned_mask[labels == label] = True
        # *areas[label] = area
        # *brightness_sums[label] = frame[labels == label].sum()
        areas.append(area)
        brightness_sums.append(frame[labels == label].sum())

    # cluster metrics
    cluster_count = len(areas)
    if cluster_count > 0:
        # *.item() is generally slightly better practice than float
        # *avg_cluster_size = np.nanmean(areas).item()
        # *avg_cluster_brightness = np.nansum(brightness_sums)/np.nansum(areas).item()
        avg_cluster_size = float(np.mean(areas))
        avg_cluster_brightness = float(np.sum(brightness_sums) / np.sum(areas))
    else:
        avg_cluster_size = 0.0
        avg_cluster_brightness = float('nan')

    return cleaned_mask, cluster_count, avg_cluster_size, \
        avg_cluster_brightness


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


# @njit(parallel=True)
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
    estimated_damaged_pixel_counts = np.full(num_frames, np.nan,
                                             dtype=np.float64)  # could just define as a np.empty

    # preprocess masks
    # *v alternatively this could all be replaced by:
    # processed_masks = damaged_pixel_masks[damaged_pixel_masks == np.nan] = np.False_ v*
    processed_masks = np.zeros((num_frames, frame_shape[0], frame_shape[1]),
                               dtype=np.bool_)

    for i in range(num_frames):
        if damaged_pixel_masks[i] is not None:
            processed_masks[i] = damaged_pixel_masks[i]
    # *^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^*

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


# @njit(parallel=True)
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
def compute_optical_flow_metric(frames):
    """
    computes the average optical flow magnitude between consecutive frames
        using the farneback method.
    assumes frames are grayscale images.
    returns an array of optical flow magnitudes for each frame (first frame is
        assigned 0).
    """
    optical_flows = np.full(len(frames), 0.0)

    for i in range(1, len(frames)):  # *Feels like this could be very easily parallised*
        prev_frame = frames[i - 1].astype(np.float32)
        current_frame = frames[i].astype(np.float32)
        flow = cv2.calcOpticalFlowFarneback(
            prev_frame,
            current_frame,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=2,
            flags=0)
        # Finds the means of the magnitudes of the flow vectors
        optical_flows[i] = np.mean(cv2.cartToPolar(flow[..., 0], flow[..., 1])[0])

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
        monolith_frames_list = monolith_frames_list[:max_chunks]

    # storage for each chunk’s raw results
    frames_count = []
    all_clusters = []
    all_sizes = []
    all_brightness = []

    # how many frames per averaging window
    step = int(round(FPS * average_time))

    # loop over video chunks and used damaged pixel detector
    for idx in range(len(monolith_frames_list)):
        start = monolith_frames_list[idx]
        end = monolith_frames_list[idx + 1]
        print(f"processing chunk {idx}: frames {start}–{end}")
        chunk = load_video_frames(video_filename,
                                  frames_start=start,
                                  frames_end=end)
        counts, clusters, sizes, brightness = detect_damaged_pixels(
            chunk, plot=False)

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
        "brightness": chunked_nanmean(brightness, step),
        "times": times
    }


if __name__ == "__main__":
    VIDEO_FILENAME = "11_01_H_170726081325.avi"

    # for a quick test on only 2 chunks:
    results = main(VIDEO_FILENAME, average_time=1.0, max_chunks=2)

    print("counts:", results["averages_counts"])
    print("clusters:", results["averages_clusters"])
    print("sizes:", results["averages_size"])
    print("brightness:", results["averages_brightness"])
    print("times:", results["times"])
