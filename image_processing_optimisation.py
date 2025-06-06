"""
image_processing_optimisation.py

detects gamma radiation damaged pixels from camera footage
for use on scarf

ella beck
11/04/2025
"""

#importing libraries
import random
import cv2
import numpy as np
import requests
from numba import njit, prange


#including functions

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
        print("Failed to download video")

    return filename


def load_video_frames(filename, frames_start = None, frames_end = None):
    """
    loads in video frames as greyscale arrays with brightness values ranging from 0 to 255
    can load in specific chunk of frames from given video (given frame start and end 
    values as integers)
    requires video filename as string
    """

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

        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame)

        frame_idx += 1

    cap.release()
    return frames


def get_video_frames_from_url(url, local_filename='temp_video.avi'):
    """
    takes video url and loads frames in directly
    """

    download_video_from_url(url, local_filename)

    return load_video_frames(local_filename)


def detect_damaged_pixels(frames, plot=False, consecutive_threshold=5, brightness_threshold = 170, flow_threshold = 2.0, number_of_plots = 20, static_threshold = 50):
    """
    main code for detecting damaged pixels
    requires video frames as greyscale arrays of brightness values

    consecutive threshold adjusts how many frames a pixel is bright consecutively before 
        being disregarded as damaged
    brightness_threshold should be cut off at the brightness point where tests start failing
    min and max cluster size adjusts how big a pixel cluster should be before being disregarded
    ssim_threshold should be adjusted depending on how similar the frames are expected to be
    """
    frames = [np.array(frame) for frame in frames]
    num_frames = len(frames)
    height, width = frames[0].shape[:2]  # Dimensions of the frame

    damaged_pixel_masks = []

    min_cluster_size = 5
    max_cluster_size = 20

    for i in range(num_frames):
        current_frame = frames[i]

        # determine sliding frame window for determining background (exclude the current frame)
        start = max(0, i - 3)
        end = min(num_frames, i + 4)
        window_frames = np.array(frames[start:i] + frames[i+1:end])

        # determine background (excluding potentially damaged pixels)
        background = find_background(window_frames)

        # get damaged pixel mask
        damaged_pixels_uint8, _ = get_damaged_pixel_mask(current_frame, height,
            width, background)

        # filter out clusters of damaged pixels
        # filtered_damaged_pixels = filter_damaged_pixel_clusters(damaged_pixels_uint8,
        #     min_cluster_size, max_cluster_size, min_circularity = 0.5)
        filtered_damaged_pixels, _, _, _ = filter_damaged_pixel_clusters(
            current_frame, damaged_pixels_uint8, min_cluster_size,
            max_cluster_size, min_circularity = 0.1, circularity_size_threshold=10
        )

        filtered_damaged_pixels = remove_bright_regions(background, brightness_threshold,
                                                        filtered_damaged_pixels, max_cluster_size)
        
        damaged_pixel_masks.append(filtered_damaged_pixels)

    # filter pixels which have been marked as damaged for too many consecutive frames
    filtered_damaged_pixel_counts, persistent_pixels = filter_consecutive_damaged_pixels(damaged_pixel_masks,
        consecutive_threshold)
    
    cleaned_masks = [None if m is None else (m & ~persistent_pixels)
                     for m in damaged_pixel_masks]
    
    #initial heatmap calculation and static hotspot suppression
    init_heatmap = find_damaged_pixel_heatmap(height, width, frames, cleaned_masks, brightness_threshold)
    static_mask = init_heatmap > static_threshold
    persistent_pixels |= static_mask

    final_masks = [m & ~persistent_pixels for m in cleaned_masks]

    # find estimated number of damaged pixels in bright areas
    bright_area_estimates = find_bright_area_estimates(frames, final_masks,
        brightness_threshold)

    total_damaged_pixel_counts = [actual + estimate if not np.isnan(estimate) else actual
        for actual, estimate in zip(filtered_damaged_pixel_counts, bright_area_estimates)]


    #compute optical flow metrics on the original frames
    optical_flows = compute_optical_flow_metric(frames)
    frames_f, masks_f, pixels_f, kept_idx = [], [], [], []

    for idx, (fr, cnt, flow, m) in enumerate(zip(frames, total_damaged_pixel_counts, optical_flows, final_masks)):
        if flow <= flow_threshold:
            frames_f.append(fr)
            masks_f. append(m)
            pixels_f.append(cnt)
            kept_idx.append(idx)

    # frames_f, total_damaged_pixel_counts, masks_f, optical_flows = \
    #     filter_frames_by_optical_flow(frames, total_damaged_pixel_counts, optical_flows, final_masks, flow_threshold)
    
    post_heatmap = find_damaged_pixel_heatmap(height, width, frames_f, masks_f, brightness_threshold)
    new_static = post_heatmap > static_threshold

    masks_f = [m & ~new_static for m in masks_f]
    counts_f = [int(m.sum()) for m in masks_f]

    #compute final cluster stats
    cluster_counts = []
    avg_cluster_sizes = []
    avg_brightnesses = []

    for frame, mask in zip(frames_f, masks_f):
        cleaned_mask, cc, avg_s, avg_b = filter_damaged_pixel_clusters(
            frame, mask.astype(np.uint8), min_cluster_size,
            max_cluster_size, min_circularity = 0.1, circularity_size_threshold=10
        )

        cluster_counts.append(cc)
        avg_cluster_sizes.append(avg_s)
        avg_brightnesses.append(avg_b)

    # # create plots
    # if plot:
    #     for i in range(number_of_plots):
    #         visualize_damaged_pixels(frames_f[i], masks_f[i], i, masks_f[i], cluster_counts[i])

    #     #calculate heatmap of damaged pixels
    #     heatmap = find_damaged_pixel_heatmap(height, width, frames_f,
    #     masks_f, brightness_threshold)#check this threshold
    #     plot_heatmap(heatmap, title = "Damaged Pixel Heatmap")

    #     #plot_damaged_pixels(counts_f)
    #     plot_cluster_metrics(cluster_counts, avg_cluster_sizes=avg_cluster_sizes, avg_brightnesses=avg_brightnesses)

    return total_damaged_pixel_counts, cluster_counts, avg_cluster_sizes, avg_brightnesses



def find_background(frames):
    """
    should take sliding window of adjacent frames as input
    finds the background for a given pixel based on mean of adjacent frames
    excludes pixels which could potentially be damaged based on their brightness values
    """

    pixel_means = np.mean(frames, axis = 0)
    pixel_std = np.std(frames, axis = 0)
    background = []

    # excludes unusually bright pixels from background calculations
    valid_background_pixels = frames <= (pixel_means + (2 * pixel_std))
    result = np.where(valid_background_pixels, frames, np.nan)
    background = np.nanmean(result, axis = 0)

    if np.isnan(background).any():
        print(f'background not accurately determined for frame')
        background = np.nan_to_num(background, nan = np.mean(frames, axis = 0))

    background = np.array(background)

    return background


@njit(parallel = True)
def get_damaged_pixel_mask(frame, height, width, background):
    """
    finds damaged pixels for a given frame
    takes background brightness as input, should be an array of brightness values
    corresponding to each pixel in the frame
    """

    damaged_pixels = np.zeros_like(frame, dtype=np.bool_)
    thresholds = np.empty((height, width), dtype = np.float64)

    for row in prange(height):
        for col in prange(width):

            # condition 1: pixel brightness should exceed background by a threshold
            #   scaled with background brightness
            threshold = max(30, 30 + (background[row, col] / 255) * (255 - 30))
            thresholds[row, col] = threshold

            if frame[row, col] > threshold:
                # condition 2: pixel's brightness should exceed mean of its
                #   neighbours in a 30x30 kernel
                kernel = frame[max(row - 10, 0) : min(row + 20, height),
                    max(col - 10, 0) : min(col + 20, width)]
                kernel_mean = np.mean(kernel)

                if frame[row, col] > (1 * kernel_mean):
                    damaged_pixels[row, col] = True

    damaged_pixels_uint8 = damaged_pixels.astype(np.uint8)

    return damaged_pixels_uint8, thresholds


def filter_damaged_pixel_clusters(frame, damaged_pixel_mask, min_cluster_size, max_cluster_size,
                                  min_circularity, circularity_size_threshold = 10):
    """
    filters large groups of damaged pixels from the mask
    prevents bright noise such as reflections or glare being misidentified as damaged pixels
    """

    # close gaps (test)

    # isolate groups of damaged pixels
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(damaged_pixel_mask,
        connectivity = 8)
    
    # prepare outputs
    cleaned_mask = np.zeros_like(damaged_pixel_mask, dtype = bool)
    areas = []
    brightness_sums = []

    # filters clusters of damaged pixels if the area is too large
    for label in prange(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]

        if area < min_cluster_size or area > max_cluster_size:
            continue

        #rule out non circular clusters
        if area >= circularity_size_threshold:
            comp_mask = (labels == label).astype(np.uint8)
            contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not contours:
                continue
            perimeter = cv2.arcLength(contours[0], True)
            if perimeter <= 0:
                continue
            circularity = 4 * np.pi * (area / (perimeter ** 2))
            if circularity < min_circularity:
                continue
        
        cleaned_mask[labels == label] = True
        areas.append(area)
        brightness_sums.append(frame[labels == label].sum())

    # cluster metrics
    cluster_count = len(areas)
    if cluster_count > 0:
        avg_cluster_size = float(np.mean(areas))
        avg_cluster_brightness = float(np.sum(brightness_sums) / np.sum(areas))
    else:
        avg_cluster_size = 0.0
        avg_cluster_brightness = float('nan')

    return cleaned_mask, cluster_count, avg_cluster_size, avg_cluster_brightness



def filter_consecutive_damaged_pixels(damaged_pixel_masks, consecutive_threshold):
    """
    removes damaged pixels from the count if they have appeared in too many consecutive frames
    prevents bright noise such as reflections or glare being misidentified as damaged pixels

    returns (filtered) damaged pixel count
    """
    if not damaged_pixel_masks:
        return []

    height, width = damaged_pixel_masks[0].shape
    num_frames = len(damaged_pixel_masks)

    current_run = np.zeros((height, width), dtype = int)
    longest_run = np.zeros((height, width), dtype = int)

    for mask in damaged_pixel_masks:

        if mask is None:
            current_run[:] = 0
            continue

        current_run[mask] += 1
        current_run[~mask] = 0

        longest_run = np.maximum(longest_run, current_run)

        persistent_pixels = longest_run >= consecutive_threshold

        #second pass
        filtered_counts = []
        for mask in damaged_pixel_masks:
            if mask is None:
                filtered_counts.append(np.nan)
            else:
                valid_mask = mask & (~persistent_pixels)
                filtered_counts.append(int(np.sum(valid_mask)))

    return filtered_counts, persistent_pixels


def remove_bright_regions(background, brightness_threshold,
    filtered_damaged_pixels, max_cluster_size):
    """
    removes damaged pixels from the mask if they exist in bright areas
    avoids inaccuracies due to the code's capabilities of operating
    in low contrast/bright background
    """

    bright_background_mask = (background > brightness_threshold).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bright_background_mask,
        connectivity = 8)

    #create a mask for large bright regions
    remove = np.zeros_like(bright_background_mask, dtype = np.bool_)

    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= max_cluster_size:
            remove[labels == label] = True


    damaged_pixel_mask_uint8 = filtered_damaged_pixels.astype(np.uint8)
    num_labels2, labels2, stats2, _ = cv2.connectedComponentsWithStats(damaged_pixel_mask_uint8, connectivity = 8)

    cleaned = np.zeros_like(filtered_damaged_pixels, dtype = bool)
    for lbl in range(1, num_labels2):
        comp = (labels2 == lbl)
        if np.any(comp & remove):
            continue
        cleaned[comp] = True

    return cleaned

@njit(parallel=True)
def estimate_damaged_pixels_in_bright_areas(frames, damaged_pixel_masks, brightness_threshold=170):
    """
    estimates the number of damaged pixels present in bright areas or areas of low contrast
    provides estimates for the correct damaged pixel count where my code would otherwise fail
    """

    num_frames = len(frames)
    frame_shape = frames[0].shape
    estimated_damaged_pixel_counts = np.full(num_frames, np.nan, dtype=np.float64)

    # preprocess masks

    processed_masks = np.zeros((num_frames, frame_shape[0], frame_shape[1]), dtype=np.bool_)

    for i in range(num_frames):
        if damaged_pixel_masks[i] is not None:
            processed_masks[i] = damaged_pixel_masks[i]

    for i in prange(len(frames)):
        frame = frames[i]
        mask = processed_masks[i]

        # identify low and high brightness regions, excluding existing damaged pixels
        low_brightness_mask = (frame < brightness_threshold) & ~mask
        high_brightness_mask = (frame >= brightness_threshold) & ~mask

        # calculate areas
        low_brightness_area = np.sum(low_brightness_mask)
        high_brightness_area = np.sum(high_brightness_mask)

        if low_brightness_area > 0:
            # density of damaged pixels in low-brightness areas
            damaged_pixel_density = np.sum(mask) / low_brightness_area

            # estimate damaged pixels in high-brightness areas
            estimated_high_brightness_damaged_pixels = round(damaged_pixel_density
                * high_brightness_area)

        else:
            estimated_high_brightness_damaged_pixels = np.nan

        estimated_damaged_pixel_counts[i] = estimated_high_brightness_damaged_pixels

    return estimated_damaged_pixel_counts


@njit(parallel = True)
def find_bright_area_estimates(frames, damaged_pixel_masks,
    brightness_threshold):
    """
    finds estimated number of damaged pixels in bright areas using 
        estimate_damaged_pixels_in_bright_areas()
    """

    bright_area_estimates = np.full(len(frames), np.nan, dtype = np.float64)

    for i, (frame, mask) in enumerate(zip(frames, damaged_pixel_masks)):
        if mask is None:
            bright_area_estimates[i] = np.nan
            continue

        high_brightness_mask = (frame > brightness_threshold) & ~mask

        if np.sum(high_brightness_mask) > 0:
            estimate = estimate_damaged_pixels_in_bright_areas(frames,
                damaged_pixel_masks)
            bright_area_estimates[i] = estimate[i]
        else:
            bright_area_estimates[i] = np.nan

    return bright_area_estimates


def compute_optical_flow_metric(frames):
    """
    computes the average optical flow magnitude between consecutive frames using the farneback method.
    assumes frames are grayscale images.
    returns an array of optical flow magnitudes for each frame (first frame is assigned 0).
    """
    optical_flows = [0.0]

    for i in range(1, len(frames)):
        prev_frame = frames[i - 1].astype(np.float32)
        current_frame = frames[i].astype(np.float32)
        flow = cv2.calcOpticalFlowFarneback(prev_frame, current_frame, None, pyr_scale=0.5,
                                            levels=3, winsize=15, iterations=3, poly_n=5,
                                            poly_sigma=2, flags=0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        optical_flows.append(np.mean(mag))

    return np.array(optical_flows)


def filter_frames_by_optical_flow(frames, pixel_counts, optical_flows, damaged_pixel_masks, threshold):
    """
    filters out frames whose optical flow exceeds the given threshold.
    frames with optical flow above the threshold are removed entirely from the frames list,
    and their corresponding pixel counts and masks are discarded.
    """
    
    removed_counter = 0
    filtered_frames = []
    filtered_counts = []
    filtered_masks = []
    filtered_flows = []
    
    for frame, count, flow, mask in zip(frames, pixel_counts, optical_flows, damaged_pixel_masks):
        if flow > threshold:
            removed_counter += 1
    
        else:
            filtered_frames.append(frame)
            filtered_counts.append(count)
            filtered_masks.append(mask)
            filtered_flows.append(flow)
            
    print(f"Removed {removed_counter} frames due to high optical flow")

    return filtered_frames, filtered_counts, filtered_masks, filtered_flows


def find_damaged_pixel_heatmap(height, width, frames, damaged_pixel_masks, brightness_threshold):
    """
    produces heatmap of damaged pixel occurrences
    can be used to verify uniformity of damaged pixels (unless frames contain 
        a lot of bright noise, which will be excluded on the heatmap)
    """
    MIN_VALID_FRAMES = 10

    mask_stack = np.stack([m.astype(np.uint8) for m in damaged_pixel_masks], axis = 0)
    frame_stack = np.stack(frames, axis = 0)

    heatmap = mask_stack.sum(axis = 0)

    bright_stack = (frame_stack > brightness_threshold) & (~mask_stack.astype(bool))
    valid_counts = (~bright_stack).sum(axis = 0)

    result = np.zeros_like(heatmap, dtype = np.float64)
    mask = valid_counts > MIN_VALID_FRAMES
    result[mask] = heatmap[mask] / valid_counts[mask] * 100

    return result


# executing main code

VIDEO_FILENAME ='11_01_H_170726081325.avi'
cap = cv2.VideoCapture(VIDEO_FILENAME)
NUM_FRAMES = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
monolith_frames_list = np.arange(0, NUM_FRAMES, 1000)
monolith_frames = np.arange(0, NUM_FRAMES, 1)

# monolith_frames_list = np.arange(0, 2000, 1000)
# monolith_frames = np.arange(0, 2000, 1)


FPS = cap.get(cv2.CAP_PROP_FPS)
AVERAGES_TIME = 1
cap.release()

frames_count = []
all_clusters      = []
all_sizes         = []
all_brightness    = []

step = int(round(FPS * AVERAGES_TIME))  # e.g. ~30 frames

for i in range(len(monolith_frames_list) - 1):
    print(f"processing chunk {i}")
    chunk_frames = load_video_frames(
        VIDEO_FILENAME, 
        frames_start = monolith_frames_list[i],
        frames_end   = monolith_frames_list[i+1]
    )
    counts, clusters, size, brightness = detect_damaged_pixels(chunk_frames, plot=False)

    # Append this chunk's data to your “global” lists
    frames_count.append(counts)
    all_clusters.append(clusters)
    all_sizes.append(size)
    all_brightness.append(brightness)

# Flatten everything so that we have one long list per metric:
counts     = [c for chunk in frames_count for c in chunk]
clusters   = [c for chunk in all_clusters  for c in chunk]
sizes      = [s for chunk in all_sizes     for s in chunk]
brightness = [b for chunk in all_brightness for b in chunk]

# Now each of these lists (counts, clusters, sizes, brightness) covers
# all processed frames in chronological order. You can window‐average them:

averages = [
    np.nanmean(counts[i : i + step])
    for i in range(0, len(counts), step)
]
averages_clusters = [
    np.nanmean(clusters[i : i + step])
    for i in range(0, len(clusters), step)
]
averages_size = [
    np.nanmean(sizes[i : i + step])
    for i in range(0, len(sizes), step)
]
averages_brightness = [
    np.nanmean(brightness[i : i + step])
    for i in range(0, len(brightness), step)
]

n = len(averages)
times = ((np.arange(n) * step) + step/2) / FPS

print("averages (counts):", averages)
print("averages_clusters:", averages_clusters)
print("averages_size:", averages_size)
print("averages_brightness:", averages_brightness)
print("times (s):", times)