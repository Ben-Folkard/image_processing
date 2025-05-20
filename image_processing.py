# #image_processing.py
# #used to find the number of damaged pixels present in a video, suitable for gamma dosimetry monitoring in cmos cameras
#importing libraries
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
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


def get_video_frames_from_url(url, local_filename = 'temp_video.avi', frames_start = None, frames_end = None):
    """
    takes video url and loads frames in directly
    """

    download_video_from_url(url, local_filename)

    return load_video_frames(local_filename, frames_start, frames_end)


def detect_damaged_pixels(frames, plot=False, consecutive_threshold=5, brightness_threshold = 170, flow_threshold = 2.0, number_of_plots = 20):
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

    min_cluster_size = 1
    max_cluster_size = 100

    for i in range(num_frames):
        current_frame = frames[i]

        # determine sliding frame window for determining background (exclude the current frame)
        start = max(0, i - 3)
        end = min(num_frames, i + 4)
        window_frames = np.array(frames[start:i] + frames[i+1:end])

        # determine background (excluding potentially damaged pixels)
        background = find_background(window_frames)

        # get damaged pixel mask
        damaged_pixels_uint8, threshold = get_damaged_pixel_mask(current_frame, height,
            width, background)

        # filter out clusters of damaged pixels
        filtered_damaged_pixels = filter_damaged_pixel_clusters(damaged_pixels_uint8,
            min_cluster_size, max_cluster_size, min_circularity = 0.5)

        # remove detected damaged pixels which lie in bright regions
        filtered_damaged_pixels = remove_bright_regions(background, threshold,
            filtered_damaged_pixels, max_cluster_size)

        damaged_pixel_masks.append(filtered_damaged_pixels)

    # filter pixels which have been marked as damaged for too many consecutive frames
    filtered_damaged_pixel_counts, persistent_pixels = filter_consecutive_damaged_pixels(damaged_pixel_masks,
        consecutive_threshold)
    
    cleaned_masks = [None if m is None else (m & ~persistent_pixels)
                     for m in damaged_pixel_masks]

    # find estimated number of damaged pixels in bright areas
    bright_area_estimates = find_bright_area_estimates(frames, cleaned_masks,
        brightness_threshold)


    total_damaged_pixel_counts = [actual + estimate if not np.isnan(estimate) else actual
        for actual, estimate in zip(filtered_damaged_pixel_counts, bright_area_estimates)]


    #compute optical flow metrics on the original frames
    optical_flows = compute_optical_flow_metric(frames)

    frames, total_damaged_pixel_counts, masks_after_flow, optical_flows = \
        filter_frames_by_optical_flow(frames, total_damaged_pixel_counts, optical_flows, cleaned_masks, flow_threshold)
    

    # create plots
    if plot:
        for i in range(number_of_plots):
            visualize_damaged_pixels(frames[i], masks_after_flow[i], total_damaged_pixel_counts[i])

        #calculate heatmap of damaged pixels
        heatmap = find_damaged_pixel_heatmap(height, width, frames,
        masks_after_flow, brightness_threshold)#check this threshold
        plot_heatmap(heatmap, title = "Damaged Pixel Heatmap")

        plot_damaged_pixels(total_damaged_pixel_counts)

    return total_damaged_pixel_counts



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


def filter_damaged_pixel_clusters(damaged_pixel_mask, min_cluster_size, max_cluster_size,
                                  min_circularity):
    """
    filters large groups of damaged pixels from the mask
    prevents bright noise such as reflections or glare being misidentified as damaged pixels
    """

    filtered_damaged_pixels = np.zeros_like(damaged_pixel_mask, dtype = np.bool_)

    # isolate groups of damaged pixels
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(damaged_pixel_mask,
        connectivity = 8)

    # filters clusters of damaged pixels if the area is too large
    for label in prange(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]

        if area < min_cluster_size or area > max_cluster_size:
            continue

        #rule out non circular clusters
        comp_mask = (labels == label).astype(np.uint8)

        contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue

        perimeter = cv2.arcLength(contours[0], True)
        if perimeter <= 0:
            continue

        circularity = 4 * np.pi * (area / (perimeter ** 2))
        if circularity >= min_circularity:
            filtered_damaged_pixels[labels == label] = True


    return filtered_damaged_pixels



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
    large_bright_regions_mask = np.zeros_like(bright_background_mask, dtype = np.bool_)

    for label in range(num_labels):
        region_size = stats[label, cv2.CC_STAT_AREA]
        if region_size >= max_cluster_size:
            large_bright_regions_mask[labels == label] = True

    return filtered_damaged_pixels


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


@njit(parallel = True)
def find_damaged_pixel_heatmap(height, width, frames, damaged_pixel_masks, brightness_threshold):
    """
    produces heatmap of damaged pixel occurrences
    can be used to verify uniformity of damaged pixels (unless frames contain 
        a lot of bright noise, which will be excluded on the heatmap)
    """

    heatmap = np.zeros((height, width), dtype = np.float32)
    valid_pixel_counts = np.zeros((height, width), dtype = np.float32)

    for i, (frame, mask) in enumerate(zip(frames, damaged_pixel_masks)):
        if mask is not None:
            heatmap += mask.astype(np.float32)
            high_brightness_mask = (frame > brightness_threshold) & ~mask
            valid_pixel_counts += (~high_brightness_mask).astype(np.float32)


    result = np.zeros_like(heatmap, dtype = np.float64)

    for x in range(height):
        for y in range(width):
            if valid_pixel_counts[x, y] > 0:
                result[x, y] = (heatmap[x, y] / valid_pixel_counts[x, y]) * 100

    return result


def visualize_damaged_pixels(frame, damaged_pixels, frame_index, bright_threshold = 170):
    """
    plots two versions of a given frame side by side, the second frame
        highlighting detected damaged pixels

    plots detected damaged pixels in red
    plots bright areas (where the code has estimated the damaged pixel count) in green
    """

    height, width = frame.shape

    bright_areas = frame > bright_threshold
    highlighted_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    # highlighted damaged pixels in red
    damaged_pixels_colored = np.zeros((height, width, 3), dtype=np.uint8)
    damaged_pixels_colored[damaged_pixels] = [255, 0, 0]
    highlighted_frame = cv2.addWeighted(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR),
        1.0, damaged_pixels_colored, 1.0, 0)

    # highlighted bright pixel mask in green
    bright_pixels_coloured = np.zeros((height, width, 3), dtype = np.uint8)
    bright_pixels_coloured[bright_areas] = [0, 255, 0]
    highlighted_frame = cv2.addWeighted(highlighted_frame, 1.0, bright_pixels_coloured, 1.0, 0)

    # #scatter red pixels for estimated damaged pixels
    # if estimate_count and bright_areas.any():
    #     bright_coords = np.column_stack(np.where(bright_areas))
    #     if len(bright_coords) > estimate_count:
    #         selected_coords = bright_coords[np.random.choice(len(bright_coords),
    #           round(estimate_count), replace = False)]

    #     else:
    #         selected_coords = bright_coords

    #     for coord in selected_coords:
    #         highlighted_frame[coord[0], coord[1]] = [0, 0, 255]

    plt.figure(figsize = (15, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(frame, cmap = 'gray')
    plt.title(f'Original Frame {frame_index}')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(highlighted_frame, cv2.COLOR_BGR2RGB))
    plt.title(f'Damaged Pixels Highlighted {frame_index}')
    plt.axis('off')
    plt.show()


def plot_heatmap(heatmap, title = "Damaged Pixel Heatmap"):
    """
    plots heatmap showing damaged pixel distribution over every frame
    """

    plt.figure(figsize = (15, 10))
    plt.imshow(heatmap, cmap = 'viridis', interpolation ='nearest')
    plt.colorbar(label = "Percentage of frames (%)")
    plt.title(title)
    plt.show()


def plot_damaged_pixels(damaged_pixel_counts):
    """
    plots the count of damaged pixels across frames
    """

    plt.figure(figsize=(10, 5))
    plt.plot(damaged_pixel_counts, label='Damaged Pixels Count', color='blue')
    plt.xlabel('Frame Number')
    plt.ylabel('Number of Damaged Pixels')
    plt.title('Damaged Pixels Detected Over Time')
    plt.legend()
    plt.show()


def create_isotropic_test_video(num_frames=1000, width=928, height=576, damaged_pixel_count=1000):
    """
    creates test video with isotropically distributed damaged pixels
        in order to visually verify heatmap
    """

    frames = []

    for i in range(num_frames):
        frame = np.full((height, width), 0, dtype=np.uint8)
        damaged_pixels = np.random.choice(height * width, damaged_pixel_count, replace=False)
        damaged_coords = np.unravel_index(damaged_pixels, (height, width))

        frame[damaged_coords] = 255

        frames.append(frame)

    return frames


def create_clustered_test_video(num_frames = 100, width = 928, height = 576,
                                cluster_count = 50, cluster_size_range = (10, 20),
                                background_intensity = 0):
    """
    creates test video comprising of small clusters of damaged pixels
    in order to test large damaged pixel region filtering
    """

    frames = []
    cluster_pixel_count_records = []

    for _ in range(num_frames):
        frame = np.full((height, width), background_intensity, dtype = np.uint8)
        total_damaged_pixels = 0
        occupied_pixels = set()
        cluster_centers = []

        for _ in range(cluster_count):
            cluster_size = random.randint(*cluster_size_range)
            cluster_pixels = set()
            overlap_detected = True #avoids damaged pixels being placed in the
                # same place twice to avoid double counting
            tries = 0

            while overlap_detected and tries < 10:
                cluster_center_x = random.randint(0, width - 1)
                cluster_center_y = random.randint(0, height - 1)

                overlap_detected = any(abs(cluster_center_x - cx) < 20 and abs(cluster_center_y -
                    cy) < 20 for cx, cy in cluster_centers)

                if not overlap_detected:
                    cluster_centers.append((cluster_center_x, cluster_center_y))

                    placed_pixels = 0
                    cluster_pixels.clear()

                    while placed_pixels < cluster_size:
                        dx = random.randint(-3, 3)
                        dy = random.randint(-3, 3)

                        x = np.clip(cluster_center_x + dx, 0, width - 1)
                        y = np.clip(cluster_center_y + dy, 0, height - 1)

                        if (x, y) not in occupied_pixels:
                            cluster_pixels.add((x, y))
                            placed_pixels += 1

                tries += 1

            if not overlap_detected:
                for x, y in cluster_pixels:
                    frame[y, x] = 255
                    occupied_pixels.add((x, y))

                total_damaged_pixels += len(cluster_pixels)

        frames.append(frame)
        cluster_pixel_count_records.append(total_damaged_pixels)

    return frames, cluster_pixel_count_records


def create_temporal_test_video(num_frames, width, height, damaged_pixel_count=100, duration = 5,
                               background_intensity = 0):
    
    frames = [np.full((height, width), background_intensity, dtype = np.uint8)
               for _ in range(num_frames)]
    
    #randomly choose damaged pixek coordinates
    total_pixels = height * width
    chosen = np.random.choice(total_pixels, size = damaged_pixel_count, replace = False)
    ys, xs = np.unravel_index(chosen, (height, width))
    coords = list(zip(ys, xs))

    start_frame = random.randint(0, max(0, num_frames - duration))
    end_frame = start_frame + duration

    damage_schedule = [0] * num_frames

    for f in range(start_frame, end_frame):
        for y, x in coords:
            frames[f][y, x] = 255
        damage_schedule[f] = damaged_pixel_count

    return frames, damage_schedule