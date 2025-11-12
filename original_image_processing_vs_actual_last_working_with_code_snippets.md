# Function Diff `image_processing_original.py` → `image_processing_actual_last_working_backup.py`

### Changed function `_find_background`
Altered _find_background so that numba could be used to run it in parallel.

From:
```python
def _find_background(frames):
    """
    mean background calculation, used in compute_background()
    """
    # find mean/std of brightness of each pixel in sliding window
    pixel_means = frames.mean(axis=0)
    pixel_std = frames.std(axis=0)

    # only take pixels which are not likely to be damaged
    valid = frames <= pixel_means + 1 * pixel_std

    # find background, excluding unusually bright pixels
    masked = np.where(valid, frames, np.nan)
    bg = np.nanmean(masked, axis=0)

    if np.isnan(bg).any():
        bg = np.nan_to_num(bg, nan=frames.mean(axis=0))

    return bg
```
To:
```python
@njit(parallel=True, fastmath=True)
def _find_background(frames, pixel_std_coeff=1.0):
    """
    Numba-accelerated version of the NumPy background estimator.
    Matches np.nanmean(np.where(frames <= mean + std, frames, nan), axis=0)
    with NaN fallback handling.
    """
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
                # faithfully match np.nan_to_num(nanmean(masked)) fallback
                s_all = 0.0
                for k in range(n):
                    s_all += frames[k, i, j]
                bg[i, j] = s_all / n

    return bg
```

#### Function call changes for `_find_background`

From:
```python
 136: return _find_background(np.stack(neighbours))
```
To:
```python
 198: return _find_background(neighbours)
```

### Changed function `_generate_plots`
Added flags so that _generate_plots' output is more customisable (i.e. show_plots, save_plots, output_folder, visualise_damaged_pixels_plot, plot_damaged_pixels_plot, and plot_heatmap_plot).

From:
```python
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
```
To:
```python
def _generate_plots(
        frames,
        masks,
        counts,
        flows,
        settings,
        show_plots=True,
        save_plots=False,
        output_folder="results",
        visualise_damaged_pixels_plot=True,
        plot_damaged_pixels_plot=True,
        plot_heatmap_plot=True,
        ):
    """
    generates plots for user
    """
    if visualise_damaged_pixels_plot:
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

    if plot_damaged_pixels_plot:
        plot_damaged_pixels(
                            counts,
                            show_plots=show_plots,
                            save_plots=save_plots,
                            output_folder=output_folder,
                            )

    if plot_heatmap_plot:
        mask_shape = frames[0].shape
        heatmap = find_damaged_pixel_heatmap(
            frames,
            [m.astype(np.uint8) if m is not None else np.zeros(mask_shape, dtype=bool) for m in masks],
            settings.brightness_threshold,
        )
        plot_heatmap(
                     heatmap,
                     show_plots=show_plots,
                     save_plots=save_plots,
                     output_folder=output_folder,
                     )
```

#### Function call changes for `_generate_plots`

From:
```python
 411: _generate_plots(
          frames,
          final_masks,
          total_counts,
          optical_flows,
          settings)
```
To:
```python
 610: _generate_plots(
          frames,
          final_masks,
          total_counts,
          optical_flows,
          settings,
          show_plots=show_plots,
          save_plots=save_plots,
          visualise_damaged_pixels_plot=visualise_damaged_pixels_plot,
          plot_damaged_pixels_plot=plot_damaged_pixels_plot,
          plot_heatmap_plot=plot_heatmap_plot
      )
```

### Changed function `_get_final_count`
Tweaked to get a very slight performance increase from vectorisation.

From:
```python
def _get_final_count(masks, bright_estimates):
    """
    gets final damaged pixel count, adding in the bright area estimates to
    the raw dark area count
    """
    counts = []

    for mask, estimate in zip(masks, bright_estimates):
        base = int(mask.sum()) if mask is not None else np.nan
        counts.append(base + (estimate if not np.isnan(estimate) else 0))

    return np.array(counts, dtype=float)
```
To:
```python
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
```

### Added function `chunked_nanmean`
Added so the code is more readable and so that indicies that didn't get included before as they weren't divided up by the step nicley can now be included.

```python
def chunked_nanmean(array, step):
    if len(array) == 0:
        return [np.nan]

    # Trim to multiple of step
    n_full = (len(array) // step) * step

    # Compute means on all full chunks
    reshaped = array[:n_full].reshape(-1, step)
    means = np.nanmean(reshaped, axis=1)

    # Includes left over stuff
    remainder = len(array) % step
    if remainder > 0:
        last_chunk_mean = np.nanmean(array[-remainder:])
        means = np.append(means, last_chunk_mean)

    return means.tolist()
```

#### Function call changes for `chunked_nanmean`

From:
```python
899: # find time averages
     averages = [
         np.nanmean(counts[i: i + step])
         for i in range(0, len(counts), step)
     ]
     averages_clusters = [
         np.nanmean(clusters[i: i + step])
         for i in range(0, len(clusters), step)
     ]
     averages_size = [
         np.nanmean(sizes[i: i + step])
         for i in range(0, len(sizes), step)
     ]
     averages_brightness = [
         np.nanmean(brightness[i: i + step])
         for i in range(0, len(brightness), step)
     ]
    
     # get time interval midpoints
     n_windows = len(averages)
     times = ((np.arange(n_windows) * step) + step / 2) / FPS
     
     # return everything in a dict
     return {
         "averages_counts": averages,
         "averages_clusters": averages_clusters,
         "averages_size": averages_size,
         "averages_brightness": averages_brightness,
         "times": times,
     }
```
To:
```python
1158: averages_counts = chunked_nanmean(counts, step)

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
```

### Changed function `compute_background`
Fixes a logical bug that caused the neighbours to be found at the radius value for values of index < radius rather than finding the neighbours of the index.

From:
```python
def compute_background(frames, index, radius):
    """
    estimate background brightness for each pixel of a given frame, based on
    the mean brightness of that pixel in the frames in a sliding window
    (providing the pixel is undamaged in those frames)
    """
    start = max(0, index - radius)
    end = min(len(frames), index + radius + 1)
    neighbours = [f for i, f in enumerate(frames[start:end]) if i != radius]

    return _find_background(np.stack(neighbours))
```
To:
```python
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
```

### Changed function `compute_cluster_stats`
It's been parallised (now runs on threads)

From:
```python
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
        if mask is None or flows[i] > settings.flow_threshold:
            continue
        clean_uint8 = mask.astype(np.uint8)
        _, count, avg_size, avg_brightness = filter_damaged_pixel_clusters(
            frame,
            clean_uint8,
            settings.min_cluster_size,
            settings.max_cluster_size,
            settings.min_circularity
        )

        cluster_count[i], cluster_size[i], cluster_brightness[i] = \
            count, avg_size, avg_brightness

    return cluster_count, cluster_size, cluster_brightness
```
To:
```python
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
```

### Changed function `compute_optical_flow_metric`
It's been parallised (now runs on threads)

From:
```python
def compute_optical_flow_metric(frames):
    """
    computes the average optical flow magnitude between consecutive frames
        using the farneback method.
    assumes frames are grayscale images.
    returns an array of optical flow magnitudes for each frame (first frame is
        assigned 0).
    """
    optical_flows = [0.0]

    for i in range(1, len(frames)):
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
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        optical_flows.append(np.mean(mag))

    return np.array(optical_flows)
```
To:
```python
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
```

### Changed function `compute_static_mask`
Fixed a bug where, as masks aren't homogenous (i.e. sometimes instead of booleans arrays, there are Nones), the following logical operations made on the numpy arrays couldn't be made.

From:
```python
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

    mask_stack = np.stack([m.astype(np.uint8) for m in masks], axis=0)
    frame_stack = np.stack(frames, axis=0)

    heatmap = mask_stack.sum(axis=0)
    bright = (frame_stack > brightness_threshold) & \
        (~mask_stack.astype(bool))
    valid_counts = (~bright).sum(axis=0)

    percentage_map = np.zeros_like(heatmap, dtype=float)
    good = valid_counts > min_valid_frames
    percentage_map[good] = (heatmap[good] / valid_counts[good]) * 100

    return percentage_map > static_threshold
```
To:
```python
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
```

### Changed function `detect_damaged_pixels`
Added extra flags that could be passed into the plotting functions, converted frames to a numpy array of numpy array more efficiently, converted raw_masks into a numpy array, converted, and restructured if statement in for loop for slightly more efficient parsing given raw_masks now had a default value of None.

From:
```python
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
    frames = [np.array(f) for f in frames]

    # optical flow screening
    optical_flows = compute_optical_flow_metric(frames)

    # find damaged pixel mask for each frame
    raw_masks = []
    for i, frame in enumerate(frames):

        if optical_flows[i] > settings.flow_threshold:
            raw_masks.append(None)
            continue

        background = compute_background(
            frames,
            i,
            settings.sliding_window_radius
            )
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
```
To:
```python
def detect_damaged_pixels(
    frames,
    plot=False,
    show_plots=False,
    save_plots=False,
    output_folder="results",
    visualise_damaged_pixels_plot=True,
    plot_damaged_pixels_plot=True,
    plot_heatmap_plot=True,
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
                settings.sliding_window_radius
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
            settings,
            show_plots=show_plots,
            save_plots=save_plots,
            visualise_damaged_pixels_plot=visualise_damaged_pixels_plot,
            plot_damaged_pixels_plot=plot_damaged_pixels_plot,
            plot_heatmap_plot=plot_heatmap_plot
        )

    return total_counts, cluster_counts, avg_sizes, avg_brightnesses
```

#### Function call changes for `detect_damaged_pixels`

From:
```python
 885: counts, clusters, sizes, brightness = detect_damaged_pixels(
          chunk, plot=False)
```
To:
```python
1135: counts, clusters, sizes, brightness = detect_damaged_pixels(
          chunk,
          plot=plot,
          show_plots=show_plots,
          save_plots=save_plots,
          output_folder=output_folder,
          visualise_damaged_pixels_plot=visualise_damaged_pixels_plot,
          plot_damaged_pixels_plot=plot_damaged_pixels_plot,
          plot_heatmap_plot=plot_heatmap_plot,
          params=params
      )
```

### Changed function `filter_damaged_pixel_clusters`
Vectorises label operations, replacing some of the old ones for faster operations, precomputes circularities for valid labels, skips computing contours for small clusters, reshufles the order things are computed to try and skip unnecessary calculations, and returns early if the inputted frame ends up giving invlaid results.

From:
```python
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

    # close gaps (test)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed_mask = cv2.morphologyEx(damaged_pixel_mask.astype(np.uint8),
                                   cv2.MORPH_CLOSE, kernel)

    # isolate groups of damaged pixels
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        closed_mask,
        connectivity=8)

    # prepare outputs
    cleaned_mask = np.zeros_like(damaged_pixel_mask, dtype=bool)
    areas = []
    brightness_sums = []

    # filters clusters of damaged pixels if the area is too large
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

    return cleaned_mask, cluster_count, avg_cluster_size, \
        avg_cluster_brightness
```
To:
```python
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

    # close gaps # Dilation followed by erosion to get rid of noise by bluring
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
                perimeter = cv2.arcLength(contours[0], True)
                if perimeter > 0:
                    circularities[label] = 4 * np.pi * (stats[label, cv2.CC_STAT_AREA] / (perimeter ** 2))

    kept_labels = []
    for label in valid_labels:
        if (
            stats[label, cv2.CC_STAT_AREA] < circularity_size_threshold
            or circularities[label] >= min_circularity
        ):
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
     if cluster_count > 0:
        avg_cluster_size = float(np.mean(areas))
        avg_cluster_brightness = float(np.sum(brightness_sums) / np.sum(areas))
    else:
        avg_cluster_size = 0.0
        avg_cluster_brightness = float('nan')

    return cleaned_mask, cluster_count, avg_cluster_size, avg_cluster_brightness
```

#### Function call changes for `filter_damaged_pixel_clusters`

From:
```python
 230: _, count, avg_size, avg_brightness = filter_damaged_pixel_clusters(
          frame,
          clean_uint8,
          settings.min_cluster_size,
          settings.max_cluster_size,
          settings.min_circularity
      )
```
To:
```python
 334: _, count, size, bright = filter_damaged_pixel_clusters(
          frame, mask, settings.min_cluster_size,
          settings.max_cluster_size, settings.min_circularity
      )
```

### Changed function `filter_frames_by_optical_flow`
Vectorised the operations using numpy functions.

From:
```python
def filter_frames_by_optical_flow(
        frames,
        pixel_counts,
        optical_flows,
        damaged_pixel_masks,
        threshold):
    """
    filters out frames whose optical flow exceeds the given threshold.
    frames with optical flow above the threshold are removed entirely from the
    frames list, and their corresponding pixel counts and masks are discarded.
    """

    removed_counter = 0
    filtered_frames = []
    filtered_counts = []
    filtered_masks = []
    filtered_flows = []

    for frame, count, flow, mask in zip(
        frames,
        pixel_counts,
        optical_flows,
        damaged_pixel_masks
    ):
        if flow > threshold:
            removed_counter += 1

        else:
            filtered_frames.append(frame)
            filtered_counts.append(count)
            filtered_masks.append(mask)
            filtered_flows.append(flow)

    print(f"Removed {removed_counter} frames due to high optical flow")

    return filtered_frames, filtered_counts, filtered_masks, filtered_flows
```
To:
```python
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
```

### Changed function `find_bright_area_estimates`
Removed estimate_damaged_pixels_in_bright_areas from being inside the for loop, meaning it no longer unnecessarily recomputes for every single frame.

From:
```python
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

    for i, (frame, mask) in enumerate(zip(frames, damaged_pixel_masks)):
        if mask is None:
            bright_area_estimates[i] = np.nan
            continue

        high_brightness_mask = (frame > brightness_threshold) & ~mask

        if np.sum(high_brightness_mask) > 0:
            estimate = estimate_damaged_pixels_in_bright_areas(
                frames, damaged_pixel_masks)
            bright_area_estimates[i] = estimate[i]
        else:
            bright_area_estimates[i] = np.nan

    return bright_area_estimates
```
To:
```python
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
```

### Changed function `find_damaged_pixel_heatmap`
Adds protection against division by 0 errors.

From:
```python
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

    mask_stack = np.stack([m.astype(np.uint8) for m in damaged_pixel_masks],
                          axis=0)
    frame_stack = np.stack(frames, axis=0)

    heatmap = mask_stack.sum(axis=0)

    bright_stack = (frame_stack > brightness_threshold) & \
        (~mask_stack.astype(bool))
    valid_counts = (~bright_stack).sum(axis=0)

    result = np.zeros_like(heatmap, dtype=np.float64)
    mask = valid_counts > MIN_VALID_FRAMES
    result[mask] = heatmap[mask] / valid_counts[mask] * 100

    return result
```
To:
```python
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
```

#### Function call changes for `find_damaged_pixel_heatmap`
In order for the logical operations to be completed, masks had to be of the same dimensions as the others (i.e. it couldn't be inhomogeonous, sometimes containing Nones instead of an array of booleans).

From:
```python
 273: heatmap = find_damaged_pixel_heatmap(
          frames,
          [m.astype(np.uint8) for m in masks if m is not None]
      )
```
To:
```python
 419: heatmap = find_damaged_pixel_heatmap(
          frames,
          [m.astype(np.uint8) if m is not None else np.zeros(mask_shape, dtype=bool) for m in masks],
          settings.brightness_threshold,
      )
```

### Changed function `get_damaged_pixel_mask`
Completely vectorised the logic by swapping out the blurring logic for an equivalent openCV function.

From:
```python
def get_damaged_pixel_mask(frame, height, width, background):
    """
    finds damaged pixels for a given frame
    takes background brightness as input, should be an array of brightness
    values corresponding to each pixel in the frame
    """

    damaged_pixels = np.zeros_like(frame, dtype=np.bool_)
    thresholds = np.empty((height, width), dtype=np.float64)

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
```
To:
```python
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
```

### Changed function `load_video_frames`
Added a flag to control whether the video is loaded in greyscale or not.

From:
```python
def load_video_frames(filename, frames_start=None, frames_end=None):
    """
    loads in video frames as greyscale arrays with brightness values ranging
    from 0 to 255 can load in specific chunk of frames from given video
    (given frame start and end values as integers)
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
```
To:
```python
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
```

### Function call changes for `main`
Added the ability to parse in variables via running the program through the command line (i.e. video_filename, average_time, max_chunks, step_size, show_plots, save_plots, output_folder, consecutive_threshold, brightness_threshold, flow_threshold, static_threshold, min_cluster_size, max_cluster_size, min_circularity, sliding_window_radius, number_of_plots).

From:
```python
 932: VIDEO_FILENAME = "11_01_H_170726081325.avi"

      # for a quick test on only 2 chunks:
      results = main(VIDEO_FILENAME, average_time=1.0, max_chunks=2)

      print("counts:", results["averages_counts"])
      print("clusters:", results["averages_clusters"])
      print("sizes:", results["averages_size"])
      print("brightness:", results["averages_brightness"])
      print("times:", results["times"])
```
To:
```python
1177: import argparse
      parser = argparse.ArgumentParser()
      parser.add_argument("-vf", "--video_filename", help="Path to the AVI file")
      parser.add_argument("-at", "--average_time", help="How many seconds to average over in the final summaries")
      parser.add_argument("-mc", "--max_chunks", help="If not None, only process that many chunks (for quick tests)")
      parser.add_argument("-ss", "--step_size", help="The number of steps each chunk is split into")
      # parser.add_argument("-p", "--plot", help="True or False, determines whether the program plots")
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
      show_plots = (args.show_plots.lower() == "true") if args.show_plots else False
      save_plots = (args.save_plots.lower() == "true") if args.save_plots else False
      plot = show_plots or save_plots
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
          visualise_damaged_pixels_plot=False,
          plot_damaged_pixels_plot=True,
          plot_heatmap_plot=True,
          params=params,
      )

      print("counts:", results["averages_counts"])
      print("clusters:", results["averages_clusters"])
      print("sizes:", results["averages_size"])
      print("brightness:", results["averages_brightness"])
      print("times:", results["times"])
```

### Changed function `plot_damaged_pixels`
Added flags to decide whether the plot is shown and/or saved, and customise which file location the images are saved (i.e. show_plots, save_plots, and output_folder).

From:
```python
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
```
To:
```python
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
```

#### Function call changes for `plot_damaged_pixels`

From:
```python
 272: plot_damaged_pixels(counts)
```
To:
```python
 410: plot_damaged_pixels(
                          counts,
                          show_plots=show_plots,
                          save_plots=save_plots,
                          output_folder=output_folder,
                          )
```

### Changed function `plot_heatmap`
Added flags to decide whether the plot is shown and/or saved, and customise which file location the images are saved (i.e. show_plots, save_plots, and output_folder).

From:
```python
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
```
To:
```python
def plot_heatmap(
                 heatmap,
                 title="Damaged Pixel Heatmap",
                 show_plots=True,
                 save_plots=False,
                 output_folder="results",
                 figsize=(15, 10)
                 ):
    """
    plots heatmap showing damaged pixel distribution over every frame
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available - skipping heatmap plot")

    else:
        plt.figure(figsize=figsize)
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
```

#### Function call changes for `plot_heatmap`

From:
```python
 277: plot_heatmap(heatmap)
```
To:
```python
 424: plot_heatmap(
                   heatmap,
                   show_plots=show_plots,
                   save_plots=save_plots,
                   output_folder=output_folder,
                   )
```

### Changed function `visualise_damaged_pixels`
Added flags to decide whether the plot is shown and/or saved, and customise which file location the images are saved (i.e. show_plots, save_plots, and output_folder).

From:
```python
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
        vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # overlay clusters in red
        cluster_overlay = np.zeros_like(vis)
        cluster_overlay[cluster_mask] = (0, 165, 255)
        vis = cv2.addWeighted(vis, 0.8, cluster_overlay, 1.0, 0)

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
```
To:
```python
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
        vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # overlay clusters in red
        cluster_overlay = np.zeros_like(vis)
        cluster_overlay[cluster_mask] = (0, 165, 255)
        vis = cv2.addWeighted(vis, 0.8, cluster_overlay, 1.0, 0)

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
```

#### Function call changes for `visualise_damaged_pixels`

From:
```python
 270: visualise_damaged_pixels(frames[i], masks[i], i)
```
To:
```python
 399: visualise_damaged_pixels(
                               frames[i],
                               i,
                               masks[i],
                               counts[i],
                               show_plots=show_plots,
                               save_plots=save_plots,
                               output_folder=output_folder,
                              )
```
