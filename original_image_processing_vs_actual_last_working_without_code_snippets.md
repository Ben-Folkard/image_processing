# Function Diff `image_processing_original.py` → `image_processing_actual_last_working_backup.py`

### Changed function `_find_background`
Altered _find_background so that numba could be used to run it in parallel.

### Changed function `_generate_plots`
Added flags so that _generate_plots' output is more customisable (i.e. show_plots, save_plots, output_folder, visualise_damaged_pixels_plot, plot_damaged_pixels_plot, and plot_heatmap_plot).

### Changed function `_get_final_count`
Tweaked to get a very slight performance increase from vectorisation.

### Added function `chunked_nanmean`
Added so the code is more readable and so that indicies that didn't get included before as they weren't divided up by the step nicley can now be included.#### Function call changes for `chunked_nanmean`

### Changed function `compute_background`
Fixes a logical bug that caused the neighbours to be found at the radius value for values of index < radius rather than finding the neighbours of the index.

### Changed function `compute_cluster_stats`
It's been parallised (now runs on threads)

### Changed function `compute_optical_flow_metric`
It's been parallised (now runs on threads)

### Changed function `compute_static_mask`
Fixed a bug where, as masks aren't homogenous (i.e. sometimes instead of booleans arrays, there are Nones), the following logical operations made on the numpy arrays couldn't be made.

### Changed function `detect_damaged_pixels`
Added extra flags that could be passed into the plotting functions, converted frames to a numpy array of numpy array more efficiently, converted raw_masks into a numpy array, converted, and restructured if statement in for loop for slightly more efficient parsing given raw_masks now had a default value of None.

#### Function call changes for `detect_damaged_pixels`

### Changed function `filter_damaged_pixel_clusters`
Vectorises label operations, replacing some of the old ones for faster operations, precomputes circularities for valid labels, skips computing contours for small clusters, reshufles the order things are computed to try and skip unnecessary calculations, and returns early if the inputted frame ends up giving invlaid results.

### Changed function `filter_frames_by_optical_flow`
Vectorised the operations using numpy functions.

### Changed function `find_bright_area_estimates`
Removed estimate_damaged_pixels_in_bright_areas from being inside the for loop, meaning it no longer unnecessarily recomputes for every single frame.

### Changed function `find_damaged_pixel_heatmap`
Adds protection against division by 0 errors.

#### Function call changes for `find_damaged_pixel_heatmap`
In order for the logical operations to be completed, masks had to be of the same dimensions as the others (i.e. it couldn't be inhomogeonous, sometimes containing Nones instead of an array of booleans).

### Changed function `get_damaged_pixel_mask`
Completely vectorised the logic by swapping out the blurring logic for an equivalent openCV function.

### Changed function `load_video_frames`
Added a flag to control whether the video is loaded in greyscale or not.

### Function call changes for `main`
Added the ability to parse in variables via running the program through the command line (i.e. video_filename, average_time, max_chunks, step_size, show_plots, save_plots, output_folder, consecutive_threshold, brightness_threshold, flow_threshold, static_threshold, min_cluster_size, max_cluster_size, min_circularity, sliding_window_radius, number_of_plots).

### Changed function `plot_damaged_pixels`
Added flags to decide whether the plot is shown and/or saved, and customise which file location the images are saved (i.e. show_plots, save_plots, and output_folder).

#### Function call changes for `plot_damaged_pixels`

### Changed function `plot_heatmap`
Added flags to decide whether the plot is shown and/or saved, and customise which file location the images are saved (i.e. show_plots, save_plots, and output_folder).

#### Function call changes for `plot_heatmap`

### Changed function `visualise_damaged_pixels`
Added flags to decide whether the plot is shown and/or saved, and customise which file location the images are saved (i.e. show_plots, save_plots, and output_folder).
