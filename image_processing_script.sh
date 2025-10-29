#!/bin/bash

#SBATCH -p scarf
#SBATCH -n 32
#SBATCH -t 24:00:00
#SBATCH -o results/%J.log
#SBATCH -e results/%J.err

# Default input values
filename="image_processing.py"
video_filename="11_01_H_170726081325.avi"
average_time=1.0
max_chunks=2
step_size=1000
plot="False"
show_plots="False"
save_plots="False"
consecutive_threshold=2
brightness_threshold=170
flow_threshold=2.0
static_threshold=50
min_cluster_size=5
max_cluster_size=20
min_circularity=0.1
sliding_window_radius=3
number_of_plots=20

if [ -n "$SLURM_JOB_ID" ]; then
	output_folder="results/job_${SLURM_JOB_ID}"
else
	output_folder="results/job_manual"
fi

# Parsing keyword arguments
while [[ "$#" -gt 0 ]]; do
	case $1 in
		--filename) filename="$2"; shift ;;
		-f) filename="$2"; shift ;;
		--video_filename) video_filename="$2"; shift ;;
		-vf) video_filename="$2"; shift ;;
		--average_time) average_time="$2"; shift ;;
		-at) average_time="$2"; shift ;;
		--max_chunks) max_chunks="$2"; shift ;;
		-mc) max_chunks="$2"; shift ;;
		--step_size) step_size="$2"; shift ;;
		-ss) step_size="$2"; shift ;;
		--plot) plot="$2"; shift ;;
		-p) plot="$2"; shift ;;
		--show_plots) show_plots="$2"; shift ;;
		-shp) show_plots="$2"; shift ;;
		--save_plots) save_plots="$2"; shift ;;
		-svp) save_plots="$2"; shift ;;
		--output_folder) output_folder="$2"; shift ;;
		-of) output_folder="$2"; shift ;;
		--consecutive_threshold) consecutive_threshold="$2"; shift ;;
		-ct) consecutive_threshold="$2"; shift ;;
		--brightness_threshold) brightness_threshold="$2"; shift ;;
		-bt) brightness_threshold="$2"; shift ;;
		--flow_threshold) flow_threshold="$2"; shift ;;
		-ft) flow_threshold="$2"; shift ;;
		--static_threshold) static_threshold="$2"; shift ;;
		-st) static_threshold="$2"; shift ;;
		--min_cluster_size) min_cluster_size="$2"; shift ;;
		-mncs) min_cluster_size="$2"; shift ;;
		--max_cluster_size) max_cluster_size="$2"; shift ;;
		-mxcs) max_cluster_size="$2"; shift ;;
		--min_circularity) min_circularity="$2"; shift ;;
		-mnc) min_circularity="$2"; shift ;;
		--sliding_window_radius) sliding_window_radius="$2"; shift ;;
		-swr) sliding_window_radius="$2"; shift ;;
		--number_of_plots) number_of_plots="$2"; shift ;;
		-np) number_of_plots="$2"; shift ;;
		*) echo "Unknown parameter passed: $1"; exit 1;;
	esac
	shift
done

mkdir -p "$output_folder"

echo "Running ${filename} with:"
echo "    video_filename = ${video_filename}"
echo "    average_time = ${average_time}"
echo "    max_chunks = ${max_chunks}"
echo "    step_size = ${step_size}"
echo "    plot = ${plot}"
echo "    show_plots = ${show_plots}"
echo "    save_plots = ${save_plots}"
echo "    output_folder = ${output_folder}"
echo "    consecutive_threshold = ${consecutive_threshold}"
echo "    brightness_threshold = ${brightness_threshold}"
echo "    flow_threshold = ${flow_threshold}"
echo "    static_threshold = ${static_threshold}"
echo "    min_cluster_size = ${min_cluster_size}"
echo "    max_cluster_size = ${max_cluster_size}"
echo "    min_circularity = ${min_circularity}"
echo "    sliding_window_radius = ${sliding_window_radius}"
echo "    number_of_plots = ${number_of_plots}"


/home/vol03/scarf1493/myenv/bin/python3 "$filename" --video_filename "$video_filename" --average_time "$average_time" --max_chunks "$max_chunks" --step_size "$step_size" --plot "$plot" --show_plots "$show_plots" --save_plots "$save_plots" --output_folder "$output_folder" --consecutive_threshold "$consecutive_threshold" --brightness_threshold "$brightness_threshold" --flow_threshold "$flow_threshold" --static_threshold "$static_threshold" --min_cluster_size "$min_cluster_size" --max_cluster_size "$max_cluster_size" --min_circularity "$min_circularity" --sliding_window_radius "$sliding_window_radius" --number_of_plots "$number_of_plots"

if [ -n "$SLURM_JOB_ID" ]; then
	mv "results/${SLURM_JOB_ID}.log" "$output_folder"
	mv "results/${SLURM_JOB_ID}.err" "$output_folder"
fi