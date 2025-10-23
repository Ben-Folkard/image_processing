#!/bin/bash

#SBATCH -p scarf
#SBATCH -n 32
#SBATCH -t 24:00:00
#SBATCH -o results/%J.log
#SBATCH -e results/%J.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=Ben.Folkard@stfc.ac.uk

# Default input values
video_filename="11_01_H_170726081325.avi"
average_time=1.0
max_chunks=2
step_size=1000

# Parsing keyword arguments
while [[ "$#" -gt 0 ]]; do
	case $1 in
		--video_filename) video_filename="$2"; shift ;;
		--average_time) average_time="$2"; shift ;;
		--max_chunks) max_chunks="$2"; shift ;;
		--step_size) step_size="$2"; shift ;;
		*) echo "Unknown parameter passed: $1"; exit 1;;
	esac
	shift
done

echo "Running job with:"
echo "    video_filename = $video_filename"
echo "    average_time = $average_time"
echo "    max_chunks = $max_chunks"
echo "    step_size = $step_size"

/home/vol03/scarf1493/myenv/bin/python3 image_processing_with_roll_and_with_nones.py --video_filename "$video_filename" --average_time "$average_time" --max_chunks "$max_chunks" --step_size "$step_size"

# mailx -s "SLURM job $SLURM_JOB_ID finished on $HOSTNAME" \
#	-a results/${SLURM_JOB_ID}.log \ 
#	-a results/${SLURM_JOB_ID}.err \
#	Ben.Folkard@stfc.ac.uk <<< "Job $SLURM_JOB_ID completed on $HOSTNAME. See attached logs."
