#!/bin/bash

#SBATCH -p scarf
#SBATCH -n 100
#SBATCH -t 48:00:00
#SBATCH -o %J.log
#SBATCH -e %J.err

module load OpenCV/4.8.1-foss-2023a-contrib
module load numba/0.58.1-foss-2023a

python image_processing_video_creator.py 11_01_H_170726081325.avi

