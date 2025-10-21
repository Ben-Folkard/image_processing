#!/bin/bash

#SBATCH -p scarf
#SBATCH -n 32
#SBATCH -t 24:00:00
#SBATCH -o results/%J.log
#SBATCH -e results/%J.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=Ben.Folkard@stfc.ac.uk

/home/vol03/scarf1493/myenv/bin/python3 image_processing_old.py --video_filename 11_01_H_170726081325.avi

# mailx -s "SLURM job $SLURM_JOB_ID finished on $HOSTNAME" \
#	-a results/${SLURM_JOB_ID}.log \ 
#	-a results/${SLURM_JOB_ID}.err \
#	Ben.Folkard@stfc.ac.uk <<< "Job $SLURM_JOB_ID completed on $HOSTNAME. See attached logs."
