#!/bin/env bash

#SBATCH -A C3SE512-22-1         # find your project with the "projinfo" command
#SBATCH -p alvis                # what partition to use (usually not necessary)
#SBATCH -t 0-00:01:00           # how long time it will take to run
#SBATCH -C NOGPU                # run on cpu
#SBATCH -J cpu_job              # the jobname (not necessary)


# Specify the path to the container and script
CONTAINER=/mimer/NOBACKUP/priv/chair/echo_anouka_luismi/streamlined/full_container.sif
SCRIPT=/mimer/NOBACKUP/priv/chair/echo_anouka_luismi/streamlined/p8_compare_augmentations.py

# Run
apptainer exec $CONTAINER python $SCRIPT
