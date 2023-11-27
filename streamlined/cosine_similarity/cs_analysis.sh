#!/bin/env bash

#SBATCH -A C3SE512-22-1         # find your project with the "projinfo" command
#SBATCH -p alvis                # what partition to use (usually not necessary)
#SBATCH -t 0-00:10:00           # how long time it will take to run
#SBATCH -C NOGPU                # run on cpu
#SBATCH -J cosine_similarity    # the jobname (not necessary)
##SBATCH --array=0-5
##SBATCH --gpus-per-node=A100:1  # choosing no. GPUs and their type

# Make sure to remove any already loaded modules
module purge


# =========================================================================================
# UPDATE PARAMETERS:
# =========================================================================================
# Create a list of parameters for array job
PARAMS=(1000 2000 4000 8000 16000 full)

# Fetch one param from the list based on the task ID (index starts from 0)
CURRENT_PARAM=${PARAMS[$SLURM_ARRAY_TASK_ID]}
CURRENT_PARAM=1000

# Image dataset directories
REAL_DIR="/mimer/NOBACKUP/priv/chair/echo_anouka_luismi/data/final/train_$CURRENT_PARAM"
FAKE_DIR="/mimer/NOBACKUP/priv/chair/echo_anouka_luismi/streamlined/synthetic_samples/train_$CURRENT_PARAM/PNG"
LOG_DIR="/mimer/NOBACKUP/priv/chair/echo_anouka_luismi/streamlined/cosine_similarity/train_$CURRENT_PARAM"


# =========================================================================================
# FIXED PARAMETERS. DO NOT EDIT ANYTHING AFTER THIS POINT.
# =========================================================================================

# Specify the path to the container and script
CONTAINER=/mimer/NOBACKUP/priv/chair/echo_anouka_luismi/streamlined/full_container.sif
SCRIPT=/mimer/NOBACKUP/priv/chair/echo_anouka_luismi/streamlined/cosine_similarity/p_cs_analysis.py

# Flags
DIR_FLAGS="--real_dir $REAL_DIR --fake_dir $FAKE_DIR --log_dir $LOG_DIR"

# Run
apptainer exec $CONTAINER python $SCRIPT $DIR_FLAGS