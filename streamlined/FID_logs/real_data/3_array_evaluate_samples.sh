#!/bin/env bash

#SBATCH -A C3SE512-22-1         # find your project with the "projinfo" command
#SBATCH -p alvis                # what partition to use (usually not necessary)
#SBATCH -t 0-00:10:00           # how long time it will take to run
#SBATCH --gpus-per-node=A40:1  # choosing no. GPUs and their type
#SBATCH -J eval_diff            # the jobname (not necessary)
#SBATCH --array=0-5             # array job


# Make sure to remove any already loaded modules
module purge

# =========================================================================================
# UPDATE PARAMETERS:
# =========================================================================================

# Create a list of parameters for array job
PARAMS=(1000 2000 4000 8000 16000 full)

# Fetch one param from the list based on the task ID (index starts from 0)
CURRENT_PARAM=${PARAMS[$SLURM_ARRAY_TASK_ID]}

# Image dataset directories
REAL_DATA="/mimer/NOBACKUP/priv/chair/echo_anouka_luismi/data/final/val"
FAKE_DATA="/mimer/NOBACKUP/priv/chair/echo_anouka_luismi/data/final/train_$CURRENT_PARAM"

# Subsample sizes
N_REAL=10000  # subsample size of real dataset
N_FAKE=1000  # subsample size of fake dataset (optional. zero means same as real)

# Subsample sizes
N_RUNS=10  # number of independent runs (subsets) to evaluate

# Logging path
LOG_PATH="/mimer/NOBACKUP/priv/chair/echo_anouka_luismi/streamlined/FID_logs/real_data/train_"$CURRENT_PARAM"_val"


# =========================================================================================
# FIXED PARAMETERS. DO NOT EDIT ANYTHING AFTER THIS POINT.
# =========================================================================================

# Specify the path to the container and script
CONTAINER=/mimer/NOBACKUP/priv/chair/echo_anouka_luismi/streamlined/full_container.sif
SCRIPT=/mimer/NOBACKUP/priv/chair/echo_anouka_luismi/streamlined/p3_evaluate_samples.py

# Run
apptainer exec $CONTAINER python $SCRIPT --real_data $REAL_DATA --fake_data $FAKE_DATA --n_real $N_REAL --n_fake $N_FAKE --n_runs $N_RUNS --log_path $LOG_PATH

# run command: sbatch --array=0-5 3_evaluate_samples.sh