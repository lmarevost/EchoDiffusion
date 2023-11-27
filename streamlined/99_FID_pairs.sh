#!/bin/env bash

#SBATCH -A C3SE512-22-1         # find your project with the "projinfo" command
#SBATCH -p alvis                # what partition to use (usually not necessary)
#SBATCH -t 0-01:00:00           # how long time it will take to run
#SBATCH --gpus-per-node=A100:1  # choosing no. GPUs and their type
#SBATCH -J FID_cont             # the jobname (not necessary)

# Make sure to remove any already loaded modules
module purge

# =========================================================================================
# UPDATE PARAMETERS:
# =========================================================================================

# Image dataset directories
REAL_DATA="/mimer/NOBACKUP/priv/chair/echo_anouka_luismi/data/processed_split/Dynamic_LVH_TMED2_split_rebalanced_random_4class_Classification/train"
FAKE_DATA="/mimer/NOBACKUP/priv/chair/echo_anouka_luismi/streamlined/saved_models/example_test/A100:4/samples/ema_0.9999_057000/PNG"

# Subsample sizes
N_REAL=100  # subsample size of real dataset
N_FAKE=0  # subsample size of fake dataset (optional. zero means same as real)

# Logging directory
LOG_DIR="/mimer/NOBACKUP/priv/chair/echo_anouka_luismi/streamlined/experiments"


# =========================================================================================
# FIXED PARAMETERS. DO NOT EDIT ANYTHING AFTER THIS POINT.
# =========================================================================================

# Specify the path to the container and script
CONTAINER=/mimer/NOBACKUP/priv/chair/echo_anouka_luismi/streamlined/full_container.sif
SCRIPT=/mimer/NOBACKUP/priv/chair/echo_anouka_luismi/streamlined/p99_FID_pairs.py

# Run
apptainer exec $CONTAINER python $SCRIPT --real_data $REAL_DATA --fake_data $FAKE_DATA --n_real $N_REAL --n_fake $N_FAKE --log_dir $LOG_DIR
