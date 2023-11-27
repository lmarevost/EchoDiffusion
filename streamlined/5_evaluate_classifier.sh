#!/bin/env bash

#SBATCH -A C3SE512-22-1         # find your project with the "projinfo" command
#SBATCH -p alvis                # what partition to use (usually not necessary)
#SBATCH -t 0-00:20:00           # how long time it will take to run
#SBATCH --gpus-per-node=A40:1   # choosing no. GPUs and their type
#SBATCH -J e_s32k1k           # the jobname (not necessary)

# Make sure to remove any already loaded modules
module purge

# =========================================================================================
# UPDATE PARAMETERS:
# =========================================================================================

# Image dataset directories
VAL_DIR="/mimer/NOBACKUP/priv/chair/echo_anouka_luismi/data/final/val"  # keep validation set fixed for fair comparisons
LOG_DIR=/mimer/NOBACKUP/priv/chair/echo_anouka_luismi/streamlined/saved_classifiers/synth_data/train_1000_val

# Model flags
MODEL_CHECKPOINT=/mimer/NOBACKUP/priv/chair/echo_anouka_luismi/streamlined/saved_classifiers/synth_data/train_1000_val/VGG16_FF_2023_05_11_220742.pt

# Hyperparameters
BATCH_SIZE=32


# =========================================================================================
# FIXED PARAMETERS. DO NOT EDIT ANYTHING AFTER THIS POINT.
# =========================================================================================

# Specify the path to the container and script
CONTAINER=/mimer/NOBACKUP/priv/chair/echo_anouka_luismi/streamlined/full_container.sif
SCRIPT=/mimer/NOBACKUP/priv/chair/echo_anouka_luismi/streamlined/p5_evaluate_classifier.py

# Flags
DIR_FLAGS="--val_dir $VAL_DIR --log_dir $LOG_DIR"
MODEL_FLAGS="--model_name $MODEL_NAME --model_checkpoint $MODEL_CHECKPOINT"
HYPERPARAMS="--batch_size $BATCH_SIZE"

# Run
apptainer exec $CONTAINER python $SCRIPT $DIR_FLAGS --model_checkpoint $MODEL_CHECKPOINT $HYPERPARAMS