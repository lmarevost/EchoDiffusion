#!/bin/env bash

#SBATCH -A C3SE512-22-1         # find your project with the "projinfo" command
#SBATCH -p alvis                # what partition to use (usually not necessary)
#SBATCH -t 0-02:00:00           # how long time it will take to run
#SBATCH --gpus-per-node=A100:1   # choosing no. GPUs and their type
#SBATCH -J cls-p20i1kR         # the jobname (not necessary)
#SBATCH --array=0-5

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
#DATA_DIR="/mimer/NOBACKUP/priv/chair/echo_anouka_luismi/streamlined/synthetic_samples/train_$CURRENT_PARAM"
DATA_DIR="/mimer/NOBACKUP/priv/chair/echo_anouka_luismi/data/final/train_$CURRENT_PARAM"
#DATA_DIR="/mimer/NOBACKUP/priv/chair/echo_anouka_luismi/data/final/train_1000"  # keep validation set fixed for fair comparisons
#DATA_DIR="/mimer/NOBACKUP/priv/chair/echo_anouka_luismi/streamlined/synthetic_samples/train_1000"

# Model flags
MODEL_CHECKPOINT=/mimer/NOBACKUP/priv/chair/echo_anouka_luismi/streamlined/saved_classifiers/real_data/train_full_val/VGG16_FF_2023_05_11_221428.pt
#                /mimer/NOBACKUP/priv/chair/echo_anouka_luismi/streamlined/saved_classifiers/real_data/train_full_val/VGG16_FF_2023_05_06_094445.pt
MODEL_NAME='VGG16'
#MODEL_NAME='ResNet18'
#MODEL_NAME='DenseNet121'
#MODEL_NAME='InceptionV3'
PRETRAINED=False
FREEZE=False

# Hyperparameters
BATCH_SIZE=32


# =========================================================================================
# FIXED PARAMETERS. DO NOT EDIT ANYTHING AFTER THIS POINT.
# =========================================================================================

# Specify the path to the container and script
CONTAINER=/mimer/NOBACKUP/priv/chair/echo_anouka_luismi/streamlined/full_container.sif
SCRIPT=/mimer/NOBACKUP/priv/chair/echo_anouka_luismi/streamlined/p9_cluster_viz.py

# Flags
DIR_FLAGS="--data_dir $DATA_DIR"
MODEL_FLAGS="--model_checkpoint $MODEL_CHECKPOINT --pretrained $PRETRAINED --freeze $FREEZE"
HYPERPARAMS="--batch_size $BATCH_SIZE"

# Run
apptainer exec $CONTAINER python $SCRIPT $DIR_FLAGS $MODEL_FLAGS $HYPERPARAMS