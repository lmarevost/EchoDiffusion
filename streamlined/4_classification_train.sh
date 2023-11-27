#!/bin/env bash

#SBATCH -A C3SE512-22-1         # find your project with the "projinfo" command
#SBATCH -p alvis                # what partition to use (usually not necessary)
#SBATCH -t 0-1:30:00            # how long time it will take to run
#SBATCH --gpus-per-node=A100:1  # choosing no. GPUs and their type
#SBATCH -J tr_upsMax          # the jobname (not necessary)

# Make sure to remove any already loaded modules
module purge

# =========================================================================================
# UPDATE PARAMETERS:
# =========================================================================================

# Create a list of parameters for array job
# PARAMS=(1000 2000 4000 8000 16000 full)

# Fetch one param from the list based on the task ID (index starts from 0)
#CURRENT_PARAM=${PARAMS[$SLURM_ARRAY_TASK_ID]}

# Image dataset directories
#TRAIN_DIR="/mimer/NOBACKUP/priv/chair/echo_anouka_luismi/data/final/train_$CURRENT_PARAM"
TRAIN_DIR="/mimer/NOBACKUP/priv/chair/echo_anouka_luismi/data/final/train_1000"
VAL_DIR="/mimer/NOBACKUP/priv/chair/echo_anouka_luismi/data/final/val"  # keep validation set fixed for fair comparisons
#SAVE_DIR="/mimer/NOBACKUP/priv/chair/echo_anouka_luismi/streamlined/saved_classifiers/real_data/train_"$CURRENT_PARAM"_val"
SAVE_DIR="/mimer/NOBACKUP/priv/chair/echo_anouka_luismi/streamlined/saved_classifiers/real_data/sampler_train_1000_counter"

# Model flags
MODEL_NAME='VGG16'
#MODEL_NAME='ResNet18'
# MODEL_NAME='DenseNet121'
# MODEL_NAME='InceptionV3'
PRETRAINED=False
FREEZE=False

# Hyperparameters
EPOCHS=25
BATCH_SIZE=32
LR=0.001  # learning rate (0.0001 for val_cu3c)


# =========================================================================================
# FIXED PARAMETERS. DO NOT EDIT ANYTHING AFTER THIS POINT.
# =========================================================================================

# Specify the path to the container and script
CONTAINER=/mimer/NOBACKUP/priv/chair/echo_anouka_luismi/streamlined/full_container.sif
# SCRIPT=/mimer/NOBACKUP/priv/chair/echo_anouka_luismi/streamlined/p4_classification_train.py
SCRIPT=/mimer/NOBACKUP/priv/chair/echo_anouka_luismi/streamlined/p4_classification_train_with_sampler.py

# Flags
DIR_FLAGS="--train_dir $TRAIN_DIR --val_dir $VAL_DIR --save_dir $SAVE_DIR"
MODEL_FLAGS="--model_name $MODEL_NAME --pretrained $PRETRAINED --freeze $FREEZE"
HYPERPARAMS="--epochs $EPOCHS --batch_size $BATCH_SIZE --lr $LR"

# Run
apptainer exec $CONTAINER python $SCRIPT $DIR_FLAGS $MODEL_FLAGS $HYPERPARAMS

# run command: sbatch --array=0-5 4_classification_train.sh