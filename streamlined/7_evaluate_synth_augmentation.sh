#!/bin/env bash

#SBATCH -A C3SE512-22-1         # find your project with the "projinfo" command
#SBATCH -p alvis                # what partition to use (usually not necessary)
#SBATCH -t 0-00:10:00           # how long time it will take to run
#SBATCH --gpus-per-node=A100:1  # choosing no. GPUs and their type
#SBATCH -J 10908A               # the jobname (not necessary)

# Make sure to remove any already loaded modules
module purge

# =========================================================================================
# UPDATE PARAMETERS:
# =========================================================================================

# Image dataset directories
REAL_DIR="/mimer/NOBACKUP/priv/chair/echo_anouka_luismi/data/new/4class_balanced_10908/train"
FAKE_DIR="/mimer/NOBACKUP/priv/chair/echo_anouka_luismi/streamlined/synthetic_samples/4class_balanced_10908/PNG"
VAL_DIR="/mimer/NOBACKUP/priv/chair/echo_anouka_luismi/data/new/4class_balanced_10908/val"  # keep validation set fixed for fair comparisons
SAVE_DIR="/mimer/NOBACKUP/priv/chair/echo_anouka_luismi/streamlined/synth_augmentation/4class_balanced_10908"

# Model flags
MODEL_NAME='VGG16'
#MODEL_NAME='ResNet18'
#MODEL_NAME='DenseNet121'
#MODEL_NAME='InceptionV3'
PRETRAINED=False
FREEZE=False
FRAC=0.1  # how much synthetic data to use, as a fraction of real dataset size

# Hyperparameters
EPOCHS=25
BATCH_SIZE=32
LR=0.001  # learning rate


# =========================================================================================
# FIXED PARAMETERS. DO NOT EDIT ANYTHING AFTER THIS POINT.
# =========================================================================================

# Specify the path to the container and script
CONTAINER=/mimer/NOBACKUP/priv/chair/echo_anouka_luismi/streamlined/full_container.sif
SCRIPT=/mimer/NOBACKUP/priv/chair/echo_anouka_luismi/streamlined/p7_evaluate_synth_augmentation.py

# Flags
DIR_FLAGS="--real_dir $REAL_DIR --fake_dir $FAKE_DIR --val_dir $VAL_DIR --save_dir $SAVE_DIR"
MODEL_FLAGS="--model_name $MODEL_NAME --pretrained $PRETRAINED --freeze $FREEZE --frac $FRAC"
HYPERPARAMS="--epochs $EPOCHS --batch_size $BATCH_SIZE --lr $LR"

# Run
apptainer exec $CONTAINER python $SCRIPT $DIR_FLAGS $MODEL_FLAGS $HYPERPARAMS