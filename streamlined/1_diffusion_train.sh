#!/bin/env bash

#SBATCH -A C3SE512-22-1             # find your project with the "projinfo" command
#SBATCH -p alvis                    # partition
#SBATCH -t 0-10:00:00               # how long time it will take to run
#SBATCH --gpus-per-node=A100:4      # choosing no. GPUs and their type
#SBATCH -J diffusion_train          # the jobname (not necessary)

# Make sure to remove any already loaded modules
module purge 

# =========================================================================================
# UPDATE PARAMETERS:
# =========================================================================================

# Create a list of parameters for array job
PARAMS=(1000 2000 4000 8000 16000 full)

# Fetch one param from the list based on the task ID (index starts from 0)
CURRENT_PARAM=${PARAMS[$SLURM_ARRAY_TASK_ID]}

# Dataset
DATA="/mimer/NOBACKUP/priv/chair/echo_anouka_luismi/data/final/train_$CURRENT_PARAM"
NUM_CLASSES=4

# Hyperparameters
MODEL_FLAGS="--image_size 112 --num_channels 128 --num_res_blocks 2 --class_cond True --learn_sigma True"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear"
TRAIN_FLAGS="--lr 1e-4 --batch_size 32"

# Save folder
export OPENAI_LOGDIR="/mimer/NOBACKUP/priv/chair/echo_anouka_luismi/streamlined/saved_diffusion/train_$CURRENT_PARAM"


# =========================================================================================
# FIXED PARAMETERS. DO NOT EDIT ANYTHING AFTER THIS POINT.
# =========================================================================================

# MPI
NUM_GPUS="$(cut -d':' -f2 <<<$SLURM_GPUS_PER_NODE)"

# Specify the path to the container and script
CONTAINER=/mimer/NOBACKUP/priv/chair/echo_anouka_luismi/streamlined/full_container.sif
SCRIPT=/mimer/NOBACKUP/priv/chair/echo_anouka_luismi/streamlined/p1_diffusion_train.py

# Run
apptainer exec $CONTAINER mpiexec -n $NUM_GPUS python $SCRIPT --data_dir $DATA --num_classes $NUM_CLASSES $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS

# run command: sbatch --array=0-5 1_diffusion_train.sh