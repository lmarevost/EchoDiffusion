#!/bin/env bash

#SBATCH -A C3SE512-22-1             # find your project with the "projinfo" command
#SBATCH -p alvis                    # what partition to use (usually not necessary)
#SBATCH -t 0-03:00:00               # how long time it will take to run
#SBATCH --gpus-per-node=A100:4      # choosing no. GPUs and their type
#SBATCH -J sampling                 # the jobname (not necessary)

# Make sure to remove any already loaded modules
module purge

# =========================================================================================
# UPDATE PARAMETERS:
# =========================================================================================

# Create a list of parameters for array job
PARAMS=(1000 2000 4000 8000 16000 full)

# Fetch one param from the list based on the task ID (index starts from 0)
CURRENT_PARAM=${PARAMS[$SLURM_ARRAY_TASK_ID]}

# Number of samples to generate
NUM_SAMPLES=32000

# Number of classes the model was trained with
NUM_CLASSES=4

# Hyperparameters
MODEL_FLAGS="--image_size 112 --num_channels 128 --num_res_blocks 2 --class_cond True --learn_sigma True"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear" #--timestep_respacing ddim250 --use_ddim True"

# Path to model checkpoint
MODEL_PATH="/mimer/NOBACKUP/priv/chair/echo_anouka_luismi/streamlined/saved_diffusion/train_$CURRENT_PARAM"

# IMPORTANT: Save folder (careful not to override previous results)
export OPENAI_LOGDIR="/mimer/NOBACKUP/priv/chair/echo_anouka_luismi/streamlined/synthetic_samples/train_$CURRENT_PARAM"


# =========================================================================================
# FIXED PARAMETERS. DO NOT EDIT ANYTHING AFTER THIS POINT.
# =========================================================================================

# MPI
NUM_GPUS="$(cut -d':' -f2 <<<$SLURM_GPUS_PER_NODE)"

# Specify the path to the container and script
CONTAINER=/mimer/NOBACKUP/priv/chair/echo_anouka_luismi/streamlined/full_container.sif
SCRIPT=/mimer/NOBACKUP/priv/chair/echo_anouka_luismi/streamlined/p2_image_sample.py

# Run
apptainer exec $CONTAINER mpiexec -n $NUM_GPUS python $SCRIPT --model_path $MODEL_PATH --num_samples $NUM_SAMPLES --num_classes $NUM_CLASSES $MODEL_FLAGS $DIFFUSION_FLAGS --clip_denoised False

# run command: sbatch --array=0-5 2_image_sample.sh