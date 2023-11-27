# Logger
import logging
from pathlib import Path 
# Parser
import argparse
# Data science tools
import numpy as np
import os
import shutil
import random
# Image manipulations
from PIL import Image
# Timing utility
import time
from datetime import datetime
from timeit import default_timer as timer
from tqdm import tqdm
# Visualizations
import matplotlib.pyplot as plt
import seaborn as sns


# --------------------------------------------
# ARG PARSER
# --------------------------------------------
parser = argparse.ArgumentParser()
# dirs
parser.add_argument('--real_dir', type=str, required=True, help='Path to real dataset')
parser.add_argument('--fake_dir', type=str, required=True, help='Path to fake dataset')
parser.add_argument('--log_dir', type=str, required=True, help='Path to logging dir')
args = parser.parse_args()

# set current time to know when script ran
time_now = datetime.now().strftime("%Y_%m_%d_%H%M%S")

# --------------------------------------------
# TRAIN, VAL, SAVE PATH
# --------------------------------------------
# train & val datasets
real_dir = Path(args.real_dir)
fake_dir = Path(args.fake_dir)
log_dir = Path(args.log_dir)
log_dir.mkdir(exist_ok=True)

# --------------------------------------------
# LOGGING
# --------------------------------------------
# save_basename = log_dir/f"{time_now}"  # e.g. 2023_04_23_115719
# logger = logging.getLogger()
# logging.basicConfig(level=logging.INFO)
# logger.addHandler(logging.FileHandler(f'{save_basename}.log', 'w'))
# logger.info(f"Logging to:\n'{save_basename}.log'")

# --------------------------------------------
# SETTING HYPERPARAMETERS
# --------------------------------------------
subset_size = 1000


def get_elapsed_time(start_time):
    # Elapsed_time in seconds
    elapsed_time = time.time() - start_time
    # Convert time in seconds to days, hours, minutes, and seconds
    days, remainder = divmod(elapsed_time, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    # Format the time as a string in d:hh:mm:ss format
    time_string = f"{days:,.0f}d {hours:02.0f}h {minutes:02.0f}m {seconds:02.0f}s"
    return time_string


def get_image_vectors(root_dir, k):
    VIEWS = ['A2C', 'A4C', 'PLAX', 'PSAX']
    filepaths = []
    for view in VIEWS:
        view_dir = root_dir/view
        filepaths.extend([view_dir/f for f in os.listdir(view_dir)])
    selected_filepaths = random.sample(filepaths, k=k)
    images = [Image.open(f).convert('L') for f in selected_filepaths]
    arrs = [np.array(f.getdata()) for f in images]
    arrs = np.stack(arrs, axis=0)
    return arrs, selected_filepaths


def compute_cosine_similarities(X, Y):
    # Normalize X and Y
    X_normalized = X / np.linalg.norm(X, axis=1, keepdims=True)
    Y_normalized = Y / np.linalg.norm(Y, axis=1, keepdims=True)

    # Compute cosine similarities using matrix multiplication
    cosine_similarities = np.dot(X_normalized, Y_normalized.T)
    
    return cosine_similarities


def get_highest_indices(matrix, n=5):
    # Flatten the matrix and find the indices of the highest n elements
    flattened_indices = np.argpartition(matrix.flatten(), -n)[-n:]

    # Convert flattened indices to row, col indices
    indices = np.unravel_index(flattened_indices, matrix.shape)

    return [(a, b) for a, b in zip(*indices)]


real_vectors, real_paths = get_image_vectors(real_dir, subset_size)
fake_vectors, fake_paths = get_image_vectors(fake_dir, subset_size)
cs_matrix = compute_cosine_similarities(real_vectors, fake_vectors)  # rows: real, cols: synth
indices = get_highest_indices(cs_matrix, 5)  # get the top 5 most similar real-fake image pairs
highest_cs = [cs_matrix[i, j] for i, j in indices]  # i-th real and j-th fake images


def compute_rmse_matrix(X, Y):
    # Compute the squared differences between X and Y using broadcasting
    squared_diff = np.square(X[:, np.newaxis, :] - Y)

    # Compute the mean of the squared differences along the last axis (columns)
    mean_squared_diff = np.mean(squared_diff, axis=-1)

    # Compute the RMSE by taking the square root of the mean squared difference
    rmse = np.sqrt(mean_squared_diff)

    return rmse


# dst_dir = log_dir/"similar_images"
# dst_dir.mkdir(exist_ok=True)
# for n, (i, j) in enumerate(indices):
#     shutil.copy(real_paths[i], dst_dir/f'{n}r.png')
#     shutil.copy(fake_paths[j], dst_dir/f'{n}f.png')

# sns.histplot(x=cs_matrix.max(axis=0))  # max similarity for each fake image
# plt.savefig(log_dir/'hist_synth.png')
# plt.close()

# sns.histplot(x=cs_matrix.max(axis=1))  # max similarity for each real image
# plt.savefig(log_dir/'hist_real.png')
# plt.close()


def plot():
    PARAMS=("1000", "2000", "4000", "8000", "16000", "full")
    names = ['1k', '2k', '4k', '8k', '16k', '32k']
    m = 3
    for func in compute_cosine_similarities, compute_rmse_matrix:
        fname = 'Cosine similarity' if func is compute_cosine_similarities else 'RMSE'
        print(fname)
        fig, axes = plt.subplots(6, 2, figsize=(6.4*m, 4.8*m))
        for i, (x, name) in enumerate(zip(PARAMS, names)):
            print(x)
            real_dir=Path(f"/mimer/NOBACKUP/priv/chair/echo_anouka_luismi/data/final/train_{x}")
            fake_dir=Path(f"/mimer/NOBACKUP/priv/chair/echo_anouka_luismi/streamlined/synthetic_samples/train_{x}/PNG")

            real_vectors, _ = get_image_vectors(real_dir, subset_size)
            fake_vectors, _ = get_image_vectors(fake_dir, subset_size)
            metric = func(real_vectors, fake_vectors)  # rows: real, cols: synth
            
            # fake
            ax = axes[i, 0]
            ax.set_title(f'Fake ({name})')
            if fname == 'RMSE':
                sns.histplot(x=metric.min(axis=0), ax=ax)  # max similarity for each fake image
            else:
                sns.histplot(x=metric.max(axis=0), ax=ax)  # max similarity for each fake image

            # real
            ax = axes[i, 1]
            ax.set_title(f'Real ({name})')
            if fname == 'RMSE':
                sns.histplot(x=metric.min(axis=1), ax=ax)  # max similarity for each real image
            else:
                sns.histplot(x=metric.max(axis=1), ax=ax)  # max similarity for each real image

        # saving
        plt.tight_layout()        
        plt.savefig(log_dir/f'{fname}_hist.png')
        print('Saved plot to', log_dir/f'{fname}_hist.png')
        plt.close()


sns.set_theme()
plot()