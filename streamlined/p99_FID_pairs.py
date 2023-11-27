import os
import argparse
import random
from pathlib import Path 
import logging
import time
from datetime import datetime
from tqdm import tqdm

import numpy as np 
import torch
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import ImageFolder
from torch.utils.data import TensorDataset, Subset
from torchmetrics.image.fid import FrechetInceptionDistance as FID
from torchmetrics.image.inception import InceptionScore as IS

from medical_diffusion.metrics.torchmetrics_pr_recall import ImprovedPrecessionRecall

# Approx. time taken to evaluate:
# 5k = 2h40m
# 10k = 5h15m
# 20k = 10h30m 
# 50k = 26h15m

def log(n_block, real_filenames, fake_filenames, fid, precision, recall, log_file):
    with open(log_file, 'a') as f:
        f.write(f'Block {str(n_block).zfill(4)} FID {fid} Precision {precision} Recall {recall}\n')
        f.write(f"Real: {' '.join(real_filenames)}\n")
        f.write(f"Fake: {' '.join(fake_filenames)}\n")
        f.write('\n')


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


def get_filenames(ds_full, indices):
    # get the file paths for the images at the specified indices
    file_paths = [ds_full.imgs[i][0] for i in indices]
    
    # extract the base names of the files from the file paths
    base_names = [os.path.basename(fpath) for fpath in file_paths]
    
    return base_names



def eval_subsample(n_block, ds_real, ds_fake, n_real, n_fake, path_out, batch_size, num_workers):

    # random subset of real dataset
    subset_indices_real = random.sample(range(len(ds_real)), n_real)
    real_filenames = get_filenames(ds_real, subset_indices_real)
    ds_real = Subset(ds_real, subset_indices_real)

    # random subset of fake dataset
    subset_indices_fake = random.sample(range(len(ds_fake)), n_fake)
    fake_filenames = get_filenames(ds_fake, subset_indices_fake)
    ds_fake = Subset(ds_fake, subset_indices_fake)

    dm_real = DataLoader(ds_real, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False)
    dm_fake = DataLoader(ds_fake, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False)

    # ------------- Init Metrics ----------------------
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    calc_fid = FID().to(device) # requires uint8
    # calc_is = IS(splits=1).to(device) # requires uint8, features must be 1008 see https://github.com/openai/guided-diffusion/blob/22e0df8183507e13a7813f8d38d51b072ca1e67c/evaluations/evaluator.py#L603 
    calc_pr = ImprovedPrecessionRecall(splits_real=1, splits_fake=1).to(device)

    # --------------- Start Calculation -----------------
    for real_batch in dm_real:
        imgs_real_batch = real_batch[0].to(device)

        # -------------- FID -------------------
        calc_fid.update(imgs_real_batch, real=True)

        # ------ Improved Precision/Recall--------
        calc_pr.update(imgs_real_batch, real=True)

    for fake_batch in dm_fake:
        imgs_fake_batch = fake_batch[0].to(device)

        # -------------- FID -------------------
        calc_fid.update(imgs_fake_batch, real=False)

        # -------------- IS -------------------
        # calc_is.update(imgs_fake_batch)

        # ---- Improved Precision/Recall--------
        calc_pr.update(imgs_fake_batch, real=False)

    # -------------- Summary -------------------
    fid = calc_fid.compute()
    precision, recall = calc_pr.compute()
    log(n_block, real_filenames, fake_filenames, fid, precision, recall, path_out)
    print(f'Logged block {n_block} to {path_out}.')


def main():
    start_time = time.time()

    # Arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--real_data', type=str, required=True, help='Real dataset')
    parser.add_argument('--fake_data', type=str, required=True, help='Fake dataset')
    parser.add_argument('--n_real', type=int, required=True, help='Subsample size of real dataset')
    parser.add_argument('--n_fake', type=int, default=None, help='Subsample size of fake dataset (optional)')
    parser.add_argument('--log_dir', type=str, required=True, help='Directory to save logs.')
    args = parser.parse_args()
    if args.n_fake in (0, None):
        # set subsample of fake dataset to same size as the real, if no size is provided
        args.n_fake = args.n_real

    # ---------------- Logging --------------
    current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    path_out = Path(args.log_dir)/f'FID_{current_time}.log'
    
    # ----------------Settings --------------
    batch_size = 100
    num_workers = 8

    # -------------- Helpers ---------------------
    pil2torch = lambda x: torch.as_tensor(np.array(x)).moveaxis(-1, 0) # In contrast to ToTensor(), this will not cast 0-255 to 0-1 and destroy uint8 (required later)

    # ---------------- Dataset/Dataloader ----------------
    real_path = args.real_data
    fake_path = args.fake_data
    n_real = args.n_real
    n_fake = args.n_fake

    print(f'\nEVALUATED SAMPLES FROM DATASETS:\nReal: {real_path}\nFake: {fake_path}')
    print(f"Logging to {path_out}")
    print(f"Block size: {batch_size} real-fake pairs\n")

    ds_real = ImageFolder(real_path, transform=pil2torch)
    ds_fake = ImageFolder(fake_path, transform=pil2torch)

    # WHILE TRUE KEEP RUNNING eval_subsample, increment n_block
    n_block = 0
    while True:
        n_block += 1
        eval_subsample(n_block, ds_real, ds_fake, n_real, n_fake, path_out, batch_size, num_workers)
        if n_block % 10 == 0:
            print(f'Elapsed time: {get_elapsed_time(start_time)}\n')

    
if __name__ == "__main__":
    main()
