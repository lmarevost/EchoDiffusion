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


def run_subset(i_subset, ds_real, ds_fake, n_real, n_fake, logger, batch_size, num_workers=8):
    subset_start_time = time.time()

    # random subset of real dataset
    if n_real < len(ds_real):
        subset_indices_real = random.sample(range(len(ds_real)), n_real)
        ds_real = Subset(ds_real, subset_indices_real)

    # random subset of fake dataset
    if n_fake < len(ds_fake):
        subset_indices_fake = random.sample(range(len(ds_fake)), n_fake)
        ds_fake = Subset(ds_fake, subset_indices_fake)

    dm_real = DataLoader(ds_real, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False)
    dm_fake = DataLoader(ds_fake, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False)

    # ------------- Init Metrics ----------------------
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    calc_fid = FID().to(device) # requires uint8
    calc_pr = ImprovedPrecessionRecall(splits_real=1, splits_fake=1).to(device)

    # --------------- Start Calculation -----------------
    for real_batch in tqdm(dm_real):
        imgs_real_batch = real_batch[0].to(device)

        # -------------- FID -------------------
        calc_fid.update(imgs_real_batch, real=True)

        # ------ Improved Precision/Recall--------
        calc_pr.update(imgs_real_batch, real=True)

    for fake_batch in tqdm(dm_fake):
        imgs_fake_batch = fake_batch[0].to(device)

        # -------------- FID -------------------
        calc_fid.update(imgs_fake_batch, real=False)

        # ---- Improved Precision/Recall--------
        calc_pr.update(imgs_fake_batch, real=False)

    # -------------- Summary -------------------
    fid = calc_fid.compute()
    precision, recall = calc_pr.compute()
    logger.info(f"\nSubset {i_subset}")
    logger.info(f"FID: {fid:.2f}")
    logger.info(f"Precision: {precision:.2f}")
    logger.info(f"Recall {recall:.2f}")
    logger.info(f"Subset time: {get_elapsed_time(subset_start_time)}")
    
    return fid.cpu(), precision.cpu(), recall.cpu()


def main():

    start_time = time.time()

    # ---------------- Arg Parser --------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--real_data', type=str, required=True, help='Path to real dataset.')
    parser.add_argument('--fake_data', type=str, required=True, help='Path to fake dataset.')
    parser.add_argument('--n_real', type=int, required=True, help='Subsample size of real dataset.')
    parser.add_argument('--n_fake', type=int, default=None, help='Subsample size of fake dataset (optional).')
    parser.add_argument('--n_runs', type=int, default=5, help='Number of independent runs (subsets) to evaluate.')
    parser.add_argument('--log_path', type=str, required=True, help='Path to save logs.')
    args = parser.parse_args()

    real_path = args.real_data
    fake_path = args.fake_data
    n_real = args.n_real
    n_fake = args.n_fake
    n_runs = args.n_runs

    # if no size is provided set size of fake subsample same as the real
    if n_fake in (0, None):
        n_fake = n_real

    # ----------------Settings --------------
    batch_size = 32
    path_out = Path(args.log_path)
    path_out.mkdir(parents=True, exist_ok=True)


    # ----------------- Logging -----------
    current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO)
    logger.addHandler(logging.FileHandler(path_out/f'metrics_{current_time}.log', 'w'))

    # -------------- Helpers ---------------------
    pil2torch = lambda x: torch.as_tensor(np.array(x)).moveaxis(-1, 0) # In contrast to ToTensor(), this will not cast 0-255 to 0-1 and destroy uint8 (required later)

    # ---------------- Dataset/Dataloader ----------------
    ds_real = ImageFolder(real_path, transform=pil2torch)
    ds_fake = ImageFolder(fake_path, transform=pil2torch)

    logger.info(f'FID EVALUATION')
    logger.info(f'\nReal dataset: {real_path}')
    logger.info(f"Samples evaluated: {min(n_real, len(ds_real)):,}")
    logger.info(f'\nFake dataset: {fake_path}')
    logger.info(f"Samples evaluated: {min(n_fake, len(ds_fake)):,}")
    logger.info(f"\nResults saved to:\n{path_out/f'metrics_{current_time}.log'}")
    logger.info(f"\n{'-'*100}")

    # ---------------- Metric Lists ----------------
    fid_list = []
    precision_list = []
    recall_list = []    
    
    # Running subsets
    for i_subset in range(n_runs):
        fid, precision, recall = run_subset(i_subset+1, ds_real, ds_fake, n_real, n_fake, logger, batch_size)
        fid_list.append(fid)
        precision_list.append(precision)
        recall_list.append(recall)
    
    # mean fid, p&r
    logger.info(f"\n{'-'*100}")
    logger.info(f'\nMean FID: {np.mean(fid_list):.2f} \u00B1 {np.std(fid_list):.2f}')
    logger.info(f'Mean Precision: {np.mean(precision_list):.2f} \u00B1 {np.std(precision_list):.2f}')
    logger.info(f'Mean Recall: {np.mean(recall_list):.2f} \u00B1 {np.std(recall_list):.2f}')
    logger.info(f'Total run time: {get_elapsed_time(start_time)}')

if __name__ == "__main__":
    # run time reference: A100:1 -> 10k vs 10k images took 1 min
    main()
