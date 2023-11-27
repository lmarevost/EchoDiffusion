"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

from PIL import Image

import time

import numpy as np
import torch as th
import torch.distributed as dist

from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


SAVE_EVERY = 1000  # saves PNGs in batches of 1000 at a time (as opposed to generating all and only then saving as PNGs)


def get_last_checkpoint(ckpt_dir, mode='ema'):
    if mode not in ('ema', 'model'):
        raise ValueError('Invalid mode argument. Must be "ema" or "model".')
    ckpts = [f for f in os.listdir(ckpt_dir) if mode in f and f.endswith('.pt')]
    return os.path.join(ckpt_dir, sorted(ckpts)[-1])
        

def main():

    def gen_batch():
        # Step 1: creating samples
        all_images = []
        all_labels = []
        while len(all_images) * args.batch_size < SAVE_EVERY:
            model_kwargs = {}
            if args.class_cond:
                classes = th.randint(
                    low=0, high=args.num_classes, size=(args.batch_size,), device=dist_util.dev()
                )  # MODIFIED NUM_CLASSES to args.num_classes
                model_kwargs["y"] = classes
            sample_fn = (
                diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
            )
            sample = sample_fn(
                model,
                (args.batch_size, 1, args.image_size, args.image_size), # 1 instead of 3 channels
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )
            sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
            sample = sample.permute(0, 2, 3, 1)
            sample = sample.contiguous()

            gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
            all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
            if args.class_cond:
                gathered_labels = [
                    th.zeros_like(classes) for _ in range(dist.get_world_size())
                ]
                dist.all_gather(gathered_labels, classes)
                all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
            logger.log(f"created {len(all_images) * args.batch_size} samples")

        arr = np.concatenate(all_images, axis=0)
        arr = arr[:SAVE_EVERY]
        if args.class_cond:
            label_arr = np.concatenate(all_labels, axis=0)
            label_arr = label_arr[:SAVE_EVERY]

        dist.barrier()
        logger.log("batch of 1,000 complete")
        return arr, label_arr


    def save_batch(arr, label_arr, i):
        # Step 2: converting to PNGs
        imgs, labels = arr, label_arr

        # save the images
        for (image, label) in zip(imgs, labels):
            i += 1
            str_label=idx_label[label] # get string label from label index, e.g. 'A2C'
            filename='sample_{}_{}.png'.format(str_label,str(i).zfill(5)) # e.g. sample_A2C_00021.png
            save_img=os.path.join(save_path,str_label,filename) # e.g. C:\Users\Anouka\Downloads\master-thesis\diffusion\save_samples\A2C\sample_A2C_0021.png
            # create subdirectory if it does not exist
            os.makedirs(os.path.join(save_path,str_label), exist_ok=True)
            # convert image to PIL Image object
            image = Image.fromarray(image[:, :, -1])
            # save image
            image.save(save_img)
            
    start_time = time.time()
    
    args = create_argparser().parse_args()
    dist_util.setup_dist()
    logger.configure()
    
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    
    # load checkpoint if explicitly passed, else load the latest available checkpoint if passed a directory full of checkpoints
    if os.path.isfile(args.model_path):
        ckpt = args.model_path
    elif os.path.isdir(args.model_path):
        ckpt = get_last_checkpoint(args.model_path, mode='ema')
    else:
        raise ValueError('Invalid model path. Must be path to a model checkpoint, or to a directory of model checkpoints (in which case last EMA checkpoint is loaded).')
    logger.log(f"Loading model checkpoint:\n{ckpt}")
    model.load_state_dict(
        dist_util.load_state_dict(ckpt, map_location="cpu")
    )

    model.to(dist_util.dev())
    model.eval()

    if args.num_classes == 2:
        # ATTENTION! 2 CLASSES:
        idx_label={0:'A4C', 1:'PLAX'}
    elif args.num_classes == 4:
        # ATTENTION! 4 CLASSES:
        idx_label={0:'A2C',1:'A4C',2:'PLAX',3:'PSAX'}
    else:
        raise ValueError(f'Invalid number of classes in .npz file: {len(set(labels))}.')
    print(idx_label)    

    # Main loop
    save_path = os.path.join(logger.get_dir(), "PNG")  # PNG save path
    n_saved = 0
    logger.log("Sampling...")
    while n_saved < args.num_samples:
        # save PNGs in batches of 1,000 images at a time
        arr, label_arr = gen_batch()
        save_batch(arr, label_arr, n_saved)
        n_saved += SAVE_EVERY

    # note: except-pass is usually not a good idea, but this just catches printing errors when using multiple GPUs
    try:
        logger.log(f'\nDone! Saved images to:\n{save_path}')
        logger.log(f'Elapsed time: {get_elapsed_time(start_time)}')
    except:
        pass


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


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


if __name__ == "__main__":
    # run time reference: A100:4 -> 25k images took 6h 40m
    # run time reference: A100:4 -> 50k images took 13h 14m
    # run time reference: A40:4 -> 10k images SHOULD take just under 6h
    main()


    # changes
    # made it so it defaults to loading last checkpoint in checkpoint directory, unless a specific checkpoint path is passed
    # made it so PNGs are saved on the fly in batches of 1000s