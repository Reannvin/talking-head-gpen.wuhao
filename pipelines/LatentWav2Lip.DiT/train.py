# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os

from models import DiT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
import wandb
from torchvision.utils import save_image

from datasets import HybridDataset
import yaml
import math
from helpers import encode_frames

#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup(args):
    """
    Cleanup function to ensure that all processes are properly shut down.
    """
    
    # finish the wandb run
    if args.wandb and dist.get_rank() == 0:
        wandb.finish()
        
    # Ensure all processes are properly shut down:
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

def sample_and_save_images(model, x, i, a, diffusion, vae, args, device, logger, train_steps, experiment_dir):
    """
    Sample images from the model and save them to disk
    """
    n = args.syncnet_T
    x = x[:n]
    i = i[:n]
    a = a[:n]
    
    # Prepare inputs for the model:
    z = torch.randn_like(x, device=device)
    zz = torch.cat([z, z], 0)
    ii = torch.cat([i, i], 0)
    a_null = torch.zeros_like(a, device=device)
    aa = torch.cat([a, a_null], 0)
    
    # Sample images:
    model_kwargs = dict(i=ii, a=aa, cfg_scale=args.cfg_scale)
    samples = diffusion.p_sample_loop(
        model.forward_with_cfg, zz.shape, zz, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    )
    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples    
    
    # Prepare reference, masked, and ground truth images:
    ref = i[:, 4:]
    masked = i[:, :4]
    with torch.no_grad():
        ref_images = vae.decode(ref / 0.18215).sample
        masked_images = vae.decode(masked / 0.18215).sample
        samples = vae.decode(samples / 0.18215).sample
        gt_images = vae.decode(x / 0.18215).sample
    
    # Save and display images as [ref, masked, sample, gt] * n:
    samples_in_line = []
    for i in range(n):
        samples_in_line.append(ref_images[i])
        samples_in_line.append(masked_images[i])
        samples_in_line.append(samples[i])
        samples_in_line.append(gt_images[i])
    os.makedirs(f"{experiment_dir}/sample_images", exist_ok=True)
    save_image(samples_in_line, f"{experiment_dir}/sample_images/samples_{train_steps:07d}.png", nrow=4, normalize=True, value_range=(-1, 1))
    logger.info(f"Saved sample images at step {train_steps:07d}")

#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        if args.wandb:
            wandb.init(project="dit-wav2lip", name=f"{model_string_name}-{experiment_index}", dir=experiment_dir)
            wandb.config.update(args)
    else:
        logger = create_logger(None)
    
    if args.ckpt:
        checkpoint = torch.load(args.ckpt, map_location='cpu')

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        syncnet_T=args.syncnet_T,
        input_size=latent_size,
        in_channels=4, # ref image + masked image + latent z
        out_channels=4,
        audio_embedding_dim=384,
        temporal_attention=args.temporal_attention,
    )
    if args.ckpt:
        if args.train_temporal_only:
            model.load_state_dict(checkpoint['model'], strict=False)
            # freeze all layers except temporal attention
            for name, param in model.named_parameters():
                if 'tmp' not in name:
                    param.requires_grad = False
        else:
            model.load_state_dict(checkpoint['model'])
        logger.info(f"Loaded model from {args.ckpt}")
        
    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    if args.ckpt:
        if args.train_temporal_only:
            ema.load_state_dict(checkpoint['ema'], strict=False)
        else: 
            ema.load_state_dict(checkpoint['ema'])
        logger.info(f"Loaded EMA from {args.ckpt}")
        
    model = DDP(model.to(device), device_ids=[rank])
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}", local_files_only=True).to(device)
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    if args.train_temporal_only:
        opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=0)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
    logger.info(f"Optimizing {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters")
    if args.ckpt:
        if args.train_temporal_only:
            #! TODO 应该只加载 temporal attention 的参数，即名称中包含 tmp 的参数的优化器状态，当前未加载
            logger.info("Do not load optimizer state dict when training temporal attention only.")
        else:
            opt.load_state_dict(checkpoint['opt'])
            logger.info(f"Loaded optimizer from {args.ckpt}")

    # Setup data:
    config = yaml.load(open(args.dataset_config, 'r'), Loader=yaml.FullLoader)
    if args.wandb and rank == 0:
        wandb.config.update({"dataset_config_content": config})
    dataset = HybridDataset(config, 'train', args, dataset_size=512000)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.dataset_config})")

    # Prepare models for training:
    # update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    if not args.ckpt:
        update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0 if not args.ckpt else checkpoint['train_steps']
    epoch_start = 0 if not args.ckpt else checkpoint['epoch']
    log_steps = 0
    running_loss = 0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(epoch_start, args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for image_cond, audio_cond, gt in loader:
            image_cond = image_cond.to(device)
            audio_cond = audio_cond.to(device)
            gt = gt.to(device)
                        
            i = encode_frames(image_cond, vae)
            x = encode_frames(gt, vae)
            
            i = i.view(-1, i.shape[2], i.shape[3], i.shape[4])
            x = x.view(-1, x.shape[2], x.shape[3], x.shape[4])
            
            a = audio_cond.view(-1, audio_cond.shape[3], audio_cond.shape[4])
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(i=i, a=a)
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model.module)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                
                if args.wandb and rank == 0:
                    wandb.log({"train_loss": avg_loss, "steps_per_sec": steps_per_sec})
                    
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Sample from the model and save images:
            if train_steps % args.sample_every == 0:
                if rank == 0:
                    model.eval()
                    sample_and_save_images(model.module, x, i, a, diffusion, vae, args, device, logger, train_steps, experiment_dir)
                    model.train()
                dist.barrier()
            
            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args,
                        "train_steps": train_steps,
                        "epoch": epoch
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup(args)


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    # parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to a checkpoint to load and resume training from")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=20_000)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--sample-every", type=int, default=1000)
    parser.add_argument("--cfg-scale",  type=float, default=1.5)
    parser.add_argument('--dataset-config', type=str, default='configs/dataset_config.yaml')
    parser.add_argument('--syncnet', action='store_true')
    parser.add_argument('--data-aug-image', action='store_true')
    parser.add_argument('--data-aug-mask', action='store_true')
    parser.add_argument('--mask-ratio', type=float, default=0.6)
    parser.add_argument('--fps', type=int, default=25)
    parser.add_argument('--syncnet-T', type=int, choices=[1, 5, 10, 25], default=5)
    parser.add_argument('--wav2vec2', action='store_true')
    parser.add_argument('--temporal-attention', action='store_true')
    parser.add_argument('--train-temporal-only', action='store_true')
    args = parser.parse_args()
    main(args)
