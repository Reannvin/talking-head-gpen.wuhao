import argparse
import math
import random
import os
import cv2
import glob
from tqdm import tqdm

import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils

import __init_paths
from training.data_loader.dataset_face import FaceDataset
from load_dataset import AudioVisualDataset

from face_model.gpen_model import FullGenerator, Discriminator

from training.loss.id_loss import IDLoss
from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)

from training import lpips
from datetime import datetime

import PIL.Image
import numpy as np

import wandb

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from audio_model import SyncNet_image_256

def x():
    import sys
    print("8R3AKP01NT")
    sys.exit()

def save_image(img, fname):
    img = np.asarray(img, dtype=np.float32)
        
    img = (img) * (255)
    img = np.rint(img).clip(0, 255).astype(np.uint8)
    if len(img.shape) == 4:
        C, H, W = img.shape[1:]
    elif len(img.shape) == 3:
        C, H, W = img.shape
        
    img = img.reshape([C, H, W])
    img = img.transpose(1, 2, 0)
    # img = img[:, :, ::-1]
    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname)
        
def data_sampler(dataset, shuffle, distributed):    
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        return data.RandomSampler(dataset)
    else:
        return data.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)

def sample_data(loader):
    while True:
        for batch in loader:
            yield batch

def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty

def grad_layer_wrt_loss(loss, layer):
    return torch.autograd.grad(loss, 
                               layer, 
                               grad_outputs=torch.ones_like(loss),retain_graph=True)[0].detach()
    
def g_image_recon_loss(real_image, fake_image, use_smooth=False):
    if use_smooth:
        return torch.nn.SmoothL1Loss()(real_image, fake_image)
    else:
        return torch.nn.functional.l1_loss(real_image, fake_image)
    
def g_perceptual_loss(real_image, fake_image):
    fake_image = torch.clamp(fake_image, 0., 1.)
    loss = lpips_loss(fake_image, real_image)
    return loss

def cosine_loss(a, v, y):
    d = nn.functional.cosine_similarity(a, v)
    # Disable autocast for unsafe operations
    with torch.cuda.amp.autocast(enabled=False):
        loss = log_loss(d.unsqueeze(1), y)
    return loss

def load_syncnet(enable_sync, syncnet_T):
    syncnet = SyncNet_image_256(enable_sync, syncnet_T)
    ckpt = torch.load("/data/wuhao/code/GPEN_wav2lip/audio_model/ckpts/syncnet.ckpt")
    new_state_dict = {k[len("model."):] if k.startswith("model.") else k: v for k, v in ckpt['state_dict'].items()}
    syncnet.load_state_dict(new_state_dict)

    for param in syncnet.parameters():
        param.requires_grad = False
    return syncnet

def get_sync_loss(mel, g):
    # mel.shape ([1, 1, 50, 384])
    # g.shape ([5, 3, 128, 256])
    mel = mel.to(device)
    g = g.to(device)
    # if image size is not [128, 256], resize the image to [128, 256]

    g = g.view(args.batch, args.syncnet_T, 3, 128, 256)
    g = g.reshape(args.batch, args.syncnet_T * 3, 128, 256)
    if g.size(2) != 128 or g.size(3) != 256:
        g = nn.functional.interpolate(g, size=(128, 256), mode='bilinear')
    # mel = mel.reshape(-1, 1, 50, 384)
    a, v = syncnet(mel, g)
    y = torch.ones(g.size(0), 1).to(device)
    sync_loss = cosine_loss(a, v, y)
    return sync_loss

def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss

def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad, = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths

def validation(model, lpips_func, args, device):
    lq_files = sorted(glob.glob(os.path.join(args.val_dir, 'lq', '*.*')))
    hq_files = sorted(glob.glob(os.path.join(args.val_dir, 'hq', '*.*')))

    assert len(lq_files) == len(hq_files)

    dist_sum = 0
    model.eval()
    for lq_f, hq_f in zip(lq_files, hq_files):
        img_lq = cv2.imread(lq_f, cv2.IMREAD_COLOR)
        img_t = torch.from_numpy(img_lq).to(device).permute(2, 0, 1).unsqueeze(0)
        img_t = (img_t/255.-0.5)/0.5
        img_t = F.interpolate(img_t, (args.size, args.size))
        img_t = torch.flip(img_t, [1])
        
        with torch.no_grad():
            img_out, __ = model(img_t)
        
            img_hq = lpips.im2tensor(lpips.load_image(hq_f)).to(device)
            img_hq = F.interpolate(img_hq, (args.size, args.size))
            dist_sum += lpips_func.forward(img_out, img_hq)
    
    return dist_sum.data/len(lq_files)


def train(args, loader, generator, discriminator, g_optim, d_optim, g_ema, device):
    loader = sample_data(loader)

    pbar = range(0, args.iter)
    experiment_time = datetime.now().strftime("%Y%m%d%H%M")

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    mean_path_length = 0

    d_loss_val = 0
    r1_loss = torch.tensor(0.0, device=device)
    g_loss_val = 0
    path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)
    # mean_path_length_avg = 0
    loss_dict = {}

    if args.distributed:
        g_module = generator.module
        d_module = discriminator.module

    else:
        g_module = generator
        d_module = discriminator
 
    accum = 0.5 ** (32 / (10 * 1000))

    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print('Done!')

            break
        if args.enable_sync_loss :
            cond_image, cond_audio, audio_crop, real_img = next(loader)
        else:
            cond_image, cond_audio, real_img = next(loader)
        cond_image = cond_image.to(device)
        cond_audio = cond_audio.to(device)
        real_img = real_img.to(device)

        cond_image = cond_image.permute(0, 2, 1, 3, 4).reshape(-1, 6, 256, 256)
        cond_audio = cond_audio.squeeze(2)
        cond_audio = cond_audio.reshape(-1, 50, 384)
        real_img = real_img.permute(0, 2, 1, 3, 4).reshape(-1, 3, 256, 256) 

        if args.ref_drop is not None and i >= args.post_drop:
            drop = (torch.rand(cond_image.shape[0]) < args.ref_drop).to(device)
            cond_image[drop, 3:] = 0
        
        requires_grad(generator, False)
        requires_grad(discriminator, True)

        # TODO: Figure out how to input cond into generator.
        fake_img, _ = generator(cond_image,cond_audio)
        
        fake_pred = discriminator(fake_img)
        real_pred = discriminator(real_img)
        
        d_loss = d_logistic_loss(real_pred, fake_pred)

        loss_dict['d'] = d_loss
        loss_dict['real_score'] = real_pred.mean()
        loss_dict['fake_score'] = fake_pred.mean()

        discriminator.zero_grad()
        d_loss.backward()
        d_optim.step()

        d_regularize = i % args.d_reg_every == 0

        if d_regularize:
            real_img.requires_grad = True
            real_pred = discriminator(real_img)
            r1_loss = d_r1_loss(real_pred, real_img)

            discriminator.zero_grad()
            (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0]).backward()

            d_optim.step()

        loss_dict['r1'] = r1_loss

        requires_grad(generator, True)
        requires_grad(discriminator, False)

        fake_img, _ = generator(cond_image, cond_audio)
        fake_pred = discriminator(fake_img)

        gen_loss = g_nonsaturating_loss(fake_pred)
        
        if args.enable_half_image_loss:
            g_recon_loss = g_image_recon_loss(fake_img[:, :, fake_img.size(3) // 2:, :], real_img[:, :, real_img.size(3) // 2:, :])
            g_lpips_loss = g_perceptual_loss(fake_img[:, :, fake_img.size(3) // 2:, :], real_img[:, :, real_img.size(3) // 2:, :])
        else:
            g_recon_loss = g_image_recon_loss(real_img, fake_img)
            g_lpips_loss = g_perceptual_loss(real_img, fake_img)
        
        if args.enable_sync_loss:
            g_sync_loss = get_sync_loss(audio_crop, fake_img[:, :, fake_img.size(3) // 2:, :])
        else:
            g_sync_loss = 0
            
        if args.enable_adaptive_weight:
            last_gen_layer = generator.generator.to_rgbs[5].conv.modulation.weight.float()
            
            norm_grad_wrt_lpips_loss = grad_layer_wrt_loss(g_lpips_loss, last_gen_layer).norm(p=2)
            norm_grad_wrt_gen_loss = grad_layer_wrt_loss(gen_loss, last_gen_layer).norm(p=2)
            adaptive_weight = (norm_grad_wrt_lpips_loss / norm_grad_wrt_gen_loss.clamp(min=1e-8)).clamp(max=1e4)

            g_loss = 100 * g_recon_loss + g_lpips_loss + adaptive_weight * gen_loss + 0.05 * g_sync_loss
        else:
            recon_weight = 5000
            lpips_weight = 1
            adv_weight = 1
            sync_weight = 0.1
            g_loss = recon_weight * g_recon_loss + lpips_weight * g_lpips_loss + adv_weight * gen_loss + sync_weight * g_sync_loss
            adaptive_weight = 0
            
        loss_dict['g'] = g_loss

        generator.zero_grad()
        g_loss.backward()
        g_optim.step()

        g_regularize = i % args.g_reg_every == 0

        if g_regularize:
            path_batch_size = max(1, args.batch // args.path_batch_shrink)

            fake_img, latents = generator(cond_image, cond_audio, return_latents=True)

            path_loss, mean_path_length, path_lengths = g_path_regularize(
                fake_img, latents, mean_path_length
            )

            generator.zero_grad()
            weighted_path_loss = args.path_regularize * args.g_reg_every * path_loss

            if args.path_batch_shrink:
                weighted_path_loss += 0 * fake_img[0, 0, 0, 0]

            weighted_path_loss.backward()

            g_optim.step()

            mean_path_length_avg = (
                reduce_sum(mean_path_length).item() / get_world_size()
            )

        loss_dict['path'] = path_loss
        loss_dict['path_length'] = path_lengths.mean()

        accumulate(g_ema, g_module, accum)

        loss_reduced = reduce_loss_dict(loss_dict)

        d_loss_val = loss_reduced['d'].mean().item()
        g_loss_val = loss_reduced['g'].mean().item()
        r1_val = loss_reduced['r1'].mean().item()
        path_loss_val = loss_reduced['path'].mean().item()
        real_score_val = loss_reduced['real_score'].mean().item()
        fake_score_val = loss_reduced['fake_score'].mean().item()
        path_length_val = loss_reduced['path_length'].mean().item()

        if get_rank() == 0:
            pbar.set_description(
                (
                    f'd: {d_loss_val:.4f}; g: {g_loss_val:.4f}; r1: {r1_val:.4f}; '
                )
            )
            
            if wandb and args.wandb:
                wandb.log(
                    {
                        "Generator": g_loss_val,
                        "Discriminator": d_loss_val,
                        "R1": r1_val,
                        "Path Length Regularization": path_loss_val,
                        "Mean Path Length": mean_path_length,
                        "Real Score": real_score_val,
                        "Fake Score": fake_score_val,
                        "Path Length": path_length_val,
                        "Adv Loss": gen_loss,
                        "L1 Loss": g_recon_loss,
                        "Perceptual Loss": g_lpips_loss,
                        "Adaptive weight": adaptive_weight,
                        "SyncLoss":g_sync_loss,
                    }
                )
                
            if i % args.save_sample_freq == 0:
                snapshot_folder = os.path.join(f"./training-run/{experiment_time}/",f'snapshots_{i:06d}')
                os.makedirs(snapshot_folder, exist_ok=True)
                
                with torch.no_grad():
                    g_ema.eval()
                    sample, _ = g_ema(cond_image,cond_audio)
                    sample = sample.cpu()
                    cond_image_1 = cond_image[:,:3].cpu()
                    cond_image_2 = cond_image[:,3:].cpu()
                    gt = real_img.cpu()
                    for idx in range(sample.shape[0]):
                        images = torch.cat([sample[idx], cond_image_1[idx], cond_image_2[idx], gt[idx]], dim=2)
                        save_image(images, os.path.join(snapshot_folder, f'sample_{idx}.png'))

                # lpips_value = validation(g_ema, lpips_func, args, device)
                # print(f'{i}/{args.iter}: lpips: {lpips_value.cpu().numpy()[0][0][0][0]}')

            if i and i % args.save_ckpt_freq == 0:
                checkpoints_folder = os.path.join(f"./training-run/{experiment_time}/",f'checkpoints')
                os.makedirs(checkpoints_folder, exist_ok=True)
                torch.save(
                    {
                        "g": g_module.state_dict(),
                        "d": d_module.state_dict(),
                        "g_ema": g_ema.state_dict(),
                        "g_optim": g_optim.state_dict(),
                        "d_optim": d_optim.state_dict(),
                        "args": args,
                        "ada_aug_p": 0, # Caution.
                    },
                    os.path.join(checkpoints_folder, f"checkpoint_{i}.pt"),
                )


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--base_dir', type=str, default='./')
    parser.add_argument('--iter', type=int, default=4000000)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--channel_multiplier', type=int, default=2)
    parser.add_argument('--narrow', type=float, default=1.0)
    parser.add_argument('--r1', type=float, default=10)
    parser.add_argument('--d_reg_every', type=int, default=16)
    parser.add_argument('--g_reg_every', type=int, default=4)
    parser.add_argument('--save_ckpt_freq', type=int, default=10000)
    parser.add_argument('--save_sample_freq', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--ckpt', type=str, default='ckpts')
    parser.add_argument('--sample', type=str, default='sample')
    parser.add_argument('--val_dir', type=str, default='val')
    
    parser.add_argument('--stage_1_pt', type=str, default=None, help='StyleGAN2 face generator loc.')
    parser.add_argument('--stage_2_pt', type=str, default=None, help='Wav2lip generator.')
    
    parser.add_argument("--wandb", action="store_true", help="use weights and biases logging")
    parser.add_argument("--data_root",type=str,default='/mnt/hdtf_btm_move/' ,help='Dataset location.')
    parser.add_argument("--dataset_name",type=str,default='hdtf' ,help='Dataset name.')
    parser.add_argument("--syncnet_T", type=int, default=1, help="For syncnet")
    
    parser.add_argument("--enable_sync_loss",action="store_true", default=False ,help='Dataset name.')
    parser.add_argument("--enable_adaptive_weight",action="store_true", default=False ,help='enable_adaptive_weight.')
    parser.add_argument("--only_D",action="store_true", default=False ,help='enable_adaptive_weight.')
    
    parser.add_argument("--concat_condition",action="store_true", default=None ,help='Concat condition image and audio.')
    parser.add_argument("--sync_audio_encoder",action="store_true", default=None ,help='Use StyleSync audio encoder.')
    parser.add_argument("--ref_drop",type=float, default=None ,help='Set a random dropout rate.')
    parser.add_argument("--enable_half_image_loss",action="store_true", default=False ,help='Only compute lower-side loss.')
    parser.add_argument("--post_drop",type=int, default = 0, help='Drop after specificed steps.')
    
    
    args = parser.parse_args()

    device = 'cuda'

    n_gpu = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    args.distributed = n_gpu > 1
    args.path_regularize = 2
    args.path_batch_shrink = 2
    
    if args.stage_1_pt is not None and args.stage_2_pt is not None:
        raise ValueError("Only one pretrained model can be loaded.")
    
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        synchronize()

    # args.latent = 512
    args.latent = 256
    
    args.n_mlp = 8

    args.start_iter = 0
    
    if args.stage_1_pt is not None and not args.only_D:
        print('load model G:', args.stage_1_pt)
        ckpt = torch.load(args.stage_1_pt)
        generator = FullGenerator(
            args.size, 
            args.latent, 
            args.n_mlp, 
            channel_multiplier=args.channel_multiplier, 
            narrow=args.narrow, 
            device=device,
            pretrained_generator=ckpt['g'],
            concat_condition = args.concat_condition,
            sync_audio_encoder = args.sync_audio_encoder
        ).to(device)
        g_ema = FullGenerator(
            args.size, 
            args.latent, 
            args.n_mlp, 
            channel_multiplier=args.channel_multiplier, 
            narrow=args.narrow, 
            device=device,
            pretrained_generator=ckpt['g_ema'],
            concat_condition = args.concat_condition,
            sync_audio_encoder = args.sync_audio_encoder
        ).to(device)
        
    else:
        generator = FullGenerator(
            args.size, 
            args.latent, 
            args.n_mlp, 
            channel_multiplier=args.channel_multiplier, 
            narrow=args.narrow,
            device=device,
            pretrained_generator=None,
            concat_condition = args.concat_condition,
            sync_audio_encoder = args.sync_audio_encoder
        ).to(device)
        g_ema = FullGenerator(
            args.size, 
            args.latent, 
            args.n_mlp, 
            channel_multiplier=args.channel_multiplier, 
            narrow=args.narrow, 
            device=device,
            pretrained_generator=None,
            concat_condition = args.concat_condition,
            sync_audio_encoder = args.sync_audio_encoder
        ).to(device)

    discriminator = Discriminator(
        args.size, channel_multiplier=args.channel_multiplier, narrow=args.narrow, device=device
    ).to(device)

    g_ema.eval()
    accumulate(g_ema, generator, 0)

    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)
    
    g_optim = optim.Adam(
        generator.parameters(),
        lr=args.lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )

    d_optim = optim.Adam(
        discriminator.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )
    
    if args.stage_1_pt is not None:
        print('load model D:', args.stage_1_pt)
        ckpt = torch.load(args.stage_1_pt)
        discriminator.load_state_dict(ckpt['d'])
        
    if args.stage_2_pt is not None:
        print('load model:', args.stage_2_pt)
        ckpt = torch.load(args.stage_2_pt)

        generator.load_state_dict(ckpt['g'])
        discriminator.load_state_dict(ckpt['d'])
        g_ema.load_state_dict(ckpt['g_ema'])
            
        g_optim.load_state_dict(ckpt['g_optim'])
        d_optim.load_state_dict(ckpt['d_optim'])
    
    smooth_l1_loss = torch.nn.SmoothL1Loss().to(device)
    id_loss = IDLoss(args.base_dir, device, ckpt_dict=None)
    log_loss = nn.BCELoss().to(device)
    
    lpips_loss = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).to(device)
    
    if args.distributed:
        generator = nn.parallel.DistributedDataParallel(
            generator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

        discriminator = nn.parallel.DistributedDataParallel(
            discriminator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

        id_loss = nn.parallel.DistributedDataParallel(
            id_loss,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )
    
    dataset = AudioVisualDataset(data_root = args.data_root, dataset_name = args.dataset_name, syncnet_T=args.syncnet_T, split = 'main')
    
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=data_sampler(dataset, shuffle=True, distributed=args.distributed),
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    
    if args.enable_sync_loss:
        syncnet = load_syncnet(args.enable_sync_loss, args.syncnet_T).eval().to(device)
    if get_rank() == 0 and wandb is not None and args.wandb:
        wandb.init(project="GPEN_wav2lip", name="Test-HDTF", entity="local-optima")
        
    train(args, loader, generator, discriminator, g_optim, d_optim, g_ema, device)
   
