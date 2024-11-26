'''
This is a simplified training code of GPEN. It achieves comparable performance as in the paper.

@Created by rosinality

@Modified by yangxy (yangtao9009@gmail.com)
'''
import argparse
import math
import random
import os
import cv2
import glob
from tqdm import tqdm
import yaml
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils
from einops import rearrange
import uuid
import __init_paths
from datasets import HybridDataset
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
import wandb

import loralib as lora

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


def g_nonsaturating_loss(fake_pred, fake_clip_pred=None, loss_funcs=None, fake_img=None, real_img=None, ref_img=None, l1_weight=1.0):
    smooth_l1_loss, id_loss = loss_funcs
    
    loss_i = F.softplus(-fake_pred).mean()
    loss_v = F.softplus(-fake_clip_pred).mean() if fake_clip_pred is not None else 0.0
    loss_l1 = smooth_l1_loss(fake_img, real_img)
    loss_id, __, __ = id_loss(fake_img, real_img, ref_img)
    loss = l1_weight*loss_l1 + 1.0*loss_id + 1.0*loss_i + 1.0*loss_v

    return loss, loss_l1, loss_id, loss_i, loss_v


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

def validation(val_loader, model, lpips_func, args, device):
    dist_sum = 0
    model.eval()
    # for lq_f, hq_f in zip(lq_files, hq_files):
    for batch in tqdm(val_loader):
        image_cond, audio_cond, real_img = batch
        image_cond = rearrange(image_cond, 'b c f h w -> (b f) c h w')
        real_img = rearrange(real_img, 'b c f h w -> (b f) c h w')
        audio_cond = rearrange(audio_cond, 'b f c t d -> (b f) c t d')
        
        image_cond = image_cond.to(device)
        audio_cond = audio_cond.to(device)
        real_img = real_img.to(device)
        
        with torch.no_grad():
            img_out, __ = model(image_cond, audio_cond)
            dist_sum += lpips_func.forward(img_out, real_img)
    
    return dist_sum.data/len(val_loader.dataset)

def train_discriminator(real_img, fake_img, i, discriminator, d_optim, args):
    loss_dict = {}
    r1_loss = torch.tensor(0.0, device=device)
    
    requires_grad(discriminator, True)
    
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
    
    return loss_dict

def train(args, loader, val_loader, generator, discriminators, losses, g_optim, d_optimizers, g_ema, lpips_func, device):
    loader = sample_data(loader)

    pbar = range(0, args.iter)

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    mean_path_length = 0

    d_loss_val = 0
    g_loss_val = 0
    path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)
    mean_path_length_avg = 0
    loss_dict = {}

    if args.distributed:
        g_module = generator.module
        d_module = discriminators[0].module
        d_v_module = discriminators[1].module if args.video_disc else None

    else:
        g_module = generator
        d_module = discriminators[0]
        d_v_module = discriminators[1] if args.video_disc else None
        
 
    accum = 0.5 ** (32 / (10 * 1000))

    for idx in pbar:
        if args.training_steps is not None and idx > args.training_steps:
            break
        
        i = idx + args.start_iter

        if i > args.iter:
            print('Done!')

            break
        
        image_cond, audio_cond, real_img = next(loader)
        
        if args.drop_ref:
            # 生成一个随机的布尔掩码矩阵，根据概率决定是否 drop 每个样本的 ref image
            drop = (torch.rand(image_cond.shape[0]) < args.drop_ref_prob).to(image_cond.device)
            # 通过掩码选择性地对某些样本进行 drop（将第 4-6 通道置为 0）
            image_cond[drop, 3:] = 0
        
        image_cond = rearrange(image_cond, 'b c f h w -> (b f) c h w')
        real_img = rearrange(real_img, 'b c f h w -> (b f) c h w')
        audio_cond = rearrange(audio_cond, 'b f c t d -> (b f) c t d')
        
        image_cond = image_cond.to(device)
        audio_cond = audio_cond.to(device)
        real_img = real_img.to(device)
        
        requires_grad(generator, False)
        fake_img, _ = generator(image_cond, audio_cond)
        
        d_loss_dict = train_discriminator(real_img, fake_img, i, discriminators[0], d_optimizers[0], args)
        loss_dict['d'] = d_loss_dict['d']
        loss_dict['real_score'] = d_loss_dict['real_score']
        loss_dict['fake_score'] = d_loss_dict['fake_score']
        loss_dict['r1'] = d_loss_dict['r1']
        
        if args.video_disc:
            fake_clip = rearrange(fake_img, '(b f) c h w -> b (f c) h w', f=args.syncnet_T)
            real_clip = rearrange(real_img.detach(), '(b f) c h w -> b (f c) h w', f=args.syncnet_T)
            d_v_loss_dict = train_discriminator(real_clip, fake_clip, i, discriminators[1], d_optimizers[1], args)
            loss_dict['d_v'] = d_v_loss_dict['d']
            loss_dict['real_score_v'] = d_v_loss_dict['real_score']
            loss_dict['fake_score_v'] = d_v_loss_dict['fake_score']
            loss_dict['r1_v'] = d_v_loss_dict['r1']

        requires_grad(generator, True)
        discriminator = discriminators[0]
        discriminator.requires_grad_(False)
        if args.video_disc:
            discriminator_v = discriminators[1]
            discriminator_v.requires_grad_(False)

        fake_img, _ = generator(image_cond, audio_cond)
        fake_pred = discriminator(fake_img)
        
        if args.video_disc:
            fake_clip = rearrange(fake_img, '(b f) c h w -> b (f c) h w', f=args.syncnet_T)
            fake_clip_pred = discriminator_v(fake_clip)
        
        ref_image = image_cond[:, 3:]
        g_loss, g_loss_l1, g_loss_id, g_disc, g_disc_v = g_nonsaturating_loss(fake_pred, 
                                                            fake_clip_pred if args.video_disc else None, 
                                                            losses, fake_img, real_img, ref_image, args.l1_weight)

        loss_dict['g'] = g_loss
        loss_dict['g_l1'] = g_loss_l1
        loss_dict['g_id'] = g_loss_id
        loss_dict['g_disc'] = g_disc
        if args.video_disc: 
            loss_dict['g_disc_v'] = g_disc_v

        generator.zero_grad()
        g_loss.backward()
        g_optim.step()

        g_regularize = i % args.g_reg_every == 0

        if g_regularize:
            path_batch_size = max(1, args.batch // args.path_batch_shrink)

            fake_img, latents = generator(image_cond, audio_cond, return_latents=True)

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
            if args.wandb:
                wandb.log(loss_reduced)
            
            pbar.set_description(
                (
                    f'd: {d_loss_val:.4f}; g: {g_loss_val:.4f}; r1: {r1_val:.4f}; '
                )
            )
            if not args.save_best_val_ckpt:
                if i % args.save_freq == 0:
                    with torch.no_grad():
                        g_ema.eval()
                        sample, _ = g_ema(image_cond, audio_cond)
                        sample = torch.cat((image_cond[:, 3:], image_cond[:, :3], sample, real_img), 0) 
                        utils.save_image(
                            sample,
                            f'{args.sample}/{str(i).zfill(6)}.png',
                            nrow=args.batch * args.syncnet_T,
                            normalize=True,
                            value_range=(-1, 1),
                        )
                        
                    lpips_value = validation(val_loader, g_ema, lpips_func, args, device)
                    print(f'{i}/{args.iter}: lpips: {lpips_value.cpu().numpy()[0][0][0][0]}')
                    if args.wandb:
                        wandb.log({'lpips': lpips_value.cpu().numpy()[0][0][0][0]})

                if i and i % args.save_freq == 0:
                    state_dict = {
                            'g': g_module.state_dict(),
                            'd': d_module.state_dict(),
                            'g_ema': g_ema.state_dict(),
                            'g_optim': g_optim.state_dict(),
                            'd_optim': d_optim.state_dict(),
                            'global_step': i,
                        }
                    if args.video_disc:
                        state_dict['d_v'] = d_v_module.state_dict()
                        state_dict['d_v_optim'] = d_v_optim.state_dict()
                    if args.lora:
                        torch.save(lora.lora_state_dict(g_module), f'{args.ckpt}/{str(i).zfill(6)}.pth')
                    else:
                        torch.save(state_dict,
                            f'{args.ckpt}/{str(i).zfill(6)}.pth',
                    )
                
            elif g_loss_val < args.min_g_loss:
                args.min_g_loss = g_loss_val
                state_dict = {
                        'g': g_module.state_dict(),
                        'd': d_module.state_dict(),
                        'g_ema': g_ema.state_dict(),
                        'g_optim': g_optim.state_dict(),
                        'd_optim': d_optim.state_dict(),
                        'global_step': i,
                    }
                if args.video_disc:
                    state_dict['d_v'] = d_v_module.state_dict()
                    state_dict['d_v_optim'] = d_v_optim.state_dict()
                if args.lora:
                    torch.save(lora.lora_state_dict(g_module), f'{args.ckpt}/{str(i).zfill(6)}_{str(g_loss_val)}_best_lora.pth')
                else:
                    torch.save(state_dict,
                        f'{args.ckpt}/{str(i).zfill(6)}_{str(g_loss_val)}_best.pth',
                )


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--base_dir', type=str, default='./')
    parser.add_argument('--iter', type=int, default=4000000)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--channel_multiplier', type=int, default=2)
    parser.add_argument('--narrow', type=float, default=1.0)
    parser.add_argument('--r1', type=float, default=10)
    parser.add_argument('--path_regularize', type=float, default=2)
    parser.add_argument('--path_batch_shrink', type=int, default=2)
    parser.add_argument('--d_reg_every', type=int, default=16)
    parser.add_argument('--g_reg_every', type=int, default=4)
    parser.add_argument('--save_freq', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--local-rank', type=int, default=0)
    parser.add_argument('--pretrain', type=str, default=None)
    parser.add_argument('--val_dir', type=str, default='val')
    parser.add_argument('--dataset_config', type=str, default='configs/dataset_config.yaml')
    parser.add_argument('--wav2vec2', action='store_true')
    parser.add_argument('--syncnet_T', type=int, choices=[1, 3, 5, 10, 25], default=1)
    parser.add_argument('--data_aug_image', action='store_true')
    parser.add_argument('--data_aug_mask', action='store_true')
    parser.add_argument('--mask_ratio', type=float, default=0.6)
    parser.add_argument('--syncnet', action='store_true')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--l1_weight', type=float, default=1.0)
    parser.add_argument('--drop_ref', action='store_true')
    parser.add_argument('--drop_ref_prob', type=float, default=0.5)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--video_disc', action='store_true')
    
    parser.add_argument('--experiment', type=str, default=None)
    parser.add_argument('--training_steps', type=int, default=None)
    parser.add_argument('--freeze_audio_encoder', action='store_true')
    parser.add_argument('--freeze_discriminator', action='store_true')
    
    parser.add_argument('--save_best_val_ckpt', action='store_true')
    parser.add_argument('--lora', action='store_true')
    
    args = parser.parse_args()
    
    device = 'cuda'

    n_gpu = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    args.distributed = n_gpu > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        synchronize()

    args.latent = 512
    args.n_mlp = 8

    args.start_iter = 0
    
    args.min_g_loss = 1000.0
    
    generator = FullGenerator(
        args.image_size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier, narrow=args.narrow, isconcat=False, device=device, freeze_audio=args.freeze_audio_encoder
    ).to(device)
        
    discriminator = Discriminator(
        args.image_size, channel_multiplier=args.channel_multiplier, narrow=args.narrow, device=device
    ).to(device)
    
    if args.video_disc:
        discriminator_v = Discriminator(
            args.image_size, T=args.syncnet_T, channel_multiplier=args.channel_multiplier, narrow=args.narrow, device=device
        ).to(device)
    
    if args.freeze_discriminator:
        for param in discriminator.parameters():
            param.requires_grad = False
        if args.video_disc:
            for param in discriminator_v.parameters():
                param.requires_grad = False
        
    g_ema = FullGenerator(
        args.image_size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier, narrow=args.narrow, isconcat=False, device=device
    ).to(device)
    g_ema.eval()
    accumulate(g_ema, generator, 0)

    lora.mark_only_lora_as_trainable(generator)
    lora.mark_only_lora_as_trainable(discriminator)

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
    
    if args.video_disc:
        d_v_optim = optim.Adam(
            discriminator_v.parameters(),
            lr=args.lr * d_reg_ratio,
            betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
        )

    if args.pretrain is not None:
        print('load pretrain:', args.pretrain)
        
        ckpt = torch.load(args.pretrain)

        generator_state_dict = {f'generator.{k}': v for k, v in ckpt['g'].items()}
        generator.load_state_dict(generator_state_dict, strict=False)
        discriminator.load_state_dict(ckpt['d'])
        generator_ema_state_dict = {f'generator.{k}': v for k, v in ckpt['g_ema'].items()}
        g_ema.load_state_dict(generator_ema_state_dict, strict=False)
            
        # g_optim.load_state_dict(ckpt['g_optim'])
        # d_optim.load_state_dict(ckpt['d_optim'])
    
    if args.ckpt is not None:
        print('load ckpt:', args.ckpt)
        ckpt = torch.load(args.ckpt)
        generator.load_state_dict(ckpt['g'])
        discriminator.load_state_dict(ckpt['d'])        
        g_ema.load_state_dict(ckpt['g_ema'])
        g_optim.load_state_dict(ckpt['g_optim'])
        d_optim.load_state_dict(ckpt['d_optim'])
        
        if args.video_disc:
            discriminator_v.load_state_dict(ckpt['d_v'])
            d_v_optim.load_state_dict(ckpt['d_v_optim'])
        
        args.start_iter = ckpt['global_step']
    
    smooth_l1_loss = torch.nn.SmoothL1Loss().to(device)
    id_loss = IDLoss(args.base_dir, device, ckpt_dict=None)
    lpips_func = lpips.LPIPS(net='alex',version='0.1').to(device)
    
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
        
        if args.video_disc:
            discriminator_v = nn.parallel.DistributedDataParallel(
                discriminator_v,
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

    config = yaml.load(open(args.dataset_config, 'r'), Loader=yaml.FullLoader)
    
    dataset = HybridDataset(config, 'train', args, dataset_size=40000)
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=data_sampler(dataset, shuffle=True, distributed=args.distributed),
        drop_last=True,
    )
    
    val_dataset = HybridDataset(config, 'val', args, dataset_size=400)
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=args.batch,
        sampler=data_sampler(val_dataset, shuffle=False, distributed=args.distributed),
        drop_last=True,
    )
    
    if get_rank() == 0:
        experiment_name = 'experiments/' + str(args.experiment) + "-" + str(uuid.uuid4()) 
        print('>>>>>>>>>> experiment_name:', experiment_name)
        args.ckpt = f'{experiment_name}/ckpts'
        args.sample = f'{experiment_name}/samples'
        os.makedirs(args.ckpt, exist_ok=True)
        os.makedirs(args.sample, exist_ok=True)
     
        if args.wandb:
            wandb.init(project="GPEN_wav2lip", name=str(args.experiment), entity="local-optima")
            wandb.config.update({'experiment_name': experiment_name})
            wandb.config.update({'config_details': config})

    discriminators = [discriminator] if not args.video_disc else [discriminator, discriminator_v]
    d_optimizers = [d_optim] if not args.video_disc else [d_optim, d_v_optim]
    train(args, loader, val_loader, generator, discriminators, [smooth_l1_loss, id_loss], g_optim, d_optimizers, g_ema, lpips_func, device)
   
