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
import wandb
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils
import numpy as np
import __init_paths
from training.data_loader.stage1_dataset import HybridDatasetStageOne
from training.data_loader.stage2_dataset import HybridDatasetStageTwo
from face_model.gpen_model import FullGenerator, Discriminator,Generator
from face_model.syncnet import SyncNet_image_256
from training.loss.id_loss import IDLoss
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)

from training import lpips


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

def convert_rgb_to_bgr(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

def save_sample_image(real_img, fake_img, con_img,exp,global_step):
    save_dir =f"{exp}/sample/{global_step}"
    real_img = (real_img + 1) / 2
    con_img = (con_img + 1) / 2 # -1 - 1 => 0 -1
    fake_img = (fake_img + 1) / 2 # -1 - 1 => 0 -1
    os.makedirs(save_dir, exist_ok=True)
    if len(real_img.size())>4:
        real_img=torch.cat([real_img[:, :, i] for i in range(real_img.size(2))], dim=0)
    if len(con_img.size())>4:
        con_img=torch.cat([con_img[:, :, i] for i in range(con_img.size(2))], dim=0)
    if len(fake_img.size())>4:
        fake_img=torch.cat([fake_img[:, :, i] for i in range(fake_img.size(2))], dim=0)
    ref_img=(con_img[:,3:].clamp(0, 1)*255).byte().cpu().numpy().transpose(0, 2, 3, 1).astype(np.uint8)
    mask_img=(con_img[:,:3].clamp(0, 1)*255).byte().cpu().numpy().transpose(0, 2, 3, 1).astype(np.uint8)
    real_img =(real_img.clamp(0, 1)*255).byte().cpu().numpy().transpose(0, 2, 3, 1).astype(np.uint8)
    fake_img =(fake_img.clamp(0, 1)*255).byte().cpu().numpy().transpose(0, 2, 3, 1).astype(np.uint8)
    sample_num=4
    for i in range(sample_num):
        fake_and_real=np.concatenate([ref_img[i][:,:,::-1],mask_img[i][:,:,::-1],real_img[i][:,:,::-1],fake_img[i][:,:,::-1]],axis=1)
        cv2.imwrite(f"{exp}/sample/{global_step}/{i}.png",fake_and_real)

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

    if real_img.dim() >4:
            real_img = tensor5to4(real_img)
    grad_real, = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty
    # with conv2d_gradfix.no_weight_gradients():
    #     grad_real, = autograd.grad(
    #         outputs=real_pred.sum(), inputs=real_img, create_graph=True
    #     )
    # grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    # return grad_penalty
def tensor5to4(input):
        input_dim_size = len(input.size())
        if input_dim_size > 4:
            b, c, t, h, w = input.size()
            input = input.permute(0, 2, 1, 3, 4).reshape(-1, c, h, w)
        return input

def g_nonsaturating_loss(fake_pred, loss_funcs=None, fake_img=None, real_img=None, input_img=None):
    smooth_l1_loss = loss_funcs[-1]
    g_loss = F.softplus(-fake_pred).mean()
    if real_img.dim() > 4:
        real_img = tensor5to4(real_img)
    loss_l1 = smooth_l1_loss(fake_img, real_img)
    
    return g_loss, loss_l1
def add_module_names(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key ='generator.'+ key
        new_state_dict[new_key]= value
    return new_state_dict

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

def grad_layer_wrt_loss(loss, layer):
    return torch.autograd.grad(
        outputs=loss,
        inputs=layer,
        grad_outputs=torch.ones_like(loss),
        retain_graph=True,
    )[0].detach()

def train(args, loader, generator, discriminator, losses, g_optim, d_optim, g_ema, lpips_func, device):
    loader = sample_data(loader)

    pbar = range(0, args.iter)
    if args.syncnet:
        syncnet=load_syncnet(args.syncnet, args.syncnet_T).eval().to(device)
    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    mean_path_length = 0

    d_loss_val = 0
    r1_loss = torch.tensor(0.0, device=device)
    g_loss_val = 0
    path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)
    mean_path_length_avg = 0
    loss_dict = {}
    os.makedirs(f'{args.exp}/ckeckpoint/', exist_ok= True)
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
        if args.syncnet:
            cond_ref, audio, audio_cropped, real_img = next(loader)
        else:
            cond_ref, audio, real_img = next(loader)
        if args.drop_ref:
            drop = (torch.rand(cond_ref.shape[0]) < args.drop_prob).to(device)
            cond_ref[drop, 3:] = 0
        if real_img.dim() >4:
            real_img = tensor5to4(real_img)

        real_img = real_img.to(device)
        requires_grad(generator, False)
        requires_grad(discriminator, True)
        if args.stage == 2:
           sample_z = cond_ref
        else:
            sample_z = torch.randn(real_img.size(0), 6, real_img.size(2), real_img.size(3))
        sample_z = sample_z.to(device)
        audio = audio.to(device)
        fake_img, _ = generator(inputs = sample_z, audio = audio)
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
       
        fake_img, _ = generator(sample_z, audio = audio)
        fake_pred = discriminator(fake_img)
        
        lpips_loss = lpips_func(real_img, fake_img).mean()
        g_loss, l1_loss = g_nonsaturating_loss(fake_pred, losses, fake_img, real_img, sample_z)
        if args.stage ==2:
            if args.adaptive_weight:
                norm_grad_wrt_perceptual_loss = grad_layer_wrt_loss(lpips_loss, generator.generator.to_rgbs[-1].conv.weight).norm(p=2)
                norm_grad_wrt_gen_loss = grad_layer_wrt_loss(g_loss, generator.generator.to_rgbs[-1].conv.weight).norm(p=2)
                adaptive_weight = norm_grad_wrt_perceptual_loss / norm_grad_wrt_gen_loss.clamp(min=1e-8)
                g_loss *= adaptive_weight
            else:
                g_loss *= args.gan_weight  
            if args.decay_weight and i > args.decay_step and (i - args.decay_step) % 1000 == 0:
                l1_weight = args.l1_weight  * args.decay_rate 
            elif args.warmup_weight and i > args.warmup_step and (i - args.warmup_step) % 1000 == 0:
                l1_weight = args.l1_weight  * args.warmup_rate
                print(l1_weight)
            else:
                l1_weight = args.l1_weight
            l1_loss = l1_weight * l1_loss
            lpips_loss = lpips_loss * args.lpips_weight 
            loss_dict['l1_div_g'] = l1_loss / g_loss
            loss_dict['lpips_div_g'] = lpips_loss / g_loss
            g_loss += l1_loss
            g_loss += lpips_loss
            loss_dict['lpips_loss'] = lpips_loss
            
        else:
            g_loss += l1_loss
        if args.syncnet and i > args.syncnet_step:
            audio_cropped=audio_cropped.to(device)
            _, c, h, w = fake_img.size()
            fake_img = fake_img.reshape(args.batch, args.syncnet_T, c, h, w).permute(0, 2, 1, 3, 4)
            fake_img = torch.cat([fake_img[:, :, i] for i in range(args.syncnet_T)], dim=1)
            a, v=syncnet(audio_cropped,fake_img[:,:,fake_img.size(2)//2:,:])
            d=F.cosine_similarity(a, v).unsqueeze(1)
            y = torch.ones(fake_img.size(0), 1).to(device)
            sync_loss=F.binary_cross_entropy(d, y) 
            sync_loss *= args.sync_loss_wt
            loss_dict["sync_loss"]=sync_loss
            g_loss += sync_loss
        loss_dict['g'] = g_loss
        loss_dict['l1_loss'] = l1_loss
        
        generator.zero_grad()
        g_loss.backward()
        g_optim.step()

        g_regularize = i % args.g_reg_every == 0

        if g_regularize:
            path_batch_size = max(1, args.batch // args.path_batch_shrink)

            fake_img, latents = generator(inputs=sample_z, audio = audio, return_latents=True)

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
        l1_loss = loss_reduced['l1_loss'].mean().item()
        sync_loss = loss_reduced.get('sync_loss',torch.tensor(0.0)).mean().item()
        l1_div_g =  loss_reduced.get('l1_div_g',torch.tensor(0.0)).mean().item()
        lpips_loss = loss_reduced.get('lpips_loss', torch.tensor(0.0)).mean()
        lpips_div_g = loss_reduced.get('lpips_div_g', torch.tensor(0.0)).mean().item()
        path_loss_val = loss_reduced['path'].mean().item()
        real_score_val = loss_reduced['real_score'].mean().item()
        fake_score_val = loss_reduced['fake_score'].mean().item()
        path_length_val = loss_reduced['path_length'].mean().item()

        if get_rank() == 0:
            pbar.set_description(
                (
                    f'd: {d_loss_val:.4f}; g: {g_loss_val:.4f}; r1: {r1_val:.4f}; l1_loss: {l1_loss:.4f}; lpips_loss:{lpips_loss:.4f}; sync_loss:{sync_loss:.4f}; \
                    l1_div_g:{l1_div_g:.4f}; lpips_div_g:{lpips_div_g:.4f}'
                )
            )
            if wandb and args.wandb:
                wandb.log(
                    {
                        "Generator": g_loss_val,
                        "Discriminator": d_loss_val,
                        "l1_loss": l1_loss,
                        "lpips_loss": lpips_loss,
                        "lpips_div_g": lpips_div_g,
                        "l1_div_g": l1_div_g,
                        "R1": r1_val,
                        "sync_loss":sync_loss,
                        "Path Length Regularization": path_loss_val,
                        "Mean Path Length": mean_path_length,
                        "Real Score": real_score_val,
                        "Fake Score": fake_score_val,
                        "Path Length": path_length_val,
                    }
                )
            if i == 0 or i % args.save_freq == 0:
                with torch.no_grad():
                    g_ema.eval()
                    sample, _ = g_ema(inputs= sample_z, audio = audio)
                    save_sample_image(real_img, sample, sample_z,args.exp,i)
                    

            if i and i % args.save_freq == 0:
                torch.save(
                    {
                        'g': g_module.state_dict(),
                        'd': d_module.state_dict(),
                        'g_ema': g_ema.state_dict(),
                        'g_optim': g_optim.state_dict(),
                        'd_optim': d_optim.state_dict(),
                    },
                    f'{args.exp}/ckeckpoint/{str(i).zfill(6)}.pth',
                )

def initialize_missing_params(required_module, g_s):
    missing_module = {}
    for name, param in required_module.items():
        if 'audio_encoder.' in name and 'bias' in name:   
            missing_module[name] = param
            g_s[name] = nn.init.constant_(param, 0.0)
        elif 'audio_encoder.' in name and 'weight' in name:
            g_s[name] = nn.init.kaiming_normal_(param)  
            missing_module[name] = param
        else:
            continue
    return g_s,missing_module

def load_syncnet(syncnet_path,syncnet_T=5):
    syncnet = SyncNet_image_256(syncnet_T=syncnet_T)
    ckpt = torch.load(syncnet_path)
    new_state_dict = {k[len("model."):] if k.startswith("model.") else k: v for k, v in ckpt['state_dict'].items()}
    syncnet.load_state_dict(new_state_dict)
    for param in syncnet.parameters():
        param.requires_grad = False
    return syncnet

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--iter', type=int, default=4000000)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--channel_multiplier', type=int, default=2)
    parser.add_argument('--narrow', type=float, default=1.0)
    parser.add_argument('--r1', type=float, default=10)
    parser.add_argument('--path_regularize', type=float, default=2)
    parser.add_argument('--path_batch_shrink', type=int, default=2)
    parser.add_argument('--d_reg_every', type=int, default=16)
    parser.add_argument('--g_reg_every', type=int, default=4)
    parser.add_argument('--save_freq', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--exp', type=str, help="experment name")
    parser.add_argument('--pretrain', type=str, default=None)
    parser.add_argument('--dataset_config', type=str, default='./data/config.yaml',required=True)
    parser.add_argument('--stage', type=int ,required=True)
    parser.add_argument('--base_dir', type=str, default='./')
    parser.add_argument("--wandb", action="store_true", help="use weights and biases logging")
    parser.add_argument("--syncnet", type= str , help="use syncnet or not")
    parser.add_argument("--syncnet_T", type= int, default=1, help="syncnet_T = 1 or 5")
    parser.add_argument("--sync_loss_wt", type= float, default=0.01, help="syncnet_T = 1 or 5")
    parser.add_argument("--syncnet_step", type= int, default=0, help="sync loss use after this step")
    parser.add_argument("--adaptive_weight", action="store_true", help="use adaptive weight")
    parser.add_argument("--l1_weight", type= float, default=100, help="l1 loss weight")
    parser.add_argument("--gan_weight", type= float, default=0.1, help="gan loss weight")
    parser.add_argument("--lpips_weight", type= float, default=1, help="lpips loss weight")
    parser.add_argument("--decay_weight", action="store_true", help="use decay method")
    parser.add_argument("--decay_step", type= int, default=10000, help="l1 loss decay_rate")
    parser.add_argument("--decay_rate", type= float, default=0.999, help="exponential l1 loss decay_rate")
    parser.add_argument("--warmup_weight", action="store_true", help="use decay method")
    parser.add_argument("--warmup_step", type= int, default=10000, help="l1 loss decay_rate")
    parser.add_argument("--warmup_rate", type= float, default=1.01, help="exponential l1 loss decay_rate")
    parser.add_argument("--apply_noise_injection_intervel", type= bool, default=False, help="add noise skip injection")
    parser.add_argument("--drop_ref", action="store_true", help="random drop ref image")
    parser.add_argument("--drop_prob", type=float, default=0.5, help="drop ref image percentage")
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
    if args.stage:
        generator = FullGenerator(
            args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier, narrow=args.narrow, device=device
        , stage = args.stage, apply_noise_injection_intervel = args.apply_noise_injection_intervel).to(device)
        g_ema = FullGenerator(args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier, narrow=args.narrow, device=device, stage = args.stage, apply_noise_injection_intervel = args.apply_noise_injection_intervel).to(device)
        g_ema.eval()
        accumulate(g_ema, generator, 0)
    else:
        raise ValueError(" required stage1 pretraning, stage2 finetuning", args.stage)
    discriminator = Discriminator(
        args.size, channel_multiplier=args.channel_multiplier, narrow=args.narrow, device=device
    ).to(device)
    
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

    if args.pretrain is not None and args.stage==2:
        print('load model:', args.pretrain)
        ckpt = torch.load(args.pretrain)
       
        required_module = {}
        for name, params in generator.named_parameters():
            required_module[name] = params
        if len(list(required_module.keys())) != len(list(ckpt['g'].keys())):
            print("Missing keys, will initialize with HeKaiming methods")
            g_s, missing_module = initialize_missing_params(required_module, ckpt['g'])
            g_e, _ = initialize_missing_params(required_module, ckpt['g_ema'])
        else:
            g_s = ckpt['g']
            g_e = ckpt['g_ema']
        generator.load_state_dict(g_s)
        discriminator.load_state_dict(ckpt['d'])
        g_ema.load_state_dict(g_e)

               
        if args.stage == 2 and required_module:
            print("using pretrained generator optimizer, add audio encoder optimizer initialize")
            old_g_optim_state_dict = ckpt['g_optim']
            new_g_optim_state_dict = g_optim.state_dict()

            for k, v in old_g_optim_state_dict['state'].items():
                if k in new_g_optim_state_dict['state']:
                    new_g_optim_state_dict['state'][k] = v
            g_optim.load_state_dict(new_g_optim_state_dict)
        else:
            g_optim.load_state_dict(ckpt['g_optim'])
        d_optim.load_state_dict(ckpt['d_optim'])
    
    smooth_l1_loss = torch.nn.SmoothL1Loss().to(device)
    # id_loss = IDLoss(args.base_dir, device, ckpt_dict=None)
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

        # id_loss = nn.parallel.DistributedDataParallel(
        #     id_loss,
        #     device_ids=[args.local_rank],
        #     output_device=args.local_rank,
        #     broadcast_buffers=False,
        # )
    import yaml
    try:
        with open(args.dataset_config, 'r') as file:  
            dataset_config = yaml.safe_load(file)
    except Exception as e:
        print(e)
    if args.stage == 2:
        dataset =HybridDatasetStageTwo(config=dataset_config, split='train', syncnet=args.syncnet, syncnet_T=args.syncnet_T, mel = False, dataset_size=8000) 
    elif args.stage == 1:
        dataset = HybridDatasetStageOne(config=dataset_config, split='train', dataset_size=8000)
    else:
        raise ValueError("unsupport stage for traing, required stage1 pretraning, stage2 finetuning", args.stage)
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=data_sampler(dataset, shuffle=False, distributed=args.distributed),
        drop_last=True,
    )
    if get_rank() == 0 and wandb is not None and args.wandb:
        wandb.init(project='image-styleGAN', entity='local-optima')
    train(args, loader, generator, discriminator, [smooth_l1_loss], g_optim, d_optim, g_ema, lpips_func, device)
   
