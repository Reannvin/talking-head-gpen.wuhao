import argparse
import math
import random
import os

import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils
from tqdm import tqdm
import cv2
try:
    import wandb

except ImportError:
    wandb = None

from torch.utils.tensorboard import SummaryWriter
from dataset import MultiResolutionDataset, ImageVideoCombineDataset,CombinedDataset,CombinedDataloader
from models import AudioVisualDataset,PerceptualLoss,SyncNet_image_256
from piq import  psnr, ssim, FID
from torchmetrics.functional.image.lpips import learned_perceptual_image_patch_similarity

from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)
from op import conv2d_gradfix
from non_leaking import augment, AdaptiveAugment


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)

def load_syncnet(syncnet_path,syncnet_T=5):
        syncnet = SyncNet_image_256(syncnet_T=syncnet_T)
        ckpt = torch.load(syncnet_path)#, map_location=lambda storage, loc: storage)
        new_state_dict = {k[len("model."):] if k.startswith("model.") else k: v for k, v in ckpt['state_dict'].items()}
        syncnet.load_state_dict(new_state_dict)

        # 冻结 Syncnet 的所有参数
        for param in syncnet.parameters():
            param.requires_grad = False
        return syncnet
def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def grad_layer_wrt_loss(loss, layer):
    return torch.autograd.grad(
        outputs=loss,
        inputs=layer,
        grad_outputs=torch.ones_like(loss),
        retain_graph=True,
    )[0].detach()


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)
    # print("real loss:",real_loss.mean().item())
    # print("fake loss:",fake_loss.mean().item())
    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    with conv2d_gradfix.no_weight_gradients():
        grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True
        )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


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


def make_noise(batch, latent_dim, n_noise, device):
    
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)

    else:
        return [make_noise(batch, latent_dim, 1, device)]


def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None

def get_lpips_loss(model,real_img,fake_img):
    perceptual_loss=PerceptualLoss()
    return perceptual_loss(real_img,fake_img)


def save_sample_image(real_img, fake_img, con_img,exp_name,global_step,ref_num=1):
    save_dir =f"experiments/{exp_name}/sample/{global_step}"
    # print("real image shape:",real_img.shape)
    # print("fake image shape:",fake_img.shape)
    # print("con image shape:",con_img.shape)
    os.makedirs(save_dir, exist_ok=True)
    if len(real_img.size())>4:
        real_img=torch.cat([real_img[:, :, i] for i in range(real_img.size(2))], dim=0)
    if len(con_img.size())>4:
        con_img=torch.cat([con_img[:, :, i] for i in range(con_img.size(2))], dim=0)
    if len(fake_img.size())>4:
        fake_img=torch.cat([fake_img[:, :, i] for i in range(fake_img.size(2))], dim=0)
       # print("fake image shape:",fake_img.shape)
    if ref_num!=0:
        ref_img=(con_img[:,3:].clamp(0, 1)*255).byte().cpu().numpy().transpose(0, 2, 3, 1).astype(np.uint8)
    mask_img=(con_img[:,:3].clamp(0, 1)*255).byte().cpu().numpy().transpose(0, 2, 3, 1).astype(np.uint8)
    real_img =(real_img.clamp(0, 1)*255).byte().cpu().numpy().transpose(0, 2, 3, 1).astype(np.uint8)
    fake_img =( fake_img.clamp(0, 1)*255).byte().cpu().numpy().transpose(0, 2, 3, 1).astype(np.uint8)
    sample_num=8
    for i in range(sample_num):
        if ref_num==0:
            fake_and_real=np.concatenate([mask_img[i],real_img[i],fake_img[i]],axis=1)
        else:
            ref_img_list=[]
            for j in range(ref_num):
                ref_img_list.append(ref_img[i,:,:,3*j:3*(j+1)])
            ref_concat_img=np.concatenate(ref_img_list,axis=1)
            
            #print("ref_concat_img shape:",ref_concat_img.shape)
            fake_and_real=np.concatenate([ref_concat_img,mask_img[i],real_img[i],fake_img[i]],axis=1)
        cv2.imwrite(f"experiments/{exp_name}/sample/{global_step}/{i}.png",fake_and_real)
    # #concat real image in 4 rows and 4 cols
    # real_img_save=np.zeros((256*sample_num,256*sample_num,3),dtype=np.uint8)
    # fake_img_save=np.zeros((256*sample_num,256*sample_num,3),dtype=np.uint8)
    # for i in range(sample_num):
    #    for j in range(sample_num):
    #        real_img_save[i*256:(i+1)*256,j*256:(j+1)*256,:]=real_img[i*sample_num+j]
    #        fake_img_save[i*256:(i+1)*256,j*256:(j+1)*256,:]=fake_img[i*sample_num+j]
    # cv2.imwrite(f"experiments/{exp_name}/sample/real_{global_step}.png",real_img_save)
    # cv2.imwrite(f"experiments/{exp_name}/sample/fake_{global_step}.png",fake_img_save)

def cal_lpips(image,gt,type='vgg'):
    image=image.clamp(0,1)
    gt=gt.clamp(0,1)
    if type=='vgg':
        loss = learned_perceptual_image_patch_similarity(image, gt,net_type=type, normalize=True)
    else:
        loss = 2*learned_perceptual_image_patch_similarity(image, gt,net_type=type, normalize=True)
    return loss

def evaluation_metric(g, gt): #B,C,T,H,W
        # refs = x[:, 3:, :, :, :]
        # inps = x[:, :3, :, :, :]
        # #reshape the tensors to [B*T, C, H, W]
        # refs = (refs.view(-1, refs.size(3), refs.size(4), refs.size(1)).clamp(0, 1) * 255).byte().cpu().numpy().astype(np.uint8)
        # inps = (inps.view(-1, inps.size(3), inps.size(4), inps.size(1)).clamp(0, 1) * 255).byte().cpu().numpy().astype(np.uint8)
        psnr_values = []
        ssim_values = []
        lpips_values = []

        if len(g.size())>4:
            g= g.view(-1, g.size(1), g.size(3), g.size(4)).clamp(0, 1) 
        if len(gt.size())>4:
            gt = gt.view(-1, gt.size(1), gt.size(3), gt.size(4)).clamp(0, 1) 
        #计算g和gt之间的PSNR,SSIM,FID
        psnr_score = psnr(g, gt, reduction='none')
        psnr_values.extend([e.item() for e in psnr_score])
        ssim_score = ssim(g, gt, data_range=1, reduction='none')
        ssim_values.extend([e.item() for e in ssim_score])
        lpips_score = cal_lpips(g, gt)
        lpips_values.extend([e.item() for e in lpips_score])

        

        return np.asarray(psnr_values).mean(), np.asarray(ssim_values).mean(), np.asarray(lpips_values).mean()

def train(args, loader,ft_loader, generator, discriminator,g_optim, d_optim, g_ema, device):
    if args.lpips_loss_wt>0:
        Lpips_loss=PerceptualLoss(network='vgg19',layers=['relu_1_1', 'relu_2_1', 'relu_3_1', 'relu_4_1', 'relu_5_1'],
                                num_scales=2,instance_normalized=True).eval().to(device)
    if args.syncnet:
        syncnet=load_syncnet(args.syncnet,args.syncnet_T).eval().to(device)


    if get_rank() == 0 and not args.wandb:
        writer = SummaryWriter(f"experiments/{args.exp_name}")
    loader = sample_data(loader)
    if ft_loader is not None:
        ft_loader = sample_data(ft_loader)

    pbar = range(args.iter)

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

    if args.distributed:
        g_module = generator.module
        d_module = discriminator.module

    else:
        g_module = generator
        d_module = discriminator

    accum = 0.5 ** (32 / (10 * 1000))
    ada_aug_p = args.augment_p if args.augment_p > 0 else 0.0
    r_t_stat = 0

    if args.augment and args.augment_p == 0:
        ada_augment = AdaptiveAugment(args.ada_target, args.ada_length, 8, device)

    #sample_z = torch.randn(args.n_sample, args.latent, device=device)
    psnr,ssim,lpips,l1=[]
    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print("Done!")

            break
        # if ft_loader is not None and i%args.ft_ratio==0:
        #     tmp_loader=ft_loader
        #     con_image,con_audio,real_img = next(tmp_loader)
        # else:
        #     tmp_loader=loader
        if args.syncnet:
            con_image,con_audio,sync_audio,real_img = next(loader)
        else:
            con_image,con_audio,real_img = next(loader)
        if ft_loader is not None and i%args.ft_ratio==0:
            _,hr_img=next(ft_loader)
            hr_img=hr_img.to(device)
        con_image = con_image.to(device)
        con_audio = con_audio.to(device)
        real_img = real_img.to(device)
        # if len(real_img.size())>4:
        #     real_img=torch.cat([real_img[:, :, i] for i in range(real_img.size(2))], dim=0)
        
        requires_grad(generator,False)
        requires_grad(discriminator, False)


        fake_img = generator(con_image,con_audio)
        psnr_val,ssim_val,lpips_val=evaluation_metric(fake_img,real_img)
        psnr_val,ssim_val,lpips_val=round(psnr_val,2),round(ssim_val,2),round(lpips_val,2)
        psnr.append(psnr_val)
        ssim.append(ssim_val)
        lpips.append(lpips_val)
        l1_loss=F.l1_loss(fake_img,real_img)
        l1_val=round(l1_loss.item(),2)
        l1.append(l1_val)
    print(f"psnr:{np.mean(psnr)},ssim:{np.mean(ssim)},lpips:{np.mean(lpips)},l1:{np.mean(l1)}")
       # print("fake image shape:",fake_img.shape)
        # if args.augment:
        #     fake_img, _ = augment(fake_img, ada_aug_p)

       
        


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="StyleGAN2 trainer")

   # parser.add_argument("path", type=str, help="path to the lmdb dataset")
    parser.add_argument('--arch', type=str, default='stylegan2', help='model architectures (stylegan2 | swagan)')
    parser.add_argument(
        "--iter", type=int, default=800000, help="total training iterations"
    )
    parser.add_argument(
        "--batch", type=int, default=16, help="batch sizes for each gpus"
    )
    parser.add_argument(
        "--n_sample",
        type=int,
        default=64,
        help="number of the samples generated during training",
    )
    parser.add_argument(
        "--size", type=int, default=256, help="image sizes for the model"
    )
    parser.add_argument(
        "--r1", type=float, default=10, help="weight of the r1 regularization"
    )
    parser.add_argument(
        "--path_regularize",
        type=float,
        default=2,
        help="weight of the path length regularization",
    )
    parser.add_argument(
        "--path_batch_shrink",
        type=int,
        default=2,
        help="batch size reducing factor for the path length regularization (reduce memory consumption)",
    )
    parser.add_argument(
        "--d_reg_every",
        type=int,
        default=16,
        help="interval of the applying r1 regularization",
    )
    parser.add_argument(
        "--g_reg_every",
        type=int,
        default=4,
        help="interval of the applying path length regularization",
    )
    parser.add_argument(
        "--mixing", type=float, default=0.9, help="probability of latent code mixing"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="path to the checkpoints to resume training",
    )
    parser.add_argument("--lr", type=float, default=0.002, help="learning rate")
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier factor for the model. config-f = 2, else = 1",
    )
    parser.add_argument(
        "--wandb", action="store_true", help="use weights and biases logging"
    )
    parser.add_argument(
        "--local_rank", type=int, default=0, help="local rank for distributed training"
    )
    parser.add_argument(
        "--local-rank", type=int, default=0, help="local rank for distributed training"
    )
    parser.add_argument(
        "--augment", action="store_true", help="apply non leaking augmentation"
    )
    parser.add_argument(
        "--augment_p",
        type=float,
        default=0,
        help="probability of applying augmentation. 0 = use adaptive augmentation",
    )
    parser.add_argument(
        "--ada_target",
        type=float,
        default=0.6,
        help="target augmentation probability for adaptive augmentation",
    )
    parser.add_argument(
        "--ada_length",
        type=int,
        default=500 * 1000,
        help="target duraing to reach augmentation probability for adaptive augmentation",
    )
    parser.add_argument(
        "--ada_every",
        type=int,
        default=256,
        help="probability update interval of the adaptive augmentation",
    )
    parser.add_argument('--wav2vec2', action='store_true', help='Use wav2vec2 embeddings')
    parser.add_argument('--init', action='store_true', help='init a new exp')
    parser.add_argument('--syncnet', type=str, help='Path to the syncnet checkpoint to load the model from')
    parser.add_argument('--syncnet_T', type=int, default=1, help='Number of frames to consider for syncnet loss')
    parser.add_argument('--dataset_size', type=int, default=8000, help='Size of the dataset')
    parser.add_argument("--data_root", help="Root folder of the preprocessed LRS2 dataset", default='/data/wangbaiqin/dataset/all_images')
    parser.add_argument("--ft_root", help="Root folder of the preprocessed LRS2 dataset", default=None)
    parser.add_argument('--ft_ratio', type=float, default=9, help='ratio of the finetune dataset')
    parser.add_argument('--audio_root', type=str, help='Root folder of the preprocessed audio dataset',default='/data/wangbaiqin/dataset/all_audios')
    parser.add_argument("--dataset_name", help="Name of the dataset to use, eg: 3300, hdtf, 3300_and_hdtf", required=True)
    parser.add_argument("--exp_name", help="Name of exp", type=str,default='exp')
    parser.add_argument("--ref_num", help="reference image number",default=1, type=int)
    parser.add_argument('--gs_blur', action='store_true', help='Enable Gaussian blur when mask image')
    parser.add_argument('--sample_images', type=int, default=1000, help='Number sample image iter')
    parser.add_argument('--save_iter', type=int, default=10000, help='checkpoint save iter')
    parser.add_argument('--gan_loss_wt', type=float, default=0.01, help='Gan loss weight')
    parser.add_argument('--lpips_loss_wt', type=float, default=0.1, help='lpips loss weight')
    parser.add_argument('--image_loss_wt', type=float, default=10, help='image loss weight')
    parser.add_argument('--sync_loss_wt', type=float, default=0.03, help='image loss weight')
    parser.add_argument('--drop_ref_prob', type=float, default=0, help='drop ref prob')
    parser.add_argument('--adap_wt', action='store_true', help='adaptive weight')
    parser.add_argument('--disc_ckpt', type=str, help='Path to the discriminator checkpoint to load the model from')
    parser.add_argument('--gen_ckpt', type=str, help='Path to the generator checkpoint to load the model from')
    parser.add_argument('--pre_type', type=str, help='pretrained type',default='ori')
   # parser.add_argument('--sync_ckpt', type=str, help='Path to the syncnet checkpoint to load the model from')
    args = parser.parse_args()

    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = n_gpu > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")#,world_size=n_gpu,rank = torch.cuda.device_count())
        synchronize()

    args.latent = 512
    args.n_mlp = 8

    args.start_iter = 0

    if args.arch == 'stylegan2':
        from models import Discriminator,FullGenerator  

    elif args.arch == 'swagan':
        from swagan import Generator, Discriminator
    
    if args.pre_type=='ori' or args.pre_type=='inpaint':
        args.end_act=True
    else:
        args.end_act=False
    generator = FullGenerator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier,
        ref_num=args.ref_num,end_act=args.end_act#,inpaint=(args.pre_type=='inpaint')
    ).to(device)
    g_ema = FullGenerator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier,
        ref_num=args.ref_num,end_act=args.end_act#,inpaint=(args.pre_type=='inpaint')
    ).to(device)
    g_ema.eval()
    discriminator = Discriminator(
        args.size, channel_multiplier=args.channel_multiplier,syncnet_T=args.syncnet_T
    ).to(device)

    accumulate(g_ema, generator, 0)
    

    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)
   
    if args.pre_type=='ori':
        param_groups=[{'params': generator.parameters(), 'lr': args.lr * g_reg_ratio},]
    elif args.pre_type=='style':
        param_to_optimize=set(generator.parameters())-set(generator.generator.parameters())
        param_to_optimize=list(param_to_optimize)
        param_groups=[{'params': generator.generator.parameters(), 'lr': args.lr * g_reg_ratio},#//10},
                      {'params': param_to_optimize, 'lr': args.lr * g_reg_ratio},]
    elif args.pre_type=='inpaint':
        #exclude_param=set(generator.generator.parameters()).union(set(generator.names[1:].parameters())) 
        #param_to_optimize=set(generator.parameters())-exclude_param
        # bcd_param=set()
        # for i in range(len(generator.names2)):
        #     bcd_param=bcd_param.union(set(generator.__getattr__(generator.names2[i]).parameters()))
        # exclude_param=set(generator.audio_encoder.parameters())#.union(bcd_param)
        # exclude_param2=set(generator.generator.style.parameters())
        # param_to_optimize=set(generator.parameters())-exclude_param-exclude_param2
        # exclude_param=list(exclude_param)
        # exclude_param2=list(exclude_param2)
        # param_to_optimize=list(param_to_optimize)
        # param_groups=[{'params': exclude_param, 'lr': args.lr * g_reg_ratio},
        #                 {'params': exclude_param2, 'lr': args.lr * g_reg_ratio},
        #               {'params': param_to_optimize, 'lr': args.lr * g_reg_ratio//10},]
        param_groups=[{'params': generator.parameters(), 'lr': args.lr * g_reg_ratio},]
    g_optim = optim.Adam(
            param_groups,
            #lr=args.lr * g_reg_ratio,
            betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
        )
    d_optim = optim.Adam(
        discriminator.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )
    if args.ckpt is not None:
        print("load model:", args.ckpt)

        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)

        try:
            ckpt_name = os.path.basename(args.ckpt)
            if not args.init:
                args.start_iter = int(os.path.splitext(ckpt_name)[0])

        except ValueError:
            pass

        generator.load_state_dict(ckpt["g"])
        discriminator.load_state_dict(ckpt["d"])
        g_ema.load_state_dict(ckpt["g_ema"])
       

        g_optim.load_state_dict(ckpt["g_optim"])
        d_optim.load_state_dict(ckpt["d_optim"])

    if args.disc_ckpt is not None:
        print("load disc model from:", args.disc_ckpt)

        ckpt = torch.load(args.disc_ckpt, map_location=lambda storage, loc: storage)

        discriminator.load_state_dict(ckpt["d"])
        #d_optim.load_state_dict(ckpt["d_optim"])
    if args.gen_ckpt is not None:
        print("load gen model from:", args.gen_ckpt)

        ckpt = torch.load(args.gen_ckpt, map_location=lambda storage, loc: storage)
        if args.pre_type=='ori':
            generator.load_state_dict(ckpt["g"])
        elif args.pre_type=='style':
            generator.generator.load_state_dict(ckpt["g"])
        else:
        #    ckpt["g"].pop('ecd0.0.0.weight')
            generator.load_state_dict(ckpt["g"],strict=False)
    if args.distributed:
        generator = nn.parallel.DistributedDataParallel(
            generator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )

        discriminator = nn.parallel.DistributedDataParallel(
            discriminator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )
        
            

    # transform = transforms.Compose(
    #     [
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
    #     ]
    # )

    # dataset = MultiResolutionDataset(args.path, transform, args.size)
    # loader = data.DataLoader(
    #     dataset,
    #     batch_size=args.batch,
    #     sampler=data_sampler(dataset, shuffle=True, distributed=args.distributed)
    #     drop_last=True,
    # )
    dataset=AudioVisualDataset(args.data_root, args.dataset_name, 'main', args, audio_root=args.audio_root, dataset_size=args.dataset_size,up_ratio=0,gs_blur=args.gs_blur)#transform=transform)
    loader=data.DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=data_sampler(dataset, shuffle=True, distributed=args.distributed),
        num_workers=4,
       # transforms=transform,
    )

    if args.ft_root!=None:
        ft_dataset=ImageVideoCombineDataset(args.ft_root,base_ratio=1)
        ft_loader=data.DataLoader(
            ft_dataset,
            batch_size=args.batch//2,
            sampler=data_sampler(ft_dataset, shuffle=True, distributed=args.distributed),
            num_workers=4,
           # transforms=transform,
        )
    else:
        ft_loader=None

    
    if get_rank() == 0 and wandb is not None and args.wandb:
        wandb.init(project="stylegan 2")

    if get_rank() == 0:
        print("gen:",sum(p.numel() for p in generator.parameters() if p.requires_grad))
        print("disc:",sum(p.numel() for p in discriminator.parameters() if p.requires_grad))
    if not os.path.exists(f"experiments/{args.exp_name}") and get_rank() == 0:
       
        os.makedirs(f"experiments/{args.exp_name}/checkpoint")
        os.makedirs(f"experiments/{args.exp_name}/sample", exist_ok=True)
    train(args, loader,ft_loader, generator, discriminator, g_optim, d_optim, g_ema, device)
