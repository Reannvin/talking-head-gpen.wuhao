# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Main training loop."""

import os
import time
import copy
import json
import pickle
import psutil
import PIL.Image
import numpy as np
import torch
import dnnlib
from torch_utils import misc
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import grid_sample_gradfix

import legacy
from metrics import metric_main

from dataset import AudioVisualDataset
from PIL import Image

import random
import wandb

#----------------------------------------------------------------------------

def save_image(img, fname):
    if img.shape[0] > 1:
        img = img[0:1]
    img = np.asarray(img, dtype=np.float32)
    img = (img) * (255)
    img = np.rint(img).clip(0, 255).astype(np.uint8)
    if len(img.shape) == 4:
        C, H, W = img.shape[1:]
    elif len(img.shape) == 3:
        C, H, W = img.shape
        
    img = img.reshape([C, H, W])
    img = img.transpose(1, 2, 0)
    img = img[:, :, ::-1]
    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname)

def training_loop(
    run_dir                 = '.',      # Output directory.
    training_set_kwargs     = {},       # Options for training set.
    G_kwargs                = {},       # Options for generator network.
    D_kwargs                = {},       # Options for discriminator network.
    G_opt_kwargs            = {},       # Options for generator optimizer.
    D_opt_kwargs            = {},       # Options for discriminator optimizer.
    augment_kwargs          = None,     # Options for augmentation pipeline. None = disable.
    loss_kwargs             = {},       # Options for loss function.
    metrics                 = [],       # Metrics to evaluate during training.
    random_seed             = 0,        # Global random seed.
    num_gpus                = 1,        # Number of GPUs participating in the training.
    rank                    = 0,        # Rank of the current process in [0, num_gpus[.
    batch_size              = 4,        # Total batch size for one training iteration. Can be larger than batch_gpu * num_gpus.
    batch_gpu               = 4,        # Number of samples processed at a time by one GPU.
    ema_kimg                = 10,       # Half-life of the exponential moving average (EMA) of generator weights.
    ema_rampup              = 0.05,     # EMA ramp-up coefficient. None = no rampup.
    G_reg_interval          = None,     # How often to perform regularization for G? None = disable lazy regularization.
    D_reg_interval          = 16,       # How often to perform regularization for D? None = disable lazy regularization.
    augment_p               = 0,        # Initial value of augmentation probability.
    ada_target              = None,     # ADA target value. None = fixed p.
    ada_interval            = 4,        # How often to perform ADA adjustment?
    ada_kimg                = 500,      # ADA adjustment speed, measured in how many kimg it takes for p to increase/decrease by one unit.
    total_kimg              = 25000,    # Total length of the training, measured in thousands of real images.
    kimg_per_tick           = 4,        # Progress snapshot interval.
    image_snapshot_ticks    = 5,        # How often to save image snapshots? None = disable.
    network_snapshot_ticks  = 50,       # How often to save network snapshots? None = disable.
    resume_pkl              = None,     # Network pickle to resume training from.
    resume_kimg             = 0,        # First kimg to report when resuming training.
    cudnn_benchmark         = True,     # Enable torch.backends.cudnn.benchmark?
    abort_fn                = None,     # Callback function for determining whether to abort training. Must return consistent results across ranks.
    progress_fn             = None,     # Callback function for updating training progress. Called for all ranks.
    # ---------------------
    data_root               = None,     # The path of the image dataset.
    audio_root              = None,     # The path of the audio dataset.
    fullname                = None,     # Filelists.
    syncnet_t               = 5,        # The syncnet_T for the dataset.
    use_wandb               = False,    # Enable wandb to log
):
    # Initialize.
    start_time = time.time()
    device = torch.device('cuda', rank)
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.backends.cudnn.benchmark = cudnn_benchmark    # Improves training speed.
    torch.backends.cuda.matmul.allow_tf32 = False       # Improves numerical accuracy.
    torch.backends.cudnn.allow_tf32 = False             # Improves numerical accuracy.
    conv2d_gradfix.enabled = True                       # Improves training speed.
    grid_sample_gradfix.enabled = True                  # Avoids errors with the augmentation pipe.
    
    if use_wandb: wandb.init(project="styleGAN_wav2lip", name="wav2lip-HDTF-L1Loss", entity="local-optima")
    # Load training set.
    if rank == 0:
        print('Loading training set...')
    
    print(training_set_kwargs)
    training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs)  # training.dataset.ImageFolderDataset类的对象
    training_set_sampler = misc.InfiniteSampler(dataset=training_set, rank=rank, num_replicas=num_gpus, seed=random_seed)
    train_dataset = AudioVisualDataset(data_root = data_root, 
                                       audio_root = audio_root, 
                                       dataset_name = fullname, 
                                       syncnet_T = syncnet_t,
                                       split = 'main')
    
    training_set_iterator = iter(torch.utils.data.DataLoader(train_dataset,
                                                             sampler=training_set_sampler,
                                                              batch_size=batch_size, 
                                                              num_workers=16,
                                                              pin_memory=True,
                                                              persistent_workers=True,
                                                              prefetch_factor=2))

    if rank == 0:
        print()
        print('Num images: ', len(training_set))
        print('Image shape:', training_set.image_shape)
        print('Label shape:', training_set.label_shape)
        print()
    
    # Construct networks.
    if rank == 0:
        print('Constructing networks...')
    common_kwargs = dict(c_dim=training_set.label_dim, img_resolution=training_set.resolution, img_channels=training_set.num_channels)
    G = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    D = dnnlib.util.construct_class_by_name(**D_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    G_ema = copy.deepcopy(G).eval()

    # Resume from existing pickle.
    if (resume_pkl is not None) and (rank == 0):
        print(f'Resuming from "{resume_pkl}"')
        with dnnlib.util.open_url(resume_pkl) as f:
            resume_data = legacy.load_network_pkl(f)
        for name, module in [('G', G), ('D', D), ('G_ema', G_ema)]:
            misc.copy_params_and_buffers(resume_data[name], module, require_all=False)

    # Print network summary tables.
    if rank == 0:
        z = torch.empty([batch_gpu, G.z_dim], device=device)
        c_i = torch.empty([batch_gpu * syncnet_t, 9, 256, 256], device=device)
        c_a = torch.empty([batch_gpu * syncnet_t, 50, 384], device=device)
        img = misc.print_module_summary(G, [z, c_i, c_a])
        misc.print_module_summary(D, [img, c_i, c_a])

    # Setup augmentation.
    if rank == 0:
        print('Setting up augmentation...')
    augment_pipe = None
    ada_stats = None
    if (augment_kwargs is not None) and (augment_p > 0 or ada_target is not None):
        augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
        augment_pipe.p.copy_(torch.as_tensor(augment_p))
        if ada_target is not None:
            ada_stats = training_stats.Collector(regex='Loss/signs/real')

    # Distribute across GPUs.
    if rank == 0:
        print(f'Distributing across {num_gpus} GPUs...')
    for module in [G, D, G_ema, augment_pipe]:
        if module is not None and num_gpus > 1:
            for param in misc.params_and_buffers(module):
                torch.distributed.broadcast(param, src=0)

    # Setup training phases.
    if rank == 0:
        print('Setting up training phases...')
    loss = dnnlib.util.construct_class_by_name(device=device, G=G, D=D, augment_pipe=augment_pipe, **loss_kwargs) # subclass of training.loss.Loss
    phases = []
    for name, module, opt_kwargs, reg_interval in [('G', G, G_opt_kwargs, G_reg_interval), ('D', D, D_opt_kwargs, D_reg_interval)]:
        if reg_interval is None:
            opt = dnnlib.util.construct_class_by_name(params=module.parameters(), **opt_kwargs) # subclass of torch.optim.Optimizer
            phases += [dnnlib.EasyDict(name=name+'both', module=module, opt=opt, interval=1)]
        else: # Lazy regularization.
            mb_ratio = reg_interval / (reg_interval + 1)
            opt_kwargs = dnnlib.EasyDict(opt_kwargs)
            opt_kwargs.lr = opt_kwargs.lr * mb_ratio
            opt_kwargs.betas = [beta ** mb_ratio for beta in opt_kwargs.betas]
            opt = dnnlib.util.construct_class_by_name(module.parameters(), **opt_kwargs) # subclass of torch.optim.Optimizer
            phases += [dnnlib.EasyDict(name=name+'main', module=module, opt=opt, interval=1)]
            phases += [dnnlib.EasyDict(name=name+'reg', module=module, opt=opt, interval=reg_interval)]
    for phase in phases:
        phase.start_event = None
        phase.end_event = None
        if rank == 0:
            phase.start_event = torch.cuda.Event(enable_timing=True)
            phase.end_event = torch.cuda.Event(enable_timing=True)

    # Export sample images.
    sample_z = None
    sample_c_i = None
    sample_c_a = None

    # Initialize logs.
    if rank == 0:
        print('Initializing logs...')
    stats_collector = training_stats.Collector(regex='.*')
    stats_metrics = dict()
    stats_jsonl = None
    if rank == 0:
        stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'wt')

    # Train.
    if rank == 0:
        print(f'Training for {total_kimg} kimg...')
        print()
    cur_nimg = resume_kimg * 1000
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    batch_idx = 0
    
    if progress_fn is not None:
        progress_fn(0, total_kimg)
    while True:

        # Fetch training data.
        with torch.autograd.profiler.record_function('data_fetch'):
            c_image , c_audio, gt = [item.to(device) for item in next(training_set_iterator)]
            noise_window = torch.randn(batch_size, 3, syncnet_t, 256, 256, dtype=torch.float32, device=device)
            # noise_window = np.random.randn(batch_size, 3, syncnet_t, 256, 256).astype(np.float32)
            
            # c_image = np.concatenate([noise_window, c_image], axis=1)
            c_image = torch.cat([noise_window, c_image], dim = 1)
            c_image = c_image.permute(0, 2, 1, 3, 4).reshape(-1, 9, 256, 256) # B*T, C, W, H (80, 9, 256, 256)

            # For VAE part.
            
            #----------------------------------------------------------------
            c_audio = c_audio.squeeze(2)
            c_audio = c_audio.reshape(-1, 50, 384) # B*T, 50, 384
            
            #----------------------------------------------------------------
            c_audio = c_audio.to(device).split(batch_gpu)
            c_image = c_image.to(device).split(batch_gpu)
            
            gt = gt.permute(0, 2, 1, 3, 4).reshape(-1, 3, 256, 256)
            
            gt = gt.to(device).split(batch_gpu)
            
            # Generate random vector as Generator's input, spilt into each GPU
            
            all_gen_z = torch.randn([len(phases) * batch_size, G.z_dim], device=device)
            all_gen_z = [phase_gen_z.split(batch_gpu) for phase_gen_z in all_gen_z.split(batch_size)]
            
            random_index = random.randint(0, len(train_dataset) - 1)
            # c_image_random, c_audio_random, gt_random = [item.unsqueeze(0).to(device) for item in train_dataset[random_index]]
            dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
            c_image_random, c_audio_random, gt_random = next(iter(dataloader))
            c_image_random = c_image_random.to(device)
            c_audio_random = c_audio_random.to(device)
            noise_window_random = torch.randn(batch_size, 3, syncnet_t, 256, 256, dtype=torch.float32, device=device)
            
            c_image_random = torch.cat([noise_window_random, c_image_random], dim = 1)
            c_image_random = c_image_random.permute(0, 2, 1, 3, 4).reshape(-1, 9, 256, 256) # B*T, C, W, H (80, 9, 256, 256)
            
            c_audio_random = c_audio_random.squeeze(2)
            c_audio_random = c_audio_random.reshape(-1, 50, 384) # B*T, 50, 384
            
            all_gen_c_i = c_image_random.to(device).split(batch_gpu)
            all_gen_c_a = c_audio_random.to(device).split(batch_gpu)
            
        sample_z = torch.randn(batch_size, 256,device=device).split(batch_gpu)
        
        sample_c_i = c_image
        sample_c_a = c_audio
            
        # Execute training phases.
        for phase, phase_gen_z in zip(phases, all_gen_z):                           
        
            if batch_idx % phase.interval != 0:
                continue
            if phase.start_event is not None:
                phase.start_event.record(torch.cuda.current_stream(device))

            # Accumulate gradients.
            phase.opt.zero_grad(set_to_none=True)
            phase.module.requires_grad_(True)
            for real_img, real_ci, real_ca, gen_z, gen_ci, gen_ca in zip(gt, c_image, c_audio, phase_gen_z, all_gen_c_i, all_gen_c_a):
                loss.accumulate_gradients(phase=phase.name,
                                          real_img=real_img, 
                                          real_ci=real_ci, 
                                          real_ca=real_ca, 
                                          gen_z=gen_z, 
                                          gen_ci=gen_ci, 
                                          gen_ca = gen_ca, 
                                          gain=phase.interval, 
                                          cur_nimg=cur_nimg)
            phase.module.requires_grad_(False)

            # Update weights.
            with torch.autograd.profiler.record_function(phase.name + '_opt'):
                params = [param for param in phase.module.parameters() if param.grad is not None]
                if len(params) > 0:
                    flat = torch.cat([param.grad.flatten() for param in params])
                    if num_gpus > 1:
                        torch.distributed.all_reduce(flat)
                        flat /= num_gpus
                    misc.nan_to_num(flat, nan=0, posinf=1e5, neginf=-1e5, out=flat)
                    grads = flat.split([param.numel() for param in params])
                    for param, grad in zip(params, grads):
                        param.grad = grad.reshape(param.shape)
                phase.opt.step()

            # Phase done.
            if phase.end_event is not None:
                phase.end_event.record(torch.cuda.current_stream(device))

        # Update G_ema.
        with torch.autograd.profiler.record_function('Gema'):
            ema_nimg = ema_kimg * 1000
            if ema_rampup is not None:
                ema_nimg = min(ema_nimg, cur_nimg * ema_rampup)
            ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))
            for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                p_ema.copy_(p.lerp(p_ema, ema_beta))
            for b_ema, b in zip(G_ema.buffers(), G.buffers()):
                b_ema.copy_(b)

        # Update state.
        cur_nimg += batch_size
        batch_idx += 1

        # Execute ADA heuristic.
        if (ada_stats is not None) and (batch_idx % ada_interval == 0):
            ada_stats.update()
            adjust = np.sign(ada_stats['Loss/signs/real'] - ada_target) * (batch_size * ada_interval) / (ada_kimg * 1000)
            augment_pipe.p.copy_((augment_pipe.p + adjust).max(misc.constant(0, device=device)))

        # Perform maintenance tasks once per tick.
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<8.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        fields += [f"augment {training_stats.report0('Progress/augment', float(augment_pipe.p.cpu()) if augment_pipe is not None else 0):.3f}"]
        training_stats.report0('Timing/total_hours', (tick_end_time - start_time) / (60 * 60))
        training_stats.report0('Timing/total_days', (tick_end_time - start_time) / (24 * 60 * 60))
        if rank == 0:
            print(' '.join(fields))

        # Check for abort.
        if (not done) and (abort_fn is not None) and abort_fn():
            done = True
            if rank == 0:
                print()
                print('Aborting...')

        # Save image snapshot.
        if (rank == 0) and (image_snapshot_ticks is not None) and (done or cur_tick % image_snapshot_ticks == 0):
            gen_images = [G_ema(z=z, c_i=c_i, c_a=c_a, noise_mode='const').cpu() for z, c_i, c_a in zip(sample_z, sample_c_i, sample_c_a)]
            c_images = [c_i[:,3:6].cpu() for c_i in sample_c_i]
            gt_images = [gt_image.cpu() for gt_image in gt]
            images = torch.cat([gen_images[0], c_images[0], gt_images[0]], dim=3)
            save_image(images, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}.png'))
            
        # Save network snapshot.
        snapshot_pkl = None
        snapshot_data = None
        if (network_snapshot_ticks is not None) and (done or cur_tick % network_snapshot_ticks == 0):
            snapshot_data = dict(G=G, D=D, G_ema=G_ema, augment_pipe=augment_pipe, training_set_kwargs=dict(training_set_kwargs))
            for key, value in snapshot_data.items():
                if isinstance(value, torch.nn.Module):
                    value = copy.deepcopy(value).eval().requires_grad_(False)
                    if num_gpus > 1:
                        misc.check_ddp_consistency(value, ignore_regex=r'.*\.[^.]+_(avg|ema)')
                        for param in misc.params_and_buffers(value):
                            torch.distributed.broadcast(param, src=0)
                    snapshot_data[key] = value.cpu()
                del value # conserve memory
            snapshot_pkl = os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl')
            if rank == 0:
                with open(snapshot_pkl, 'wb') as f:
                    pickle.dump(snapshot_data, f)

        # Evaluate metrics.
        if (snapshot_data is not None) and (len(metrics) > 0):
            if rank == 0:
                print('Evaluating metrics...')
            for metric in metrics:
                result_dict = metric_main.calc_metric(metric=metric, G=snapshot_data['G_ema'],
                    dataset_kwargs=training_set_kwargs, num_gpus=num_gpus, rank=rank, device=device)
                if rank == 0:
                    metric_main.report_metric(result_dict, run_dir=run_dir, snapshot_pkl=snapshot_pkl)
                stats_metrics.update(result_dict.results)
        del snapshot_data # conserve memory

        # Collect statistics.
        for phase in phases:
            value = []
            if (phase.start_event is not None) and (phase.end_event is not None):
                phase.end_event.synchronize()
                value = phase.start_event.elapsed_time(phase.end_event)
            training_stats.report0('Timing/' + phase.name, value)
        stats_collector.update()
        stats_dict = stats_collector.as_dict()

        # Update logs.
        timestamp = time.time()
        if stats_jsonl is not None:
            loss_keys = [key for key in stats_dict.keys() if key.startswith('Loss/')]
            for key in loss_keys:
                value = stats_dict[key]
            fields = dict(stats_dict, timestamp=timestamp)
            stats_jsonl.write(json.dumps(fields) + '\n')
            stats_jsonl.flush()
        if use_wandb:
            global_step = int(cur_nimg / 1e3)
            walltime = timestamp - start_time
            log_dict = {}
            loss_keys = [key for key in stats_dict.keys() if key.startswith('Loss/')]
            for name, value in stats_dict.items():
                log_dict[name] = value.mean

            for name, value in stats_metrics.items():
                log_dict[f'Metrics/{name}'] = value
            log_dict['global_step'] = global_step
            log_dict['walltime'] = walltime
            wandb.log(log_dict, step=global_step)

        if progress_fn is not None:
            progress_fn(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    if rank == 0:
        print()
        print('Exiting...')

#----------------------------------------------------------------------------
