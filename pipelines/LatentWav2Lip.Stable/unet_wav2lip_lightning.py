from os.path import dirname, join, basename, isfile
from glob import glob
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os, random, argparse
from hparams import hparams, get_image_list
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import RichProgressBar, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from lr_scheduler import LambdaLinearScheduler
import cv2
from diffusers import AutoencoderKL, UNet2DConditionModel
from models import SyncNet_image_256
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from diffusers import DDPMScheduler
from diffusers.training_utils import EMAModel
from tqdm import tqdm
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from diffusers.utils.import_utils import is_xformers_available


class CombinedDataset(Dataset):
    def __init__(self, base_dataset, fine_tune_dataset=None, base_ratio=0.5):
        self.base_dataset = base_dataset
        self.fine_tune_dataset = fine_tune_dataset
        self.base_ratio = base_ratio
        self.base_length = len(base_dataset)
        self.fine_tune_length = len(fine_tune_dataset) if fine_tune_dataset is not None else 0
        
    def __len__(self):
        if self.fine_tune_dataset is None:
            return self.base_length
        return self.base_length + self.fine_tune_length

    def __getitem__(self, idx):
        if self.fine_tune_dataset is None or random.random() < self.base_ratio:
            base_idx = random.randint(0, self.base_length - 1)
            return self.base_dataset[base_idx]
        else:
            fine_tune_idx = random.randint(0, self.fine_tune_length - 1)
            return self.fine_tune_dataset[fine_tune_idx]


class AudioVisualDataset(Dataset):
    def __init__(self, data_root, dataset_name, split, args, audio_root=None, dataset_size=512000):
        self.all_videos = get_image_list(data_root, dataset_name, split)
        self.dataset_size = dataset_size
        self.syncnet_audio_size = 10 * args.syncnet_T if not args.wav2vec2 else 2 * args.syncnet_T
        self.frame_audio_size = 10 * 5 if not args.wav2vec2 else 2 * 5 # always use syncnet_T=5 for each frame
        self.args = args
        self.data_root = data_root
        self.audio_root = audio_root
        
    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])

    def get_window(self, start_frame):
        start_id = self.get_frame_id(start_frame)
        vidname = dirname(start_frame)

        window_fnames = []
        for frame_id in range(start_id, start_id + self.args.syncnet_T):
            frame = join(vidname, '{}.jpg'.format(frame_id))
            if not isfile(frame):
                return None
            window_fnames.append(frame)
        return window_fnames

    def read_window(self, window_fnames):
        if window_fnames is None: return None
        window = []
        for fname in window_fnames:
            img = cv2.imread(fname)
            if img is None:
                return None
            try:
                img = cv2.resize(img, (hparams.image_size, hparams.image_size))
            except Exception as e:
                return None

            window.append(img)

        return window
    
    def get_whisper_embedding(self, vidname, frame_id, syncnet_T):
        try:
            whisper_file = f"{frame_id}.npy" if syncnet_T == 5 else f"{frame_id}.npy.{syncnet_T}.npy"
            audio_path = join(vidname, whisper_file)
            audio_embedding = np.load(audio_path)
            audio_embedding = torch.from_numpy(audio_embedding)
        except:
            print(f"Error loading {audio_path}")
            audio_embedding = None
        return audio_embedding
    
    def get_whisper_segmented_audios(self, vidname, frame_id):
        audios = []
        offset = self.args.syncnet_T // 2
        start_frame_num = frame_id + 1
        if start_frame_num - offset < 0: return None
        for i in range(start_frame_num, start_frame_num + self.args.syncnet_T):
            m = self.get_whisper_embedding(vidname, i - offset, syncnet_T=5) # always use syncnet_T=5 for each frame
            if m is None or m.shape[0] != self.frame_audio_size:
                return None
            audios.append(m)

        audios = torch.stack(audios)
        return audios
    
    def crop_audio_window(self, audio_embeddings, start_frame, syncnet_T):
        if type(start_frame) == int:
            start_frame_num = start_frame
        else:
            start_frame_num = self.get_frame_id(start_frame) # 0-indexing ---> 1-indexing
        start_idx = int(50. * (start_frame_num / float(hparams.fps)))
        end_idx = start_idx + 2 * syncnet_T
        return audio_embeddings[start_idx : end_idx]
    
    def get_segmented_audios(self, audio_embeddings, start_frame):
        audios = []
        offset = self.args.syncnet_T // 2
        start_frame_num = self.get_frame_id(start_frame) + 1
        if start_frame_num - offset < 0: return None
        for i in range(start_frame_num, start_frame_num + self.args.syncnet_T):
            m = self.crop_audio_window(audio_embeddings, i - offset, syncnet_T=5) # always use syncnet_T=5 for each frame
            if m.shape[0] != self.frame_audio_size:
                return None
            audios.append(m)

        audios = torch.stack(audios)
        return audios

    def prepare_window(self, window):
        # 3 x T x H x W
        x = np.asarray(window) / 255.
        x = np.transpose(x, (3, 0, 1, 2))

        return x

    def __len__(self):
        return self.dataset_size # len(self.all_videos)

    def __getitem__(self, idx):
        while 1:
            idx = random.randint(0, len(self.all_videos) - 1)
            vidname = self.all_videos[idx]
            img_names = list(glob(join(vidname, '*.jpg')))
            if len(img_names) <= 3 * self.args.syncnet_T:
                # print(f"Video {vidname} has less frames than required")
                continue
            
            img_name = random.choice(img_names)
            wrong_img_name = random.choice(img_names)
            # Ensure wrong_img_name is at least syncnet_T frames away from img_name
            while abs(self.get_frame_id(img_name) - self.get_frame_id(wrong_img_name)) < self.args.syncnet_T:
                wrong_img_name = random.choice(img_names)

            window_fnames = self.get_window(img_name)
            wrong_window_fnames = self.get_window(wrong_img_name)
            if window_fnames is None or wrong_window_fnames is None:
                continue

            window = self.read_window(window_fnames)
            if window is None:
                # print(f"Window is None for {vidname}")
                continue

            wrong_window = self.read_window(wrong_window_fnames)
            if wrong_window is None:
                # print(f"Wrong Window is None for {vidname}")
                continue
            
            if self.audio_root:
                # switch fro data_root to audio_root
                vidname = vidname.replace(self.data_root, self.audio_root)
                
            if not args.wav2vec2:
                # load syncnet_T frames of audio embeddings for syncnet loss
                if self.args.syncnet:
                    audio_cropped = self.get_whisper_embedding(vidname, self.get_frame_id(img_name), syncnet_T=self.args.syncnet_T)
                    
                # always load for each frame of 5 frames of audio embeddings for cross attention
                indiv_audios = self.get_whisper_segmented_audios(vidname, self.get_frame_id(img_name))
            else:
                # load audio embedding from file wav2vec2.pt
                audio_path = join(vidname, "wav2vec2.pt")
                audio_embeddings = torch.load(audio_path, map_location='cpu')[0]
                
                # load syncnet_T frames of audio embeddings for syncnet loss
                audio_cropped = self.crop_audio_window(audio_embeddings.clone(), img_name, syncnet_T=self.args.syncnet_T)
                # always load for each frame of 5 frames of audio embeddings for cross attention
                indiv_audios = self.get_segmented_audios(audio_embeddings.clone(), img_name)
            
            if self.args.syncnet and (audio_cropped is None or audio_cropped.shape[0] != self.syncnet_audio_size): continue
            
            if indiv_audios is None: continue

            window = self.prepare_window(window)
            y = window.copy()
            
            # mask lower part of the image, according to mask_ratio
            mask_height = int(window.shape[2] * self.args.mask_ratio)
            window[:, :, window.shape[2] - mask_height:] = 0.
            
            wrong_window = self.prepare_window(wrong_window)
            x = np.concatenate([window, wrong_window], axis=0)
            
            x = torch.FloatTensor(x)
            
            if self.args.syncnet:
                audio_cropped = audio_cropped.unsqueeze(0).float()
            
            indiv_audios = indiv_audios.unsqueeze(1).float()
            y = torch.FloatTensor(y)
            
            # don't return audio_cropped if syncnet is not enabled
            if not self.args.syncnet:
                return x, indiv_audios, y
            
            return x, indiv_audios, audio_cropped, y
        
class LatentWav2Lip(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        unet_config = UNet2DConditionModel.load_config(f"{self.hparams.unet_config}/config.json")
        self.unet = UNet2DConditionModel.from_config(unet_config)        
                
        # Enable memory efficient attention with xformers if available 
        if is_xformers_available():
            try:
                print("Enabling memory efficient attention with xformers...")
                self.unet.enable_xformers_memory_efficient_attention()
            except Exception as e:
                print(f"Could not enable memory efficient attention. Make sure xformers is installed correctly and a GPU is available: {e}")
        else:
            print("xformers is not available, using default attention mechanism...")
     
        # self.zero_timestep = torch.zeros([])
        
        if self.hparams.syncnet:
            self.syncnet = self.load_syncnet()
            self.syncnet.eval()  # Ensure syncnet is always in eval mode
        
        self.vae = self.load_vae('stabilityai/sd-vae-ft-mse')
        self.vae.eval()  # Ensure vae is always in eval mode
        
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

        if self.hparams.ema:
            self.ema = EMAModel(self.unet.parameters())
        
        # 定义损失函数
        self.recon_loss = nn.L1Loss() if not self.hparams.l2_loss else nn.MSELoss()
        self.log_loss = nn.BCELoss()
        self.lpips_loss = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True)
        
        self.lora_config = None  # 初始化为 None
        
        if self.unet.config.cross_attention_dim != 384:
            print(f"Cross attention dim is {self.unet.config.cross_attention_dim}, adapting from 384")
            self.audio_adapter = torch.nn.Linear(384, self.unet.config.cross_attention_dim)
    
    def load_syncnet(self):
        syncnet = SyncNet_image_256(not self.hparams.wav2vec2, self.hparams.syncnet_T)
        ckpt = torch.load(self.hparams.syncnet)
        new_state_dict = {k[len("model."):] if k.startswith("model.") else k: v for k, v in ckpt['state_dict'].items()}
        syncnet.load_state_dict(new_state_dict)

        # 冻结 Syncnet 的所有参数
        for param in syncnet.parameters():
            param.requires_grad = False
        return syncnet
    
    def load_vae(self, model_name):
        vae = AutoencoderKL.from_pretrained(model_name, local_files_only=True)
        
        # 冻结 VAE 的所有参数
        for param in vae.parameters():
            param.requires_grad = False
        return vae
    
    def load_unet(self, unet_ckpt):
        try:
            self.unet.from_pretrained(unet_ckpt, low_cpu_mem_usage=False, ignore_mismatched_sizes=True)
            print(f"Loaded U-Net from {unet_ckpt}")
        except Exception as e:
            # print(f"Error loading U-Net from {unet_ckpt}: {e}")
            print("Train UNet from scratch")
        return self.unet
    
    def add_lora(self, lora_config):
        self.lora_config = lora_config
        self.unet.add_adapter(lora_config)
        
        if self.hparams.ema:
            # reinitialize ema model
            self.ema = EMAModel(self.unet.parameters())

    def on_save_checkpoint(self, checkpoint):
        if self.lora_config is not None:
            lora_params = {k: v for k, v in self.unet.named_parameters() if 'lora' in k}
            # checkpoint['lora_params'] = lora_params
            # make checkpoint only includes lora params, not the whole model
            checkpoint['state_dict'] = lora_params
        
        # save ema model
        if self.hparams.ema:
            checkpoint['ema'] = self.ema.state_dict()
            
    def on_load_checkpoint(self, checkpoint):                
        # load ema model
        if self.hparams.ema:
            if 'ema' in checkpoint:
                self.ema.load_state_dict(checkpoint['ema'])
                self.ema.to(self.device)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        # Call the parent class's state_dict method
        state_dict = super(LatentWav2Lip, self).state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        # Remove the VAE and syncnet parameters from the state_dict
        keys_to_remove = [key for key in state_dict.keys() if key.startswith('vae') or key.startswith('syncnet')]
        for key in keys_to_remove:
            del state_dict[key]
        return state_dict

    def load_state_dict(self, state_dict, strict=True):
        # Remove the VAE and syncnet parameters from the incoming state_dict
        keys_to_remove = [key for key in state_dict.keys() if key.startswith('vae') or key.startswith('syncnet')]
        for key in keys_to_remove:
            del state_dict[key]
                
        # if lora params, append unet. in the front
        new_state_dict = {}
        for key in state_dict.keys():
            if 'lora' in key and not key.startswith('unet.'):
                new_state_dict[f"unet.{key}"] = state_dict[key]
            else:
                new_state_dict[key] = state_dict[key]
        
        # Call the parent class's load_state_dict method
        super(LatentWav2Lip, self).load_state_dict(new_state_dict, strict=False)
    
    def cosine_loss(self, a, v, y):
        d = nn.functional.cosine_similarity(a, v)
        # Disable autocast for unsafe operations
        with torch.cuda.amp.autocast(enabled=False):
            loss = self.log_loss(d.unsqueeze(1), y)
        return loss
    
    def get_sync_loss(self, mel, g):
        # B, 4 * T, H, W
        g = torch.cat([g[:, :, i] for i in range(self.hparams.syncnet_T)], dim=1)
        
        # if image size is not [128, 256], resize the image to [128, 256]
        if g.size(2) != 128 or g.size(3) != 256:
            g = nn.functional.interpolate(g, size=(128, 256), mode='bilinear')
        
        a, v = self.syncnet(mel, g)
        y = torch.ones(g.size(0), 1).to(self.device)
        loss = self.cosine_loss(a, v, y)
        return loss
    
    def get_lpips_loss(self, g, gt):
        # Expected both input arguments to be normalized tensors with shape [N, 3, H, W].
        if len(g.shape) != 4 or len(gt.shape) != 4 or g.shape[1] != 3 or gt.shape[1] != 3:
            # reshape the tensors to [N, 3, H, W]
            g = self.reshape_gt_image_for_vae(g)
            gt = self.reshape_gt_image_for_vae(gt)
        
        # make sure g is in [0, 1] range
        g = torch.clamp(g, 0., 1.)
        
        loss = self.lpips_loss(g, gt)
        return loss
    
    def inverse_reshape_face_sequences(self, tensor):
        """
        Inverse operation for the reshape_face_sequences function, reconstructing the original tensor
        from a reshaped format of [batch_size, channels * groups, height, width].
        
        Parameters:
            tensor (torch.Tensor): A tensor with dimensions [batch_size * groups, channels, height, width].
        
        Returns:
            torch.Tensor: A tensor with dimensions [batch_size, channels, groups, height, width].
        """
        total_batch_size, channels, height, width = tensor.shape
        groups = self.hparams.syncnet_T
        batch_size = total_batch_size // groups
        
        # check if the total batch size is divisible by the number of groups
        if total_batch_size % groups != 0:
            raise ValueError("Total batch size is not divisible by the number of groups.")
        
        # Reshape the tensor to its original dimensions
        original_shape_tensor = tensor.view(batch_size, groups, channels, height, width).permute(0, 2, 1, 3, 4)        
        return original_shape_tensor
    
    def reshape_face_sequences_for_vae(self, tensor): # [8, 6, 5, 768, 768] -> [80, 3, 768, 768]
        batch_size, double_channels, groups, height, width = tensor.shape
        reshaped_tensor = tensor.permute(0, 2, 1, 3, 4).contiguous().view(batch_size * groups * 2, double_channels // 2, height, width)
        return reshaped_tensor
    
    def reshape_gt_image_for_vae(self, tensor): # # [8, 3, 5, 768, 768] -> [40, 3, 768, 768]
        batch_size, channels, groups, height, width = tensor.shape
        reshaped_tensor = tensor.permute(0, 2, 1, 3, 4).contiguous().view(batch_size * groups, channels, height, width)
        return reshaped_tensor
    
    def reshape_audio_sequences_for_unet(self, tensor):
        batch_size, dim1, dim2, dim3, features = tensor.shape
        reshaped_tensor = tensor.view(batch_size * dim1, dim2 * dim3, features)
        return reshaped_tensor
    
    def reshape_latent_faces_for_unet(self, tensor):
        # [batch_size * 2, channels, height, width] -> [batch_size, channels * 2, height, width]
        batch_size_times_2, channels, height, width = tensor.shape
        batch_size = batch_size_times_2 // 2
        reshaped_tensor = tensor.view(batch_size, channels * 2, height, width)
        return reshaped_tensor

    def encode_with_vae(self, face_sequences):
        # face_sequences are images of [0, 1] range
        face_sequences = face_sequences * 2. - 1.
        
        latent_face_sequences = self.vae.encode(face_sequences).latent_dist.sample()
        
        # scale the latent space to have unit variance when training unet
        scaling_factor = self.vae.config.scaling_factor
        latent_face_sequences = latent_face_sequences * scaling_factor
        return latent_face_sequences
    
    def decode_with_vae(self, latent_face_sequences):
        # scale the latent from unet when decoding with vae
        scaling_factor = self.vae.config.scaling_factor
        latent_face_sequences = latent_face_sequences / scaling_factor
        
        image_face_sequences = self.vae.decode(latent_face_sequences).sample
        
        # convert the image to [0, 1] range
        image_face_sequences = (image_face_sequences + 1.) / 2.
        return image_face_sequences
    
    def sample_noise(self, shape):
        # Sample noise to add to the images
        noise = torch.randn(shape, device=self.device)
        return noise
    
    def random_timestep(self, batch_size):
        timestep = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (batch_size,), device=self.device, dtype=torch.long)
        return timestep
    
    def get_noisy_image(self, clean_image, timestep):
        noise = self.sample_noise(clean_image.shape)
        noisy_image = self.noise_scheduler.add_noise(clean_image, noise, timestep)
        return noisy_image, noise
    
    def forward(self, audio_condition, image_condition, noisy_target, timestep):
        image_condition = self.reshape_face_sequences_for_vae(image_condition)
        latent_image_condition = self.encode_with_vae(image_condition)

        latent_image_condition = self.reshape_latent_faces_for_unet(latent_image_condition)
        audio_condition = self.reshape_audio_sequences_for_unet(audio_condition) 
        
        # concat latent image condition with noisy target
        image_tensor = torch.cat([latent_image_condition, noisy_target], dim=1)
        
        if self.unet.config.cross_attention_dim != 384:
            audio_condition = self.audio_adapter(audio_condition)
        
        if self.hparams.cfg and not self.training:
            unconditional_audio = torch.zeros_like(audio_condition).to(self.device)
            combined_audio = torch.cat([unconditional_audio, audio_condition])
            image_tensor = torch.cat([image_tensor] * 2)
            timestep = torch.cat([timestep] * 2)
            
            pred = self.unet(image_tensor, timestep=timestep, encoder_hidden_states=combined_audio).sample
            uncond_pred, cond_pred = pred.chunk(2)
            g_niose = uncond_pred + self.hparams.guidance_scale * (cond_pred - uncond_pred)
        else:
            g_niose = self.unet(image_tensor, timestep=timestep, encoder_hidden_states=audio_condition).sample      
                  
        return g_niose
    
    def training_step(self, batch, batch_idx):
        if self.hparams.syncnet:
            x, indiv_audios, audio_cropped, gt = batch
        else:
            x, indiv_audios, gt = batch

        # to drop the reference image, based on image_drop_prob
        if self.hparams.image_drop_prob > 0.:
            need_dropout = random.random() < self.hparams.image_drop_prob
            if need_dropout:
                x = torch.zeros_like(x)
        
        # to drop the audio condition, based on audio_drop_prob
        if self.hparams.audio_drop_prob > 0.:
            need_dropout = random.random() < self.hparams.audio_drop_prob
            if need_dropout:
                indiv_audios = torch.zeros_like(indiv_audios)
                
        # get gt latent
        gt_for_vae = self.reshape_gt_image_for_vae(gt)        
        gt_latent = self.encode_with_vae(gt_for_vae)
        
        # get random timestep
        unet_batch_size = gt_latent.shape[0]
        timestep = self.random_timestep(unet_batch_size)
        
        # make latent_face_sequences noisy
        noisy_gt_latent, noise = self.get_noisy_image(gt_latent, timestep)
        
        g_noise = self(indiv_audios, x, noisy_gt_latent, timestep)
        
        loss = self.recon_loss(g_noise, noise)
        self.log('train_loss', loss, prog_bar=True, sync_dist=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.hparams.syncnet:
            x, indiv_audios, audio_cropped, gt = batch
        else:
            x, indiv_audios, gt = batch

        if self.hparams.pretrain_unconditional:
            # drop image and audio condition
            x = torch.zeros_like(x).to(self.device)
            indiv_audios = torch.zeros_like(indiv_audios).to(self.device)
        elif self.hparams.pretrain_image_condition:
            # drop audio condition
            indiv_audios = torch.zeros_like(indiv_audios).to(self.device)
            
        # get gt latent
        gt_for_vae = self.reshape_gt_image_for_vae(gt)        
        gt_latent = self.encode_with_vae(gt_for_vae)
        
        # get random timestep
        unet_batch_size = gt_latent.shape[0]
        timestep = self.random_timestep(unet_batch_size)
        
        # make latent_face_sequences noisy
        noisy_gt_latent, noise = self.get_noisy_image(gt_latent, timestep)
        
        g_noise = self(indiv_audios, x, noisy_gt_latent, timestep)
        
        loss = self.recon_loss(g_noise, noise)
        
        if not self.hparams.no_sample_images and batch_idx == 0:
            self.save_sample_images(x[:1], indiv_audios[:1], gt[:1], self.global_step, self.trainer.checkpoint_callback.dirpath)
        
        self.log('val_loss', loss, prog_bar=True, sync_dist=True, on_epoch=True)
        return loss
    
    def sample_images(self, x, audio):
        # get latent shape for x
        latent_shape = [x.shape[0] * x.shape[2], 4, x.shape[-2] // 8, x.shape[-1] // 8]
        
        # init g_latent with noise
        g_latent = self.sample_noise(latent_shape)
        
        scheduler = DDPMScheduler(num_train_timesteps=1000)
        scheduler.set_timesteps(num_inference_steps=self.hparams.num_inference_steps)
        for i, t in tqdm(enumerate(scheduler.timesteps), total=len(scheduler.timesteps)):
            with torch.no_grad():
                g_noise = self(audio, x, g_latent, t.repeat(latent_shape[0]).to(self.device))
            g_latent = scheduler.step(g_noise, t, g_latent).prev_sample
            
        # decode with vae
        g = self.decode_with_vae(g_latent)
        return g
            

    def save_sample_images(self, x, audio, gt, global_step, checkpoint_dir):
        g = self.sample_images(x, audio)
        
        sample_image_dir = join(os.path.dirname(checkpoint_dir), "sample_images")
        os.makedirs(sample_image_dir, exist_ok=True)
        
        folder = join(sample_image_dir, "samples_step_{:09d}".format(global_step))
        os.makedirs(folder, exist_ok=True)
        
        refs = x[:, 3:, :, :, :]
        inps = x[:, :3, :, :, :]
        refs = (refs.squeeze(0).clamp(0, 1) * 255).byte().cpu().numpy().transpose(1, 2, 3, 0).astype(np.uint8)
        inps = (inps.squeeze(0).clamp(0, 1) * 255).byte().cpu().numpy().transpose(1, 2, 3, 0).astype(np.uint8)
        gt = (gt.squeeze(0).clamp(0, 1) * 255).byte().cpu().numpy().transpose(1, 2, 3, 0).astype(np.uint8)
        g = (g.clamp(0, 1) * 255).byte().cpu().numpy().transpose(0, 2, 3, 1).astype(np.uint8)
        
        collage = np.concatenate((refs, inps, g, gt), axis=2)
        # print(f"collage: {collage.shape}")
        
        for t, c in enumerate(collage[:5]):
            # print(f"batch_idx: {t}, c: {c.shape}")
            cv2.imwrite('{}/{}_{}.jpg'.format(folder, 0, t), c)

    def on_validation_epoch_end(self) -> None:
        if self.hparams.syncnet:
            val_sync_loss = self.trainer.logged_metrics['val_sync_loss']
            if val_sync_loss < .75:
                print(f"Syncnet loss {val_sync_loss} is less than 0.75, setting syncnet_wt to {self.hparams.sync_loss_weight}")
                self.hparams.syncnet_wt = self.hparams.sync_loss_weight
            
    def configure_optimizers(self):
        # 优化器：用于self.unet和self.audio_adapter
        optimizer = torch.optim.AdamW(
            [p for p in self.unet.parameters() if p.requires_grad] + 
            ([p for p in self.audio_adapter.parameters() if p.requires_grad] if self.unet.config.cross_attention_dim != 384 else []), 
            lr=self.hparams.learning_rate
        )
        
        return optimizer

    def train_dataloader(self):
        train_dataset = AudioVisualDataset(self.hparams.data_root, 
                                           audio_root=self.hparams.audio_root if self.hparams.audio_root else None, 
                                           split='train' if not self.hparams.overfit else 'main', 
                                           dataset_name=self.hparams.dataset_name, 
                                           args=self.hparams, 
                                           dataset_size=self.hparams.dataset_size) 
        if self.hparams.ft_dataset:
            fine_tune_dataset = AudioVisualDataset(self.hparams.ft_root, 
                                                   split='train' if not self.hparams.overfit else 'main', 
                                                   audio_root=self.hparams.ft_audio_root if self.hparams.ft_audio_root else None,
                                                   dataset_name=self.hparams.ft_dataset, 
                                                   args=self.hparams, 
                                                   dataset_size=self.hparams.dataset_size)
            train_dataset = CombinedDataset(train_dataset, fine_tune_dataset)
        return DataLoader(train_dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers)

    def val_dataloader(self):
        test_dataset = AudioVisualDataset(self.hparams.data_root, 
                                          audio_root=self.hparams.audio_root if self.hparams.audio_root else None, 
                                          split='val', 
                                          dataset_name=self.hparams.dataset_name, 
                                          args=self.hparams, 
                                          dataset_size=self.hparams.dataset_size // 10)
        if self.hparams.ft_dataset:
            fine_tune_dataset = AudioVisualDataset(self.hparams.ft_root, 
                                                   audio_root=self.hparams.ft_audio_root if self.hparams.ft_audio_root else None,
                                                   split='val', 
                                                   dataset_name=self.hparams.ft_dataset, 
                                                   args=self.hparams, 
                                                   dataset_size=self.hparams.dataset_size // 10)
            test_dataset = CombinedDataset(test_dataset, fine_tune_dataset)
        return DataLoader(test_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)

# 回调函数，用于更新和保存 EMA 模型
class EMACallback(pl.Callback):
    def setup(self, trainer, pl_module, stage=None):
        # 将 ema_model 移动到与模型相同的设备
        pl_module.ema.to(pl_module.device)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        pl_module.ema.step(pl_module.unet.parameters())

    def on_validation_start(self, trainer, pl_module):
        pl_module.ema.store(pl_module.unet.parameters())
        pl_module.ema.copy_to(pl_module.unet.parameters())

    def on_validation_end(self, trainer, pl_module):
        pl_module.ema.restore(pl_module.unet.parameters())
        
def print_training_info(args):
    print("\nTraining Configuration:")
    print(f"Dataset Name: {args.dataset_name}")
    print(f"Data Root: {args.data_root}")
    print(f"Audio Root: {args.audio_root}")
    print(f"Clip Loss Enabled: {args.clip_loss}")
    print(f"U-Net Config File: {args.unet_config}")
    print(f"Checkpoint Path: {args.ckpt}")
    print(f"No Sample Images: {args.no_sample_images}")
    print(f"WandB Logging Enabled: {args.wandb}")
    print(f"Overfit Mode Enabled: {args.overfit}")
    print(f"Wav2Vec2 Embeddings Enabled: {args.wav2vec2}")
    print(f"Gradient Accumulation Steps: {args.accu_grad}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Sync Loss Weight: {args.sync_loss_weight}")
    print(f"Fine-Tune Dataset: {args.ft_dataset}")
    print(f"Fine-Tune Dataset Root: {args.ft_root}")
    print(f"Fine-Tune Audio Root: {args.ft_audio_root}")
    print(f"L2 Loss Enabled: {args.l2_loss}")
    print(f"Image Size: {args.image_size}")
    print(f"Syncnet Checkpoint: {args.syncnet}")
    print(f"Syncnet T: {args.syncnet_T}")
    print(f"Dataset Size: {args.dataset_size}")
    print(f"Lora Fine Tuning Enbaled: {args.lora}")
    print(f"Lora Rank: {args.lora_rank}")
    print(f"Lora Alpha: {args.lora_alpha}")
    print(f"Lora Checkpoint: {args.lora_ckpt}")
    print(f"Classifier Free Guidance Enabled: {args.cfg}")
    print(f"Guidance Scale: {args.guidance_scale}")
    print(f"Probability of Dropping Image Condition: {args.image_drop_prob}")
    print(f"Probability of Dropping Audio Condition: {args.audio_drop_prob}")
    print(f"EMA Enabled: {args.ema}")
    print(f"Mask Ratio: {args.mask_ratio}")
    print(f"Pretrain Unconditional Model Enabled: {args.pretrain_unconditional}")
    print(f"Pretrain Image Condition Model Enabled: {args.pretrain_image_condition}")
    print(f"Set UNet learning rate: {args.learning_rate}")
    print("\nStarting training...\n")

def get_checkpoint_filename(args):
    filename = 'wav2lip-s' if not args.lora else 'wav2lip-lora'
    
    # append key training args to the filename
    filename += f'-{args.dataset_name}'
    filename += f'-s={args.image_size}'
    filename += f'-t={args.syncnet_T}'
    
    # append epoch, step, train loss and val loss to the filename
    filename += '-{epoch}-{step}-{train_loss:.3f}-{val_loss:.3f}'
    return filename

    
if __name__ == "__main__":
    # Set the matrix multiplication precision
    # Use 'medium' for better performance with acceptable precision loss
    torch.set_float32_matmul_precision('medium')

    parser = argparse.ArgumentParser(description='Code to train latent wav2lip with lightning')
    parser.add_argument("--data_root", help="Root folder of the preprocessed LRS2 dataset", required=True)
    parser.add_argument('--audio_root', type=str, help='Root folder of the preprocessed audio dataset')
    parser.add_argument("--dataset_name", help="Name of the dataset to use, eg: 3300, hdtf, 3300_and_hdtf", required=True)
    parser.add_argument('--clip_loss', action='store_true', help='Enable clip loss.')
    parser.add_argument('--unet_config', type=str, help='Path to the unet config file', default='customized_unet_v5_large')
    parser.add_argument('--ckpt', type=str, help='Path to the checkpoint to load the model from')
    parser.add_argument('--no_sample_images', action='store_true', help='Disable saving sample images')
    parser.add_argument('--wandb', action='store_true', help='Enable wandb logging')
    parser.add_argument('--overfit', action='store_true', help='Enable overfitting mode to focus training on training loss.')
    parser.add_argument('--wav2vec2', action='store_true', help='Use wav2vec2 embeddings')
    parser.add_argument('--accu_grad', type=int, default=1, help='Number of gradient accumulation steps')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--sync_loss_weight', type=float, default=0.01, help='Weight for sync loss')
    parser.add_argument('--ft_dataset', type=str, help='Fine-tune dataset name')
    parser.add_argument('--ft_root', type=str, help='Root folder of the fine-tune dataset')
    parser.add_argument('--ft_audio_root', type=str, help='Root folder of the fine-tune audio dataset')
    parser.add_argument('--l2_loss', action='store_true', help='Enable L2 loss')
    parser.add_argument('--image_size', type=int, default=256, help='Image size')
    parser.add_argument('--syncnet', type=str, help='Path to the syncnet checkpoint to load the model from')
    parser.add_argument('--syncnet_T', type=int, default=5, help='Number of frames to consider for syncnet loss')
    parser.add_argument('--dataset_size', type=int, default=8000, help='Size of the dataset')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_inference_steps', type=int, default=50, help='Number of inference steps')
    parser.add_argument('--lora', action='store_true', help='Enable LoRA fine-tuning')  # 新增 LoRA 选项
    parser.add_argument('--lora_rank', type=int, default=64, help='Rank of Lora Matrix')
    parser.add_argument('--lora_alpha', type=int, default=64, help='Alpha of Lora Matrix')
    parser.add_argument('--lora_ckpt', type=str, help='Path to the lora checkpoint to load the model from')
    parser.add_argument('--cfg', action='store_true', help='Enable Classifier Free Guidance')
    parser.add_argument('--guidance_scale', type=float, default=7.5, help='Guidance Scale')
    parser.add_argument('--image_drop_prob', type=float, default=0.1, help='Probability of dropping the image condition')
    parser.add_argument('--audio_drop_prob', type=float, default=0.1, help='Probability of dropping the audio condition')
    parser.add_argument('--ema', action='store_true', help='Enable EMA')
    parser.add_argument('--mask_ratio', type=float, default=0.5, help='Mask ratio for the image')
    parser.add_argument('--pretrain_unconditional', action='store_true', help='Enable pretraining unconditional model')
    parser.add_argument('--pretrain_image_condition', action='store_true', help='Enable pretraining image condition model')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Set learning rate, default to 1e-5')
    args = parser.parse_args()
    
    if args.pretrain_unconditional and args.pretrain_image_condition:
        raise RuntimeError("Please enable only one of the pretraining options")
    elif args.pretrain_unconditional:
        print("Pretraining unconditional model")
        args.image_drop_prob = 1.
        args.audio_drop_prob = 1.
    elif args.pretrain_image_condition:
        print("Pretraining image condition model")
        args.audio_drop_prob = 1.
    elif not args.cfg:
        print("Training with both reference image and audio, but without cfg")
        args.image_drop_prob = 0.
        args.audio_drop_prob = 0.
    else:
        print("Training with both reference image and audio, with cfg")
    
    if args.ft_dataset and not args.ft_root:
        raise RuntimeError("Please specify the root folder of the fine-tune dataset by --ft_root")
    elif args.ft_root and not args.ft_dataset:
        raise RuntimeError("Please specify the fine-tune dataset name by --ft_dataset")
    
    # Print the training information
    print_training_info(args)
    
    # Convert hparams instance to a dictionary
    hparams_dict = hparams.data

    # Update hparams with args
    hparams_dict.update(vars(args))

    # Apply LoRA fine-tuning if enabled
    if args.lora:
        if not args.ckpt:
            raise RuntimeError("Please specify the base model for lora fine-tuning by --ckpt")
        
        # Load the Base model from the checkpoint
        model = LatentWav2Lip.load_from_checkpoint(args.ckpt, hparams=hparams_dict)
        
        # Create an instance of LoraConfig
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=[
                "to_q",
                "to_k",
                "to_v",
                "to_out.0",
                "proj_in",
                "proj_out",
                "ff.net.0.proj",
                "ff.net.2",
                "conv1",
                "conv2",
                "conv_shortcut",
                "downsamplers.0.conv",
                "upsamplers.0.conv",
                "time_emb_proj",
            ],
        )
        
        # Add LoRA to the model
        model.add_lora(lora_config)
    else:
        # Create an instance of LatentWav2Lip with merged parameters
        model = LatentWav2Lip(hparams_dict)
        model.load_unet(model.hparams.unet_config)

    # Checkpoint callback to save the model periodically

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filename=get_checkpoint_filename(args),
        save_top_k=3,
        verbose=True,
        monitor='val_loss' if not args.overfit else 'train_loss',
        mode='min'
    )
    
    # 设置日志目录和实验名称
    if args.wandb:
        logger = WandbLogger(project='stable_wav2lip', entity='local-optima')
    else:
        logger = TensorBoardLogger('experiments', name='stable_wav2lip_experiment')

    callbacks = [checkpoint_callback, RichProgressBar(), LearningRateMonitor(logging_interval='step')]
        
    # Include EMA callback if enabled
    if args.ema:
        ema_callback = EMACallback()
        callbacks.append(ema_callback)
        
    # Trainer setup for multi-GPU training
    trainer = Trainer(
        logger=logger,
        max_epochs=model.hparams.nepochs,
        strategy='ddp', 
        precision='16-mixed',
        accumulate_grad_batches=model.hparams.accu_grad,
        gradient_clip_val=0.5,
        callbacks=callbacks
    )

    if args.lora:
        trainer.fit(model, ckpt_path=args.lora_ckpt)
    else:
        trainer.fit(model, ckpt_path=args.ckpt)
