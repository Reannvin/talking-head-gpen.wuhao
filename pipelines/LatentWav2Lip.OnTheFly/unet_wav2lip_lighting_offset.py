from os.path import dirname, join, basename, isfile
import os.path as osp
import re
from glob import glob
import torch, torchvision
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
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
import yaml
from PIL import Image


def parse_info_file_to_hashmap(txt_file):
    # 创建一个哈希表来存储每张图片的offset
    offset_map = {}

    with open(txt_file, 'r') as f:
        for line in f:
            # 按空格分割每行内容，忽略多余的空格
            parts = line.strip().split()
            pattern = re.compile(r'(\S+)\s+([\w\s（）()#.，/-]+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)')
            match = pattern.match(line.strip())
            if match is None:
                print('match is None why?????????????????')

            if len(match.groups()) == 7:
                _, video_relative_path, start_img_file, end_img_file, offset, _, _ = match.groups()
                start_img = int(start_img_file.replace('.jpg', ''))
                end_img = int(end_img_file.replace('.jpg', ''))
                offset = int(float(parts[4]))

                # 将该视频片段中的每一张图片对应的offset存入哈希表
                for img_num in range(start_img, end_img + 1):
                    img_name = f"{img_num}.jpg"
                    key = f"{video_relative_path}/{img_name}"  # 使用视频路径和图片名作为key
                    offset_map[key] = offset

    return offset_map


def find_offset_from_hashmap(offset_map, video_relative_path, img_name):
    key = f"{video_relative_path}/{img_name}"
    return offset_map.get(key, 0)  # 直接在哈希表中查找


def modify_image_name(image_path, N):
    # 获取文件目录和文件名
    directory, filename = os.path.split(image_path)
    # 提取图片名的数字部分，并将其转为整数
    img_num = int(os.path.splitext(filename)[0])
    # 计算新的图片名
    new_img_num = img_num - N
    # 构造新的文件名
    new_filename = f"{new_img_num}.jpg"
    # 构造新的完整路径
    new_image_path = os.path.join(directory, new_filename)

    return new_image_path


class HybridDataset(Dataset):
    def __init__(self, config, split, args, dataset_size):
        self.datasets = []
        self.ratios = []
        self.dataset_size = dataset_size
        self.load_config(config, split, args)

    def load_config(self, config, split, args):
        dataset_configs = config['datasets'][split]
        for dataset_config in dataset_configs:
            dataset = AudioVisualDataset(
                data_root=dataset_config['path'],
                audio_root=dataset_config.get('audio_path'),
                split=dataset_config['split'],
                dataset_name=dataset_config['name'],
                args=args,
                dataset_size=self.dataset_size
            )
            self.datasets.append(dataset)
            self.ratios.append(dataset_config['ratio'])

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        dataset_choice = random.choices(self.datasets, weights=self.ratios, k=1)[0]
        return dataset_choice[idx]


class AudioVisualDataset(Dataset):
    def __init__(self, data_root, dataset_name, split, args, audio_root=None, dataset_size=512000):
        self.all_videos = get_image_list(data_root, dataset_name, split)
        self.dataset_size = dataset_size
        self.syncnet_audio_size = 10 * args.syncnet_T if not args.wav2vec2 else 2 * args.syncnet_T
        self.frame_audio_size = 10 * 5 if not args.wav2vec2 else 2 * 5 # always use syncnet_T=5 for each frame
        self.args = args
        self.data_root = data_root
        self.audio_root = audio_root
        if self.args.offset:
            self.offset_info = parse_info_file_to_hashmap(f'filelists_{dataset_name}/info.txt')

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
        # print('video:', vidname, frame_id)
        audios = []
        offset = self.args.syncnet_T // 2
        start_frame_num = frame_id + 1
        # start_frame_num = frame_id
        if start_frame_num - offset < 0: return None
        for i in range(start_frame_num, start_frame_num + self.args.syncnet_T):
            m = self.get_whisper_embedding(vidname, i - offset, syncnet_T=5) # always use syncnet_T=5 for each frame
            # print('audio:', vidname, i - offset)
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
            
            # transform window and wrong_window to with data augmentation if enabled
            if self.args.data_aug_image:
                data_aug_transforms = torchvision.transforms.Compose([
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
                ])
                window = [data_aug_transforms(Image.fromarray(w)) for w in window]
                wrong_window = [data_aug_transforms(Image.fromarray(w)) for w in wrong_window]
            
            if self.audio_root:
                # switch fro data_root to audio_root
                if self.args.offset:
                    video_relative_path = os.path.relpath(vidname, self.data_root)
                    offset = find_offset_from_hashmap(self.offset_info, video_relative_path, osp.basename(img_name))
                    vidname = vidname.replace(self.data_root, self.audio_root)
                    offset = offset if abs(offset) <= 5 else 9999999
                    img_name = modify_image_name(img_name, offset)
                else:
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
            
            if self.args.data_aug_mask:
                # random mask ratio around self.args.mask_ratio with range [-0.1, +0.1]
                mask_ratio = self.args.mask_ratio + random.uniform(-0.1, 0.1)
            else:
                mask_ratio = self.args.mask_ratio
            # mask lower part of the image, according to mask_ratio
            mask_height = int(window.shape[2] * mask_ratio)
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

class LatentWav2LipOutput:
    def __init__(self, g_latent, g_image=None, g_base_image=None, latent_face_sequences=None, audio_sequences=None):
        self.g_latent = g_latent
        self.g_image = g_image
        self.g_base_image = g_base_image
        self.latent_face_sequences = latent_face_sequences
        self.audio_sequences = audio_sequences
        
class LatentWav2Lip(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        unet_config = UNet2DConditionModel.load_config(f"{self.hparams.unet_config}/config.json")
        self.unet = UNet2DConditionModel.from_config(unet_config)
        if self.hparams.balance_loss:
            self.unet.set_default_attn_processor()
        self.zero_timestep = torch.zeros([])
        
        # init refine model if needed
        if self.hparams.refine:
            refine_config = UNet2DConditionModel.load_config(f"{self.hparams.refine_config}/config.json")
            self.refine = UNet2DConditionModel.from_config(refine_config)
            
            # freeze the base model            
            self.unet.eval()
            for param in self.unet.parameters():
                param.requires_grad = False
        
        if self.hparams.syncnet:
            self.syncnet = self.load_syncnet()
            self.syncnet.eval()  # Ensure syncnet is always in eval mode
        
        self.vae = self.load_vae('stabilityai/sd-vae-ft-mse')
        self.vae.eval()  # Ensure vae is always in eval mode
        
        if self.hparams.fid:
            from torchmetrics.image.fid import FrechetInceptionDistance
            self.fid = FrechetInceptionDistance(feature=64, normalize=True)
            
        if self.hparams.ema:
            from diffusers.training_utils import EMAModel
            self.ema = EMAModel(self.unet.parameters())
        
        # 定义损失函数
        self.recon_loss = nn.L1Loss() if not self.hparams.l2_loss else nn.MSELoss()
        self.log_loss = nn.BCELoss()
        self.lpips_loss = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True)
        
        self.lora_config = None  # 初始化为 None
    
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
            if isfile(unet_ckpt):
                s = torch.load(unet_ckpt)['state_dict']
                s = {k.replace('unet.', ''): v for k, v in s.items() if k.startswith('unet.')}
                self.unet.load_state_dict(s)
            else:
                self.unet.from_pretrained(unet_ckpt)
            
            print(f"Loaded U-Net from {unet_ckpt}")
        except Exception as e:
            print(f"Error loading U-Net from {unet_ckpt}: {e}")
            print("Train UNet from scratch")
            
        return self.unet
    
    def add_lora(self, lora_config):
        self.lora_config = lora_config
        self.unet.add_adapter(lora_config)

    def on_save_checkpoint(self, checkpoint):
        if self.hparams.lora:
            lora_params = {'unet.' + k: v for k, v in self.unet.named_parameters() if 'lora' in k}
            checkpoint['state_dict'] = lora_params
        elif self.hparams.refine:
            refine_params = {'refine.' + k: v for k, v in self.refine.named_parameters()}
            checkpoint['state_dict'] = refine_params
        else:
            unet_params = {'unet.' + k: v for k, v in self.unet.named_parameters()}
            checkpoint['state_dict'] = unet_params
        
        # save ema model
        if self.hparams.ema:
            checkpoint['ema'] = self.ema.state_dict()
            
    def on_load_checkpoint(self, checkpoint):
        # load ema model
        if self.hparams.ema:
            if 'ema' in checkpoint:
                self.ema.load_state_dict(checkpoint['ema'])
                self.ema.to(self.device)

    def load_state_dict(self, state_dict, strict=True): 
        # Call the parent class's load_state_dict method
        super(LatentWav2Lip, self).load_state_dict(state_dict, strict=False)
    
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
            if torch.isnan(g).any():
                g = torch.nan_to_num(g, nan=0.0)
            if torch.isnan(gt).any():
                gt = torch.nan_to_num(gt, nan=0.0)
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
    
    def crop_by_percentage(self, image: torch.Tensor, crop_percentage: float) -> torch.Tensor:
        if not (0 <= crop_percentage <= 1):
            raise ValueError("crop_percentage must be between 0 and 1")

        # Get the height and width of the image
        H, W = image.shape[-2], image.shape[-1]

        # Calculate the number of pixels to crop
        crop_h = int(H * crop_percentage)
        crop_w = int(W * crop_percentage)

        # Calculate the new height and width
        new_H = H - 2 * crop_h
        new_W = W - 2 * crop_w

        # Perform the crop
        cropped_image = image[..., crop_h:crop_h + new_H, crop_w:crop_w + new_W]
        
        # Resize the cropped image to the original size
        cropped_image = self.reshape_gt_image_for_vae(cropped_image)
        cropped_image = nn.functional.interpolate(cropped_image, size=(H, W), mode='bilinear')
        cropped_image = self.inverse_reshape_face_sequences(cropped_image)
        
        return cropped_image

    def image_audio_balance_loss(self, image, audio, output, weight):
        assert image.shape[0] == audio.shape[0]
        image_grad, audio_grad = torch.autograd.grad(
            outputs=output,
            inputs=(image, audio),
            grad_outputs=torch.ones_like(output),
            create_graph=True,
            retain_graph=True,
        )
        assert image.shape == image_grad.shape and audio.shape == audio_grad.shape
        
        image_grad = torch.reshape(image_grad, (image_grad.shape[0], -1))
        image_grad_norm = image_grad.norm(2, dim=1)
        audio_grad = torch.reshape(audio_grad, (audio_grad.shape[0], -1))
        audio_grad_norm = audio_grad.norm(2, dim=1)
        
        return weight * ((image_grad_norm / audio_grad_norm - 1) ** 2).mean()
    
    def forward(self, audio_sequences, face_sequences, with_image=True, with_base_image=False):
        if self.hparams.pretrain_unconditional or self.hparams.pretrain_image_condition or self.hparams.cfg:
            if self.unet.config.in_channels != 12:
                raise ValueError("Pretraining or CFG requires input channels to be 12")
            
        face_sequences = self.reshape_face_sequences_for_vae(face_sequences)
        latent_face_sequences = self.encode_with_vae(face_sequences)

        latent_face_sequences = self.reshape_latent_faces_for_unet(latent_face_sequences)
        audio_sequences = self.reshape_audio_sequences_for_unet(audio_sequences)
        
        if self.unet.config.in_channels == 12:
            latent_noise_shape = [latent_face_sequences.size(0), 4, latent_face_sequences.size(2), latent_face_sequences.size(3)]
            latent_noise = self.sample_noise(latent_noise_shape)
            latent_face_sequences = torch.cat([latent_face_sequences, latent_noise], dim=1)
        
        if self.hparams.cfg and not self.training:
            # use classifier free guidance
            unconditional_audio = torch.zeros_like(audio_sequences).to(self.device)
            combined_audio = torch.cat([unconditional_audio, audio_sequences])
            latent_face_sequences = torch.cat([latent_face_sequences] * 2)
            pred = self.unet(latent_face_sequences, timestep=self.zero_timestep, encoder_hidden_states=combined_audio).sample
            uncond_pred, cond_pred = pred.chunk(2)
            g_latent = uncond_pred + self.hparams.guidance_scale * (cond_pred - uncond_pred)
        elif self.hparams.refine:
            g_base = self.unet(latent_face_sequences, timestep=self.zero_timestep, encoder_hidden_states=audio_sequences).sample
            # replace the first part of the latent with the base prediction
            latent_face_sequences[:, :4] = g_base
            g_delta = self.refine(latent_face_sequences, timestep=self.zero_timestep, encoder_hidden_states=audio_sequences).sample    
            g_latent = g_base + g_delta
        else:
            if self.hparams.balance_loss:
                latent_face_sequences.requires_grad_()
                audio_sequences.requires_grad_()
            g_latent = self.unet(latent_face_sequences, timestep=self.zero_timestep, encoder_hidden_states=audio_sequences).sample   
         
        
        if with_image:
            g_image = self.decode_with_vae(g_latent)
            g_image = self.inverse_reshape_face_sequences(g_image)
        else:
            g_image = None
        
        g_latent = self.inverse_reshape_face_sequences(g_latent)
        
        if self.hparams.refine and with_base_image:
            g_base_image = self.decode_with_vae(g_base)
            g_base_image = self.inverse_reshape_face_sequences(g_base_image)
        else:
            g_base_image = None

        return LatentWav2LipOutput(g_latent=g_latent, g_image=g_image, g_base_image=g_base_image, latent_face_sequences=latent_face_sequences, audio_sequences=audio_sequences)
    
    def training_step(self, batch, batch_idx):
        if self.hparams.syncnet:
            x, indiv_audios, audio_cropped, gt = batch
        else:
            x, indiv_audios, gt = batch
        
        if self.hparams.crop_percentage > 0.:
            x = self.crop_by_percentage(x, self.hparams.crop_percentage)
            gt = self.crop_by_percentage(gt, self.hparams.crop_percentage)
        
        # to drop the reference image, based on drop_ref_prob    
        if self.hparams.dropout_ref and self.hparams.drop_ref_prob > 0.:
            need_dropout = random.random() < self.hparams.drop_ref_prob
            if need_dropout:
                masked, ref = x.split([3, 3], dim=1)
                zero_ref = torch.zeros_like(ref)
                x = torch.cat([masked, zero_ref], dim=1)
            
        # to drop the image condition, based on image_drop_prob
        if self.hparams.image_drop_prob > 0.:
            need_dropout = random.random() < self.hparams.image_drop_prob
            if need_dropout:
                x = torch.zeros_like(x)
        
        # to drop the audio condition, based on audio_drop_prob
        if self.hparams.audio_drop_prob > 0.:
            need_dropout = random.random() < self.hparams.audio_drop_prob
            if need_dropout:
                indiv_audios = torch.zeros_like(indiv_audios)
                
        model_out = self(indiv_audios, x, not self.hparams.no_image_loss)
        g_latent, g_image, latent_x, audio_sequences = model_out.g_latent, model_out.g_image, model_out.latent_face_sequences, model_out.audio_sequences
        
        if self.hparams.no_image_loss:
            # no image loss, set image_recon_loss and lpips_loss to 0
            image_recon_loss = 0.
            lpips_loss = 0.
        elif self.hparams.no_full_image_loss:
            # only calculate the loss for the lower half of the image
            image_recon_loss = self.recon_loss(g_image[:, :, :, g_image.size(3) // 2:, :], gt[:, :, :, gt.size(3) // 2:, :])
            lpips_loss = self.get_lpips_loss(g_image[:, :, :, g_image.size(3) // 2:, :], gt[:, :, :, gt.size(3) // 2:, :])
        else:
            # calculate the loss for the full image
            image_recon_loss = self.recon_loss(g_image, gt)
            lpips_loss = self.get_lpips_loss(g_image, gt)
            
        image_loss = self.hparams.lpips_loss_weight * lpips_loss + image_recon_loss
        
        gt_for_vae = self.reshape_gt_image_for_vae(gt)        
        gt_latent = self.encode_with_vae(gt_for_vae)
        gt_latent = self.inverse_reshape_face_sequences(gt_latent)
        latent_recon_loss = self.recon_loss(g_latent, gt_latent)
        
        if self.hparams.syncnet and self.hparams.syncnet_wt > 0.:
            sync_loss = self.get_sync_loss(audio_cropped, g_image[:, :, :, g_image.size(3) // 2:, :])
        else:
            sync_loss = 0.            
        
        recon_loss = latent_recon_loss + self.hparams.image_loss_wt * image_loss
        
        if self.hparams.balance_loss:
            # add image-audio balance loss
            balance_loss = self.image_audio_balance_loss(image=latent_x, audio=audio_sequences, output=g_latent, weight=self.hparams.balance_loss_weight)
        else:
            balance_loss = 0.
        
        loss = self.hparams.syncnet_wt * sync_loss + (1 - self.hparams.syncnet_wt) * recon_loss + balance_loss

        if not self.hparams.no_sample_images and batch_idx % 500 == 0:
            self.save_sample_images(x[:1], None, g_image[:1], gt[:1], self.global_step, self.trainer.checkpoint_callback.dirpath, name='train_samples_step')
        
        self.log('train_loss', loss, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log('train_image_recon_loss', image_recon_loss, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log('train_lpips_loss', lpips_loss, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log('train_latent_recon_loss', latent_recon_loss, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log('train_sync_loss', sync_loss, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log('train_balance_loss', balance_loss, prog_bar=True, sync_dist=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.hparams.syncnet:
            x, indiv_audios, audio_cropped, gt = batch
        else:
            x, indiv_audios, gt = batch
            
        if self.hparams.crop_percentage > 0.:
            x = self.crop_by_percentage(x, self.hparams.crop_percentage)
            gt = self.crop_by_percentage(gt, self.hparams.crop_percentage)
        
        if self.hparams.pretrain_unconditional:
            # drop image and audio condition
            x = torch.zeros_like(x).to(self.device)
            indiv_audios = torch.zeros_like(indiv_audios).to(self.device)
        elif self.hparams.pretrain_image_condition:
            # drop audio condition
            indiv_audios = torch.zeros_like(indiv_audios).to(self.device)
        
        model_out = self(indiv_audios, x, not self.hparams.no_image_loss or not self.hparams.no_sample_images, with_base_image=True)
        g_latent, g_image, g_base_image = model_out.g_latent, model_out.g_image, model_out.g_base_image        
        
        if self.hparams.no_image_loss:
            # no image loss, set image_recon_loss and lpips_loss to 0
            image_recon_loss = 0.
            lpips_loss = 0.
        elif self.hparams.no_full_image_loss:
            # only calculate the loss for the lower half of the image
            image_recon_loss = self.recon_loss(g_image[:, :, :, g_image.size(3) // 2:, :], gt[:, :, :, gt.size(3) // 2:, :])
            lpips_loss = self.get_lpips_loss(g_image[:, :, :, g_image.size(3) // 2:, :], gt[:, :, :, gt.size(3) // 2:, :])
        else:
            # calculate the loss for the full image
            image_recon_loss = self.recon_loss(g_image, gt)
            lpips_loss = self.get_lpips_loss(g_image, gt)
            
        image_loss = self.hparams.lpips_loss_weight * lpips_loss + image_recon_loss
        
        gt_for_vae = self.reshape_gt_image_for_vae(gt)        
        gt_latent = self.encode_with_vae(gt_for_vae)
        gt_latent = self.inverse_reshape_face_sequences(gt_latent)
        latent_recon_loss = self.recon_loss(g_latent, gt_latent)
        
        if self.hparams.syncnet:
            sync_loss = self.get_sync_loss(audio_cropped, g_image[:, :, :, g_image.size(3) // 2:, :])
        else:
            sync_loss = 0.
        
        recon_loss = latent_recon_loss + self.hparams.image_loss_wt * image_loss
        val_loss = self.hparams.syncnet_wt * sync_loss + (1 - self.hparams.syncnet_wt) * recon_loss
        
        if not self.hparams.no_sample_images and batch_idx == 0:
            self.save_sample_images(x[:1], g_base_image[:1] if g_base_image is not None else None , g_image[:1], gt[:1], self.global_step, self.trainer.checkpoint_callback.dirpath)
            
        if self.hparams.fid:
            g_image = self.reshape_gt_image_for_vae(g_image)
            gt = self.reshape_gt_image_for_vae(gt)
            
            # use torch metrics to compute FID
            self.fid.update(gt, real=True)
            self.fid.update(g_image, real=False)
        
        self.log('val_loss', val_loss, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log('val_image_recon_loss', image_recon_loss, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log('val_lpips_loss', lpips_loss, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log('val_latent_recon_loss', latent_recon_loss, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log('val_sync_loss', sync_loss, prog_bar=True, sync_dist=True, on_epoch=True)
        return val_loss
    
    def save_sample_images(self, x, g_base, g, gt, global_step, checkpoint_dir, name='samples_step'):
        refs = x[:, 3:, :, :, :]
        inps = x[:, :3, :, :, :]
        
        sample_image_dir = join(os.path.dirname(checkpoint_dir), "sample_images")
        os.makedirs(sample_image_dir, exist_ok=True)
        
        folder = join(sample_image_dir, "{:s}_{:09d}".format(name, global_step))
        os.makedirs(folder, exist_ok=True)
        
        refs = (refs.squeeze(0).clamp(0, 1) * 255).byte().cpu().numpy().transpose(1, 2, 3, 0).astype(np.uint8)
        inps = (inps.squeeze(0).clamp(0, 1) * 255).byte().cpu().numpy().transpose(1, 2, 3, 0).astype(np.uint8)
        g_base = (g_base.squeeze(0).clamp(0, 1) * 255).byte().cpu().numpy().transpose(1, 2, 3, 0).astype(np.uint8) if g_base is not None else None
        g = (g.squeeze(0).clamp(0, 1) * 255).byte().cpu().numpy().transpose(1, 2, 3, 0).astype(np.uint8)
        gt = (gt.squeeze(0).clamp(0, 1) * 255).byte().cpu().numpy().transpose(1, 2, 3, 0).astype(np.uint8)
            
        if g_base is not None:
            collage = np.concatenate((refs, inps, g_base, g, gt), axis=2)
        else:
            collage = np.concatenate((refs, inps, g, gt), axis=2)
                    
        for t, c in enumerate(collage[:4]):
            # print(f"batch_idx: {t}, c: {c.shape}")
            cv2.imwrite('{}/{}_{}.jpg'.format(folder, 0, t), c)

    def on_validation_epoch_end(self) -> None:
        if self.hparams.syncnet:
            val_sync_loss = self.trainer.logged_metrics['val_sync_loss']
            print(f"Syncnet loss {val_sync_loss} , syncnet_wt is {self.hparams.sync_loss_weight}")
            if val_sync_loss < .75:
                print(f"Syncnet loss {val_sync_loss} is less than 0.75, setting syncnet_wt to {self.hparams.sync_loss_weight}")
                self.hparams.syncnet_wt = self.hparams.sync_loss_weight
        
        if self.hparams.fid:
            fid_score = self.fid.compute() # need 28 seconds for this       
            self.log('fid_score', fid_score, prog_bar=True, sync_dist=True, on_epoch=True)
            self.fid.reset()
            
    def configure_optimizers(self):
        if self.hparams.refine:
            optimizer = torch.optim.AdamW([p for p in self.refine.parameters() if p.requires_grad], lr=1e-4)
        else:
            optimizer = torch.optim.AdamW([p for p in self.unet.parameters() if p.requires_grad], lr=1e-4)
        
        # 设置 LambdaLinearScheduler 参数
        warm_up_steps = [10000]  # 预热步数
        f_min = [1.0]  # 最小学习率
        f_max = [1.0]  # 最大学习率
        f_start = [1.e-6]  # 开始学习率
        cycle_lengths = [10000000000000]  # 周期长度
        
        # 创建 LambdaLinearScheduler 实例
        scheduler = LambdaLinearScheduler(
            warm_up_steps=warm_up_steps,
            f_min=f_min,
            f_max=f_max,
            f_start=f_start,
            cycle_lengths=cycle_lengths,
            # verbosity_interval=1000  # 每1000步打印一次学习率信息
        )
        
        # 使用 LambdaLR 包装自定义的调度器
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler.schedule)
        
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": lr_scheduler, "interval": "step", "frequency": 1}}

    def train_dataloader(self):
        train_dataset = HybridDataset(config=self.hparams.dataset_config, split='train', args=self.hparams, dataset_size=self.hparams.dataset_size)
        return DataLoader(train_dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers)

    def val_dataloader(self):
        val_dataset = HybridDataset(config=self.hparams.dataset_config, split='val', args=self.hparams, dataset_size=self.hparams.dataset_size // 10)
        return DataLoader(val_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)

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
    print(f"Dataset Config File: {args.dataset_config}")
    print(f"Clip Loss Enabled: {args.clip_loss}")
    print(f"U-Net Config File: {args.unet_config}")
    print(f"Checkpoint Path: {args.ckpt}")
    print(f"No Sample Images: {args.no_sample_images}")
    print(f"WandB Logging Enabled: {args.wandb}")
    print(f"Overfit Mode Enabled: {args.overfit}")
    print(f"Dropout on Reference Frames Enabled: {args.dropout_ref}")
    print(f"Wav2Vec2 Embeddings Enabled: {args.wav2vec2}")
    print(f"Gradient Accumulation Steps: {args.accu_grad}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Sync Loss Weight: {args.sync_loss_weight}")
    print(f"L2 Loss Enabled: {args.l2_loss}")
    print(f"Image Size: {args.image_size}")
    print(f"Image Loss Weight: {args.image_loss_wt}")
    print(f"Syncnet Checkpoint: {args.syncnet}")
    print(f"Syncnet T: {args.syncnet_T}")
    print(f"Dataset Size: {args.dataset_size}")
    print(f"No Image Loss: {args.no_image_loss}")
    print(f"LPIPS Loss Weight: {args.lpips_loss_weight}")
    print(f"No Full Image Loss: {args.no_full_image_loss}")
    print(f"Lora Fine Tuning Enbaled: {args.lora}")
    print(f"Lora Rank: {args.lora_rank}")
    print(f"Lora Alpha: {args.lora_alpha}")
    print(f"Lora Checkpoint: {args.lora_ckpt}")
    print(f"WandB Entity: {args.wandb_entity}")
    print(f"Classifier Free Guidance Enabled: {args.cfg}")
    print(f"Guidance Scale: {args.guidance_scale}")
    print(f"Image Drop Probability: {args.image_drop_prob}")
    print(f"Audio Drop Probability: {args.audio_drop_prob}")
    print(f"EMA Enabled: {args.ema}")
    print(f"Crop Percentage: {args.crop_percentage}")
    print(f"Mask Ratio: {args.mask_ratio}")
    print(f"Pretrain Unconditional: {args.pretrain_unconditional}")
    print(f"Pretrain Image Condition: {args.pretrain_image_condition}")
    print(f"Data Augmentation for Images: {args.data_aug_image}")
    print(f"Data Augmentation for Masks: {args.data_aug_mask}")
    print(f"Refine Model Enabled: {args.refine}")
    print(f"Refine Config File: {args.refine_config}")
    print(f"Refine Checkpoint: {args.refine_ckpt}")
    print(f"Balance Loss Enabled: {args.balance_loss}")
    print(f"Balance Loss Weight: {args.balance_loss_weight}")
    print("\nStarting training...\n")

def get_checkpoint_filename(args):
    if args.lora:
        filename = 'wav2lip-lora'
    elif args.refine:
        filename = 'wav2lip-refine'
    else:
        filename = 'wav2lip-base'
    
    # append key training args to the filename
    filename += f'-{args.dataset_config["name"]}'
    filename += f'-s={args.image_size}'
    filename += f'-t={args.syncnet_T}'
    
    # append epoch, step, train loss and val loss to the filename
    filename += '-{epoch}-{step}-{train_loss:.3f}-{val_loss:.3f}-{val_sync_loss:.3f}'
    return filename
    
    
if __name__ == "__main__":
    # Set the matrix multiplication precision
    # Use 'medium' for better performance with acceptable precision loss
    torch.set_float32_matmul_precision('medium')

    parser = argparse.ArgumentParser(description='Code to train latent wav2lip with lightning')
    parser.add_argument('--clip_loss', action='store_true', help='Enable clip loss.')
    parser.add_argument('--unet_config', type=str, help='Path to the unet config file', default='customized_unet_v4_large')
    parser.add_argument('--ckpt', type=str, help='Path to the checkpoint to load the model from')
    parser.add_argument('--no_sample_images', action='store_true', help='Disable saving sample images')
    parser.add_argument('--wandb', action='store_true', help='Enable wandb logging')
    parser.add_argument('--overfit', action='store_true', help='Enable overfitting mode to focus training on training loss.')
    parser.add_argument('--dropout_ref', action='store_true', help='Enable dropout on the reference frames.')
    parser.add_argument('--drop_ref_prob', type=float, default=0.2, help='Probability of dropping the reference frames')
    parser.add_argument('--wav2vec2', action='store_true', help='Use wav2vec2 embeddings')
    parser.add_argument('--accu_grad', type=int, default=4, help='Number of gradient accumulation steps')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--sync_loss_weight', type=float, default=0.01, help='Weight for sync loss')
    parser.add_argument('--l2_loss', action='store_true', help='Enable L2 loss')
    parser.add_argument('--image_size', type=int, default=256, help='Image size')
    parser.add_argument('--image_loss_wt', type=float, default=1.0, help='Weight for image reconstruction loss')
    parser.add_argument('--syncnet', type=str, help='Path to the syncnet checkpoint to load the model from')
    parser.add_argument('--syncnet_T', type=int, default=5, help='Number of frames to consider for syncnet loss')
    parser.add_argument('--dataset_size', type=int, default=8000, help='Size of the dataset')
    parser.add_argument('--no_image_loss', action='store_true', help='Disable image loss')
    parser.add_argument('--lpips_loss_weight', type=float, default=1.0, help='Weight for LPIPS loss')
    parser.add_argument('--no_full_image_loss', action='store_true', help='Disable full image loss')
    parser.add_argument('--lora', action='store_true', help='Enable LoRA fine-tuning')  # 新增 LoRA 选项
    parser.add_argument('--lora_rank', type=int, default=64, help='Rank of Lora Matrix')
    parser.add_argument('--lora_alpha', type=int, default=64, help='Alpha of Lora Matrix')
    parser.add_argument('--lora_ckpt', type=str, help='Path to the lora checkpoint to load the model from')
    parser.add_argument('--wandb_entity', type=str, default='local-optima', help='wandb_entity')
    parser.add_argument('--fid', action='store_true', help='Enable FID computation')
    parser.add_argument('--cfg', action='store_true', help='Enable Classifier Free Guidance')
    parser.add_argument('--guidance_scale', type=float, default=7.5, help='Guidance Scale')
    parser.add_argument('--image_drop_prob', type=float, default=0.2, help='Probability of dropping the image condition')
    parser.add_argument('--audio_drop_prob', type=float, default=0.2, help='Probability of dropping the audio condition')
    parser.add_argument('--ema', action='store_true', help='Enable EMA')
    parser.add_argument('--crop_percentage', type=float, default=0, help='Crop percentage for the image')
    parser.add_argument('--mask_ratio', type=float, default=0.5, help='Mask ratio for the image')
    parser.add_argument('--pretrain_unconditional', action='store_true', help='Enable pretraining unconditional model')
    parser.add_argument('--pretrain_image_condition', action='store_true', help='Enable pretraining image condition model')
    parser.add_argument('--dataset_config', type=str, default='data/dataset_config.yaml', help='Path to the dataset config file')
    parser.add_argument('--data_aug_image', action='store_true', help='Enable data augmentation for images')
    parser.add_argument('--data_aug_mask', action='store_true', help='Enable data augmentation for mask ratio')
    parser.add_argument('--refine', action='store_true', help='Train refine model')
    parser.add_argument('--refine_config', type=str, help='Path to the refine config file', default='customized_unet_v4_large')
    parser.add_argument('--refine_ckpt', type=str, help='Path to the refine checkpoint to load the model from')
    parser.add_argument('--balance_loss', action='store_true', help='Enable balance loss')
    parser.add_argument('--balance_loss_weight', type=float, default=10.0, help='Weight for balance loss')

    parser.add_argument('--wandb_name', type=str, default='', help='wandb_name')
    parser.add_argument('--wandb_id', type=str, default='', help='wandb_name')
    parser.add_argument('--offset', action='store_true', help='Enable overfitting mode to focus training on training loss.')
    args = parser.parse_args()
    
    # load the dataset config
    try:
        with open(args.dataset_config, 'r') as file:
            args.dataset_config = yaml.safe_load(file)
    except FileNotFoundError:
        raise RuntimeError("Dataset config file not found")
    
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
    elif args.refine:
        # init model
        model = LatentWav2Lip(hparams_dict)
        
        if not model.hparams.ckpt:
            raise RuntimeError("Please specify the base model for refinement by --ckpt")
        
        # load base model
        model.load_unet(model.hparams.ckpt)        
    else:
        # Create an instance of LatentWav2Lip with merged parameters
        model = LatentWav2Lip(hparams_dict)
        model.load_unet(model.hparams.unet_config)
        
    # Checkpoint callback to save the model periodically

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=f'image_wav2lip/{args.wandb_name}/checkpoints',
        filename=get_checkpoint_filename(args),
        save_top_k=3,
        verbose=True,
        monitor='val_loss' if not args.overfit else 'train_loss',
        mode='min'
    )
    
    # 设置日志目录和实验名称
    if args.wandb:
        # logger = WandbLogger(project='image_wav2lip', entity=args.wandb_entity)
        # logger = WandbLogger(project='image_wav2lip', entity=args.wandb_entity, name=args.wandb_name, mode='offline')
        # logger = WandbLogger(project='image_wav2lip', entity=args.wandb_entity, name=args.wandb_name, mode='disabled')
        if args.wandb_id:
            logger = WandbLogger(project='image_wav2lip', entity=args.wandb_entity, name=args.wandb_name,  resume='must', id=args.wandb_id)
            print(f'wandb resume from id:{args.wandb_id }')
        else:
            logger = WandbLogger(project='image_wav2lip', entity=args.wandb_entity, name=args.wandb_name)
    else:
        logger = TensorBoardLogger('experiments', name='image_wav2lip_experiment')

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
    elif args.refine:
        trainer.fit(model, ckpt_path=args.refine_ckpt)
    else:
        trainer.fit(model, ckpt_path=args.ckpt)
