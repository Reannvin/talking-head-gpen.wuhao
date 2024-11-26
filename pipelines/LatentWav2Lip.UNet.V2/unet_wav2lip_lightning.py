from os.path import dirname, join, basename, isfile
from glob import glob
from models import SyncNet_latent, SyncNet_latent_xl
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
    def __init__(self, args, data_root, dataset_name, split, dataset_size=512000):
        self.all_videos = get_image_list(data_root, dataset_name, split)
        self.dataset_size = dataset_size
        self.syncnet_audio_size = 10 * args.syncnet_T if args.whisper else 2 * args.syncnet_T
        self.frame_audio_size = 10 * 5 if args.whisper else 2 * 5 # always use syncnet_T=5 for each frame
        self.args = args
        
    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])

    def get_window(self, start_frame):
        start_id = self.get_frame_id(start_frame)
        vidname = dirname(start_frame)

        window_fnames = []
        for frame_id in range(start_id, start_id + self.args.syncnet_T):
            frame = join(vidname, '{}.pt'.format(frame_id))
            if not isfile(frame):
                return None
            window_fnames.append(frame)
        return window_fnames

    def read_window(self, window_fnames):
        if window_fnames is None: return None
        window = []
        for fname in window_fnames:
            latent = torch.load(fname)
            if latent is None:
                return None
            window.append(latent)
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
            m = self.get_whisper_embedding(vidname, i - offset, syncnet_T=5)
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
            m = self.crop_audio_window(audio_embeddings, i - offset, syncnet_T=5)
            if m.shape[0] != self.frame_audio_size:
                return None
            audios.append(m)

        audios = torch.stack(audios)
        return audios

    def prepare_window(self, window):
        x = torch.stack(window) # N x [C, H, W] -> [N, C, H, W]
        x = x.permute(1, 0, 2, 3) # [N, C, H, W] -> [C, N, H, W]
        return x

    def __len__(self):
        return self.dataset_size # len(self.all_videos)

    def __getitem__(self, idx):
        while 1:
            idx = random.randint(0, len(self.all_videos) - 1)
            vidname = self.all_videos[idx]
                        
            # get the list of *.pt but not wav2vec2.pt
            img_names = list(glob(join(vidname, '*.pt')))
            img_names = [img_name for img_name in img_names if not img_name.endswith("wav2vec2.pt") and not img_name.endswith("whisper.pt")]
            
            if len(img_names) <= 3 * self.args.syncnet_T:
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
                continue
            
            window_upper_half = [l['upper_half'] for l in window]
            window_full_image = [l['full_image'] for l in window]
            if None in window_upper_half or None in window_full_image:
                continue

            wrong_window = self.read_window(wrong_window_fnames)
            if wrong_window is None:
                continue
            
            wrong_window_full_image = [l['full_image'] for l in wrong_window]
            
            if args.whisper:
                # load audio embedding from {frame_id}.npy
                audio_cropped = self.get_whisper_embedding(vidname, self.get_frame_id(img_name), syncnet_T=args.syncnet_T)
                indiv_audios = self.get_whisper_segmented_audios(vidname, self.get_frame_id(img_name))
                # print(f"audio_cropped: {audio_cropped.shape}, indiv_audios: {indiv_audios.shape}")
                # audio_cropped: torch.Size([50, 384]), indiv_audios: torch.Size([5, 50, 384])
            else:
                # load audio embedding from file wav2vec2.pt
                audio_path = join(vidname, "wav2vec2.pt")
                audio_embeddings = torch.load(audio_path, map_location='cpu')[0]
                
                # crop audio embedding, video frame is 25fps, audio frame is 50fps
                audio_cropped = self.crop_audio_window(audio_embeddings.clone(), img_name, syncnet_T=args.syncnet_T)
                indiv_audios = self.get_segmented_audios(audio_embeddings.clone(), img_name)
            
            if audio_cropped is None or audio_cropped.shape[0] != self.syncnet_audio_size: continue
            if indiv_audios is None: continue

            # N x [C, H, W] -> [C, N, H, W]
            window_upper_half = self.prepare_window(window_upper_half)
            window_full_image = self.prepare_window(window_full_image)
            wrong_window_full_image = self.prepare_window(wrong_window_full_image)
            
            y = window_full_image
            
            # [C, N, H/2, W] + [C, N, H/2, W] -> [C, N, H, W]
            # window_black_lower_half = torch.zeros_like(window_upper_half)
            # window_stitch_image = torch.cat([window_upper_half, window_black_lower_half], dim=2)

            # [C, N, H, W] + [C, N, H, W] -> [2C, N, H, W]
            x = torch.cat([window_upper_half, wrong_window_full_image], dim=0)
            x = torch.FloatTensor(x)
            
            audio_cropped = audio_cropped.unsqueeze(0).float()
            indiv_audios = indiv_audios.unsqueeze(1).float()
            
            y = torch.FloatTensor(y)
            return x, indiv_audios, audio_cropped, y
        
class LatentWav2Lip(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        unet_config = UNet2DConditionModel.load_config(f"{self.hparams.unet_config}/config.json")
        self.unet = UNet2DConditionModel.from_config(unet_config)
        self.unet.train()
        self.zero_timestep = torch.zeros([])
        
        if self.hparams.enable_syncnet:
            self.syncnet = self.load_syncnet()
            self.syncnet.eval()  # Ensure syncnet is always in eval mode
        
        self.vae = self.load_vae('stabilityai/sd-vae-ft-mse')
        self.vae.eval()  # Ensure vae is always in eval mode
        
        # 为了只加载 Wav2Lip 的参数，我们需要将 strict_loading 设置为 False
        self.strict_loading = False
        
        # 定义损失函数
        self.recon_loss = nn.L1Loss() if not self.hparams.l2_loss else nn.MSELoss()
        self.log_loss = nn.BCELoss()
        
    def load_syncnet(self):
        syncnet = SyncNet_latent(self.hparams.syncnet_T, self.hparams.whisper) if self.hparams.syncnet else SyncNet_latent_xl(self.hparams.syncnet_T, self.hparams.whisper)
        ckpt = torch.load(self.hparams.syncnet if self.hparams.syncnet else self.hparams.syncnet_xl)
        new_state_dict = {k[len("model."):] if k.startswith("model.") else k: v for k, v in ckpt['state_dict'].items()}
        syncnet.load_state_dict(new_state_dict)

        # 冻结 Syncnet 的所有参数
        for param in syncnet.parameters():
            param.requires_grad = False
        return syncnet
    
    def load_vae(self, model_name):
        vae = AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-mse', local_files_only=True)
        
        # 冻结 VAE 的所有参数
        for param in vae.parameters():
            param.requires_grad = False
        return vae
    
    def load_unet(self, unet_ckpt):
        try:
            self.unet.from_pretrained(unet_ckpt)
        except Exception as e:
            print(f"Error loading UNet: {e}")
            print("Train UNet from scratch")
        self.unet.train()
        return self.unet
    
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        # 从模型的状态字典中删除 VAE 和 Syncnet 的参数
        # 传递额外的参数给父类的 state_dict
        original_state_dict = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        # 过滤掉包含 "vae" 和 "syncnet" 的键
        filtered_state_dict = {k: v for k, v in original_state_dict.items() if "vae" not in k and "syncnet" not in k}
        return filtered_state_dict

    def forward(self, audio_sequences, face_sequences):
        # scale vae latents
        scaling_factor = self.vae.config.scaling_factor
        face_sequences = face_sequences * scaling_factor
                
        face_sequences = self.reshape_face_sequences(face_sequences)
        audio_sequences = self.reshape_audio_sequences(audio_sequences)        
        g = self.unet(face_sequences, timestep=self.zero_timestep, encoder_hidden_states=audio_sequences).sample
        g = self.inverse_reshape_face_sequences(g)
                
        # scale back vae latents
        g = g / scaling_factor
        return g
    
    def cosine_loss(self, a, v, y):
        d = nn.functional.cosine_similarity(a, v)
        # Disable autocast for unsafe operations
        with torch.cuda.amp.autocast(enabled=False):
            loss = self.log_loss(d.unsqueeze(1), y)
        return loss

    def clip_loss(self, logits):
        # 为每个视频和音频生成正确的标签
        labels = torch.arange(logits.size(0)).long().to(self.device)
        
        # 计算损失，同时考虑视频到音频和音频到视频的匹配
        loss_audio_to_face = nn.functional.cross_entropy(logits, labels)
        loss_face_to_audio = nn.functional.cross_entropy(logits.T, labels)
    
        # 计算总损失
        loss = (loss_audio_to_face + loss_face_to_audio) / 2
        return loss
    
    def get_sync_loss(self, mel, g):
        # B, 4 * T, H, W
        g = torch.cat([g[:, :, i] for i in range(self.hparams.syncnet_T)], dim=1)

        if self.hparams.clip_loss:
            logits = self.syncnet.get_logits(mel, g)
            loss = self.clip_loss(logits)
        else:
            a, v = self.syncnet(mel, g)
            y = torch.ones(g.size(0), 1).to(self.device)
            loss = self.cosine_loss(a, v, y)
        return loss

    def reshape_face_sequences(self, tensor):
        """
        Reshape and concatenate a tensor assuming a format similar to [64, 8, 5, 96, 96],
        but dimensions are taken from the input tensor to increase flexibility.

        Parameters:
            tensor (torch.Tensor): A tensor with dimensions [batch_size, channels, groups, height, width].
        
        Returns:
            torch.Tensor: A reshaped tensor with dimensions [batch_size * groups, channels, height, width].
        """
        # 获取输入tensor的维度
        batch_size, channels, groups, height, width = tensor.shape
                
        # Reshape the tensor
        reshaped_tensor = tensor.permute(0, 2, 1, 3, 4).contiguous().view(batch_size * groups, channels, height, width)        
        return reshaped_tensor
    
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

    def reshape_audio_sequences(self, tensor):
        """
        Reshape a tensor from [batch_size, dim1, dim2, dim3, features] to [batch_size * dim1, dim2 * dim3, features].
        
        Parameters:
            tensor (torch.Tensor): A tensor with dimensions [batch_size, dim1, dim2, dim3, features].
        
        Returns:
            torch.Tensor: A reshaped tensor with dimensions [batch_size * dim1, dim2 * dim3, features].
        """
        batch_size, dim1, dim2, dim3, features = tensor.shape
        
        # Reshape the tensor
        reshaped_tensor = tensor.view(batch_size * dim1, dim2 * dim3, features)
        
        # print(f"Reshaped tensor shape: {reshaped_tensor.shape}")
        return reshaped_tensor

    def training_step(self, batch, batch_idx):
        x, indiv_audios, audio_cropped, gt = batch
        # print(f"x: {x.shape}, indiv_mels: {indiv_audios.shape}, mel: {audio_cropped.shape}, gt: {gt.shape}")
        # x: torch.Size([64, 8, 5, 96, 96]), indiv_mels: torch.Size([64, 5, 1, 10, 768]), mel: torch.Size([64, 1, 10, 768]), gt: torch.Size([64, 4, 5, 96, 96])
        
        # dropout reference frames if enabled
        if self.hparams.dropout_ref:
            # to drop the ref frames, based on dropout_ref_prob
            need_dropout = random.random() < self.hparams.dropout_ref_prob
            # print(f"need_dropout: {need_dropout}")
            if need_dropout:
                upper_half, ref = x.chunk(2, dim=1)
                # print(f"upper_half: {upper_half.shape}, ref: {ref.shape}")
                # upper_half: torch.Size([8, 4, 5, 96, 96]), ref: torch.Size([8, 4, 5, 96, 96])
                x = torch.cat([upper_half, torch.zeros_like(ref)], dim=1)
        
        g = self(indiv_audios, x)
        
        if self.hparams.enable_syncnet and self.hparams.syncnet_wt > 0.:
            sync_loss = self.get_sync_loss(audio_cropped, g)
        else:
            sync_loss = 0.
                
        recon_loss = self.recon_loss(g, gt)
        
        loss = self.hparams.syncnet_wt * sync_loss + (1 - self.hparams.syncnet_wt) * recon_loss    
        self.log('train_loss', loss, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log('train_sync_loss', sync_loss, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log('train_recon_loss', recon_loss, prog_bar=True, sync_dist=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, indiv_audios, audio_cropped, gt = batch
        # print(f"x: {x.shape}, indiv_mels: {indiv_audios.shape}, mel: {audio_cropped.shape}, gt: {gt.shape}")
        # x: torch.Size([64, 8, 5, 96, 96]), indiv_mels: torch.Size([64, 5, 1, 10, 768]), mel: torch.Size([64, 1, 10, 768]), gt: torch.Size([64, 4, 5, 96, 96])
        g = self(indiv_audios, x)        
        
        if self.hparams.enable_syncnet:
            sync_loss = self.get_sync_loss(audio_cropped, g)
        else:
            sync_loss = 0.
            
        recon_loss = self.recon_loss(g, gt)
        val_loss = self.hparams.syncnet_wt * sync_loss + (1 - self.hparams.syncnet_wt) * recon_loss
        
        if self.hparams.sample_images and batch_idx == 0:
            self.save_sample_images(x[:1], g[:1], gt[:1], self.global_step, self.trainer.checkpoint_callback.dirpath)
        
        self.log('val_loss', val_loss, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log('val_sync_loss', sync_loss, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log('val_recon_loss', recon_loss, prog_bar=True, sync_dist=True, on_epoch=True)
        return val_loss

    def save_sample_images(self, x, g, gt, global_step, checkpoint_dir):
        refs = x[:, 4:, :, :, :]
        inps = x[:, :4, :, :, :]
        
        refs = self.decode_latent(refs)
        inps = self.decode_latent(inps)
        g = self.decode_latent(g)
        gt = self.decode_latent(gt)
        
        sample_image_dir = join(os.path.dirname(checkpoint_dir), "sample_images")
        os.makedirs(sample_image_dir, exist_ok=True)
        
        folder = join(sample_image_dir, "samples_step_{:09d}".format(global_step))
        os.makedirs(folder, exist_ok=True)
            
        collage = np.concatenate((refs, inps, g, gt), axis=-2)
        for batch_idx, c in enumerate(collage[:4]):
            for t in range(len(c)):
                cv2.imwrite('{}/{}_{}.jpg'.format(folder, batch_idx, t), c[t][:, :, ::-1])
    
    def decode_latent(self, face_sequences):        
        input_dim_size = len(face_sequences.size())
        if input_dim_size > 4:
            face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)
            
        output_tensor=None
        for i in range(0, face_sequences.shape[0]):
            batch_input_tensor = face_sequences[i: i + 1]
            with torch.no_grad():
                rec = self.vae.decode(batch_input_tensor).sample
                output_tensor = rec if output_tensor is None else torch.cat([output_tensor, rec], dim=0)
                
        if input_dim_size > 4:
                output_tensor = torch.split(output_tensor, face_sequences.size(0), dim=0) # [(B, C, H, W)]
                output_tensor = torch.stack(output_tensor, dim=1) # (B, T, C, H, W)

        outputs = (((output_tensor + 1) / 2).clamp(0, 1) * 255).byte().cpu().numpy().transpose(0, 1, 3, 4, 2).astype(np.uint8)
        return outputs
    
    def on_validation_epoch_end(self) -> None:
        if self.hparams.enable_syncnet:
            val_sync_loss = self.trainer.logged_metrics['val_sync_loss']
            if val_sync_loss < .75:
                print(f"Syncnet loss {val_sync_loss} is less than 0.75, setting syncnet_wt to {self.hparams.sync_loss_weight}")
                self.hparams.syncnet_wt = self.hparams.sync_loss_weight

    def configure_optimizers(self):
        # optimizer = FusedAdam(self.unet.parameters(), lr=1e-4)
        optimizer = torch.optim.AdamW(self.unet.parameters(), lr=1e-4)
        
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
        train_dataset = AudioVisualDataset(args=self.hparams, data_root=self.hparams.data_root, split='train' if not self.hparams.overfit else 'main', dataset_name=self.hparams.dataset_name, dataset_size=2000) 
        if self.hparams.ft_dataset:
            fine_tune_dataset = AudioVisualDataset(args=self.hparams, data_root=self.hparams.ft_root, split='train' if not self.hparams.overfit else 'main', dataset_name=self.hparams.ft_dataset, dataset_size=2000)
            train_dataset = CombinedDataset(train_dataset, fine_tune_dataset)
        return DataLoader(train_dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers)

    def val_dataloader(self):
        test_dataset = AudioVisualDataset(args=self.hparams, data_root=self.hparams.data_root, split='val', dataset_name=self.hparams.dataset_name, dataset_size=200)
        if self.hparams.ft_dataset:
            fine_tune_dataset = AudioVisualDataset(args=self.hparams, data_root=self.hparams.ft_root, split='val', dataset_name=self.hparams.ft_dataset, dataset_size=200)
            test_dataset = CombinedDataset(test_dataset, fine_tune_dataset)
        return DataLoader(test_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)

def print_training_info(args):
    print("\nTraining Configuration:")
    print(f"Data Root: {args.data_root}")
    print(f"Dataset Name: {args.dataset_name}")
    print(f"Clip Loss Enabled: {args.clip_loss}")
    print(f"U-Net Config File: {args.unet_config}")
    print(f"Checkpoint Path: {args.ckpt}")
    print(f"SyncNet Checkpoint Path: {args.syncnet}")
    print(f"SyncNet XL Checkpoint Path: {args.syncnet_xl}")
    print(f"Syncnet Enabled: {args.enable_syncnet}")
    print(f"Sample Images Enabled: {args.sample_images}")
    print(f"WandB Logging Enabled: {args.wandb}")
    print(f"Overfit Mode Enabled: {args.overfit}")
    print(f"Dropout on Reference Frames Enabled: {args.dropout_ref}")
    print(f"Whisper Enabled: {args.whisper}")
    print(f"Gradient Accumulation Steps: {args.accu_grad}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Sync Loss Weight: {args.sync_loss_weight}")
    print(f"Fine-Tune Dataset: {args.ft_dataset}")
    print(f"Fine-Tune Dataset Root: {args.ft_root}")
    print(f"L2 Loss Enabled: {args.l2_loss}")
    print(f"SyncNet T: {args.syncnet_T}")
    print("\nStarting training...\n")
    
if __name__ == "__main__":
    # Set the matrix multiplication precision
    # Use 'medium' for better performance with acceptable precision loss
    torch.set_float32_matmul_precision('medium')

    parser = argparse.ArgumentParser(description='Code to train latent wav2lip with lightning')
    parser.add_argument("--data_root", help="Root folder of the preprocessed LRS2 dataset", required=True)
    parser.add_argument("--dataset_name", help="Name of the dataset to use, eg: 3300, hdtf, 3300_and_hdtf", required=True)
    parser.add_argument('--clip_loss', action='store_true', help='Enable clip loss.')
    parser.add_argument('--unet_config', type=str, help='Path to the unet config file', default='customized_unet_v4')
    parser.add_argument('--ckpt', type=str, help='Path to the checkpoint to load the model from')
    parser.add_argument('--syncnet', type=str, help='Path to the syncnet checkpoint to load the model from')
    parser.add_argument('--syncnet_xl', type=str, help='Path to the syncnet xl checkpoint to load the model from')
    parser.add_argument('--sample_images', action='store_true', help='Save sample images')
    parser.add_argument('--wandb', action='store_true', help='Enable wandb logging')
    parser.add_argument('--overfit', action='store_true', help='Enable overfitting mode to focus training on training loss.')
    parser.add_argument('--dropout_ref', action='store_true', help='Enable dropout on the reference frames.')
    parser.add_argument('--whisper', action='store_true', help='Enable whisper as the audio feature extractor.')
    parser.add_argument('--accu_grad', type=int, default=4, help='Number of gradient accumulation steps')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--sync_loss_weight', type=float, default=0.1, help='Weight for sync loss')
    parser.add_argument('--ft_dataset', type=str, help='Fine-tune dataset name')
    parser.add_argument('--ft_root', type=str, help='Root folder of the fine-tune dataset')
    parser.add_argument('--l2_loss', action='store_true', help='Enable L2 loss')
    parser.add_argument('--syncnet_T', type=int, default=5, help='Number of frames to consider for syncnet')
    args = parser.parse_args()
    
    if not args.syncnet and not args.syncnet_xl:
        args.enable_syncnet = False
    else:
        args.enable_syncnet = True
    
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

    # Create an instance of LatentWav2Lip with merged parameters
    model = LatentWav2Lip(hparams_dict)
    
    # Load the UNet model if a checkpoint is not provided
    if not model.hparams.ckpt:
        model.load_unet(model.hparams.unet_config)

    # Checkpoint callback to save the model periodically
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filename='wav2lip-u-' + args.dataset_name + '-t=' + str(args.syncnet_T) + '-{epoch}-{step}-{train_loss:.3f}-{val_loss:.3f}-{val_sync_loss:.3f}',
        save_top_k=3,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )
    
    # 设置日志目录和实验名称
    if args.wandb:
        logger = WandbLogger(project='latent_wav2lip')
    else:
        logger = TensorBoardLogger('experiments', name='latent_wav2lip_experiment')

    callbacks = [checkpoint_callback, RichProgressBar(), LearningRateMonitor(logging_interval='step')]

    # Include EarlyStopping if overfitting is enabled
    if args.overfit:
        early_stopping_callback = EarlyStopping(monitor='train_loss', min_delta=0.001, patience=100, verbose=True, mode='min', stopping_threshold=0.6)
        callbacks.append(early_stopping_callback)
        
    # Trainer setup for multi-GPU training
    trainer = Trainer(
        logger=logger,
        max_epochs=model.hparams.nepochs,
        strategy='ddp', 
        precision='16-mixed',
        accumulate_grad_batches=model.hparams.accu_grad,
        gradient_clip_val=1.0,
        callbacks=callbacks
    )

    trainer.fit(model, ckpt_path=model.hparams.ckpt if model.hparams.ckpt else None)

