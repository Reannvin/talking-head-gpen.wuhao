from os.path import dirname, join, basename, isfile
from glob import glob
from models import SyncNet_latent as SyncNet
from models import Wav2Lip as Wav2Lip
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os, random, argparse
from hparams import hparams, get_image_list
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import RichProgressBar, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from lr_scheduler import LambdaLinearScheduler
import cv2
from diffusers import AutoencoderKL

syncnet_T = 5
syncnet_audio_step_size = 10

class AudioVisualDataset(Dataset):
    def __init__(self, split, dataset_size=512000):
        self.all_videos = get_image_list(args.data_root, split)
        self.dataset_size = dataset_size

    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])

    def get_window(self, start_frame):
        start_id = self.get_frame_id(start_frame)
        vidname = dirname(start_frame)

        window_fnames = []
        for frame_id in range(start_id, start_id + syncnet_T):
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
    
    def crop_audio_window(self, audio_embeddings, start_frame):
        if type(start_frame) == int:
            start_frame_num = start_frame
        else:
            start_frame_num = self.get_frame_id(start_frame) # 0-indexing ---> 1-indexing
        start_idx = int(50. * (start_frame_num / float(hparams.fps)))
        end_idx = start_idx + syncnet_audio_step_size
        return audio_embeddings[start_idx : end_idx]
    
    def get_segmented_audios(self, audio_embeddings, start_frame):
        audios = []
        assert syncnet_T == 5
        start_frame_num = self.get_frame_id(start_frame) + 1
        if start_frame_num - 2 < 0: return None
        for i in range(start_frame_num, start_frame_num + syncnet_T):
            m = self.crop_audio_window(audio_embeddings, i - 2)
            if m.shape[0] != syncnet_audio_step_size:
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
            img_names = [img_name for img_name in img_names if not img_name.endswith("wav2vec2.pt")]
            
            if len(img_names) <= 3 * syncnet_T:
                continue
            
            img_name = random.choice(img_names)
            wrong_img_name = random.choice(img_names)
            while wrong_img_name == img_name:
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
            
            # load audio embedding from file wav2vec2.pt
            audio_path = join(vidname, "wav2vec2.pt")
            audio_embeddings = torch.load(audio_path, map_location='cpu')[0]
            
            # crop audio embedding, video frame is 25fps, audio frame is 50fps
            audio_cropped = self.crop_audio_window(audio_embeddings.clone(), img_name)
            if audio_cropped.shape[0] != syncnet_audio_step_size:
                continue

            indiv_audios = self.get_segmented_audios(audio_embeddings.clone(), img_name)
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
            
            audio_cropped = audio_cropped.unsqueeze(0)
            indiv_audios = indiv_audios.unsqueeze(1)
            
            y = torch.FloatTensor(y)
            return x, indiv_audios, audio_cropped, y


class LatentWav2Lip(pl.LightningModule):
    recon_loss = nn.L1Loss()
    log_loss = nn.BCELoss()
            
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        
        self.wav2lip = Wav2Lip()
        self.syncnet = self.load_syncnet(self.hparams.syncnet)
        
        if self.hparams.sample_images:
            self.vae = self.load_vae('stabilityai/sd-vae-ft-mse')
        
        # 为了只加载 Wav2Lip 的参数，我们需要将 strict_loading 设置为 False
        self.strict_loading = False
        
    def load_syncnet(self, syncnet_ckpt):
        syncnet = SyncNet()
        ckpt = torch.load(syncnet_ckpt)
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
    
    def state_dict(self):
        # 从模型的状态字典中删除 VAE 和 Syncnet 的参数
        return {k: v for k, v in super().state_dict().items() if "vae" not in k and "syncnet" not in k}

    def forward(self, x):
        return self.model(x)
    
    def cosine_loss(self, a, v, y):
        d = nn.functional.cosine_similarity(a, v)
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
        g = torch.cat([g[:, :, i] for i in range(syncnet_T)], dim=1)
        # B, 4 * T, H, W
        a, v = self.syncnet(mel, g)

        if self.hparams.clip_loss:
            logits = self.syncnet.get_logits(mel, g)
            loss = self.clip_loss(logits)
        else:
            y = torch.ones(g.size(0), 1).float().to(self.device)
            loss = self.cosine_loss(a, v, y)
        return loss

    def training_step(self, batch, batch_idx):
        x, indiv_mels, mel, gt = batch
        
        g = self.wav2lip(indiv_mels, x)
        
        if self.hparams.syncnet_wt > 0.:
            sync_loss = self.get_sync_loss(mel, g)
        else:
            sync_loss = 0.
                
        recon_loss = self.recon_loss(g, gt)
        
        loss = self.hparams.syncnet_wt * sync_loss + (1 - self.hparams.syncnet_wt) * recon_loss    
        self.log('train_loss', loss, prog_bar=True, sync_dist=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, indiv_mels, mel, gt = batch
        
        g = self.wav2lip(indiv_mels, x)
        sync_loss = self.get_sync_loss(mel, g)
        recon_loss = self.recon_loss(g, gt)
        
        if self.hparams.sample_images and batch_idx == 0:
            self.save_sample_images(x[:1], g[:1], gt[:1], self.global_step, self.trainer.checkpoint_callback.dirpath)
        
        self.log('val_loss', recon_loss, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log('val_sync_loss', sync_loss, prog_bar=True, sync_dist=True, on_epoch=True)
        return recon_loss

    def save_sample_images(self, x, g, gt, global_step, checkpoint_dir):
        refs = x[:, 4:, :, :, :]
        inps = x[:, :4, :, :, :]
        
        refs = self.decode_latent(refs)
        inps = self.decode_latent(inps)
        g = self.decode_latent(g)
        gt = self.decode_latent(gt)
        
        sample_image_dir = join(os.path.dirname(checkpoint_dir), "sample_images")
        if not os.path.exists(sample_image_dir):
            os.mkdir(sample_image_dir)
        
        folder = join(sample_image_dir, "samples_step_{:09d}".format(global_step))
        if not os.path.exists(folder):
            os.mkdir(folder)
            
        collage = np.concatenate((refs, inps, g, gt), axis=-2)
        for batch_idx, c in enumerate(collage):
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
        val_sync_loss = self.trainer.logged_metrics['val_sync_loss']
        if val_sync_loss < .75:
            print(f"Syncnet loss {val_sync_loss} is less than 0.75, setting syncnet_wt to 0.01")
            self.hparams.syncnet_wt = 0.01 # switch on syncnet loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.wav2lip.parameters(), lr=1e-4)
        
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
        train_dataset = AudioVisualDataset('train', dataset_size=512000) 
        return DataLoader(train_dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers)

    def val_dataloader(self):
        test_dataset = AudioVisualDataset('val', dataset_size=5120)
        return DataLoader(test_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)

if __name__ == "__main__":
    # Set the matrix multiplication precision
    # Use 'medium' for better performance with acceptable precision loss
    torch.set_float32_matmul_precision('medium')

    parser = argparse.ArgumentParser(description='Code to train latent wav2lip with lightning')
    parser.add_argument("--data_root", help="Root folder of the preprocessed LRS2 dataset", required=True)
    parser.add_argument('--clip_loss', action='store_true', help='Enable clip loss.')
    parser.add_argument('--ckpt', type=str, help='Path to the checkpoint to load the model from')
    parser.add_argument('--syncnet', type=str, help='Path to the syncnet checkpoint to load the model from', required=True)
    parser.add_argument('--sample_images', action='store_true', help='Save sample images')
    args = parser.parse_args()
    
    print(f"Train wav2lip with {args.data_root}" + (" with clip loss" if args.clip_loss else " with cosine loss"))
    
    # Convert hparams instance to a dictionary
    hparams_dict = hparams.data

    # Update hparams with args
    hparams_dict.update(vars(args))

    # Create an instance of LatentWav2Lip with merged parameters
    model = LatentWav2Lip(hparams_dict)

    # Checkpoint callback to save the model periodically
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filename='wav2lip-w-{epoch}-{step}-{train_loss:.3f}-{val_loss:.3f}-{val_sync_loss:.3f}',
        save_top_k=3,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )
    
    # 设置日志目录和实验名称
    logger = TensorBoardLogger('experiments', name='latent_wav2lip_experiment')

    # Trainer setup for multi-GPU training
    trainer = Trainer(
        logger=logger,
        max_epochs=model.hparams.nepochs,
        strategy='ddp',
        callbacks=[checkpoint_callback, RichProgressBar(), LearningRateMonitor(logging_interval='step')]
    )

    trainer.fit(model, ckpt_path=model.hparams.ckpt if model.hparams.ckpt else None)

