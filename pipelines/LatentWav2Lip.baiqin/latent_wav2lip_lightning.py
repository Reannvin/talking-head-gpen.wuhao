from os.path import dirname, join, basename, isfile
from glob import glob
from models import SyncNet_latent as SyncNet
from models import Wav2Lip as Wav2Lip
import audio
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os, random, argparse
from hparams import hparams, get_image_list
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import cv2
from diffusers import AutoencoderKL

syncnet_T = 5
syncnet_mel_step_size = 16

class AudioVisualDataset(Dataset):
    def __init__(self, split, multiply):
        self.all_videos = get_image_list(args.data_root, split) * multiply

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

    def crop_audio_window(self, spec, start_frame):
        if type(start_frame) == int:
            start_frame_num = start_frame
        else:
            start_frame_num = self.get_frame_id(start_frame) # 0-indexing ---> 1-indexing
        start_idx = int(80. * (start_frame_num / float(hparams.fps)))
        
        end_idx = start_idx + syncnet_mel_step_size

        return spec[start_idx : end_idx, :]

    def get_segmented_mels(self, spec, start_frame):
        mels = []
        assert syncnet_T == 5
        start_frame_num = self.get_frame_id(start_frame) + 1 # 0-indexing ---> 1-indexing
        if start_frame_num - 2 < 0: return None
        for i in range(start_frame_num, start_frame_num + syncnet_T):
            m = self.crop_audio_window(spec, i - 2)
            if m.shape[0] != syncnet_mel_step_size:
                return None
            mels.append(m.T)

        mels = np.asarray(mels)

        return mels

    def prepare_window(self, window):
        x = torch.stack(window) # N x [C, H, W] -> [N, C, H, W]
        x = x.permute(1, 0, 2, 3) # [N, C, H, W] -> [C, N, H, W]
        return x

    def __len__(self):
        return len(self.all_videos)

    def __getitem__(self, idx):
        while 1:
            idx = random.randint(0, len(self.all_videos) - 1)
            vidname = self.all_videos[idx]
            img_names = list(glob(join(vidname, '*.pt')))
            if len(img_names) <= 3 * syncnet_T:
                continue
            
            img_name = random.choice(img_names)
            #print(img_name)
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

            try:
                mel_out_path = join(vidname, "mel.npy")
                if os.path.isfile(mel_out_path):  # x50 times faster - 0.002 -> 0.01s
                    with open(mel_out_path, "rb") as f:
                        orig_mel = np.load(f)
                else:
                    wavpath = os.path.join(vidname, "audio.wav")
                    wav = audio.load_wav(wavpath, hparams.sample_rate)

                    orig_mel = audio.melspectrogram(wav).T  # 0.2 -> 0.9s
                    with open(mel_out_path, "wb") as f:
                        np.save(f, orig_mel)
            except Exception as e:
                continue

            mel = self.crop_audio_window(orig_mel.copy(), img_name)
            
            if (mel.shape[0] != syncnet_mel_step_size):
                continue

            indiv_mels = self.get_segmented_mels(orig_mel.copy(), img_name)
            if indiv_mels is None: continue

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
            mel = torch.FloatTensor(mel.T).unsqueeze(0)
            indiv_mels = torch.FloatTensor(indiv_mels).unsqueeze(1)
            y = torch.FloatTensor(y)
            return x, indiv_mels, mel, y


class LatentWav2Lip(pl.LightningModule):
    recon_loss = nn.L1Loss()
    log_loss = nn.BCELoss()
            
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        
        self.wav2lip = Wav2Lip()
        self.syncnet = self.load_syncnet(self.hparams.syncnet)
        
        if self.hparams.sample_images:
            self.vae = self.load_vae(self.hparams.vae_path)
        
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
        vae = AutoencoderKL.from_pretrained(model_name, local_files_only=True)
        
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
        
        if self.hparams.sample_images and batch_idx == 0 and self.global_rank==0:
            self.save_sample_images(x, g, gt, self.global_step, self.trainer.checkpoint_callback.dirpath)
        
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
        batch_size=2
        for i in range(0, face_sequences.shape[0],batch_size):
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
        optimizer = torch.optim.Adam(self.wav2lip.parameters(), lr=self.hparams.initial_learning_rate)
        return optimizer

    def train_dataloader(self):
        train_dataset = AudioVisualDataset('train', multiply=self.hparams.multiply) 
        return DataLoader(train_dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers)

    def val_dataloader(self):
        test_dataset = AudioVisualDataset('val', multiply=1)
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
    parser.add_argument('--multiply', type=int, default=1, help='Multiply the number of videos in the dataset')
    parser.add_argument('--sample_images', action='store_true', help='Save sample images')
    parser.add_argument('--vae_path', type=str,  help='VAE LOADER',default='stabilityai/sd-vae-ft-mse')
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
        filename='wav2lip-f-{epoch}-{step}-{train_loss:.3f}-{val_loss:.3f}-{val_sync_loss:.3f}',
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
        callbacks=[checkpoint_callback]
    )

    trainer.fit(model, ckpt_path=model.hparams.ckpt if model.hparams.ckpt else None)

