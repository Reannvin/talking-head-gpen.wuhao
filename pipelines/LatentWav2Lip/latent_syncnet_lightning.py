from os.path import dirname, join, basename, isfile
from glob import glob
from models import SyncNet_latent as SyncNet
import audio
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os, random, argparse
from hparams import hparams, get_image_list
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import RichProgressBar, LearningRateMonitor
from lr_scheduler import LambdaLinearScheduler

syncnet_T = 5
syncnet_mel_step_size = 16

class AudioVisualDataset(Dataset):
    def __init__(self, split, need_negative_sample, dataset_size=512000):
        self.all_videos = get_image_list(args.data_root, split)
        self.need_negative_sample = need_negative_sample
        self.split = split
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
    
    def crop_audio_window(self, spec, start_frame):
        # num_frames = (T x hop_size * fps) / sample_rate
        start_frame_num = self.get_frame_id(start_frame)
        start_idx = int(80. * (start_frame_num / float(hparams.fps)))

        end_idx = start_idx + syncnet_mel_step_size

        return spec[start_idx : end_idx, :]

    def __len__(self):
        return self.dataset_size # len(self.all_videos)
    
    def __getitem__(self, idx):
        while 1:
            idx = random.randint(0, len(self.all_videos) - 1)
            vidname = self.all_videos[idx]

            img_names = list(glob(join(vidname, '*.pt')))
            img_names = [img_name for img_name in img_names if not img_name.endswith("wav2vec2.pt")]
            if len(img_names) <= 3 * syncnet_T:
                # print(f"Error getting img_names from {vidname}")
                continue
            img_name = random.choice(img_names)
            wrong_img_name = random.choice(img_names)
            while wrong_img_name == img_name:
                wrong_img_name = random.choice(img_names)

            if not self.need_negative_sample or random.choice([True, False]):
                y = torch.ones(1).float()
                chosen = img_name
            else:
                y = torch.zeros(1).float()
                chosen = wrong_img_name

            window_fnames = self.get_window(chosen)
            if window_fnames is None:
                # print(f"Error getting window from {vidname}")
                continue

            window = []
            all_read = True
            for fname in window_fnames:
                latent = torch.load(fname)
                if latent is None:
                    all_read = False
                    break
                window.append(latent['full_image'])

            if not all_read: 
                # print(f"Error reading latent from window_fnames {vidname}")
                continue
            
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
                print(f"Loading audio from {vidname} error {e}.")
                continue

            mel = self.crop_audio_window(orig_mel.copy(), img_name)

            if (mel.shape[0] != syncnet_mel_step_size):
                # print(f"mel wrong shape {mel.shape}")
                continue
            
            # window_latents 5 * [4, 48, 96] -> [5 * 4, 48, 96]
            x = torch.cat(window, dim=0)

            x = torch.FloatTensor(x)
            mel = torch.FloatTensor(mel.T).unsqueeze(0)

            return x, mel, y

logloss = nn.BCELoss()
def cosine_loss(a, v, y):
    d = nn.functional.cosine_similarity(a, v)
    loss = logloss(d.unsqueeze(1), y)
    return loss

def clip_loss(logits):
    # 为每个视频和音频生成正确的标签
    labels = torch.arange(logits.size(0)).long().to(logits.device)
    
    # 计算损失，同时考虑视频到音频和音频到视频的匹配
    loss_audio_to_face = nn.functional.cross_entropy(logits, labels)
    loss_face_to_audio = nn.functional.cross_entropy(logits.T, labels)
    
    # 计算总损失
    loss = (loss_audio_to_face + loss_face_to_audio) / 2
    return loss

class LatentSyncNet(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = SyncNet()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, mel, y = batch
        
        if self.hparams.clip_loss:
            # y must be ones
            assert torch.all(y == 1)
            logits = self.model.get_logits(mel, x)
            loss = clip_loss(logits)
        else:
            a, v = self.model(mel, x)
            loss = cosine_loss(a, v, y)
            
        self.log('train_loss', loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, mel, y = batch
        a, v = self.model(mel, x)
        loss = cosine_loss(a, v, y)

        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        
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
        if self.hparams.clip_loss:
            # train with clip loss, doesn't want negative sample
            train_dataset = AudioVisualDataset('main', need_negative_sample=False, dataset_size=512000) 
        else:
            # train with cosine loss, negative sample wanted
            train_dataset = AudioVisualDataset('main', need_negative_sample=True, dataset_size=512000)
            
        return DataLoader(train_dataset, batch_size=self.hparams.syncnet_batch_size, shuffle=True, num_workers=self.hparams.num_workers)

    def val_dataloader(self):
        test_dataset = AudioVisualDataset('val', need_negative_sample=True, dataset_size=51200) # val with cosine loss, negative sample wanted
        return DataLoader(test_dataset, batch_size=self.hparams.syncnet_batch_size, num_workers=self.hparams.num_workers)

if __name__ == "__main__":
    # Set the matrix multiplication precision
    # Use 'medium' for better performance with acceptable precision loss
    torch.set_float32_matmul_precision('medium')

    parser = argparse.ArgumentParser(description='Code to train the expert lip-sync discriminator with lightning')
    parser.add_argument("--data_root", help="Root folder of the preprocessed LRS2 dataset", required=True)
    parser.add_argument('--clip_loss', action='store_true', help='Enable clip loss.')
    parser.add_argument('--ckpt', type=str, help='Path to the checkpoint to load the model from')
    parser.add_argument('--wandb', action='store_true', help='Enable wandb logging')
    args = parser.parse_args()
    
    print(f" Train SyncNet with {args.data_root}" + (" with clip loss" if args.clip_loss else " with cosine loss"))
    
    # Convert hparams instance to a dictionary
    hparams_dict = hparams.data

    # Update hparams with args
    hparams_dict.update(vars(args))

    # Create an instance of LitSyncNet with merged parameters
    model = LatentSyncNet(hparams_dict)

    # Checkpoint callback to save the model periodically
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filename='syncnet-f-{epoch}-{step}-{train_loss:.3f}-{val_loss:.3f}',
        save_top_k=3,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )
    
    # 设置日志目录和实验名称
    if args.wandb:
        logger = WandbLogger(project='latent_syncnet')
    else:
        logger = TensorBoardLogger('experiments', name='latent_syncnet_experiment')
    
    # Trainer setup for multi-GPU training
    trainer = Trainer(
        logger=logger,
        max_epochs=model.hparams.nepochs,
        strategy='ddp' if model.hparams.clip_loss else 'ddp_find_unused_parameters_true',
        callbacks=[checkpoint_callback, RichProgressBar(), LearningRateMonitor(logging_interval='step')]
    )

    trainer.fit(model, ckpt_path=model.hparams.ckpt if model.hparams.ckpt else None)

