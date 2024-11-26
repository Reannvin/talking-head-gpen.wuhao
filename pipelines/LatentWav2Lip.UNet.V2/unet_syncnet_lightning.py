from os.path import dirname, join, basename, isfile
from glob import glob
from models import SyncNet_latent, SyncNet_latent_xl
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import os, random, argparse
from hparams import hparams, get_image_list
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import RichProgressBar, LearningRateMonitor, EarlyStopping
from lr_scheduler import LambdaLinearScheduler
import numpy as np


class AudioVisualDataset(Dataset):
    def __init__(self, split, args, need_negative_sample, dataset_name, dataset_size=512000):
        self.all_videos = get_image_list(args.data_root, dataset_name, split)
        self.need_negative_sample = need_negative_sample
        self.split = split
        self.dataset_size = dataset_size
        self.syncnet_audio_step_size = 10 * args.syncnet_T if args.whisper else 2 * args.syncnet_T
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
    
    def get_whisper_embedding(self, vidname, frame_id):
        try:
            whisper_file = f"{frame_id}.npy" if self.args.syncnet_T == 5 else f"{frame_id}.npy.{self.args.syncnet_T}.npy"
            audio_path = join(vidname, whisper_file)
            audio_embedding = np.load(audio_path)
            audio_embedding = torch.from_numpy(audio_embedding)
        except:
            print(f"Error loading {audio_path}")
            audio_embedding = None
        return audio_embedding
    
    def crop_audio_window(self, audio_embeddings, start_frame):
        start_frame_num = self.get_frame_id(start_frame)
        start_idx = int(50. * (start_frame_num / float(hparams.fps)))
        end_idx = start_idx + self.syncnet_audio_step_size
        return audio_embeddings[start_idx:end_idx]

    def __len__(self):
        return self.dataset_size # len(self.all_videos)
    
    def __getitem__(self, idx):
        while 1:
            idx = random.randint(0, len(self.all_videos) - 1)
            vidname = self.all_videos[idx]

            # get the list of *.pt but not wav2vec2.pt
            img_names = sorted(glob(join(vidname, '*.pt')))
            img_names = [img_name for img_name in img_names if not img_name.endswith("wav2vec2.pt")]
            
            if len(img_names) <= 3 * self.args.syncnet_T:
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
                continue

            window = []
            all_read = True
            for fname in window_fnames:
                latent = torch.load(fname)
                if latent is None:
                    all_read = False
                    break
                window.append(latent['full_image'])

            if not all_read: continue
            
            if args.whisper:
                audio_cropped = self.get_whisper_embedding(vidname, self.get_frame_id(img_name))
            else:
                # load audio embedding from file wav2vec2.pt
                audio_path = join(vidname, "wav2vec2.pt")
                audio_embeddings = torch.load(audio_path, map_location='cpu')[0]
                
                # crop audio embedding, video frame is 25fps, audio frame is 50fps
                audio_cropped = self.crop_audio_window(audio_embeddings, img_name)
                
            # print("audio_cropped shape: ", audio_cropped.shape)
            if audio_cropped is None or audio_cropped.shape[0] != self.syncnet_audio_step_size:
                continue
            
            audio_cropped = audio_cropped.unsqueeze(0)
            
            # window_latents 5 * [4, 48, 96] -> [5 * 4, 48, 96]
            x = torch.cat(window, dim=0)
            x = torch.FloatTensor(x)
            
            audio_cropped = audio_cropped.float()

            return x, audio_cropped, y

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
        self.model = SyncNet_latent_xl(self.hparams.syncnet_T, self.hparams.whisper) if self.hparams.xl else SyncNet_latent(self.hparams.syncnet_T, self.hparams.whisper)

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
            train_dataset = AudioVisualDataset(args=self.hparams, split='train' if not self.hparams.overfit else 'main', need_negative_sample=False, dataset_name=self.hparams.dataset_name, dataset_size=512000) 
        else:
            # train with cosine loss, negative sample wanted
            train_dataset = AudioVisualDataset(args=self.hparams, split='train' if not self.hparams.overfit else 'main', need_negative_sample=True, dataset_name=self.hparams.dataset_name, dataset_size=512000)
            
        return DataLoader(train_dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers)

    def val_dataloader(self):
        test_dataset = AudioVisualDataset(args=self.hparams, split='val', need_negative_sample=True, dataset_name=self.hparams.dataset_name, dataset_size=51200) # val with cosine loss, negative sample wanted
        return DataLoader(test_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)

def print_training_info(args):
    print("\nSyncnet Training Configuration:")
    print(f"Data Root: {args.data_root}")
    print(f"Dataset Name: {args.dataset_name}")
    print(f"Clip Loss Enabled: {args.clip_loss}")
    print(f"Checkpoint Path: {args.ckpt}")
    print(f"WandB Logging Enabled: {args.wandb}")
    print(f"Overfit Mode Enabled: {args.overfit}")
    print(f"Whisper Enabled: {args.whisper}")
    print(f"Syncnet T: {args.syncnet_T}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Gradient Accumulation Steps: {args.accu_grad}")
    print("\nStarting training...\n")
    
if __name__ == "__main__":
    # Set the matrix multiplication precision
    # Use 'medium' for better performance with acceptable precision loss
    torch.set_float32_matmul_precision('medium')

    parser = argparse.ArgumentParser(description='Code to train the expert lip-sync discriminator with lightning')
    parser.add_argument("--data_root", help="Root folder of the preprocessed LRS2 dataset", required=True)
    parser.add_argument("--dataset_name", help="Name of the dataset to use, eg: 3300, hdtf, 3300_and_hdtf", required=True)
    parser.add_argument('--clip_loss', action='store_true', help='Enable clip loss.')
    parser.add_argument('--ckpt', type=str, help='Path to the checkpoint to load the model from')
    parser.add_argument('--wandb', action='store_true', help='Enable wandb logging')
    parser.add_argument('--xl', action='store_true', help='Use SyncNet_latent_xl for training')
    parser.add_argument('--overfit', action='store_true', help='Enable overfitting mode to focus training on training loss.')
    parser.add_argument('--whisper', action='store_true', help='Enable whisper as the audio feature extractor.')
    parser.add_argument('--syncnet_T', type=int, default=5, help='Number of frames to consider for syncnet')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--accu_grad', type=int, default=1, help='Number of gradient accumulation steps')
    args = parser.parse_args()
    
    print_training_info(args)

    # Convert hparams instance to a dictionary and update with args
    hparams_dict = hparams.data
    hparams_dict.update(vars(args))

    # Create an instance of LatentSyncNet with merged parameters
    model = LatentSyncNet(hparams_dict)

    # Setup checkpoint callback
    monitor = 'train_loss' if args.overfit else 'val_loss'
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filename='syncnet-' + ('xl' if args.xl else 'u') + '-' + args.dataset_name + ('-whisper-' if args.whisper else '-wav2vec2-') + '{epoch}-{step}-{train_loss:.3f}-{val_loss:.3f}',
        save_top_k=3,
        verbose=True,
        monitor=monitor,
        mode='min'
    )
    
    # 设置日志目录和实验名称
    if args.wandb:
        logger = WandbLogger(project='latent_syncnet')
    else:
        logger = TensorBoardLogger('experiments', name='latent_syncnet_experiment')
    
    callbacks = [checkpoint_callback, RichProgressBar(), LearningRateMonitor(logging_interval='step')]
    
    # Include EarlyStopping if overfitting is enabled
    if args.overfit:
        early_stopping_callback = EarlyStopping(monitor='train_loss', min_delta=0.001, patience=100, verbose=True, mode='min', stopping_threshold=0.2)
        callbacks.append(early_stopping_callback)

    # Trainer setup
    trainer = Trainer(
        logger=logger,
        max_epochs=model.hparams.nepochs,
        accumulate_grad_batches=model.hparams.accu_grad,
        strategy='ddp' if model.hparams.clip_loss else 'ddp_find_unused_parameters_true',
        callbacks=callbacks
    )

    trainer.fit(model, ckpt_path=model.hparams.ckpt if model.hparams.ckpt else None)

