from os.path import dirname, join, basename, isfile
from glob import glob
from models import SyncNet_image_256
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
import cv2
import yaml
from PIL import Image
import re
import os.path as osp



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
                dataset_size=self.dataset_size,
                need_negative_sample=True
            )
            self.datasets.append(dataset)
            self.ratios.append(dataset_config['ratio'])


    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        dataset_choice = random.choices(self.datasets, weights=self.ratios, k=1)[0]
        return dataset_choice[idx]


class AudioVisualDataset(Dataset):

    def __init__(self, data_root, audio_root, split, need_negative_sample, dataset_name, args, dataset_size=512000):
        self.all_videos = get_image_list(data_root, dataset_name, split)
        self.need_negative_sample = need_negative_sample
        self.split = split
        self.dataset_size = dataset_size
        self.syncnet_audio_step_size = 10 * args.syncnet_T if args.whisper else 2 * args.syncnet_T
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
    
    def get_whisper_embedding(self, vidname, frame_id, syncnet_T):
        try:
            # whisper_file = f"{frame_id}.npy" if syncnet_T == 5 else f"{frame_id}.npy.{syncnet_T}.npy"
            whisper_file = f"{frame_id}.npy" 
            audio_path = join(vidname, whisper_file)
            audio_embedding = np.load(audio_path)
            audio_embedding = torch.from_numpy(audio_embedding)
        except:
            print(f"Error loading {audio_path}")
            audio_embedding = None
        return audio_embedding
    
    def crop_audio_window(self, audio_embeddings, start_frame):
        start_frame_num = self.get_frame_id(start_frame)
        start_idx = int(50. * (start_frame_num / float(self.args.fps)))
        end_idx = start_idx + self.syncnet_audio_step_size
        return audio_embeddings[start_idx:end_idx]

    def __len__(self):
        return self.dataset_size # len(self.all_videos)
    
    def __getitem__(self, idx):
        while 1:
            idx = random.randint(0, len(self.all_videos) - 1)
            vidname = self.all_videos[idx]
            img_names = list(glob(join(vidname, '*.jpg')))
            
            if len(img_names) <= 3 * self.args.syncnet_T:
                # print(f"Video {vidname} has less than {3 * self.args.syncnet_T} frames, skipping...")
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
                # print(fname)
                img = cv2.imread(fname)
                if img is None:
                    all_read = False
                    break
                try:
                    img = cv2.resize(img, (self.args.image_size, self.args.image_size))
                except Exception as e:
                    all_read = False
                    break

                window.append(img)

            if not all_read: continue
            
            if self.audio_root:
                # switch from data_root to audio_root
                if self.args.offset:
                    video_relative_path = os.path.relpath(vidname, self.data_root)
                    offset = find_offset_from_hashmap(self.offset_info, video_relative_path, osp.basename(img_name))
                    vidname = vidname.replace(self.data_root, self.audio_root)
                    offset = offset if offset <= 5 else 9999999
                    img_name = modify_image_name(img_name, offset)
                else:
                    vidname = vidname.replace(self.data_root, self.audio_root)


                
            if self.args.whisper:
                audio_cropped = self.get_whisper_embedding(vidname, self.get_frame_id(img_name), syncnet_T=self.args.syncnet_T)
            else:
                # load audio embedding from file wav2vec2.pt
                audio_path = join(vidname, "wav2vec2.pt")
                audio_embeddings = torch.load(audio_path, map_location='cpu')[0]
                
                # crop audio embedding, video frame is 25fps, audio frame is 50fps
                audio_cropped = self.crop_audio_window(audio_embeddings, img_name)
                
            if audio_cropped is None or audio_cropped.shape[0] != self.syncnet_audio_step_size:
                continue
            
            audio_cropped = audio_cropped.unsqueeze(0).float()            
            
            x = np.concatenate(window, axis=2) / 255.
            x = x.transpose(2, 0, 1)
            x = x[:, x.shape[1]//2:]
            x = torch.FloatTensor(x)

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
        self.model = SyncNet_image_256(self.hparams.whisper, self.hparams.syncnet_T)

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
            print('clip loss')
            # train with clip loss, doesn't want negative sample
            # train_dataset = AudioVisualDataset('train' if not self.hparams.overfit else 'main', need_negative_sample=False, dataset_name=self.hparams.dataset_name, args=self.hparams, dataset_size=51200) 
            train_dataset = HybridDataset(config=self.hparams.dataset_config, split='train', args=self.hparams, dataset_size=self.hparams.dataset_size)
        else:
            # train with cosine loss, negative sample wanted
            # train_dataset = AudioVisualDataset('train' if not self.hparams.overfit else 'main', need_negative_sample=True, dataset_name=self.hparams.dataset_name, args=self.hparams, dataset_size=51200)
            train_dataset = HybridDataset(config=self.hparams.dataset_config, split='train', args=self.hparams, dataset_size=self.hparams.dataset_size)
            print('no clip loss')
            
        return DataLoader(train_dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers)

    def val_dataloader(self):
        # test_dataset = AudioVisualDataset('val', need_negative_sample=True, dataset_name=self.hparams.dataset_name, args=self.hparams, dataset_size=5120) # val with cosine loss, negative sample wanted
        test_dataset = HybridDataset(config=self.hparams.dataset_config, split='train', args=self.hparams, dataset_size=self.hparams.dataset_size//10)
        return DataLoader(test_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)

def print_training_info(args):
    print("\nSyncnet Training Configuration:")
    print(f"Dataset Config File: {args.dataset_config}")
    # print(f"Data Root: {args.data_root}")
    # print(f"Dataset Name: {args.dataset_name}")
    print(f"Clip Loss Enabled: {args.clip_loss}")
    print(f"Checkpoint Path: {args.ckpt}")
    print(f"WandB Logging Enabled: {args.wandb}")
    print(f"Overfit Mode Enabled: {args.overfit}")
    print(f"Whisper Enabled: {args.whisper}")
    print(f"Image Size: {args.image_size}")
    print(f"Syncnet T: {args.syncnet_T}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Gradient Accumulation Steps: {args.accu_grad}")
    print(f"Audio Root: {args.audio_root}")
    print("\nStarting training...\n")
    
if __name__ == "__main__":
    # Set the matrix multiplication precision
    # Use 'medium' for better performance with acceptable precision loss
    torch.set_float32_matmul_precision('medium')

    parser = argparse.ArgumentParser(description='Code to train the expert lip-sync discriminator with lightning')
    # parser.add_argument("--data_root", help="Root folder of the preprocessed LRS2 dataset", required=True)
    # parser.add_argument("--dataset_name", help="Name of the dataset to use, eg: 3300, hdtf, 3300_and_hdtf", required=True)
    parser.add_argument('--clip_loss', action='store_true', help='Enable clip loss.')
    parser.add_argument('--ckpt', type=str, help='Path to the checkpoint to load the model from')
    parser.add_argument('--wandb', action='store_true', help='Enable wandb logging')
    parser.add_argument('--overfit', action='store_true', help='Enable overfitting mode to focus training on training loss.')
    parser.add_argument('--whisper', action='store_true', help='Enable whisper as the audio feature extractor.')
    parser.add_argument('--image_size', type=int, default=256, help='Image size')
    parser.add_argument('--syncnet_T', type=int, default=5, help='Number of frames to consider for syncnet')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--accu_grad', type=int, default=4, help='Number of gradient accumulation steps')
    parser.add_argument('--audio_root', type=str, help='Root folder of the preprocessed audio dataset')

    parser.add_argument('--dataset_config', type=str, default='data/dataset_config.yaml', help='Path to the dataset config file')
    parser.add_argument('--wandb_entity', type=str, default='local-optima', help='wandb_entity')
    parser.add_argument('--wandb_name', type=str, default='', help='wandb_name')
    parser.add_argument('--dataset_size', type=int, default=8000, help='Size of the dataset')
    parser.add_argument('--offset', action='store_true', help='Enable overfitting mode to focus training on training loss.')
    args = parser.parse_args()


    # load the dataset config
    try:
        with open(args.dataset_config, 'r') as file:
            args.dataset_config = yaml.safe_load(file)
    except FileNotFoundError:
        raise RuntimeError("Dataset config file not found")
    
    # Print the training information
    print_training_info(args)

    # Convert hparams instance to a dictionary and update with args
    hparams_dict = hparams.data
    hparams_dict.update(vars(args))

    # Create an instance of LatentSyncNet with merged parameters
    model = LatentSyncNet(hparams_dict)

    # Setup checkpoint callback
    monitor = 'train_loss' if args.overfit else 'val_loss'
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=f'image_syncnet/{args.wandb_name}/checkpoints',
        filename='syncnet-i-' + f'-{args.dataset_config["name"]}' + ('-whisper-' if args.whisper else '-wav2vec2-') + ('-T=') + str(args.syncnet_T) + '-{epoch}-{step}-{train_loss:.3f}-{val_loss:.3f}',
        save_top_k=3,
        verbose=True,
        monitor=monitor,
        mode='min'
    )
    
    # 设置日志目录和实验名称
    if args.wandb:
        # logger = WandbLogger(project='image_syncnet')
        logger = WandbLogger(project='image_syncnet', entity=args.wandb_entity, name=args.wandb_name)
    else:
        logger = TensorBoardLogger('experiments', name='image_syncnet_experiment')
    
    callbacks = [checkpoint_callback, RichProgressBar(), LearningRateMonitor(logging_interval='step')]
    
    # Include EarlyStopping if overfitting is enabled
    # if args.overfit:
        # early_stopping_callback = EarlyStopping(monitor='train_loss', min_delta=0.001, patience=20, verbose=True, mode='min', stopping_threshold=0.2)
        # callbacks.append(early_stopping_callback)

    # Trainer setup
    trainer = Trainer(
        logger=logger,
        max_epochs=model.hparams.nepochs,
        accumulate_grad_batches=model.hparams.accu_grad,
        strategy='ddp',
        callbacks=callbacks
    )

    trainer.fit(model, ckpt_path=model.hparams.ckpt if model.hparams.ckpt else None)

