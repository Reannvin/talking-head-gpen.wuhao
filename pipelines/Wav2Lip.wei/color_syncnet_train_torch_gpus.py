from os.path import dirname, join, basename, isfile
from tqdm import tqdm

from models import SyncNet_color as SyncNet
import audio

import torch
from torch import nn
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils
import torchaudio
import numpy as np

from glob import glob

import os, random, cv2, argparse
from hparams import hparams, get_image_list
import utils

parser = argparse.ArgumentParser(description='Code to train the expert lip-sync discriminator')

parser.add_argument("--data_root", help="Root folder of the preprocessed LRS2 dataset", required=True)

parser.add_argument('--checkpoint_dir', help='Save checkpoints to this directory', required=True, type=str)
parser.add_argument('--checkpoint_path', help='Resumed from this checkpoint', default=None, type=str)

# distributed training parameters
parser.add_argument('--world_size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--local_rank', default=-1, type=int)
parser.add_argument('--dist_url', default='env://',
                    help='url used to set up distributed training')
# add log parameters
parser.add_argument('--log_dir', default=None,
                    help='path where to tensorboard log')


args = parser.parse_args()


global_step = 0
global_epoch = 0
use_cuda = torch.cuda.is_available()
print('use_cuda: {}'.format(use_cuda))

syncnet_T = 5
syncnet_mel_step_size = 16

class Dataset(object):
    def __init__(self, split):
        self.all_videos = get_image_list(args.data_root, split)
        self.all_videos = self.all_videos * 100
        self.audio_transforms = audio.get_audio_transforms()

    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])

    def get_window(self, start_frame):
        start_id = self.get_frame_id(start_frame)
        vidname = dirname(start_frame)

        window_fnames = []
        for frame_id in range(start_id, start_id + syncnet_T):
            frame = join(vidname, '{}.jpg'.format(frame_id))
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

    def crop_audio_window_torch(self, spec, start_frame):
        # num_frames = (T x hop_size * fps) / sample_rate
        start_frame_num = self.get_frame_id(start_frame)
        start_idx = int(80. * (start_frame_num / float(hparams.fps)))

        end_idx = start_idx + syncnet_mel_step_size

        return spec[:, :, start_idx : end_idx]


    def __len__(self):
        return len(self.all_videos)

    def __getitem__(self, idx):
        while 1:
            idx = random.randint(0, len(self.all_videos) - 1)
            vidname = self.all_videos[idx]

            img_names = list(glob(join(vidname, '*.jpg')))
            if len(img_names) <= 3 * syncnet_T:
                continue
            img_name = random.choice(img_names)
            wrong_img_name = random.choice(img_names)
            while wrong_img_name == img_name:
                wrong_img_name = random.choice(img_names)

            if random.choice([True, False]):
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
                img = cv2.imread(fname)
                if img is None:
                    all_read = False
                    break
                try:
                    img = cv2.resize(img, (hparams.img_size, hparams.img_size))
                except Exception as e:
                    all_read = False
                    break

                window.append(img)

            if not all_read: continue

            try:
                wavpath = join(vidname, "audio.wav")
                # wav = audio.load_wav(wavpath, hparams.sample_rate)
                wav1, sample_rate = torchaudio.load(wavpath, normalize=True)
                wav1 = wav1.mean(dim=0, keepdim=True)
                if sample_rate != hparams.sample_rate:
                    print('resample')
                    wav1 = torchaudio.functional.resample(wav1, sample_rate,  hparams.sample_rate)

                # print('wav:', wav.shape, wav1.shape)
                # orig_mel = audio.melspectrogram(wav).T
                orig_mel1 = self.audio_transforms(wav1)
                # print('mel:', orig_mel.shape, orig_mel1.shape)
            except Exception as e:
                print(e)
                continue

            # mel = self.crop_audio_window(orig_mel.copy(), img_name)
            mel = self.crop_audio_window_torch(orig_mel1.clone(), img_name)
            # print('mel:', mel.shape, mel1.shape, mel1.dtype)

            # if (mel.shape[0] != syncnet_mel_step_size):
            if (mel.shape[-1] != syncnet_mel_step_size):
                continue

            # H x W x 3 * T
            x = np.concatenate(window, axis=2) / 255.
            x = x.transpose(2, 0, 1)
            x = x[:, x.shape[1]//2:]

            x = torch.FloatTensor(x)
            # mel = torch.FloatTensor(mel.T).unsqueeze(0)
            # print('mel 2:', mel.shape)

            return x, mel, y

logloss = nn.BCELoss()
def cosine_loss(a, v, y):
    d = nn.functional.cosine_similarity(a, v)
    loss = logloss(d.unsqueeze(1), y)

    return loss

def train(device, model, train_data_loader, test_data_loader, optimizer,
          checkpoint_dir=None, checkpoint_interval=None, nepochs=None, log_writer=None):

    global global_step, global_epoch
    resumed_step = global_step
    log_writer.set_step(global_step)

    while global_epoch < nepochs:
        train_data_loader.sampler.set_epoch(global_epoch)
        running_loss = 0.
        print(f'{len(train_data_loader)} iters in one epoch')
        current_lr = optimizer.param_groups[0]['lr']
        log_writer.update(lr=current_lr, head="train")
        prog_bar = tqdm(enumerate(train_data_loader))
        for step, (x, mel, y) in prog_bar:
            model.train()
            optimizer.zero_grad()

            # Transform data to CUDA device
            x = x.to(device)

            mel = mel.to(device)

            a, v = model(mel, x)
            y = y.to(device)

            loss = cosine_loss(a, v, y)
            loss.backward()
            optimizer.step()

            global_step += 1
            cur_session_steps = global_step - resumed_step
            running_loss += loss.item()

            #********log***********
            loss_value = loss.item()
            loss_mean_value = running_loss / (step + 1)
            # if log_writer:
            log_writer.update(loss=loss_value, head="train")
            log_writer.update(loss_mean=loss_mean_value, head="train")

            if global_step == 1 or global_step % checkpoint_interval == 0 and utils.is_main_process():
                save_checkpoint(
                    model.module, optimizer, global_step, checkpoint_dir, global_epoch)

            if global_step % hparams.syncnet_eval_interval == 0:
                with torch.no_grad():
                    val_mean_loss = eval_model(test_data_loader, global_step, device, model, checkpoint_dir)
                # if log_writer:
                log_writer.update(loss_mean=val_mean_loss, head="val")

            prog_bar.set_description('Loss: {}'.format(running_loss / (step + 1)))
            log_writer.set_step()

        global_epoch += 1

def eval_model(test_data_loader, global_step, device, model, checkpoint_dir):
    eval_steps = 1400
    print('Evaluating for {} steps'.format(eval_steps))
    losses = []
    while 1:
        for step, (x, mel, y) in enumerate(test_data_loader):

            model.eval()

            # Transform data to CUDA device
            x = x.to(device)

            mel = mel.to(device)

            a, v = model(mel, x)
            y = y.to(device)

            loss = cosine_loss(a, v, y)
            losses.append(loss.item())

            if step > eval_steps: break

        averaged_loss = sum(losses) / len(losses)
        print(averaged_loss)

        return averaged_loss 

def save_checkpoint(model, optimizer, step, checkpoint_dir, epoch):

    checkpoint_path = join(
        checkpoint_dir, "checkpoint_step{:09d}.pth".format(global_step))
    optimizer_state = optimizer.state_dict() if hparams.save_optimizer_state else None
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "global_step": step,
        "global_epoch": epoch,
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)

def _load(checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint

def load_checkpoint(path, model, optimizer, reset_optimizer=False):
    global global_step
    global global_epoch

    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    model.load_state_dict(checkpoint["state_dict"])
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    global_step = checkpoint["global_step"]
    global_epoch = checkpoint["global_epoch"]

    return model

if __name__ == "__main__":
    utils.init_distributed_mode(args)

    checkpoint_dir = args.checkpoint_dir
    checkpoint_path = args.checkpoint_path

    if not os.path.exists(checkpoint_dir) and utils.is_main_process(): os.mkdir(checkpoint_dir)

    # Dataset and Dataloader setup
    train_dataset = Dataset('train_bili')
    test_dataset = Dataset('test_xiaohongshu')
    # train_dataset = Dataset('train')
    # test_dataset = Dataset('val')

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True,
    )


    train_data_loader = data_utils.DataLoader(
        train_dataset, sampler=sampler_train, batch_size=hparams.syncnet_batch_size, shuffle=False,
        num_workers=hparams.num_workers, drop_last=True)

    test_data_loader = data_utils.DataLoader(
        test_dataset, batch_size=hparams.syncnet_batch_size,
        num_workers=8)

    device = torch.device("cuda" if use_cuda else "cpu")
    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
    log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    # else:
        # log_writer = None

    # Model
    model = SyncNet().to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
    model_without_ddp = model.module
    print('total trainable params {}'.format(sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)))

    hparams.syncnet_lr = hparams.syncnet_lr * hparams.syncnet_batch_size * utils.get_world_size() / 64
    optimizer = optim.Adam([p for p in model_without_ddp.parameters() if p.requires_grad],
                           lr=hparams.syncnet_lr)

    if checkpoint_path is not None:
        load_checkpoint(checkpoint_path, model_without_ddp, optimizer, reset_optimizer=False)

    train(device, model, train_data_loader, test_data_loader, optimizer,
          checkpoint_dir=checkpoint_dir,
          checkpoint_interval=hparams.syncnet_checkpoint_interval,
          nepochs=hparams.nepochs, log_writer=log_writer)
