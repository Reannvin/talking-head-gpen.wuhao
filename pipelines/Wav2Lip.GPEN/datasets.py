import os
from os.path import join, isfile, basename, dirname
from glob import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision
import yaml

class ImageTransform:
    def __init__(self, data_aug_image, image_size):
        self.data_aug_image = data_aug_image
        self.image_size = image_size
        self.basic_transforms = torchvision.transforms.Resize((image_size, image_size))
        self.data_aug_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((image_size, image_size)),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        ])

    def __call__(self, image):
        # Apply data augmentation if enabled
        if self.data_aug_image:
            image = self.data_aug_transforms(image)
        else:
            image = self.basic_transforms(image)
        
        return image

class MaskTransform:
    def __init__(self, data_aug_mask, mask_ratio):
        self.data_aug_mask = data_aug_mask
        self.mask_ratio = mask_ratio

    def __call__(self, image):
        # Random mask ratio around self.mask_ratio with range [-0.1, +0.1]
        if self.data_aug_mask:
            mask_ratio = self.mask_ratio + random.uniform(-0.1, 0.1)
        else:
            mask_ratio = self.mask_ratio
            
        image = self.apply_mask(image, mask_ratio)
        return image

    def apply_mask(self, image, mask_ratio):
        # Apply a mask to the lower part of the image based on the mask_ratio
        mask_height = int(image.shape[2] * mask_ratio)
        image[:, :, -mask_height:, :] = 0.  # Setting masked pixels to 0
        return image

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
        self.all_videos = self.get_image_list(data_root, dataset_name, split)
        self.all_offsets = self.get_offset_dict(data_root, dataset_name)
        self.dataset_size = dataset_size
        self.syncnet_audio_size = 10 * args.syncnet_T if not args.wav2vec2 else 2 * args.syncnet_T
        self.frame_audio_size = 10 * 5 if not args.wav2vec2 else 2 * 5 # always use syncnet_T=5 for each frame
        self.args = args
        self.data_root = data_root
        self.audio_root = audio_root
        self.image_transform = ImageTransform(args.data_aug_image, args.image_size)
        self.mask_transform = MaskTransform(args.data_aug_mask, args.mask_ratio)
        
    import os

    def get_offset_dict(self, data_root, dataset_name):
        offset_dict = {}
        try:
            with open('filelists_{}/offsets.txt'.format(dataset_name)) as f:
                for line in f:
                    line = line.strip().split()
                    # video path 是从第二个元素到倒数第五个元素
                    video_path = ' '.join(line[1:-5])
                    # 去掉 .jpg 扩展名
                    min_frame = line[-5].replace('.jpg', '')
                    max_frame = line[-4].replace('.jpg', '')
                    offset = float(line[-3])  # 倒数第三个值是 offset
                    # 将 value 设置为包含 offset, min_frame, max_frame 的字典
                    offset_dict[os.path.join(data_root, video_path)] = {
                        'offset': offset,
                        'min_frame': min_frame,
                        'max_frame': max_frame
                    }
        except FileNotFoundError:
            # 如果文件不存在，返回空的字典
            print(f"Offsets file for {dataset_name} not found. Returning empty dictionary.")
            return offset_dict

        return offset_dict

    def get_image_list(self, data_root, dataset_name, split):
        filelist = []

        with open('filelists_{}/{}.txt'.format(dataset_name, split)) as f:
            for line in f:
                line = line.strip()
                filelist.append(join(data_root, line))

        return filelist
        
    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])

    def get_window(self, start_frame):
        start_id = self.get_frame_id(start_frame)
        
        # 通过 dirname(start_frame) 获取 video path
        video_path = dirname(start_frame)
        
        # 一次性获取 all_offsets 的值
        offsets = self.all_offsets.get(video_path, {})
        
        # 获取 offset, min_frame, max_frame 的值，并设置默认值
        offset = offsets.get('offset', 0)  # 默认 offset 为 0
        min_frame = int(offsets.get('min_frame', 0))  # 默认 min_frame 为 0
        max_frame = int(offsets.get('max_frame', 2**31 - 1)) # 默认 max_frame 为最大的int
        
        # print(f"start_id: {start_id}, offset: {offset}, min_frame: {min_frame}, max_frame: {max_frame}")
        
        # 计算实际的 start_id
        start_id = int(start_id + offset)
        
        # 检查 start_id 是否超出范围
        if start_id < min_frame or start_id + self.args.syncnet_T > max_frame:
            # print(f"Frame {start_frame} is out of range")
            return None
            
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
            try:
                img = Image.open(fname)
                img = self.image_transform(img)
            except Exception as e:
                return None
            
            if img is None:
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
        start_idx = int(50. * (start_frame_num / float(args.fps)))
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
                print(f"Video {vidname} has less frames than required")
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
                print(f"Window is None for {vidname}")
                continue

            wrong_window = self.read_window(wrong_window_fnames)
            if wrong_window is None:
                print(f"Wrong Window is None for {vidname}")
                continue
            
            if self.audio_root:
                # switch fro data_root to audio_root
                vidname = vidname.replace(self.data_root, self.audio_root)
                
            if not self.args.wav2vec2:
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
            
            window = self.mask_transform(window)            
                        
            wrong_window = self.prepare_window(wrong_window)
            x = np.concatenate([window, wrong_window], axis=0)
            
            x = torch.FloatTensor(x)
            
            if self.args.syncnet:
                audio_cropped = audio_cropped.unsqueeze(0).float()
            
            indiv_audios = indiv_audios.unsqueeze(1).float()
            y = torch.FloatTensor(y)
            
            # make x and y [-1, 1]
            x = x * 2 - 1
            y = y * 2 - 1
            
            # don't return audio_cropped if syncnet is not enabled
            if not self.args.syncnet:
                return x, indiv_audios, y
            
            return x, indiv_audios, audio_cropped, y
        
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--syncnet', action='store_true')
    parser.add_argument('--data_aug_image', action='store_true')
    parser.add_argument('--data_aug_mask', action='store_true')
    parser.add_argument('--mask_ratio', type=float, default=0.6)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--fps', type=int, default=25)
    parser.add_argument('--syncnet_T', type=int, default=5)
    parser.add_argument('--wav2vec2', action='store_true')
    args = parser.parse_args()
    
    config = yaml.load(open('configs/dataset_config.yaml', 'r'), Loader=yaml.FullLoader)
    dataset = HybridDataset(config, 'train', args, dataset_size=512000)
    x, indiv_audios, y = dataset[0]
    print(f"x: {x.shape}, indiv_audios: {indiv_audios.shape}, y: {y.shape}")
    
    os.makedirs("temp", exist_ok=True)
    torchvision.utils.save_image(x[:3].permute(1, 0, 2, 3), "temp/masked.png", nrow=args.syncnet_T, normalize=True, value_range=(0, 1))
    torchvision.utils.save_image(x[3:].permute(1, 0, 2, 3), "temp/refrence.png", nrow=args.syncnet_T, normalize=True, value_range=(0, 1))
    torchvision.utils.save_image(y.permute(1, 0, 2, 3), "temp/gt.png", nrow=args.syncnet_T, normalize=True, value_range=(0, 1))
    