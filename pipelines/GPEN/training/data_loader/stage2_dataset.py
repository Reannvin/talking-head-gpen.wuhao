import os
from os.path import join, isfile, basename, dirname
from glob import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision

syncnet_mel_step_size = 16

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

class HybridDatasetStageTwo(Dataset):
    def __init__(self, config, split, syncnet, syncnet_T, mel, dataset_size):
        self.datasets = []
        self.ratios = []
        self.dataset_size = dataset_size
        self.syncnet_T = syncnet_T
        self.mel = mel
        self.load_config(config, split,syncnet,syncnet_T = self.syncnet_T, mel=False)

    def load_config(self, config, split, syncnet, syncnet_T, mel=False):
        dataset_configs = config['datasets'][split]
        for dataset_config in dataset_configs:
            dataset = AudioVisualDataset(
                data_root=dataset_config['path'],
                audio_root=dataset_config.get('audio_path'),
                split=dataset_config['split'],
                dataset_name=dataset_config['name'],
                syncnet=syncnet,
                syncnet_T = syncnet_T, 
                mel = mel,
                dataset_size=self.dataset_size
            )
            self.datasets.append(dataset)
            self.ratios.append(dataset_config['ratio'])

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        dataset_choice = random.choices(self.datasets, weights=self.ratios, k=1)[0]
        # print(dataset_choice)
        return dataset_choice[idx]

class AudioVisualDataset(Dataset):
    def __init__(self, data_root, dataset_name, split, syncnet, syncnet_T, audio_root=None,  mel = False, dataset_size=512000):
        self.all_videos = self.get_image_list(data_root, dataset_name, split)
        self.dataset_size = dataset_size
        self.syncnet = syncnet
        self.syncnet_T = syncnet_T
        self.mel = mel
        self.syncnet_audio_size = 10 * self.syncnet_T
        self.frame_audio_size = 10 * 5 
        self.data_root = data_root
        self.audio_root = audio_root
    
        self.image_transform = ImageTransform(data_aug_image= False, image_size = 256)
        self.mask_transform = MaskTransform(data_aug_mask = False, mask_ratio = 0.6)
    
    def get_image_list(self, data_root, dataset_name, split):
        filelist = []

        with open('filelists_{}/{}.txt'.format(dataset_name, split)) as f:
            for line in f:
                line = line.strip()
                if ' ' in line: line = line.split()[0]
                filelist.append(join(data_root, line))

        return filelist
        
    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])

    def get_window(self, start_frame):
        start_id = self.get_frame_id(start_frame)
        vidname = dirname(start_frame)

        window_fnames = []
        # print(vidname)
        for frame_id in range(start_id, start_id + self.syncnet_T):
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
                # print(e)
                return None
            
            if img is None:
                return None
            
            window.append(img)
        # print(len(window))
        return window
    
    def get_whisper_embedding(self, vidname, frame_id, syncnet_T):
        try:
            whisper_file = f"{frame_id}.npy" 
            audio_path = join(vidname, whisper_file)
            audio_embedding = np.load(audio_path)
            audio_embedding = torch.from_numpy(audio_embedding)
        except:
            print(f"Error loading {audio_path}")
            audio_embedding = None
        return audio_embedding
    
    def get_whisper_segmented_audios(self, vidname, frame_id):
        audios = []
        offset = self.syncnet_T // 2
        start_frame_num = frame_id + 1
        if start_frame_num - offset < 0: return None
        for i in range(start_frame_num, start_frame_num + self.syncnet_T):
            m = self.get_whisper_embedding(vidname, i - offset, syncnet_T=self.syncnet_T) # always use syncnet_T=5 for each frame
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
        start_idx = int(50. * (start_frame_num / float(25)))
        end_idx = start_idx + 2 * syncnet_T
        return audio_embeddings[start_idx : end_idx]
    
    def get_segmented_audios(self, audio_embeddings, start_frame):
        audios = []
        offset = self.syncnet_T // 2
        start_frame_num = self.get_frame_id(start_frame) + 1
        if start_frame_num - offset < 0: return None
        for i in range(start_frame_num, start_frame_num + self.syncnet_T):
            m = self.crop_audio_window(audio_embeddings, i - offset, syncnet_T=self.syncnet_T) # always use syncnet_T=5 for each frame
            if m.shape[0] != self.frame_audio_size:
                return None
            audios.append(m)

        audios = torch.stack(audios)
        return audios

    def crop_audio_window_mel(self, spec, start_frame):
        if type(start_frame) == int:
            start_frame_num = start_frame
        else:
            start_frame_num = self.get_frame_id(start_frame) # 0-indexing ---> 1-indexing
        start_idx = int(80. * (start_frame_num / float(25)))
        
        end_idx = start_idx + syncnet_mel_step_size
        return spec[start_idx : end_idx, :]

    def get_segmented_mels(self, spec, start_frame):
        mels = []
        assert self.syncnet_T == 1
        start_frame_num = self.get_frame_id(start_frame) + 1 # 0-indexing ---> 1-indexing
        if start_frame_num - 2 < 0: return None
        for i in range(start_frame_num, start_frame_num + self.syncnet_T):
            m = self.crop_audio_window_mel(spec, i - 2)
            if m.shape[0] != syncnet_mel_step_size:
                return None
            mels.append(m.T)

        mels = np.asarray(mels)

        return mels

    def prepare_window(self, window):
        # 3 x T x H x W
        x = np.asarray(window) / 255.
        x = np.transpose(x, (3, 0, 1, 2))
        return x

    def __len__(self):
        return self.dataset_size 

    def __getitem__(self, idx):
        while 1:
            idx = random.randint(0, len(self.all_videos) - 1)
            vidname = self.all_videos[idx]
            img_names = list(glob(join(vidname, '*.jpg')))
            if len(img_names) <= 3 * self.syncnet_T:
                continue
            img_name = random.choice(img_names)
            wrong_img_name = random.choice(img_names)
            while abs(self.get_frame_id(img_name) - self.get_frame_id(wrong_img_name)) < self.syncnet_T:
                wrong_img_name = random.choice(img_names)

            window_fnames = self.get_window(img_name)
            wrong_window_fnames = self.get_window(wrong_img_name)
            if window_fnames is None or wrong_window_fnames is None:
                continue
            window = self.read_window(window_fnames)
           
            if window is None:
               
                continue

            wrong_window = self.read_window(wrong_window_fnames)
            if wrong_window is None:
                print(f"Wrong Window is None for {vidname}")
                continue
         
            if self.audio_root:
                # switch fro data_root to audio_root
                vidname = vidname.replace(self.data_root, self.audio_root)
                
       
                if self.syncnet:
                    audio_cropped = self.get_whisper_embedding(vidname, self.get_frame_id(img_name), syncnet_T=self.syncnet_T)
                    indiv_audios = self.get_whisper_segmented_audios(vidname, self.get_frame_id(img_name))
                if not self.mel:
                    indiv_audios = self.get_whisper_segmented_audios(vidname, self.get_frame_id(img_name))
                else:
                    try:
                        wavpath = join(vidname, "audio.wav")
                        wav = audio.load_wav(wavpath, 16000)
                        orig_mel = audio.melspectrogram(wav).T
                    except Exception as e:
                        continue
                    mel = self.crop_audio_window(orig_mel.copy(), img_name)
                    if (mel.shape[0] != syncnet_mel_step_size):
                        continue
                    indiv_audios = self.crop_audio_window(orig_mel.copy(), img_name)

            if self.syncnet and (audio_cropped is None or audio_cropped.shape[0] != self.syncnet_audio_size): 
                continue
    
            if indiv_audios is None: 
                continue

            window = self.prepare_window(window)
            y = window.copy()
            mask_window = self.mask_transform(window)            
              
            wrong_window = self.prepare_window(wrong_window)
            x = np.concatenate([mask_window, wrong_window], axis=0)
            x = x * 2 - 1
            y = y * 2 - 1
            x = torch.FloatTensor(x)
            if self.syncnet:
                audio_cropped = audio_cropped.unsqueeze(0).float()
            
            indiv_audios = indiv_audios.unsqueeze(1).float()
            y = torch.FloatTensor(y)
            
            # don't return audio_cropped if syncnet is not enabled
            if not self.syncnet:
                return x, indiv_audios, y
            return x, indiv_audios, audio_cropped, y
        
