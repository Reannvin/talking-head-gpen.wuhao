from os.path import dirname, join, basename, isfile
from glob import glob
import torch, torchvision
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os, random, argparse
import cv2
from PIL import Image

def get_image_list(data_root, dataset_name, split):
	filelist = []

	with open('filelists_{}/{}.txt'.format(dataset_name, split)) as f:
		for line in f:
			line = line.strip()
			if ' ' in line: line = line.split()[0]
			filelist.append(os.path.join(data_root, line))
	return filelist
from io import BytesIO

import lmdb
from PIL import Image


class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=256):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img)

        return img

class AudioVisualDataset(Dataset):
    def __init__(self, data_root, dataset_name, split, audio_root=None, syncnet_T = 1, dataset_size=512000):
        self.all_videos = get_image_list(data_root, dataset_name, split)
        self.dataset_size = dataset_size
        if syncnet_T != 1:
            self.syncnet = True 
        else:
            self.syncnet = False
        self.syncnet_T = syncnet_T
        self.syncnet_audio_size = 10 * self.syncnet_T
        self.frame_audio_size = 10 * 5
        self.mask_ratio = 0.6
        self.data_root = data_root
        self.audio_root = audio_root
        self.wav2vec2 = None
        
    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])

    def get_window(self, start_frame):
        start_id = self.get_frame_id(start_frame)
        vidname = dirname(start_frame)

        window_fnames = []
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
            img = cv2.imread(fname)
            if img is None:
                return None
            try:
                img = cv2.resize(img, (256, 256))
            except Exception as e:
                return None

            window.append(img)

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
        offset = max(1, self.syncnet_T // 2)
        start_frame_num = frame_id + 1
        if start_frame_num - offset < 0: return None
        for i in range(start_frame_num, start_frame_num + self.syncnet_T):
            m = self.get_whisper_embedding(vidname, i - offset, syncnet_T=5) # always use syncnet_T=5 for each frame
            if m is None or m.shape[0] != self.frame_audio_size:
                return None
            audios.append(m)

        audios = torch.stack(audios)
        return audios
    
    def crop_audio_window(self, audio_embeddings, start_frame, syncnet_T, fps=25):
        if type(start_frame) == int:
            start_frame_num = start_frame
        else:
            start_frame_num = self.get_frame_id(start_frame) # 0-indexing ---> 1-indexing
        start_idx = int(50. * (start_frame_num / float(fps)))
        end_idx = start_idx + 2 * syncnet_T
        return audio_embeddings[start_idx : end_idx]
    
    def get_segmented_audios(self, audio_embeddings, start_frame):
        audios = []
        offset = max(1, self.syncnet_T // 2)
        start_frame_num = self.get_frame_id(start_frame) + 1
        if start_frame_num - offset < 0: return None
        for i in range(start_frame_num, start_frame_num + self.syncnet_T):
            m = self.crop_audio_window(audio_embeddings, i - offset, syncnet_T=5) # always use syncnet_T=5 for each frame
            if m.shape[0] != self.frame_audio_size:
                return None
            audios.append(m)

        audios = torch.stack(audios)
        return audios

    def prepare_window(self, window):
        x = np.asarray(window) / 255
        x = np.transpose(x, (3, 0, 1, 2))
        x = x[::-1, :, :, :]
        return x 

    def __len__(self):
        return self.dataset_size # len(self.all_videos)

    def __getitem__(self, idx):
        while 1:
            idx = random.randint(0, len(self.all_videos) - 1)      
            vidname = self.all_videos[idx]
            img_names = list(glob(join(vidname, '*.jpg')))
            # print(len(img_names)) # 7952
            if len(img_names) <= 3 * self.syncnet_T:
                # print(f"Video {vidname} has less frames than required")
                continue
            
            img_name = random.choice(img_names)
            wrong_img_name = random.choice(img_names)
            # Ensure wrong_img_name is at least syncnet_T frames away from img_name
            while abs(self.get_frame_id(img_name) - self.get_frame_id(wrong_img_name)) < self.syncnet_T:
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

            if self.audio_root:
                # switch fro data_root to audio_root
                vidname = vidname.replace(self.data_root, self.audio_root)
                
                # load syncnet_T frames of audio embeddings for syncnet loss
            
            if self.wav2vec2 is None:
                # load syncnet_T frames of audio embeddings for syncnet loss
                if self.syncnet:
                    audio_cropped = self.get_whisper_embedding(vidname, self.get_frame_id(img_name), syncnet_T=self.syncnet_T)
                    
                # always load for each frame of 5 frames of audio embeddings for cross attention
                indiv_audios = self.get_whisper_segmented_audios(vidname, self.get_frame_id(img_name))
            else:
                # load audio embedding from file wav2vec2.pt
                audio_path = join(vidname, "wav2vec2.pt")
                audio_embeddings = torch.load(audio_path, map_location='cpu')[0]
                
                # load syncnet_T frames of audio embeddings for syncnet loss
                audio_cropped = self.crop_audio_window(audio_embeddings.clone(), img_name, syncnet_T=self.syncnet_T)
                # always load for each frame of 5 frames of audio embeddings for cross attention
                indiv_audios = self.get_segmented_audios(audio_embeddings.clone(), img_name)

            if self.syncnet and (audio_cropped is None or audio_cropped.shape[0] != self.syncnet_audio_size): continue
            
            if indiv_audios is None: continue
            
            window = self.prepare_window(window)
            y = window.copy()
            
            # mask lower part of the image, according to mask_ratio
            mask_height = int(window.shape[2] * self.mask_ratio)
            window[:, :, window.shape[2] - mask_height:] = 0.

            wrong_window = self.prepare_window(wrong_window)
            
            # noise_window = np.random.randn(3, self.syncnet_T, 256, 256).astype(np.float32)

            x = np.concatenate([window, wrong_window], axis=0)
            
            x = torch.FloatTensor(x)
            
            if self.syncnet:
                audio_cropped = audio_cropped.unsqueeze(0).float()
            
            indiv_audios = indiv_audios.unsqueeze(1).float()
            y = torch.FloatTensor(y)
            
            # don't return audio_cropped if syncnet is not enabled
            if not self.syncnet:
                return x, indiv_audios, y
            return x, indiv_audios, audio_cropped, y

# Only for debug.
if __name__ == "__main__":
    print("123")
    data_root = "/data/wuhao/solo_video/nandi/nandi/"
    dataset_name = "nandi"
    train_dataset = AudioVisualDataset(data_root = data_root, dataset_name = dataset_name, split = 'main')
    training_set_iterator = iter(torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=16, 
                                                        num_workers=16,
                                                        pin_memory=True,
                                                        persistent_workers=True,
                                                        prefetch_factor=2))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test
    #----------------------------------------------------------------------------
    try:
        c_image, c_audio, gt = [item.to(device) for item in next(training_set_iterator)]
        
        print("Image tensor shape:", c_image.shape) # ([16, 6, 1, 256, 256])
        print("Audio tensor shape:", c_audio.shape) # ([16, 1, 1, 30, 256])
        print("Ground truth tensor shape:", gt.shape) # ([16, 3, 1, 256, 256])
        
        images = c_image.squeeze(2).cpu()
        gt = gt.squeeze(2).cpu()
        gt_image = gt[0]
        image = images[0]
        image1 = image[:3]
        image2 = image[3:]
        
        def tensor_to_image(tensor):
            array = (tensor.numpy() * 255).astype(np.uint8)
            array = np.transpose(array, (1, 2, 0))
            array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
            return array

        image1_array = tensor_to_image(image1)
        image1_pil = Image.fromarray(image1_array)
        image1_pil.save("image1.png")

        image2_array = tensor_to_image(image2)
        image2_pil = Image.fromarray(image2_array)
        image2_pil.save("image2.png")
        
        gt_image_array = tensor_to_image(gt_image)
        gt_image_pil = Image.fromarray(gt_image_array)
        gt_image_pil.save("image3.png")
        
    except StopIteration:
        print("The dataset is empty or all data has been iterated.")
    except RuntimeError as e:
        print(f"Runtime error occurred: {e}")
        print("This might be due to CUDA out of memory or other device-related issues.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    
    #----------------------------------------------------------------------------

    print("Finito.")


    