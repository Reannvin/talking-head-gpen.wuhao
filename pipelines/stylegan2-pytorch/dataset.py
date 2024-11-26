from io import BytesIO

import lmdb
from PIL import Image
from torch.utils.data import Dataset, DataLoader

import random
import os
import cv2
import torch
import numpy as np

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
    
class ImageVideoCombineDataset(Dataset):
    def __init__(self, data_root, transform=None, resolution=256,base_ratio=0.8):
        self.data_root = data_root
        self.transform = transform
        self.resolution = resolution
        self.base_ratio = base_ratio

        dirs = os.listdir(data_root)
        self.image_list=[]
        self.video_list=[]
        for dir in dirs:
            files=os.listdir(os.path.join(data_root, dir))
            for file in files:
                if os.path.isdir(os.path.join(data_root, dir, file)):
                    self.video_list.append(os.path.join(data_root, dir, file))
                elif file.endswith('.jpg') or file.endswith('.png'):
                    self.image_list.append(os.path.join(data_root, dir, file))
        
    def read_image(self, path):
        if  not os.path.exists(path):
            print(path)
            return None
        img = cv2.imread(path)
        img = cv2.resize(img, (self.resolution, self.resolution))
        #水平翻转
        if random.random()>0.7:
            img=cv2.flip(img, 1)
        img=np.transpose(img, (2, 0, 1))/255.0
        

       # img_tensor=torch.FloatTensor(img)
        return img
                
    def __len__(self):
        return len(self.image_list)+len(self.video_list)
    
    def get_mask(self, img_path):
        lmk_path=img_path.replace('.jpg','_lmd.npy')
        lmk=np.load(lmk_path)
        mouth_points=range(48,60)
        teeth_points=range(60,68)
        img=cv2.imread(img_path)
        mask=np.zeros_like(img[:,:,0])
        mask=cv2.fillConvexPoly(mask, np.int32(lmk[teeth_points]), (255))
        mask=cv2.resize(mask, (self.resolution, self.resolution))
        mask=np.repeat(mask[:,:,np.newaxis], 3, axis=2)
        mask=np.transpose(mask,(2,0,1))
        return mask



    def __getitem__(self, idx):
        while(1):
            #生成一个0-1之间的随机数
            s=random.random()
            if s>self.base_ratio:
                idx=random.randint(0,len(self.video_list)-1)
                frame_path=self.video_list[idx]
                frame_list=os.listdir(frame_path)
                frame_list=[frame for frame in frame_list if frame.endswith('.jpg') or frame.endswith('.png')]
                img_idx=random.randint(0,len(frame_list)-1)
                img_path=os.path.join(frame_path, frame_list[img_idx])
                img=self.read_image(img_path)
                mask=self.get_mask(img_path)

            else:
                idx=random.randint(0,len(self.image_list)-1)
                img_path=self.image_list[idx]
                img=self.read_image(img_path)
                mask=self.get_mask(img_path)
            if img is None:
                continue
            audio=0
            mask_img=img.copy()
            # print(mask.shape)
            # print(mask_img.shape)
            mask_img[mask==255]=0
            img=torch.FloatTensor(img)
            mask=torch.FloatTensor(mask)
            #mask_img=torch.cat([mask_img,torch.zeros_like(mask_img)],dim=0).unsqueeze(0).permute(1,0,2,3)
           # img=img.unsqueeze(0).permute(1,0,2,3)
            # print("mask_img",mask_img.shape)
            # print("img",img.shape)
            
            return mask,img


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

class CombinedDataloader(DataLoader):
    def __init__(self, base_loader, fine_tune_loader=None, base_ratio=0.5):
        self.base_loader = base_loader
        self.fine_tune_loader = fine_tune_loader
        self.base_ratio = base_ratio

    def __iter__(self):
        base_iter = iter(self.base_loader)
        fine_tune_iter = iter(self.fine_tune_loader) if self.fine_tune_loader is not None else None

        while True:
            if fine_tune_iter is None or random.random() < self.base_ratio:
                try:
                    yield next(base_iter)
                except StopIteration:
                    base_iter = iter(self.base_loader)
                    yield next(base_iter)
            else:
                try:
                    yield next(fine_tune_iter)
                except StopIteration:
                    fine_tune_iter = iter(self.fine_tune_loader)
                    yield next(fine_tune_iter)