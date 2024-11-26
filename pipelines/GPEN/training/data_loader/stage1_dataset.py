import os
from os.path import join, isfile, basename, dirname
from glob import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision

from os.path import isfile
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
class AudioVisualDataset(Dataset):
    def __init__(self, data_root, dataset_name, split, dataset_size=512000):
        self.all_images = self.get_image_list(data_root, dataset_name, split)
        self.dataset_size = dataset_size
        self.data_root = data_root
        self.image_transform = ImageTransform(data_aug_image= False, image_size = 256)
    
    def get_image_list(self, data_root, dataset_name, split):
        filelist = []

        with open('filelists_{}/{}.txt'.format(dataset_name, split)) as f:
            for line in f:
                line = line.strip()
                if ' ' in line: line = line.split()[0]
                filelist.append(join(data_root, line + '.png'))
        return filelist
        
    def __len__(self):
        return self.dataset_size 

    def __getitem__(self, idx):
        while 1:
            x = random.choice(self.all_images)
            if isfile(x):
                image = Image.open(x).convert('RGB')
                image = self.image_transform(image)
                image = torchvision.transforms.ToTensor()(image)
                image = image * 2 - 1
                return torch.empty((0, 0)), torch.empty((0, 0)), image
            else:
                print(f"Error loading image file {x}")
                continue

# Adjust HybridDatasetStageOne for possible batched index retrieval
class HybridDatasetStageOne(Dataset):
    def __init__(self, config, split, dataset_size):
        self.datasets = []
        self.ratios = []
        self.dataset_size = dataset_size
        self.load_config(config, split)

    def load_config(self, config, split):
        dataset_configs = config['datasets'][split]
        for dataset_config in dataset_configs:
            dataset = AudioVisualDataset(
                data_root=dataset_config['path'],
                split=dataset_config['split'],
                dataset_name=dataset_config['name'],
                dataset_size=self.dataset_size
            )
            self.datasets.append(dataset)
            self.ratios.append(dataset_config['ratio'])

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        dataset_choice = random.choices(self.datasets, weights=self.ratios, k=1)[0]
        return dataset_choice[idx]
