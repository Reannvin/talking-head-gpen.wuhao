import argparse
import os
import cv2
import torch
from torchvision import transforms
from pytorch_fid import fid_score
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader, Dataset
import tempfile

class VideoFrameDataset(Dataset):
    def __init__(self, video_path, transform=None):
        self.video_path = video_path
        self.transform = transform
        self.frames = self._extract_frames()

    def _extract_frames(self):
        frames = []
        cap = cv2.VideoCapture(self.video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            if self.transform is not None:
                frame = self.transform(frame)
            frames.append(frame)
        cap.release()
        return frames

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        return self.frames[idx]

def calculate_average_fid(video1_path, video2_path, gpu_id):
    transform = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset1 = VideoFrameDataset(video1_path, transform=transform)
    dataset2 = VideoFrameDataset(video2_path, transform=transform)


    with tempfile.TemporaryDirectory() as real_images_folder, tempfile.TemporaryDirectory() as generated_images_folder:
        def save_frames(dataset, folder):
            for i, frame in enumerate(dataset):
                frame = frame.permute(1, 2, 0).numpy()  
                frame = (frame * 255).astype(np.uint8) 
                frame = Image.fromarray(frame)
                frame.save(os.path.join(folder, f'frame_{i:05d}.png'))
        
        save_frames(dataset1, real_images_folder)
        save_frames(dataset2, generated_images_folder)
        
        fid_value = fid_score.calculate_fid_given_paths(
            [real_images_folder, generated_images_folder],
            batch_size=50,
            device=f'cuda:{gpu_id}',
            dims=2048
        )
    
    return fid_value
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate FID between two videos.')
    parser.add_argument('--video1', type=str,  help='Path to the first video.')
    parser.add_argument('--video2', type=str, help='Path to the second video.')
    parser.add_argument('--gpu', type=int, default=5, help='GPU ID to use for computation.')
    
    args = parser.parse_args()

    fid_value = calculate_average_fid(args.video1, args.video2, args.gpu)
    print('Average FID value:', fid_value)
