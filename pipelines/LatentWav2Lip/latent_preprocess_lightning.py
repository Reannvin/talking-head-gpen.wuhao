import argparse
import os
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from diffusers import AutoencoderKL
import pytorch_lightning as pl
from tqdm import tqdm

class RescaleTransform:
    """将图像像素值从[0, 1]缩放到[-1, 1]的转换"""
    def __call__(self, tensor):
        return (tensor * 2.0) - 1.0

class VideoFramesDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.videos = self._get_videos_list()

    def _get_videos_list(self):
        videos_list = []
        for person_id in sorted(os.listdir(self.root_dir)):
            person_path = os.path.join(self.root_dir, person_id)
            for video_id in sorted(os.listdir(person_path)):
                video_path = os.path.join(person_path, video_id)
                videos_list.append((video_path, person_id, video_id))
        return videos_list

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video_path, person_id, video_id = self.videos[idx]
        frames_paths = [f for f in os.listdir(video_path) if f.endswith('.jpg')]  # 确保只处理图像文件
        
        if not frames_paths:
            print(f"No frames found for video: {video_path}")
            return torch.tensor([]), [], video_path, person_id, video_id
        
        # sort frames_paths by frame id
        sorted_frames_paths = sorted(frames_paths, key=lambda x: int(x.split('.')[0]))
        
        # get frame ids
        frame_ids = [int(frame.split('.')[0]) for frame in sorted_frames_paths]
        
        return frame_ids, sorted_frames_paths, video_path, person_id, video_id
    
class VideoFramesDataModule(pl.LightningDataModule):
    def __init__(self, dataset_dir, batch_size=1, image_size=768):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.image_size = image_size

    def setup(self, stage=None):
        self.dataset = VideoFramesDataset(root_dir=self.dataset_dir)

    def predict_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

class VAEInferenceModule(pl.LightningModule):
    def __init__(self, model_name, latent_tensors_dir, transform=None, vae_batch_size=8):
        super().__init__()
        self.model = AutoencoderKL.from_pretrained(model_name, local_files_only=True)
        self.latent_tensors_dir = latent_tensors_dir
        self.transform = transform
        self.vae_batch_size = vae_batch_size

    def forward(self, batch):
        return self.model.encode(batch)

    def predict_step(self, batch, batch_idx):
        frame_ids, sorted_frames_paths, video_path, person_id, video_id = batch    
        sorted_frames_paths = [frame[0] for frame in sorted_frames_paths] # first ele of tuple ('0.jpeg',)    
        if len(sorted_frames_paths) == 0:
            print(f"No frames found for video: {[video_path[0]]}")
            return None
        
        save_dir = os.path.join(self.latent_tensors_dir, person_id[0], video_id[0])
        os.makedirs(save_dir, exist_ok=True)
        
        # if tensors of frames are already saved, return empty sample
        tensor_already_saved = all([os.path.exists(os.path.join(save_dir, f"{frame.split('.')[0]}.pt")) for frame in sorted_frames_paths])
        if tensor_already_saved:
            print(f"Latent tensors already saved for video: {save_dir}")
            return None

        n_frames = len(sorted_frames_paths)
        for start_idx in range(0, n_frames, self.vae_batch_size):
            end_idx = min(start_idx + self.vae_batch_size, n_frames)
            
            sub_frames_paths = sorted_frames_paths[start_idx:end_idx]
            
            frames = [Image.open(os.path.join(video_path[0], frame)) for frame in sub_frames_paths]
            if self.transform:
                frames = [self.transform(frame) for frame in frames]
            sub_tensor = torch.stack(frames).to(self.device)
                        
            # 取上半张图片
            upper_half = sub_tensor[:, :, :sub_tensor.size(2) // 2]
            zero_half = torch.zeros_like(upper_half)
            upper_half_stitch = torch.cat([upper_half, zero_half], dim=2)
            
            with torch.no_grad():
                full_image_latents = self(sub_tensor).latent_dist.sample().cpu()
                upper_half_latents = self(upper_half_stitch).latent_dist.sample().cpu()
            
            for idx, frame_id in enumerate(frame_ids[start_idx:end_idx]):
                save_filename = os.path.join(save_dir, f"{frame_id.item()}.pt")
                torch.save({"full_image": full_image_latents[idx].clone(), "upper_half": upper_half_latents[idx].clone()}, save_filename)

def main():
    torch.set_float32_matmul_precision('high')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='stabilityai/sd-vae-ft-mse')
    parser.add_argument('--latent_tensors_dir', type=str, required=True)
    parser.add_argument('--image_size', type=int, default=768)
    args = parser.parse_args()

    data_module = VideoFramesDataModule(args.dataset_dir, image_size=args.image_size)
    model = VAEInferenceModule(args.model_name, args.latent_tensors_dir, transform=transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            RescaleTransform()
        ]))
    
    trainer = pl.Trainer(enable_checkpointing=False)
    trainer.predict(model, datamodule=data_module)

if __name__ == "__main__":
    main()
