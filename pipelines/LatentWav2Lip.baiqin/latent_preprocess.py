import argparse
import os
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from diffusers import AutoencoderKL
from tqdm import tqdm
import shutil  # 导入shutil模块以复制文件
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

class RescaleTransform:
    """将图像像素值从[0, 1]缩放到[-1, 1]的转换"""
    def __call__(self, tensor):
        return (tensor * 2.0) - 1.0
    
class VideoFramesDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
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
            return torch.tensor([]), torch.tensor([]), person_id, video_id
        
        # sort frames_paths by frame id
        sorted_frames_paths = sorted(frames_paths, key=lambda x: int(x.split('.')[0]))
        
        # get frame ids
        frame_ids = [int(frame.split('.')[0]) for frame in sorted_frames_paths]
        
        frames = [Image.open(os.path.join(video_path, frame)) for frame in sorted_frames_paths]
        if self.transform:
            frames = [self.transform(frame) for frame in frames]
        frames_tensor = torch.stack(frames)
        return frame_ids, frames_tensor, person_id, video_id

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def process_dataset(rank, world_size, args):
    setup(rank, world_size)
    device = torch.device('cuda', rank)
    torch.cuda.set_device(device)

    vae = AutoencoderKL.from_pretrained(args.model_name, local_files_only=True)
    vae.to(device)
    vae = torch.nn.parallel.DistributedDataParallel(vae, device_ids=[rank])

    data_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        RescaleTransform(),  # 使用自定义的转换类替代transforms.Lambda
    ])

    dataset = VideoFramesDataset(root_dir=args.dataset_dir, transform=data_transform)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, sampler=sampler, num_workers=4)

    # 创建目录以保存潜在张量
    os.makedirs(args.latent_tensors_dir, exist_ok=True)
    
    def process_in_batches(tensor, model, batch_size, device):
        n = tensor.size(1)  # 获取帧数，tensor形状为[1, 帧数, 通道数, 高, 宽]
        full_image_latent_list = []
        upper_half_latent_list = []
        lower_half_latent_list = []
        for start_idx in range(0, n, batch_size):
            end_idx = min(start_idx + batch_size, n)
            # 注意这里的切片，我们保持批次维度，这对于大多数模型来说是必要的
            batch = tensor[:, start_idx:end_idx].to(device)  
            
            # 取每一帧的上半张图片
            upper_half = batch[:, :, :, :batch.size(3) // 2]
            zero_half = torch.zeros_like(upper_half)
            upper_half_stitch = torch.cat([upper_half, zero_half], dim=3)
            
            # 因为模型期望的输入是4D的，我们需要移除帧数维度，使其成为[批次大小, 通道数, 高, 宽]
            batch = batch.squeeze(0)  # 现在batch的形状应该是[帧数, 通道数, 高, 宽]
            upper_half_stitch = upper_half_stitch.squeeze(0)
            
            with torch.no_grad():
                full_image_latent = model.module.encode(batch)
                upper_half_latent = model.module.encode(upper_half_stitch)
                
            full_image_latent_list.append(full_image_latent.latent_dist.sample().cpu())  # 把处理后的结果移到CPU
            upper_half_latent_list.append(upper_half_latent.latent_dist.sample().cpu())
        
        # 验证上半张图片和下半张图片的数量是否相等
        return torch.cat(full_image_latent_list, dim=0), torch.cat(upper_half_latent_list, dim=0)


    # 在主循环中使用分批处理
    for frame_ids, frames_tensor, person_id, video_id in tqdm(dataloader, desc="Processing Videos"):
        if frames_tensor.nelement() == 0:  # 如果tensor是空的
            print(f"Skipping video: {person_id[0]}/{video_id[0]} because no frames were found.")
            continue  # 跳过当前迭代，处理下一个批次
                        
        batch_size = 8
        full_image_latent, upper_half_latent = process_in_batches(frames_tensor, vae, batch_size, device)

        save_dir = os.path.join(args.latent_tensors_dir, person_id[0], video_id[0])
        os.makedirs(save_dir, exist_ok=True)

        # 拆分并保存新文件
        for idx, frame_id in enumerate(frame_ids):
            image_latents = {
                'full_image': full_image_latent[idx].clone(),
                'upper_half': upper_half_latent[idx].clone(),
            }
            
            # 构建新的文件名，并确保输出路径包括在内
            save_filename = os.path.join(save_dir, f"{frame_id.item()}.pt")
            
            # 保存新的文件
            torch.save(image_latents, save_filename)

        # 如果存在，复制audio.wav文件到目标目录
        audio_path = os.path.join(args.dataset_dir, person_id[0], video_id[0], "audio.wav")
        if os.path.exists(audio_path):
            shutil.copy(audio_path, save_dir)

    cleanup()
        
def main():
    parser = argparse.ArgumentParser(description='Process videos into latent tensors using a VAE.')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Directory containing the dataset.')
    parser.add_argument('--model_name', type=str, default='stabilityai/sd-vae-ft-mse', help='Pretrained model name.')
    parser.add_argument('--latent_tensors_dir', type=str, required=True, help='Directory to save latent tensors.')
    parser.add_argument('--image_size', type=int, default=768, help='Image size for both dimensions (width and height).')
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    mp.spawn(process_dataset, args=(world_size, args), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
