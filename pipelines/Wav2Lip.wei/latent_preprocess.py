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
    def __init__(self, root_dir, rel_path, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.rel_path = rel_path
        assert len(self.rel_path.strip('/').split('/')) == 2, self.rel_path
        self.person_id, self.video_id = self.rel_path.strip('/').split('/')

        frames_paths = [f for f in os.listdir(os.path.join(self.root_dir, self.rel_path)) if f.endswith('.jpg')]  # 确保只处理图像文件
        self.sorted_frames_paths = sorted(frames_paths, key=lambda x: int(x.split('.')[0]))
        self.frame_ids = [int(frame.split('.')[0]) for frame in self.sorted_frames_paths]



    def __len__(self):
        return len(self.frame_ids)

    def __getitem__(self, idx):
        frame_name = self.sorted_frames_paths[idx]
        frame_id = self.frame_ids[idx]
         
        frame = Image.open(os.path.join(self.root_dir, self.rel_path, frame_name))
        if self.transform:
            frame = self.transform(frame)
        return frame_id, frame, self.person_id, self.video_id

def process_dataset(rank, world_size, args):
    device = torch.device('cuda', rank)
    torch.cuda.set_device(device)

    vae = AutoencoderKL.from_pretrained(args.model_name, local_files_only=True)
    vae.to(device)

    data_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        RescaleTransform(),  # 使用自定义的转换类替代transforms.Lambda
    ])


    # 省略数据处理和模型训练的其他部分...
    os.makedirs(args.latent_tensors_dir, exist_ok=True)
    index_chunck = 4
    with open('./train_bili.txt') as f:
        rel_paths = f.read().strip().splitlines()[500*index_chunck:(index_chunck+1)*500]
        # rel_paths = f.read().strip().splitlines()[500*index_chunck:]
    
    def process_in_batches(dataloader, model, device):
        full_image_latent_list = []
        upper_half_latent_list = []
        lower_half_latent_list = []
        frames_id_all = []
        for frame_ids, batch, person_id, video_id in tqdm(dataloader, desc='in video'):
            # print(frame_ids)
            batch = batch.cuda()
            upper_half = batch[:, :, :batch.size(2) // 2]
            lower_half = batch[:, :, batch.size(2) // 2:]
            assert upper_half.size(2) == lower_half.size(2)
            # print(type(frame_ids), frame_ids.shape)
            frames_id_all.append(frame_ids)
            
            with torch.no_grad():
                full_image_latent = model.encode(batch)
                upper_half_latent = model.encode(upper_half)
                lower_half_latent = model.encode(lower_half)
                
            full_image_latent_list.append(full_image_latent.latent_dist.sample().cpu())  # 把处理后的结果移到CPU
            upper_half_latent_list.append(upper_half_latent.latent_dist.sample().cpu())
            lower_half_latent_list.append(lower_half_latent.latent_dist.sample().cpu())
        
        assert len(upper_half_latent_list) == len(lower_half_latent_list)
        return torch.cat(full_image_latent_list, dim=0), torch.cat(upper_half_latent_list, dim=0), torch.cat(lower_half_latent_list, dim=0), person_id, video_id, torch.cat(frames_id_all, dim=0)


    for rel_path in tqdm(rel_paths):
        dataset = VideoFramesDataset(root_dir=args.dataset_dir, rel_path=rel_path, transform=data_transform)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False, sampler=None, num_workers=4)

        full_image_latent, upper_half_latent, lower_half_latent, person_id, video_id, frame_ids = process_in_batches(dataloader, vae, device)
        print(full_image_latent.shape, upper_half_latent.shape, lower_half_latent.shape,  person_id[0], video_id[0])
        save_dir = os.path.join(args.latent_tensors_dir, person_id[0], video_id[0])
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "latent.pt")
        torch.save(dict(frame_ids=frame_ids, full_image=full_image_latent, upper_half=upper_half_latent, lower_half=lower_half_latent), save_path)

        audio_path = os.path.join(args.dataset_dir, person_id[0], video_id[0], "audio.wav")
        if os.path.exists(audio_path):
            shutil.copy(audio_path, save_dir)

        
def main():
    parser = argparse.ArgumentParser(description='Process videos into latent tensors using a VAE.')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Directory containing the dataset.')
    parser.add_argument('--model_name', type=str, default='/home/weijinghuan/head_talk/talking-head/notebooks/sd-vae-ft-mse', help='Pretrained model name.')
    parser.add_argument('--latent_tensors_dir', type=str, required=True, help='Directory to save latent tensors.')
    parser.add_argument('--image_size', type=int, default=768, help='Image size for both dimensions (width and height).')
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    process_dataset(0, world_size, args)


if __name__ == "__main__":
    main()
