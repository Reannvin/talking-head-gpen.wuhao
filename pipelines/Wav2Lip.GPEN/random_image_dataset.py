import os
import random
import argparse
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms

class RandomImageDataset(Dataset):
    def __init__(self, image_folder, image_size=256, transform=None):
        """
        Args:
            image_folder (str): 图片存放的文件夹路径。
            image_size (int, optional): 图片的目标边长（正方形），默认是 256。
            transform (callable, optional): 传入的图片预处理操作。
        """
        self.image_folder = image_folder
        self.image_size = image_size
        self.image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]

        # 使用传入的 transform 或默认的 transform
        self.transform = transform or transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),  # 调整为正方形
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 随机选择一张图片
        random_image_file = random.choice(self.image_files)
        image_path = os.path.join(self.image_folder, random_image_file)

        # 加载图片并应用预处理
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        return image

def parse_args():
    parser = argparse.ArgumentParser(description="Random Image Dataset Loader")
    parser.add_argument('--image_folder', type=str, required=True, 
                        help='路径到存放 PNG 图片的文件夹')
    parser.add_argument('--image_size', type=int, default=256, 
                        help='图片的目标尺寸 (正方形)，默认 256')
    parser.add_argument('--batch_size', type=int, default=8, 
                        help='DataLoader 的 batch 大小，默认 8')
    parser.add_argument('--shuffle', action='store_true', 
                        help='是否打乱数据，默认为 False')
    return parser.parse_args()

def main():
    # 解析命令行参数
    args = parse_args()

    # 实例化数据集
    dataset = RandomImageDataset(image_folder=args.image_folder, 
                                 image_size=args.image_size)

    # 创建 DataLoader
    dataloader = DataLoader(dataset, batch_size=args.batch_size, 
                            shuffle=args.shuffle)

    # 测试 DataLoader, 只打印第一个 batch 的大小
    batch = next(iter(dataloader))
    
    # 保存第一个 batch 的图片，要把 tensor 从 [-1, 1] 转换到 [0, 1]
    batch = batch * 0.5 + 0.5
    torchvision.utils.save_image(batch, 'random_image_dataset.png', nrow=4)

# 只有在直接运行此脚本时才执行 main 函数
if __name__ == '__main__':
    main()
