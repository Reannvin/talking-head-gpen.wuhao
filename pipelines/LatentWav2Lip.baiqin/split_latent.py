import torch
import argparse
import os
from tqdm import tqdm

def get_videos_list(root_dir):
    videos_list = []
    for person_id in sorted(os.listdir(root_dir)):
        person_path = os.path.join(root_dir, person_id)
        for video_id in sorted(os.listdir(person_path)):
            video_path = os.path.join(person_path, video_id)
            videos_list.append((video_path, person_id, video_id))
    return videos_list

def split_latent(input_path, output_dir):
    """
    加载.pt文件并根据frame_ids将数据拆分，然后保存到指定目录。
    
    参数:
    input_path : str
        输入的PT文件路径。
    output_dir : str
        保存新PT文件的目录路径。
    """
    # 加载原始的.pt文件
    try:
        data = torch.load(input_path)
        
        # 验证数据正确性
        frame_num = len(data['frame_ids'])
        if frame_num != data['full_image'].shape[0] or frame_num != data['upper_half'].shape[0] or frame_num != data['lower_half'].shape[0]:
            print(f"Latent {input_path} corrupted")
            return
    except:
        print(f"Fail to load latent {input_path}")
        return

    # 确保输出目录存在，如果不存在则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 拆分并保存新文件
    for idx, frame_id in enumerate(data['frame_ids']):
        new_data = {
            'full_image': data['full_image'][idx].clone(),
            'upper_half': data['upper_half'][idx].clone(),
            'lower_half': data['lower_half'][idx].clone()
        }
        # 构建新的文件名，并确保输出路径包括在内
        save_filename = os.path.join(output_dir, f"{frame_id.item()}.pt")
        # 保存新的文件
        torch.save(new_data, save_filename)

    # 删除原始文件
    os.remove(input_path)

def main():
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="拆分和保存特定格式的PT文件。")
    parser.add_argument('--input_root', type=str, required=True, help="输入的PT文件路径")
    parser.add_argument('--output_root', type=str, required=True, help="保存新PT文件的目录路径")

    # 解析命令行参数
    args = parser.parse_args()

    # 调用拆分和保存函数
    video_list = get_videos_list(args.input_root)
    
    for video_path, person_id, video_id in tqdm(video_list):
        input_path = os.path.join(video_path, "latent.pt")
        output_dir = os.path.join(args.output_root, person_id, video_id)
        split_latent(input_path, output_dir)
    
if __name__ == "__main__":
    main()
