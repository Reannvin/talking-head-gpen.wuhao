import argparse
import os
import shutil
from tqdm import tqdm
from glob import glob

def get_videos_list(root_dir):
    videos_list = []
    for person_id in sorted(os.listdir(root_dir)):
        person_path = os.path.join(root_dir, person_id)
        for video_id in sorted(os.listdir(person_path)):
            video_path = os.path.join(person_path, video_id)
            videos_list.append((video_path, person_id, video_id))
    return videos_list

def main():
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="拆分和保存特定格式的PT文件。")
    parser.add_argument('--input_root', type=str, required=True, help="输入的PT文件路径")
    parser.add_argument('--output_root', type=str, required=True, help="保存新PT文件的目录路径")
    parser.add_argument('--syncnet_T', type=int, default=5, help="SyncNet的T值")

    # 解析命令行参数
    args = parser.parse_args()

    # 调用拆分和保存函数
    video_list = get_videos_list(args.input_root)
    
    for video_path, person_id, video_id in tqdm(video_list):
        # copy all the npy files: 0.npy, 1.npy, ...
        input_paths = list(glob(os.path.join(video_path, f'*.npy.{args.syncnet_T}.npy')))    
        output_dir = os.path.join(args.output_root, person_id, video_id)
        
        for input_path in input_paths:
            output_path = os.path.join(output_dir, os.path.basename(input_path))
            shutil.copy(input_path, output_path)
    
if __name__ == "__main__":
    main()