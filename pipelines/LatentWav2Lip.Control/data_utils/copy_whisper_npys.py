import argparse
import os
import shutil
from tqdm import tqdm

def get_videos_list(root_dir):
    videos_list = []
    for person_id in sorted(os.listdir(root_dir)):
        person_path = os.path.join(root_dir, person_id)
        for video_id in sorted(os.listdir(person_path)):
            video_path = os.path.join(person_path, video_id)
            videos_list.append((video_path, person_id, video_id))
    return videos_list

def is_valid_npy_file(file_name):
    # Check if the file name matches the pattern of '1957.npy'
    return file_name.endswith('.npy') and file_name.count('.') == 1

def main():
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="拆分和保存特定格式的NPY文件。")
    parser.add_argument('--input_root', type=str, required=True, help="输入的根目录路径")
    parser.add_argument('--output_root', type=str, required=True, help="保存新NPY文件的目录路径")

    # 解析命令行参数
    args = parser.parse_args()

    # 获取视频列表
    video_list = get_videos_list(args.input_root)
    
    for video_path, person_id, video_id in tqdm(video_list):
        output_dir = os.path.join(args.output_root, person_id, video_id)
        os.makedirs(output_dir, exist_ok=True)

        # 复制符合条件的 .npy 文件
        for file_name in os.listdir(video_path):
            if is_valid_npy_file(file_name):
                input_path = os.path.join(video_path, file_name)
                shutil.copy(input_path, output_dir)
    
if __name__ == "__main__":
    main()
