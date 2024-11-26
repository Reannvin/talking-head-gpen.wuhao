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
        input_path = os.path.join(video_path, "audio.wav")
        output_dir = os.path.join(args.output_root, person_id, video_id)
        
        # 复制 audio.wav 文件
        os.makedirs(output_dir, exist_ok=True)
        shutil.copy(input_path, output_dir)
    
if __name__ == "__main__":
    main()