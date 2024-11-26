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

def main():
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="删除视频文件夹中的 *.npy 文件。")
    parser.add_argument('--input_root', type=str, required=True, help="输入的根目录路径")

    # 解析命令行参数
    args = parser.parse_args()

    # 获取视频列表
    video_list = get_videos_list(args.input_root)
    
    for video_path, person_id, video_id in tqdm(video_list):
        # 删除 *.npy 文件
        files = [f for f in os.listdir(video_path) if f.endswith('.npy')]
        for file in files:
            os.remove(os.path.join(video_path, file))
        
        whisper_file  = [f for f in os.listdir(video_path) if f.startswith('whisper')]
        for file in whisper_file:
            os.remove(os.path.join(video_path, file))
            
    
if __name__ == "__main__":
    main()
