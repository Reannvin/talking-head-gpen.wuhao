import os
import argparse
import subprocess

def convert_to_25fps(source_dir, target_dir):
    # 检查目标目录是否存在，不存在则创建
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    # 遍历源目录中的所有文件
    for root, _, filenames in os.walk(source_dir):
        for filename in filenames:
            if filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                source_path = os.path.join(root, filename)
                target_path = os.path.join(target_dir, filename)
                
                # 构建FFmpeg命令
                command = [
                    'ffmpeg',
                    '-i', source_path,
                    '-r', '25',
                    target_path
                ]
                
                # 执行FFmpeg命令
                try:
                    subprocess.run(command, check=True)
                    print(f"Converted {source_path} to {target_path} at 25 fps")
                except subprocess.CalledProcessError as e:
                    print(f"Failed to convert {source_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Convert all videos in a directory to 25 fps.")
    parser.add_argument('--input', type=str, required=True, help="The source directory containing the videos.")
    parser.add_argument('--output', type=str, required=True, help="The target directory where the converted videos will be saved.")
    
    args = parser.parse_args()
    
    convert_to_25fps(args.input, args.output)

if __name__ == "__main__":
    main()
