import os
import cv2
import sys
import argparse
import subprocess
import shutil
from tqdm import tqdm
from pypinyin import lazy_pinyin
import re

def change_fps(input_folder, output_folder, target_fps):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Rename files in the input directory
    renamed_files = rename_files_in_folder(input_folder)

    for orig_filename, new_filename in tqdm(renamed_files.items()):
        input_path = os.path.join(input_folder, new_filename)
        fps = get_fps(input_path)

        output_path = os.path.join(output_folder, new_filename)
        
        if fps != int(target_fps):
            if not os.path.isfile(input_path):
                continue
            # ffmpeg_command = [
            #     'ffmpeg', '-i', input_path, '-r', str(target_fps), '-c:v', 'libx264','-y', output_path
            # ]

            cmd = f"ffmpeg -i {input_path} -filter:v \"fps=25\" -r 25 -y {output_path}"
            
            print(f"Processing {input_path} and saving to {output_path}")
            process_1 = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process_1.communicate()
            print(process_1.returncode)
            print(stderr.decode('utf-8'))
        else:
            shutil.copy(input_path, output_path)

def get_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps

def convert_filename(filename):
    # Remove parentheses and convert spaces to underscores
    filename = re.sub(r'[()]', '', filename)
    filename = filename.replace(' ', '_')
    # Convert Chinese characters to Pinyin
    pinyin_filename = ''.join(lazy_pinyin(filename))
    return pinyin_filename

def rename_files_in_folder(folder):
    renamed_files = {}
    for filename in os.listdir(folder):
        new_filename = convert_filename(filename)
        old_path = os.path.join(folder, filename)
        new_path = os.path.join(folder, new_filename)
        os.rename(old_path, new_path)
        renamed_files[filename] = new_filename
    return renamed_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Change FPS of videos in a folder")
    parser.add_argument("--data_root", default="/data/fanshen/workspace/preprocessing/test", type=str, help="Path to the input folder containing videos")
    parser.add_argument("--output_root", default="/data/fanshen/workspace/preprocessing/shensi_fps25", type=str, help="Path to the output folder to save videos with changed FPS")
    parser.add_argument("--fps", type=int, default=25, help="Desired FPS for the output videos")
    args = parser.parse_args()
    change_fps(args.data_root,args.output_root, args.fps)
