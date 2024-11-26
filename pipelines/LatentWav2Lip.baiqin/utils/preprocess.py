import os
import subprocess

import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
#from landmark_extract.landmark import extract_landmarks

def change_fps(path, out_path, fps=25):
   # print(f'[INFO] ===== change fps from {path} to {out_path} =====')
    cmd = ['ffmpeg', '-i', path, '-r', str(fps), '-q:v', '1', out_path]
    subprocess.call(cmd,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
   # print(f'[INFO] ===== fps changed to {fps} =====')

def change_fps_dir(video_dir,output_dir,fps=25):
    video_paths=glob.glob(os.path.join(video_dir,"*.mp4"))
    os.makedirs(output_dir,exist_ok=True)
    for video_path in tqdm(video_paths):
        video_name=os.path.basename(video_path)
        out_path=os.path.join(output_dir,video_name)
        change_fps(video_path,out_path,fps)

def extract_images(path, out_path, idx,fps=25):

    print(f'[INFO] ===== extract images from {path} to {out_path} =====')
    cmd = f'ffmpeg -i {path} -vf fps={fps} -qmin 1 -q:v 1 -start_number 0 {os.path.join(out_path, f"{idx}_%d.jpg")}'
    os.system(cmd)
    print(f'[INFO] ===== extracted images =====')

def video_process(video_dir, image_dir,landmark_dir):
    video_paths = glob.glob(os.path.join(video_dir, "*.mp4"))
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(landmark_dir, exist_ok=True)
    for idx,video_path in video_paths:
        extract_images(video_path, image_dir,idx)

# def landmark_process(image_dir,landmark_dir):
#     image_paths = glob.glob(os.path.join(image_dir, "*.jpg"))
#     for image_path in image_paths:
#         extract_landmarks(image_path,landmark_dir)
def extract_audio_features(path):

    print(f'[INFO] ===== extract audio labels for {path} =====')
    cmd = f'python deepspeech_features/extract_ds_features.py --input {path}'
    os.system(cmd)
    print(f'[INFO] ===== extracted audio labels =====')


def extract_audio(path, out_path, sample_rate=16000):
    print(f'[INFO] ===== extract audio from {path} to {out_path} =====')
    cmd = f'ffmpeg -i {path} -f wav -ar {sample_rate} {out_path}'
    os.system(cmd)
    print(f'[INFO] ===== extracted audio =====')

def audio_process(video_dir, audio_dir,sample_rate=16000):
    video_paths = glob.glob(os.path.join(video_dir, "*.mp4"))
    for video_path in video_paths:
        audio_path = os.path.join(audio_dir, os.path.basename(video_path).replace(".mp4", ".wav"))
        extract_audio(video_path, audio_path, sample_rate)
        extract_audio_features(audio_path)
def split_video(path,output_dir,time_intervals=5,fps=25):
    print(f'[INFO] ===== split video from {path}  =====')
    probe_command = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of',
                     'default=noprint_wrappers=1:nokey=1', path]
    duration = float(subprocess.check_output(probe_command))
    os.makedirs(output_dir, exist_ok=True)
    command = ['ffmpeg', '-i', path]
    start_time = 0
    index=0
    video_name=os.path.basename(path).split(".")[0]
    while True:
        output_file = os.path.join(output_dir, '{:}_{:04d}.mp4'.format(video_name,index))
        command.extend( ['-r', str(fps),'-ss', str(start_time), '-t', str(time_intervals), output_file])
        start_time += time_intervals
        index+=1
        if start_time >= duration:
            break
    subprocess.call(command,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    #print(f'[INFO] ===== video split =====')



def split_video_dir(video_dir,split_video_dir,time_intervals=5,fps=25):
    os.makedirs(split_video_dir,exist_ok=True)
    video_paths=glob.glob(os.path.join(video_dir,"*.mp4"))
    for video_path in tqdm(video_paths):
        split_video(video_path,split_video_dir,time_intervals,fps)

def reshuffle(path):
    dirs=os.listdir(path)
    for dir in tqdm(dirs):
        idx=dir.split("_")[-1]
        new_dir=dir.split("_")[0]+"_"+dir.split("_")[1]
        os.makedirs(os.path.join(path,new_dir),exist_ok=True)
        cmd=["mv",os.path.join(path,dir),os.path.join(path,new_dir,idx)]
        subprocess.call(cmd,stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def del_short_dir(path,threshold=25):
    dirs = os.listdir(path)
    for dir in dirs:
        pic_dirs=os.listdir(os.path.join(path,dir))
        for pic_dir in pic_dirs:
            pic_file=glob.glob(os.path.join(path,dir,pic_dir,"*.jpg"))
            pic_num=len(pic_file)
            if pic_num<threshold:
                cmd = ["rm", "-rf", os.path.join(path, dir, pic_dir)]
                subprocess.call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
if __name__=="__main__":
    #change_fps_dir("/psyai/wangbaiqin/dataset/3300ID/bilibili","/psyai/wangbaiqin/dataset/3300ID/bilibili_fps25")
    #change_fps_dir("/psyai/wangbaiqin/dataset/3300ID/xiaohongshu","/psyai/wangbaiqin/dataset/3300ID/xiaohongshu_fps25")
    #split_video_dir("/psyai/wangbaiqin/dataset/3300ID/xiaohongshu_fps25","/psyai/wangbaiqin/dataset/3300ID/xiaohongshu_split")
    #split_video_dir("/psyai/wangbaiqin/dataset/3300ID/bilibili_fps25","/psyai/wangbaiqin/dataset/3300ID/bilibili_split")
    #reshuffle("/psyai/wangbaiqin/dataset/3300ID/xiaohongshu_split")
   # del_short_dir("/psyai/wangbaiqin/dataset/3300ID/xiaohongshu_preprocessed",threshold=25)
    