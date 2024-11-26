import subprocess
import os
import argparse
import sys
from yoloface.face_detector import YoloDetector
from PIL import Image
import cv2
from collections import defaultdict
import re
from tqdm import tqdm
from utils.scenedetect import detect_scenes, split_video_by_scenes
import shutil
from Evaluation.syncnet_python.batch_subproc import *
from utils.histgram import detect_transitions
from os import path
from Evaluation.face_detection import FaceAlignment,LandmarksType
import torch
from tqdm import tqdm
import numpy as np
import cv2
from utils.lm_detection import get_landmark_and_bbox
from mmpose.apis import  init_model
sys.path.append("./yoloface/")

def get_personid_list(data_root):
    person_ids = []
    for person_id in os.listdir(data_root):
        person_ids.append(os.path.splitext(person_id)[0])
    return person_ids
def get_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps
def read_img_from_video(video_path, method):
    video_stream = cv2.VideoCapture(video_path)
    frames = []
    while True:
        still_reading, frame = video_stream.read()
        if not still_reading:
            video_stream.release()
            break
        if args.method == 'yolo':
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    return frames

def Yolo(video_path, gpu_id, frames, output_dir):
    # target size of smaller image axis (choose lower for faster work). e.g. 480, 720, 1080. Choose None for original resolution.
    model = YoloDetector(target_size=None, device=f"cuda:{gpu_id}", min_face=90)
    crop_images = []
    for i, frame in enumerate(frames):
        
        bboxes,points = model.predict(frame)
        x1, y1, x2, y2 = bboxes[0][0]
        x1, x2 = max(0, x1), min(frame.shape[1], x2)
        y1, y2 = max(0, y1), min(frame.shape[0], y2)
        crop_img = frame[y1:y2, x1:x2]
        crop_images.append(crop_img)
        img = Image.fromarray(crop_img)
        img.save(f"{output_dir}/{i}.png")
def split_video(input_file, split_times, output_folder):
    video_split_path = []
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


    name =  os.path.basename(input_file).split('.mp4')[0]

    split_times = sorted([0] + split_times)

    for i in range(len(split_times) - 1):
        start_time = split_times[i]
        end_time = split_times[i + 1]
        part_output = os.path.join(output_folder, f"{name}_{i+1}.mp4")
        video_split_path.append(part_output)
        ffmpeg_command = [
            'ffmpeg','-y', '-i', input_file, '-ss', str(start_time), '-to', str(end_time), '-c:v', 'libx264', '-c:a', 'aac', '-y', part_output
        ]
        # print(f"Processing segment from {start_time}s to {end_time}s")
        subprocess.run(ffmpeg_command)
    final_start_time = split_times[-1]
    final_part_output = os.path.join(output_folder, f"{name}_{len(split_times)}.mp4")
    video_split_path.append(final_part_output)
    final_ffmpeg_command = [
        'ffmpeg', '-y', '-i', input_file, '-ss', str(final_start_time), '-c:v', 'libx264', '-c:a', 'aac', '-y', final_part_output
    ]
    # print(f"Processing final segment from {final_start_time}s to end")
    subprocess.run(final_ffmpeg_command)
    return video_split_path


def extract_audio(video_file, audio_file):
    cmd = f'ffmpeg -loglevel panic -y -i {video_file} -strict -2 {audio_file}'

    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        error_msg = f"分离音视频失败: %s" % str(stderr.decode('utf-8'))
        print(error_msg)
    else:
        print(f"提取音频成功，保存为: {audio_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='data preprocessing')
    parser.add_argument("--data_root", default='/data/fanshen/avspeech_fps25',help="input root path", type=str)
    parser.add_argument("--output_root", default= '/data/fanshen/workspace/preprocessed_avspeech', help="your output root path", type=str)
    parser.add_argument("--temp_root", default='tmp',help="your temp root path", type=str)
    parser.add_argument("--gpu_id", default=0,help="gpu use for face detection", type=int)
    parser.add_argument("--lse_d", default=8,help="index:distance", type=float)
    parser.add_argument("--lse_c", default=6,help="index: confidence", type=float)
    parser.add_argument("--scene_detect", default=0,help="whether need scene_detect, if you dataset is public, we advise you turn it to 1", type=int)
    parser.add_argument("--method", default='mmpose', choices = ['mmpose', 'yolo'],help="face detection method", type=str)
    args = parser.parse_args()
    print("args:", args)
      
    current_dir = os.path.dirname(os.path.abspath(__file__))
    person_ids =  get_personid_list(args.data_root)
    temp_path = os.path.join(current_dir, args.temp_root)
    os.makedirs(temp_path, exist_ok = True)
    temp = {}
    temp = defaultdict(list)
    for person_id in tqdm(person_ids):
        video_file = os.path.join(args.data_root, person_id + '.mp4')
        print("video_file:", video_file)
        output_person_path = os.path.join(args.output_root, person_id)
        os.makedirs(output_person_path, exist_ok=True)

        if args.scene_detect == 0:
            sence_list = []
            transitions = []
        else:
            sence_list = detect_scenes(video_file)
            transitions = detect_transitions(video_file, threshold= 0.5,fps=25)
        # print("sence_list:", len(sence_list), sence_list)
        # print("transitions:", transitions)
        if len(sence_list) == 0 and transitions==[]:
            temp[video_file].append({"person_id": person_id})
        else:
            sence_split_path = os.path.join(temp_path, person_id)
            if len(sence_list) > 0:
                os.makedirs(sence_split_path, exist_ok = True)
                split_path = split_video_by_scenes(video_file, sence_list, sence_split_path)
            else:
                split_path = split_video(video_file, transitions, sence_split_path)
            if split_path:
                for p in split_path:
                    temp[p].append({"person_id": person_id})
    with open('videosence.txt', 'w') as v:
        for video_file,info in tqdm(temp.items()):
            person_id =  info[0].get("person_id")
            v.writelines("f'{video_file} {person_id}'\n")
    person_ids = []
    video_files = []
    with open('videosence.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            video_files.append(line.split(' ')[0])
            person_ids.append(line.split(' ')[-1])
    print("*********sence detect over, continue to evaluation and face crop************")
    for video_file,person_id in tqdm(zip(video_files,person_ids)):
        # print(f"Processing... {video_file}")
        dists, conf = run_eval(video_file)
        person_id =  person_id.strip()  
        # print(person_id)
        if dists < args.lse_d and conf > args.lse_c:   
            # info[0]["LSE-D"] = dists
            # info[0]["LSE-C"] = conf
            frames = read_img_from_video(video_file, args.method)
            save_dir = os.path.join(args.output_root, person_id)
            os.makedirs(save_dir, exist_ok=True)
            if args.method == 'yolo':
                Yolo(video_file, args.gpu_id,frames,save_dir )
            else:
                device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
                config_file = './models/dwpose/rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py'
                checkpoint_file = './models/dwpose/dw-ll_ucoco_384.pth'
                model = init_model(config_file, checkpoint_file, device=device)     
                fa = FaceAlignment(LandmarksType._2D, flip_input=False,device=device)
                coord_placeholder = (0.0,0.0,0.0,0.0)
                get_landmark_and_bbox(frames, fa, model,coord_placeholder,save_dir,upperbondrange =0, btm_scalw=0.1)
            # 提取音频
            audio_file = os.path.join(os.path.join(args.output_root, person_id), "audio.wav")
            print(audio_file)
            extract_audio(video_file, audio_file)
        else:
            with open('flaw_data.txt', 'a') as f:
                f.writelines(f'file:{video_file}, lse_d:{dists}, lse_c:{conf}\n')
    # shutil.rmtree(temp_path)
    
        
        
       
        
        
        
        
        
        
        
        
