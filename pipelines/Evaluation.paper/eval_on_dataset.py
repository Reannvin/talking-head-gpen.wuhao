import os
import random
import string
import ffmpeg
import cv2
import argparse
import torch
from shutil import copytree, copy2
from tqdm import tqdm

def generate_random_experiment_name(length=6):
    """生成随机的 experiment 名字，如 experiment-xxxxxx，其中 xxxxxx 是随机字母或数字。"""
    random_str = ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))
    return f"run-{random_str}"

class EvaluationDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, experiment_dir, num_videos=100, fps=25, image_size=256, frame_count=16, audio_sample_rate=16000, dry_run=False, clone_from=None, eval_tof= False):
        self.root_dir = root_dir
        self.dry_run = dry_run
        self.num_videos = num_videos
        self.fps = fps
        self.image_size = image_size
        self.frame_count = frame_count
        self.audio_sample_rate = audio_sample_rate
        self.experiment_dir = experiment_dir
        self.clone_from = clone_from
        self.eval_tof = eval_tof
        # 如果指定了 clone-from，则从之前的实验复制预处理数据
        if self.clone_from:
            self._clone_previous_experiment()

        self.video_paths = self._get_all_videos()
        self.selected_videos = random.sample(self.video_paths, min(len(self.video_paths), self.num_videos))

        # 如果有克隆路径，加载处理后的数据
        if self.clone_from:
            self.processed_data = self._load_experiment_metadata()
        else:
            self.processed_data = self._preprocess_videos()
            self.save_experiment_metadata()

    def _clone_previous_experiment(self):
        """从指定的实验中克隆处理数据到当前实验目录。"""
        
        # 复制处理后的视频和音频文件
        clone_video_dir = os.path.join(self.clone_from, 'processed_videos')
        clone_video_clips_dir = os.path.join(self.clone_from, 'video_clips')
        clone_audio_clips_dir = os.path.join(self.clone_from, 'audio_clips')

        if os.path.exists(clone_video_dir):
            copytree(clone_video_dir, os.path.join(self.experiment_dir, 'processed_videos'), dirs_exist_ok=True)
        if os.path.exists(clone_video_clips_dir):
            copytree(clone_video_clips_dir, os.path.join(self.experiment_dir, 'video_clips'), dirs_exist_ok=True)
        if os.path.exists(clone_audio_clips_dir):
            copytree(clone_audio_clips_dir, os.path.join(self.experiment_dir, 'audio_clips'), dirs_exist_ok=True)

        # 复制 eval.txt 文件
        eval_file = os.path.join(self.clone_from, 'eval.txt')
        if os.path.exists(eval_file):
            copy2(eval_file, os.path.join(self.experiment_dir, 'eval.txt'))
        
        # 创建 generated 目录
        os.makedirs(f"{self.experiment_dir}/generated", exist_ok=True)

    def save_experiment_metadata(self):
        """保存当前实验的处理数据到 eval.txt。"""
        with open(os.path.join(self.experiment_dir, 'eval.txt'), 'w') as f:
            for data in self.processed_data:
                f.write(f"{data['video_name']} {data['start_frame']} {data['end_frame']}\n")

    def _load_experiment_metadata(self):
        """从 eval.txt 加载处理数据，使用 start_time 和 end_time 来读取音频。"""
        processed_data = []
        eval_file = os.path.join(self.experiment_dir, 'eval.txt')

        with open(eval_file, 'r') as f:
            for line in f:
                video_name, start_frame, end_frame = line.strip().split()
                video_clip = os.path.join(self.experiment_dir, 'video_clips', f"{video_name}_{start_frame}_{end_frame}.mp4")

                # 使用 start_frame 和 end_frame 来构建音频路径
                audio_clip = os.path.join(self.experiment_dir, 'audio_clips', f"{video_name}_{start_frame}_{end_frame}.wav")

                processed_data.append({
                    "video_clip": video_clip,
                    "audio_clip": audio_clip,
                    "start_frame": int(start_frame),
                    "end_frame": int(end_frame)
                })

        return processed_data

    def _get_all_videos(self):
        video_paths = []
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith('.mp4'):
                    video_paths.append(os.path.join(root, file))
        print(f"Total videos found: {len(video_paths)}")
        return video_paths

    def _check_and_process_video(self, video_path):
        try:
            probe = ffmpeg.probe(video_path)
            video_stream = next(stream for stream in probe['streams'] if stream['codec_type'] == 'video')
            audio_stream = next(stream for stream in probe['streams'] if stream['codec_type'] == 'audio')

            current_fps = eval(video_stream['r_frame_rate'])
            current_sample_rate = int(audio_stream['sample_rate'])

            if current_fps != self.fps or current_sample_rate != self.audio_sample_rate:
                os.makedirs(os.path.join(self.experiment_dir, 'processed_videos'), exist_ok=True)
                processed_video_path = os.path.join(self.experiment_dir, 'processed_videos', os.path.basename(video_path))
                ffmpeg.input(video_path).output(processed_video_path, r=self.fps, ar=self.audio_sample_rate).run()
                return processed_video_path
            return video_path
        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
            return None
        
    def _preprocess_videos(self):
        processed_data = []
        for idx, video_path in enumerate(self.selected_videos):
            print(f"Processing video {idx + 1}/{self.num_videos}: {video_path}")
            processed_video_path = self._check_and_process_video(video_path)
            if processed_video_path is None:
                raise ValueError(f"Could not process video {video_path}")
            
            video_clip, audio_clip, start_frame, end_frame = self._extract_video_and_audio_clips(processed_video_path, self.frame_count)

            processed_data.append({
                "video_name": os.path.splitext(os.path.basename(video_path))[0],
                "video_clip": video_clip,
                "audio_clip": audio_clip,
                "start_frame": start_frame,
                "end_frame": end_frame
            })
        return processed_data
    
    def _extract_video_and_audio_clips(self, input_video_path, frame_count):
        video_name = os.path.splitext(os.path.basename(input_video_path))[0]
        
        # 打开视频文件
        video_capture = cv2.VideoCapture(input_video_path)

        # 获取视频的总帧数、帧率、宽度和高度
        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_rate = int(video_capture.get(cv2.CAP_PROP_FPS))
        width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 确保 frame_count 小于总帧数，避免超出范围
        if frame_count > total_frames:
            raise ValueError(f"frame_count {frame_count} 超过了视频的总帧数 {total_frames}")

        # 随机选择一个开始的帧索引，确保不会超出帧数范围
        start_frame_idx = random.randint(0, total_frames - frame_count)
        end_frame_idx = start_frame_idx + frame_count

        # 定义输出视频文件
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 编码格式
        os.makedirs(os.path.join(self.experiment_dir, 'video_clips'), exist_ok=True)
        output_video_path = os.path.join(self.experiment_dir, 'video_clips', f"{video_name}_{start_frame_idx}_{end_frame_idx}.mp4")
        output_video = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

        current_frame_idx = 0

        # 读取并写入帧
        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if not ret:
                break

            if start_frame_idx <= current_frame_idx < end_frame_idx:
                output_video.write(frame)  # 写入帧

            current_frame_idx += 1

            if current_frame_idx >= end_frame_idx:
                break

        # 释放视频读取和写入资源
        video_capture.release()
        output_video.release()

        # 计算对应的音频时间段
        start_time = start_frame_idx / frame_rate  # 开始时间（秒）
        duration = frame_count / frame_rate        # 持续时间（秒）

        # 使用 ffmpeg 提取指定时间段的音频
        os.makedirs(os.path.join(self.experiment_dir, 'audio_clips'), exist_ok=True)
        output_audio_path = os.path.join(self.experiment_dir, 'audio_clips', f"{video_name}_{start_frame_idx}_{end_frame_idx}.wav")
        ffmpeg.input(input_video_path, ss=start_time, t=duration).output(output_audio_path).run()
        
        return output_video_path, output_audio_path, start_frame_idx, end_frame_idx

    def __len__(self):
        return len(self.selected_videos)
    
    def __getitem__(self, idx):
        data = self.processed_data[idx]

        if self.dry_run or self.eval_tof:
            other_video_idx = idx
        else:
            other_video_idx = random.choice([i for i in range(len(self.selected_videos)) if i != idx])
            
        other_data = self.processed_data[other_video_idx]     

        return dict(
            video_clip=data['video_clip'],
            video_start_frame=data['start_frame'],
            video_end_frame=data['end_frame'],
            audio_clip=other_data['audio_clip'],
            audio_start_frame=other_data['start_frame'],
            audio_end_frame=other_data['end_frame'],
        )

def get_reference_from_videofile(videofile):
    """根据视频文件路径生成默认的参考名（去掉扩展名）"""
    return os.path.splitext(os.path.basename(videofile))[0]


import subprocess
import re
import numpy as np
import json

import subprocess
def run_command(command, report_file):
    """执行命令并捕获异常，输出重定向到 report.txt"""
    try:
        print(f"Running: {' '.join(command)}")
        with open(report_file, 'a') as f: # 打开 report.txt 文件，附加模式
            subprocess.run(command, check=True, stdout=f, stderr=f) # 重定向 stdout 和 stderr 到文件
        return True
    except subprocess.CalledProcessError as e:
        print(f"Command failed with error: {e}")
        return False

def execute_and_analyze(command, report_file, type="lip_sync"):
    """执行命令并捕获结果，将输出重定向到报告文件，然后分析结果"""
    try:
        print(f"Running: {' '.join(command)}")
        with open(report_file, 'a') as f:
            subprocess.run(command, check=True, stdout=f, stderr=f)
        print("Command executed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Command failed with error: {e}")
        return None

    # 分析报告文件中的结果
    try:
        if type == "lip_sync":
            result = extract_lip_sync_metrics(report_file)
        else:
            raise ValueError(f"Unknown type: {type}")
        return result
    except Exception as e:
        print(f"Failed to extract metrics with error: {e}")
        return None

def extract_tof_metrics(file_path):
    """从报告文件中提取 TOF 指标"""
    psnr = None
    lpips = None
    tof = None
    tlp = None

    psnr_pattern = r"PSNR:\s+([\d\.]+)"
    lpips_pattern = r"LPIPS:\s+([\d\.]+)"
    tof_pattern = r"TOF:\s+([\d\.]+)"
    tlp_pattern = r"TLP:\s+([\d\.]+)"

    # 打开文件读取内容
    with open(file_path, 'r') as file:
        content = file.read()

        # 提取指标数据
        psnr_match = re.search(psnr_pattern, content)
        lpips_match = re.search(lpips_pattern, content)
        tof_match = re.search(tof_pattern, content)
        tlp_match = re.search(tlp_pattern, content)

        if psnr_match:
            psnr = float(psnr_match.group(1))
        
        if lpips_match:
            lpips = float(lpips_match.group(1))

        if tof_match:
            tof = float(tof_match.group(1))

        if tlp_match:
            tlp = float(tlp_match.group(1))

    result = {
        "PSNR": psnr,
        "LPIPS": lpips,
        "TOF": tof,
        "TLP": tlp
    }

    return result
def extract_lip_sync_metrics(file_path):
    """从报告文件中提取单个指标"""
    av_offset = None
    min_dist = None
    confidence = None

    av_offset_pattern = r"AV offset:\s+(-?\d+)"
    min_dist_pattern = r"Min dist:\s+([\d\.]+)"
    confidence_pattern = r"Confidence:\s+([\d\.]+)"

    with open(file_path, 'r') as file:
        content = file.read()

        # 提取单个匹配结果
        av_offset_match = re.search(av_offset_pattern, content)
        min_dist_match = re.search(min_dist_pattern, content)
        confidence_match = re.search(confidence_pattern, content)

        if av_offset_match:
            av_offset = int(av_offset_match.group(1))
        
        if min_dist_match:
            min_dist = float(min_dist_match.group(1))

        if confidence_match:
            confidence = float(confidence_match.group(1))

    result = {
        "Offset": av_offset,
        "LSE-D": min_dist,
        "LSE-C": confidence
    }

    return result

import subprocess
import re

def run_dataset_FID(real_videos_dir, fake_videos_dir, fid_temp_dir, dataset_fid_report_file):
    command = [
        "python", "eval_FID_new.py",
        "--real", real_videos_dir,
        "--fake", fake_videos_dir,
        "--faces_temp_dir", fid_temp_dir,
    ]
    
    try:
        # 执行命令并捕获输出
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        output = result.stdout
        
        # 如果指定了文件路径，则将结果写入文件
        if dataset_fid_report_file:
            with open(dataset_fid_report_file, "w") as file:
                file.write(output)
    except subprocess.CalledProcessError as e:
        error_message = f"Error occurred while running datset fid evaluation: {e.stderr}"
        print(error_message)
        if dataset_fid_report_file:
            with open(dataset_fid_report_file, "w") as file:
                file.write(error_message)
        return None
    
    # 使用正则表达式从输出中提取 FID 或 FVD 值
    match = re.search(r'FID:\s*([0-9.]+)', output)
    if match:
        fid_value = float(match.group(1))
        print(f"Dataset FID value: {fid_value}")
        return fid_value
    else:
        print("FID value not found in the output.")
        return None

def run_visual_evaluation(real_video_path, fake_video_path, visual_eval_report_file, type="FID"):
    """
    运行视觉评估（FID 或 FVD）并提取结果。

    :param real_video_path: 真实视频的路径。
    :param fake_video_path: 生成视频的路径。
    :param type: 评估类型，"FID" 或 "FVD"，默认值为 "FID"。
    :param visual_eval_report_file: 指定一个文件路径，保存所有运行命令的结果。
    :return: 提取到的 FID 或 FVD 值（浮点数）或者 None 如果未找到。
    """
  
    # 构建命令
    if type == "FID":
        eval_script = "eval_FID.py"  
    elif type=="TOF":
        eval_script = "eval_TOF.py"
    else:
        eval_script =  "eval_FVD.py"
    command = [
        "python", eval_script,
        "--real", real_video_path,
        "--fake", fake_video_path
    ]

    try:
        # 执行命令并捕获输出
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        output = result.stdout
        # 如果指定了文件路径，则将结果写入文件
        if visual_eval_report_file:
            with open(visual_eval_report_file, "w") as file:
                file.write(output)

    except subprocess.CalledProcessError as e:
        error_message = f"Error occurred while running {type} evaluation: {e.stderr}"
        print(error_message)
        if visual_eval_report_file:
            with open(visual_eval_report_file, "w") as file:
                file.write(error_message)
        return None

    # 使用正则表达式从输出中提取 FID 或 FVD 值
    if type == "FID":
        match = re.search(r'FID:\s*([0-9.]+)', output)
        if match:
            fid_value = float(match.group(1))
            print(f"Extracted FID value: {fid_value}")
            return fid_value
        else:
            print("FID value not found in the output.")
            return None
    elif type == "TOF":
        tof_dict = {}
        psnr_pattern = r"PSNR, max ([\d\.]+), min ([\d\.]+), avg ([\d\.]+)"
        lpips_pattern = r"LPIPS, max ([\d\.]+), min ([\d\.]+), avg ([\d\.]+)"
        tof_pattern = r"tOF, max ([\d\.]+), min ([\d\.]+), avg ([\d\.]+)"
        tlp_pattern = r"tLP, max ([\d\.]+), min ([\d\.]+), avg ([\d\.]+)"
        psnr_match = re.search(psnr_pattern, output)
        lpips_match = re.search(lpips_pattern, output)
        tof_match = re.search(tof_pattern, output)
        tlp_match = re.search(tlp_pattern, output)
        psnr_results = {}
        lpips_results = {}
        tof_results = {}
        tlp_results = {}

        if psnr_match:
            tof_dict["psnr"] = float(psnr_match.group(3))

        if lpips_match:
            tof_dict["lpips"] = float(lpips_match.group(3))

        if tof_match:
            tof_dict["tof"] = float(tof_match.group(3))

        if tlp_match:
            tof_dict["tlp"] = float(tlp_match.group(3))
        return tof_dict

    elif type == "FVD":
         # 定义正则表达式，用于匹配 JSON 块
        json_regex = r'\{\s*"fvd"\s*:\s*\{.*?\}\s*\}'
        
        # 使用 re.DOTALL 使正则表达式中的点号匹配换行符，以便匹配多行字符串
        match = re.search(json_regex, output, re.DOTALL)
        if match:
            json_string = match.group(0)
            try:
                # 将 JSON 字符串解析为字典
                fvd_dict = json.loads(json_string)
                print(f"Extracted FVD dictionary:\n{fvd_dict}")
                return fvd_dict
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                return None
        else:
            print("FVD value not found in the output.")
            return None
    else:
        raise ValueError(f"Unknown type: {type}")
    
import numpy as np
def compute_metrics_statistics_tof(eval_results):
    """计算所有评估结果的 TOF 指标均值和方差"""
    psnrs = []
    lpips = []
    tofs = []        
    tlp = []
    for result in eval_results.values():
        if psnrs is not None:
           psnrs.append(result.get("psnr"))
        if lpips is not None:
           lpips.append(result.get("lpips"))            
        if tofs is not None:
           tofs.append(result.get("tof"))
        if tlp is not None:
           tlp.append(result.get("tlp"))

    # 计算均值和方差
    statistics = {
        "PSNR": {
            "mean": np.mean(psnrs) if psnrs else None,
            "std": np.std(psnrs, ddof=1) if psnrs else None  
        },
        "LPIPS": {
            "mean": np.mean(lpips) if lpips else None,
            "std": np.std(lpips, ddof=1) if lpips else None
        },
        "TOF": {
            "mean": np.mean(tofs) if tofs else None,
            "std": np.std(tofs, ddof=1) if tofs else None
        },
        "TLP": {
            "mean": np.mean(tlp) if tlp else None,
            "std": np.std(tlp, ddof=1) if tlp else None
        }
    }

    return statistics
def compute_metrics_statistics(eval_results):
    """计算所有评估结果的指标均值和方差"""
    offsets = []
    lse_ds = []
    lse_cs = []
    fid_values = []
    fvd_values = []

    for result in eval_results.values():
        if "lip_sync" in result:
            lip_sync = result["lip_sync"]
            offset = lip_sync.get("Offset")
            lse_d = lip_sync.get("LSE-D")
            lse_c = lip_sync.get("LSE-C")

            # 如果存在值，则将其添加到对应的列表中
            if offset is not None:
                offsets.append(offset)
            if lse_d is not None:
                lse_ds.append(lse_d)
            if lse_c is not None:
                lse_cs.append(lse_c)
        
        if "FID" in result:
            fid = result["FID"]
            if fid is not None:
                fid_values.append(fid)
        
        if "FVD" in result:
            fvd = result["FVD"]
            if fvd is not None:
                fvd_values.append(fvd)

    # 计算均值和方差
    statistics = {
        "Offset": {
            "mean": np.mean(offsets) if offsets else None,
            "std": np.std(offsets, ddof=1) if offsets else None  # ddof=1 for sample std
        },
        "LSE-D": {
            "mean": np.mean(lse_ds) if lse_ds else None,
            "std": np.std(lse_ds, ddof=1) if lse_ds else None
        },
        "LSE-C": {
            "mean": np.mean(lse_cs) if lse_cs else None,
            "std": np.std(lse_cs, ddof=1) if lse_cs else None
        },
        "FID": {
            "mean": np.mean(fid_values) if fid_values else None,
            "std": np.std(fid_values, ddof=1) if fid_values else None
        },
        "FVD": {
            "mean": np.mean(fvd_values) if fvd_values else None,
            "std": np.std(fvd_values, ddof=1) if fvd_values else None
        }
    }

    return statistics

import time
def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Evaluation Dataset for Video and Audio Processing")
    parser.add_argument('--root-dir', type=str, required=True, help="Path to the dataset root directory")
    parser.add_argument('--num-videos', type=int, default=100, help="Number of videos to sample")
    parser.add_argument('--fps', type=int, default=25, help="Frames per second to ensure for each video")
    parser.add_argument('--audio-sample-rate', type=int, default=16000, help="Audio sample rate to ensure for each video")
    parser.add_argument('--eval-dir', type=str, default="./eval-runs", help="Directory to store all experiment folders")
    parser.add_argument('--image-size', type=int, default=256, help="Image size for face detection")
    parser.add_argument('--frame-count', type=int, default=250, help="Number of frames to extract from each video")
    parser.add_argument('--dry-run', action='store_true', help="Only test data loading and processing")
    parser.add_argument('--clone-from', type=str, help="Path to the previous experiment directory to clone from")
    parser.add_argument('--mask-ratio', type=float, default=0.6, help="Mask ratio for the lower part of the face")
    parser.add_argument('--crop-down', type=float, default=0.1, help="Crop the video from the top")
    parser.add_argument('--ckpt', type=str, required=True, help="Path to the GAN model checkpoint")
    parser.add_argument('--device', type=str, default="0", help="Device to run the model on")
    parser.add_argument('--algo', type=str, required=True, choices=['gpen', 'unet', 'wav2lip', 'musetalk'])
    parser.add_argument('--eval_tof',  action='store_true', help="use matched audio to infer")
    parser.add_argument('--exp_name',  type=str, required=True, help="result to save root dir")
    args = parser.parse_args()
    
    # 设置 CUDA_VISIBLE_DEVICES 环境变量
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

    # 创建随机的 experiment 文件夹，并确保它位于 --eval-dir 中
    if args.exp_name:
        experiment_dir = os.path.join(args.eval_dir, args.exp_name)
    else:
        experiment_dir = os.path.join(args.eval_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(f"{experiment_dir}/generated", exist_ok=True)
    os.makedirs(f"{experiment_dir}/reports", exist_ok=True)
    args.experiment_dir = experiment_dir
    eval_results_file = os.path.join(experiment_dir, 'eval_results.txt')
    if args.eval_tof:
        eval_results_file = os.path.join(experiment_dir, 'eval_results_tof.txt')
    eval_results_dict = {}

    print(f"Experiment directory created: {experiment_dir}")

    # 创建数据集实例，并使用 experiment_dir 作为 temp_dir
    dataset = EvaluationDataset(
        root_dir=args.root_dir, 
        num_videos=args.num_videos, 
        fps=args.fps, 
        audio_sample_rate=args.audio_sample_rate, 
        image_size=args.image_size, 
        frame_count=args.frame_count,
        experiment_dir=experiment_dir,
        dry_run=args.dry_run,
        clone_from=args.clone_from,
        eval_tof = args.eval_tof
    )
    
    print(f"Total videos in dataset: {len(dataset)}")    
    
    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        video = sample['video_clip']
        audio = sample['audio_clip']
        
        # 直接调用 inference.py 来进行模型推理
        video_name = os.path.basename(video).split('.')[0]
        audio_name = os.path.basename(audio).split('.')[0]
        inference_out_video = f"{experiment_dir}/generated/{video_name}_{audio_name}.mp4"
        reference = get_reference_from_videofile(inference_out_video)
        os.makedirs(f"{experiment_dir}/reports/{reference}", exist_ok=True)
        eval_results_dict[reference] = {}
        
        if args.dry_run:
            # Just combine the video and audio together
            dry_run_command = [
                "ffmpeg", "-y", "-i", video, "-i", audio, "-c:v", "copy", "-c:a", "aac", inference_out_video
            ]
            
            dry_run_report = f"{experiment_dir}/reports/dry_run_report.txt"
            completed_successful = run_command(dry_run_command, dry_run_report)
            if not completed_successful: continue
        else:
            # 创建 temp 目录，inference 需要这个目录
            inference_temp_dir = f"{experiment_dir}/tmp/infer"
            os.makedirs(inference_temp_dir, exist_ok=True)
            
            if args.algo == 'gpen':
                inference_command = [
                    "python", "gpen_inference.py",
                    "--resize",
                    "--face_parsing",
                    "--mask_ratio", str(args.mask_ratio),
                    "--crop_down", str(args.crop_down),
                    "--face", video,
                    "--audio", audio,
                    "--ckpt", args.ckpt,
                    "--temp_dir", inference_temp_dir,
                    "--outfile", inference_out_video
                ]
            elif args.algo == 'unet':
                inference_command = [
                    "python", "unet_inference.py",
                    "--resize",
                    "--face_parsing",
                    "--mask_ratio", str(args.mask_ratio),
                    "--crop_down", str(args.crop_down),
                    "--face", video,
                    "--audio", audio,
                    "--checkpoint_path", args.ckpt,
                    "--temp_dir", inference_temp_dir,
                    "--outfile", inference_out_video
                ]
            elif args.algo == 'wav2lip':
                inference_command = [
                    "python", "wav2lip_inference.py",
                    "--face", video,
                    "--audio", audio,
                    "--checkpoint_path", args.ckpt,
                    "--temp_dir", inference_temp_dir,
                    "--outfile", inference_out_video
                ]
            elif args.algo =='musetalk':
                inference_command = [
                    "python", "musetalk_inference.py",
                    "--face", video,
                    "--audio", audio,
                    "--checkpoint_path", args.ckpt,
                    "--temp_dir", inference_temp_dir,
                    "--outfile", inference_out_video
                ]
            else:
                raise RuntimeError(f"Unknown algo {args.algo}")
            inference_report = f"{experiment_dir}/reports/{reference}/inference_report.txt"
            completed_successful = run_command(inference_command, inference_report)
            if not completed_successful: continue
        
        # 运行 SyncNetInstance 评估
        if not args.eval_tof:
            pipeline_command = [
                "python", "run_pipeline.py",
                "--videofile", inference_out_video,
                "--reference", reference,
                "--data_dir", f"{experiment_dir}/tmp",
            ]
            lip_sync_pipeline_report = f"{experiment_dir}/reports/{reference}/lip_sync_pipeline_report.txt"
            run_command(pipeline_command, lip_sync_pipeline_report)

            syncnet_command = [
                "python", "run_syncnet.py",
                "--videofile", inference_out_video,
                "--reference", reference,
                "--data_dir", f"{experiment_dir}/tmp",
            ]
            
            lip_sync_syncnet_report = f"{experiment_dir}/reports/{reference}/lip_sync_syncnet_report.txt"
            lip_sync_result = execute_and_analyze(syncnet_command, lip_sync_syncnet_report, type="lip_sync")
            print(f"Lip sync result: {lip_sync_result}")
            eval_results_dict[reference]['lip_sync'] = lip_sync_result
            
            # 运行单视频 FID 评估
            fid_eval_report_file = f"{experiment_dir}/reports/{reference}/fid_eval_report.txt"
            fid_result = run_visual_evaluation(video, inference_out_video, fid_eval_report_file, type="FID")
            if fid_result is not None:
                eval_results_dict[reference]['FID'] = fid_result
                
            # 运行单视频 FVD 评估
            fvd_eval_report_file = f"{experiment_dir}/reports/{reference}/fvd_eval_report.txt"
            fvd_result = run_visual_evaluation(video, inference_out_video, fvd_eval_report_file, type="FVD")
            if fvd_result is not None:
                eval_results_dict[reference]['FVD'] = fvd_result["fvd"]["value"]["16"] # 使用 16 帧的 FVD 值，参考 StyleGAN2-V 论文
                eval_results_dict[reference]['FVD_dict'] = fvd_result
        else:
            # 運行單視頻 tof evaluation
            tof_eval_report_path = f"{experiment_dir}/reports/{reference}/tof_eval_report.txt"
            tof_result = run_visual_evaluation(video, inference_out_video, tof_eval_report_path, type="TOF")
            if tof_result is not None:
                eval_results_dict[reference]['psnr'] = tof_result['psnr']
                eval_results_dict[reference]['lpips'] = tof_result['lpips']
                eval_results_dict[reference]['tof'] = tof_result['tof']
                eval_results_dict[reference]['tlp'] = tof_result['tlp']
        
    # 计算所有评估结果的指标均值和方差
    if not args.eval_tof:
        statistics = compute_metrics_statistics(eval_results_dict)
        eval_results_dict['statistics'] = statistics
        
        # 运行整个数据集的 FID
        original_videos = f"{experiment_dir}/video_clips"
        generated_videos = f"{experiment_dir}/generated"
        dataset_fid_report_file = f"{experiment_dir}/reports/fid_eval_report.txt"
        fid_temp_dir = f"{experiment_dir}/tmp/fid"
        os.makedirs(fid_temp_dir, exist_ok=True)
        dataset_fid = run_dataset_FID(original_videos, generated_videos, fid_temp_dir, dataset_fid_report_file)
        statistics['dataset_FID'] = dataset_fid
    else:
        statistics = compute_metrics_statistics_tof(eval_results_dict)
        eval_results_dict['statistics'] = statistics
    
    # save the evaluation results to a file
    with open(eval_results_file, 'w') as f:
        json.dump(eval_results_dict, f, indent=4)
    print(f"Evaluation results: {eval_results_dict['statistics']}")  
    
    end_time = time.time()              
    # 输出实验完成时间，用分钟表示
    print(f"Experiment completed in {(end_time - start_time) / 60:.2f} minutes")
    # 告诉用户实验结果保存在哪里
    print(f"Experiment results saved in: {eval_results_file}")

if __name__ == '__main__':
    main()
