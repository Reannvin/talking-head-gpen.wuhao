import os
import argparse
import numpy as np
from PIL import Image
import cv2
import face_detection
import torch
from torch.utils.data import Dataset
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import subprocess


# 全局变量，保存每个进程的 FaceDetectionService 实例
face_detection_service = None

def init_face_detection_service(gpu_id):
    """
    初始化全局 face_detection_service，并指定 GPU
    如果已经初始化，则跳过。
    """
    global face_detection_service
    if face_detection_service is None:
        face_detection_service = FaceDetectionService(gpu_id)

class FaceDetectionService:
    """
    Simplified Face detection service that loads model on a specific GPU and performs detection.
    """
    def __init__(self, gpu_id):
        device = f'cuda:{gpu_id}'
        print(f"Loading model on GPU {gpu_id}")
        self.model = face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False, device=device)
        print(f"Model loaded on GPU {gpu_id}")

    def get_detections(self, np_frames):
        """
        Perform face detection on the provided frames using the loaded model.
        """
        preds = self.model.get_detections_for_batch(np_frames)
        return preds

class VideoDataset(Dataset):
    def __init__(self, root_dir):
        """
        Args:
            root_dir (string): Directory with all the video files.
        """
        self.root_dir = root_dir
        self.video_files = self._collect_video_files(root_dir)
    
    def _collect_video_files(self, folder):
        video_files = []
        for root, _, files in os.walk(folder):
            for file in files:
                if file.endswith(('.mp4', '.avi', '.mkv')):
                    video_files.append(os.path.join(root, file))
        return video_files
    
    def __len__(self):
        return len(self.video_files)
    
    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        return video_path

def get_video_frames(video_path):
    video_stream = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = video_stream.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))

    video_stream.release()
    return frames

def resize_and_crop_frames(frames, resize_to, crop_down_ratio, batch_size):
    """
    对提取的视频帧进行批量裁剪并调整大小，包含 S3FD 检测和裁剪逻辑。
    最终裁剪的图片是基于原始输入 frames。
    """
    cropped_frames = []
    
    for i in range(0, len(frames), batch_size):
        batch_frames = frames[i:i + batch_size]
        
        # 只用于人脸检测的 resized_frames 列表
        resized_frames = []
        
        # 记录每帧的原始尺寸
        original_sizes = []
        
        for frame in batch_frames:
            original_width, original_height = frame.size
            original_sizes.append((original_width, original_height))
            
            # 根据原始宽高比确定新的宽度和高度
            if original_width > original_height:
                # 宽大于高，宽度调整为 resize_to，等比例缩放高度
                new_width = resize_to
                new_height = int(resize_to * original_height / original_width)
            else:
                # 高大于宽，高度调整为 resize_to，等比例缩放宽度
                new_height = resize_to
                new_width = int(resize_to * original_width / original_height)

            # 调整大小，保持宽高比 (仅用于人脸检测)
            resized_frame = frame.resize((new_width, new_height), Image.LANCZOS)
            resized_frames.append(resized_frame)

        # 将批次转换为 numpy 数组
        np_batch_frames = np.array([np.array(frame) for frame in resized_frames])

        # 请求 face_detection_service 进行批量检测
        preds = face_detection_service.get_detections(np_batch_frames)

        for j, frame in enumerate(batch_frames):
            if preds[j] is None:
                continue

            # 获取检测到的人脸在 resized_frame 上的坐标
            x1, y1, x2, y2 = preds[j]

            # 获取原始帧的宽高
            original_width, original_height = original_sizes[j]
            resized_width, resized_height = resized_frames[j].size

            # 计算比例，将 resized_frame 的坐标映射回原始帧的坐标
            scale_x = original_width / resized_width
            scale_y = original_height / resized_height

            # 将坐标转换回原始图像中的坐标
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)

            # 计算裁剪区域的高度，并根据 crop_down_ratio 调整
            crop_height = y2 - y1
            y1 += int(crop_height * crop_down_ratio)
            y2 += int(crop_height * crop_down_ratio)

            # 在原始帧上进行裁剪
            cropped_frame = frame.crop((x1, y1, x2, y2))
            cropped_frames.append(cropped_frame)

        # 清理 GPU 显存（如有必要）
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return cropped_frames

def save_frames(frames, video_path, input_folder, output_folder):
    relative_path = os.path.relpath(video_path, input_folder)
    relative_dir = os.path.splitext(relative_path)[0]
    output_dir = os.path.join(output_folder, relative_dir)
    os.makedirs(output_dir, exist_ok=True)

    for i, frame in enumerate(frames):
        frame.save(os.path.join(output_dir, f"{i:05d}.jpg"))

def save_audio(video_path, input_folder, output_folder, sample_rate=16000):
    """
    Extracts audio from the video, converts it to 16kHz mono WAV format, and saves it
    in the output folder with the same relative path structure as the input folder.
    
    Args:
        video_path (str): Path to the video file.
        input_folder (str): Root folder of the input videos.
        output_folder (str): Root folder where extracted audio will be saved.
        sample_rate (int): Desired sample rate for the output WAV file (default: 16kHz).
    """
    # Create relative output path for audio
    relative_path = os.path.relpath(video_path, input_folder)
    relative_dir = os.path.splitext(relative_path)[0]
    output_dir = os.path.join(output_folder, relative_dir)
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Define the output audio path
    output_audio_path = os.path.join(output_dir, "audio.wav")
    
    # Extract audio and save it
    extract_audio(video_path, output_audio_path, sample_rate)

def extract_audio(video_path, output_audio_path, sample_rate=16000):
    """
    Extract audio from a video and save it as a WAV file with the specified sample rate.
    
    Args:
        video_path (str): Path to the video file.
        output_audio_path (str): Path where the extracted audio will be saved.
        sample_rate (int): Desired audio sample rate (default: 16kHz).
    """
    command = [
        'ffmpeg',
        '-i', video_path,                # Input video file
        '-vn',                           # No video stream
        '-ar', str(sample_rate),         # Set audio sample rate
        '-ac', '1',                      # Convert audio to mono
        '-acodec', 'pcm_s16le',          # Set audio codec to PCM signed 16-bit little-endian
        output_audio_path                # Output WAV file path
    ]
    
    # Suppress ffmpeg output by redirecting stdout and stderr to subprocess.DEVNULL
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

def process_single_video(video_path, input_folder, output_folder, resize_to, crop_down_ratio, batch_size):
    """
    Process a single video using the global face_detection_service initialized per process.
    """
    # Process frames
    frames = get_video_frames(video_path)
    cropped_resized_frames = resize_and_crop_frames(frames, resize_to, crop_down_ratio, batch_size)
    save_frames(cropped_resized_frames, video_path, input_folder, output_folder)
    
    # Save audio
    save_audio(video_path, input_folder, output_folder)

def process_videos_parallel(dataset, input_folder, output_folder, resize_to, crop_down_ratio, num_workers_per_gpu, batch_size, available_gpus, max_pending_futures=100):
    """
    Parallel processing of videos with controlled task submission to prevent resource exhaustion.
    """
    total_workers = num_workers_per_gpu * len(available_gpus)
    print(f"Using {total_workers} workers across {len(available_gpus)} GPUs.")

    futures = []
    with ProcessPoolExecutor(max_workers=total_workers) as executor:
        pbar = tqdm(total=len(dataset))

        for idx, video_path in enumerate(dataset):
            gpu_id = available_gpus[idx % len(available_gpus)]  # 循环分配 GPU

            # 提交任务到进程池，但限制未完成任务数量
            if len(futures) >= max_pending_futures:
                # 如果当前未完成任务的数量达到上限，等待部分任务完成
                for future in as_completed(futures):
                    try:
                        future.result()  # 处理结果
                    except Exception as e:
                        print(f"Error processing video: {e}")
                    pbar.update(1)
                    futures.remove(future)  # 移除已完成任务
                    if len(futures) < max_pending_futures:  # 当有足够空间时，跳出等待循环
                        break

            # 提交新任务到进程池
            future = executor.submit(process_single_video_with_gpu, video_path, input_folder, output_folder,
                                     resize_to, crop_down_ratio, batch_size, gpu_id)
            futures.append(future)

        # 确保所有任务都处理完成
        for future in as_completed(futures):
            try:
                future.result()  # 处理结果
            except Exception as e:
                print(f"Error processing video: {e}")
            pbar.update(1)

        pbar.close()  # 确保进度条在任务完成后正确关闭


def process_single_video_with_gpu(video_path, input_folder, output_folder, resize_to, crop_down_ratio, batch_size, gpu_id):
    """
    Wrapper for process_single_video that initializes the face detection service with a specific GPU.
    """
    init_face_detection_service(gpu_id)  # 初始化指定 GPU 的人脸检测服务
    process_single_video(video_path, input_folder, output_folder, resize_to, crop_down_ratio, batch_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process videos into frames with center crop and resize.")
    parser.add_argument('--input', type=str, required=True, help='Path to the input folder containing videos.')
    parser.add_argument('--output', type=str, required=True, help='Path to the output folder to save frames.')
    parser.add_argument('--resize-to', type=int, default=640, help='Resize frames to this size before face detection.')
    parser.add_argument('--crop-down', type=float, default=0.1, help='Move crop area downward by ratio of the bbox.')
    parser.add_argument('--workers', type=int, default=4, help='Number of worker processes per GPU.')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for face detection.')

    args = parser.parse_args()

    input_folder = args.input
    output_folder = args.output
    resize_to = args.resize_to
    crop_down_ratio = args.crop_down
    num_workers_per_gpu = args.workers
    batch_size = args.batch_size

    # 创建数据集
    dataset = VideoDataset(input_folder)
    print(f"Found {len(dataset)} videos in the input folder.")

    # 获取可用的 GPU 列表
    available_gpus = [i for i in range(torch.cuda.device_count())]

    # 并行处理视频
    process_videos_parallel(dataset, input_folder, output_folder, resize_to, crop_down_ratio, num_workers_per_gpu, batch_size, available_gpus)
