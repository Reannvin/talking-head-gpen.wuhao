import os
import cv2
import glob
import argparse
import subprocess
import shutil  # 确保正确导入 shutil
from detectors import S3FD  # 确保正确导入 S3FD
from tqdm import tqdm

def extract_faces_from_video(video_path, output_face_dir, detector, temp_dir, facedet_scale=0.25):
    """提取视频中的人脸并保存为图像。"""
    video_name = os.path.splitext(os.path.basename(video_path))[0]  # 提取视频文件名（不带扩展名）
    # video_output_dir = os.path.join(output_face_dir, video_name)  # 为每个视频创建一个单独的文件夹

    # 创建输出文件夹（如果不存在）
    # os.makedirs(video_output_dir, exist_ok=True)

    # 创建临时帧目录，用于存储解压的视频帧
    temp_frames = os.path.join(temp_dir, f'{video_name}_frames')
    os.makedirs(temp_frames, exist_ok=True)

    try:
        # 使用 ffmpeg 将视频解压为帧
        subprocess.run([
            'ffmpeg', '-i', video_path, 
            os.path.join(temp_frames, '%06d.png'), 
            '-hide_banner', '-loglevel', 'error'
        ], check=True)

        # 遍历所有帧并检测人脸
        frame_paths = sorted(glob.glob(os.path.join(temp_frames, '*.png')))
        for fidx, frame_path in tqdm(enumerate(frame_paths), total=len(frame_paths)): 
            image = cv2.imread(frame_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 使用 S3FD 检测人脸
            bboxes = detector.detect_faces(image_rgb, conf_th=0.9, scales=[facedet_scale])
            
            for idx, bbox in enumerate(bboxes):
                x1, y1, x2, y2 = map(int, bbox[:4])
                face = image[y1:y2, x1:x2]
                
                # resize to 256x256
                face = cv2.resize(face, (256, 256))

                # 保存人脸图像到单独的文件夹
                face_path = os.path.join(output_face_dir, f'{video_name}_{fidx:06d}_{idx}.png')
                cv2.imwrite(face_path, face)

    finally:
        # 删除临时帧目录
        shutil.rmtree(temp_frames)

def extract_faces_from_videos_in_dir(video_dir, output_face_dir, detector, temp_dir):
    """从文件夹中的所有视频提取人脸并保存为图像。"""
    video_paths = glob.glob(os.path.join(video_dir, '*.mp4'))  # 获取所有 mp4 文件
    for video_path in video_paths:
        print(f"Processing video: {video_path}")
        extract_faces_from_video(video_path, output_face_dir, detector, temp_dir)

def calculate_fid(real_dir, fake_dir):
    """使用 pytorch-fid 计算 FID。"""
    print(f"real_dir {real_dir}, fake_dir {fake_dir}")
    command = ['python', '-m', 'pytorch_fid', real_dir, fake_dir]
    subprocess.run(command, check=True)

def process_path(input_path, output_face_dir, detector, temp_dir):
    """根据路径是文件还是文件夹，处理单个视频或文件夹中的视频。"""
    if os.path.isfile(input_path):  # 如果是单个视频文件
        print(f"Processing video file: {input_path}")
        extract_faces_from_video(input_path, output_face_dir, detector, temp_dir)
    elif os.path.isdir(input_path):  # 如果是文件夹
        print(f"Processing video directory: {input_path}")
        extract_faces_from_videos_in_dir(input_path, output_face_dir, detector, temp_dir)
    else:
        raise ValueError(f"Invalid input path: {input_path}. It should be either a file or a directory.")

def main(args):
    """主逻辑：从视频或文件夹提取人脸并计算 FID。"""
    detector = S3FD(device='cuda')  # 初始化 S3FD

    # 在指定的 faces_temp_dir 下创建 real 和 fake 子文件夹
    real_faces_dir = os.path.join(args.faces_temp_dir, 'real')
    fake_faces_dir = os.path.join(args.faces_temp_dir, 'fake')
    temp_dir = os.path.join(args.faces_temp_dir, 'temp')  # 临时帧文件夹

    os.makedirs(real_faces_dir, exist_ok=True)
    os.makedirs(fake_faces_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    # 处理 real 和 fake 路径
    print(f"Extracting faces from real input: {args.real}...")
    process_path(args.real, real_faces_dir, detector, temp_dir)

    print(f"Extracting faces from fake input: {args.fake}...")
    process_path(args.fake, fake_faces_dir, detector, temp_dir)

    # 计算 FID
    print("Calculating FID between the two sets of faces...")
    calculate_fid(real_faces_dir, fake_faces_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute FID between two inputs (either video files or folders of videos) using face regions only.")
    
    # 统一使用相同参数，路径可以是单个视频或者文件夹
    parser.add_argument('--real', required=True, help='Path to the real video or folder containing real videos.')
    parser.add_argument('--fake', required=True, help='Path to the fake video or folder containing fake videos.')

    # 使用一个参数指定保存人脸图片和临时帧的顶级目录
    parser.add_argument('--faces_temp_dir', required=True, help='Directory to save extracted face images and temporary frames.')

    args = parser.parse_args()

    main(args)
