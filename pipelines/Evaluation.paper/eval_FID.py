import os
import cv2
import glob
import tempfile
import shutil
import argparse
import subprocess
from detectors import S3FD  # 正确导入 S3FD
from tqdm import tqdm

def extract_faces_from_video(video_path, output_face_dir, detector, facedet_scale=0.25):
    """提取视频中的人脸并保存为图像。"""
    temp_frames = tempfile.mkdtemp()  # 临时目录存储帧

    try:
        # 使用 ffmpeg 将视频解压为帧
        subprocess.run([
            'ffmpeg', '-i', video_path, 
            # '-vf', 'scale=256:256', 
            os.path.join(temp_frames, '%06d.png'), 
            '-hide_banner', '-loglevel', 'error'
        ], check=True)

        # 遍历所有帧并检测人脸
        frame_paths = sorted(glob.glob(os.path.join(temp_frames, '*.png')))
        for fidx, frame_path in tqdm(enumerate(frame_paths), total=len(frame_paths)): 
            # print(f"frame_path: {frame_path}")
            image = cv2.imread(frame_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # print(f"image_rgb: {image_rgb.shape}")
            
            # 使用 S3FD 检测人脸
            bboxes = detector.detect_faces(image_rgb, conf_th=0.9, scales=[facedet_scale])
            # print(f"bboxes: {bboxes}")
            
            for idx, bbox in enumerate(bboxes):
                x1, y1, x2, y2 = map(int, bbox[:4])
                face = image[y1:y2, x1:x2]
                
                # resize to 256x256
                face = cv2.resize(face, (256, 256))

                # 保存人脸图像
                face_path = os.path.join(output_face_dir, f'{fidx:06d}_{idx}.png')
                cv2.imwrite(face_path, face)

    finally:
        # 删除解压的临时帧目录
        shutil.rmtree(temp_frames)

def calculate_fid(real_dir, fake_dir):
    """使用 pytorch-fid 计算 FID。"""
    command = ['python', '-m', 'pytorch_fid', real_dir, fake_dir]
    subprocess.run(command, check=True)

def main(args):
    """主逻辑：提取人脸并计算 FID。"""
    detector = S3FD(device='cuda')  # 初始化 S3FD

    # 创建用于存储人脸的临时目录
    temp_real_faces = tempfile.mkdtemp()
    temp_fake_faces = tempfile.mkdtemp()

    try:
        print(f"Extracting faces from real video: {args.real}...")
        extract_faces_from_video(args.real, temp_real_faces, detector)

        print(f"Extracting faces from fake video: {args.fake}...")
        extract_faces_from_video(args.fake, temp_fake_faces, detector)

        print("Calculating FID between the two sets of faces...")
        calculate_fid(temp_real_faces, temp_fake_faces)

    finally:
        # 删除存储人脸的临时目录
        shutil.rmtree(temp_real_faces)
        shutil.rmtree(temp_fake_faces)
        print("Temporary directories cleaned up.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute FID between two videos using face regions only.")
    parser.add_argument('--real', required=True, help='Path to the real video.')
    parser.add_argument('--fake', required=True, help='Path to the fake video.')
    args = parser.parse_args()

    main(args)
