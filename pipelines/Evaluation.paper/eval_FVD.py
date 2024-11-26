import torch
import argparse
import json
import torchvision.transforms as transforms
import cv2
import numpy as np
from detectors import S3FD  # 确保正确导入
from common_metrics.calculate_fvd import calculate_fvd
import torchvision


def parse_args():
    parser = argparse.ArgumentParser(description="Calculate FVD, PSNR, SSIM, LPIPS between real and fake videos")
    parser.add_argument("--real", type=str, required=True, help="Path to the real video (mp4 format)")
    parser.add_argument("--fake", type=str, required=True, help="Path to the fake video (mp4 format)")
    parser.add_argument("--use-cuda", type=bool, default=True, help="Use CUDA if available")
    parser.add_argument("--method", type=str, default='styleganv', choices=['styleganv', 'videogpt'],
                        help="Method to use for FVD calculation")
    parser.add_argument("--limit", type=int, default=16, help="Limit the number of frames to process")
    return parser.parse_args()

def detect_faces(frame, detector, facedet_scale=0.25):
    """使用 S3FD 检测人脸，并返回裁剪的第一个人脸区域。"""
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    bboxes = detector.detect_faces(image_rgb, conf_th=0.9, scales=[facedet_scale])

    if bboxes is None or len(bboxes) == 0:
        return None  # 如果未检测到人脸，返回 None

    # 获取第一个检测到的人脸区域
    x1, y1, x2, y2 = map(int, bboxes[0][:4])
    face = frame[y1:y2, x1:x2]

    # 如果人脸区域有效，则调整大小并返回
    if face.size > 0:
        face = cv2.resize(face, (256, 256))
        return face

    return None  # 如果人脸区域无效，则返回 None

def load_video_as_tensor(video_path, detector, limit):
    """将视频加载为张量，仅使用检测到的第一个人脸，并限制帧数。"""
    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened() and len(frames) < limit:
        ret, frame = cap.read()
        if not ret:
            break

        # 检测并提取第一个人脸区域
        face = detect_faces(frame, detector)

        # 如果未检测到人脸，则跳过该帧
        if face is None:
            continue

        # 将人脸转换为张量并添加到帧列表
        face_tensor = transforms.ToTensor()(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        frames.append(face_tensor)

    cap.release()

    if not frames:
        raise ValueError(f"No faces detected in video: {video_path}")

    # 将所有帧堆叠为张量，格式为 [1, T, C, H, W]
    video_tensor = torch.stack(frames).permute(0, 1, 2, 3)  # [T, C, H, W]
    return video_tensor.unsqueeze(0)  # [1, T, C, H, W]

def main():
    args = parse_args()

    # 初始化人脸检测器
    detector = S3FD(device='cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu')

    # 加载视频并转换为张量，仅使用人脸区域，并限制帧数
    print(f"Loading real video: {args.real}")
    videos1 = load_video_as_tensor(args.real, detector, args.limit)
    
    # 用 torchvision 保存图片
    # torchvision.utils.save_image(videos1[0], 'real.png', nrow=8, normalize=True)

    print(f"Loading fake video: {args.fake}")
    videos2 = load_video_as_tensor(args.fake, detector, args.limit)
    
    # 用 torchvision 保存图片
    # torchvision.utils.save_image(videos2[0], 'fake.png', nrow=8, normalize=True)

    # 设备配置
    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")

    # 将视频张量移动到指定设备
    videos1 = videos1.to(device)
    videos2 = videos2.to(device)

    # 计算 FVD
    result = {}
    result['fvd'] = calculate_fvd(videos1, videos2, device, method=args.method)

    # 打印结果为 JSON
    print(json.dumps(result, indent=4))

if __name__ == "__main__":
    main()
