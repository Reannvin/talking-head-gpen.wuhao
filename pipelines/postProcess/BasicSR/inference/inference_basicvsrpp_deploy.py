import argparse
import cv2
import glob
import os
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import face_alignment

from basicsr.archs.basicvsrpp_arch import BasicVSRPlusPlus
from basicsr.data.data_util import read_img_seq, read_img_seq_crop
from basicsr.utils.img_util import tensor2img
from basicsr.utils import FileClient, get_root_logger, img2tensor



def get_max_area_face(fa, input, max_size=1024):
    """
    input 图像缩放到max_size进行检测，再逆变会input尺度人脸框, 最后返回最大人脸框
    :param input: H*W*3 RGB
    :return:
    """
    height, width = input.shape[:2]
    scale = max_size / max(height, width)
    new_height = round(height * scale)
    new_width = round(width * scale)
    input = cv2.resize(input, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    landmarks, landmarks_scores, detected_faces = fa.get_landmarks(input, return_bboxes=True,
                                                                   return_landmark_score=True)
    if landmarks is None:
        return None, None, None

    sort_inds = sorted(enumerate(detected_faces), key=lambda x: (x[1][3] - x[1][1]) * (x[1][2] - x[1][0]), reverse=True)
    ind = sort_inds[0][0]
    landmarks = landmarks[ind]
    landmarks_scores = landmarks_scores[ind]
    detected_faces = detected_faces[ind]
    landmarks = landmarks / scale
    detected_faces[:4] = detected_faces[:4] / scale

    return landmarks, landmarks_scores, detected_faces

def read_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        H, W = frame.shape[:2]
        frames.append(frame)

    cap.release()

    return frames, fps, W, H

def get_video_face_box(frames):
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)
    expand_ratio = 2
    overall_boxes = []

    for frame in frames:
        frame_width, frame_height = frame.shape[:2]
        _, _, bbox = get_max_area_face(fa, frame, max_size=480)
        if bbox is not None:
            overall_boxes.append(bbox)

    overall_boxes = np.array(overall_boxes)

    if len(overall_boxes) == 0:
        return None

    # Merge all detected face boxes
    x_min = int(np.min(overall_boxes[:, 0]))
    y_min = int(np.min(overall_boxes[:, 1]))
    x_max = int(np.max(overall_boxes[:, 2]))
    y_max = int(np.max(overall_boxes[:, 3]))

    # Calculate the face box dimensions
    width = x_max - x_min
    height = y_max - y_min

    # Determine the longer side and calculate the expansion
    long_side = max(width, height)
    expansion = int((long_side * expand_ratio - long_side) / 2)

    # Expand the box and ensure it stays within video boundaries
    x_min = max(0, x_min - expansion)
    y_min = max(0, y_min - expansion)
    x_max = min(frame_width, x_max + expansion)
    y_max = min(frame_height, y_max + expansion)

    return (x_min, y_min, x_max, y_max)

def add_suffix_to_filename(file_path, suffix):
    # 分离路径和文件名
    dir_name, base_name = os.path.split(file_path)
    # 分离文件名和扩展名
    file_name, file_ext = os.path.splitext(base_name)
    # 添加后缀
    new_file_name = f"{file_name}{suffix}{file_ext}"
    # 重新组合成新的路径
    return os.path.join(dir_name, new_file_name)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument(
        '--input_path', type=str, help='input test video file')
    parser.add_argument('--save_path', type=str, default='results/result.mp4', help='save image path')
    parser.add_argument('--interval', type=int, default=7, help='interval size')
    parser.add_argument('--save_crop', action='store_true', help='save video restore crop img')
    parser.add_argument('--org_video_path', type=str, default='', help='save video restore crop img')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set up model
    model = BasicVSRPlusPlus(mid_channels=64, num_blocks=7, is_low_res_input=False)
    model.load_state_dict(torch.load(args.model_path)['params'], strict=True)
    model.eval()
    model = model.to(device)

    frames, fps, W, H = read_video_frames(args.input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.save_path, fourcc, fps, (W, H))
    x1, y1, x2, y2 = get_video_face_box(frames)
    if args.save_crop:
        if os.path.exists(args.org_video_path):
            out_crop_cap = cv2.VideoWriter(add_suffix_to_filename(args.save_path, '_crop'), fourcc, fps, (x2-x1, y2-y1))
            out_tK_crop_cap = cv2.VideoWriter(add_suffix_to_filename(args.save_path, '_wav2lip_crop'), fourcc, fps, (x2-x1, y2-y1))
            out_org_crop_cap = cv2.VideoWriter(add_suffix_to_filename(args.save_path, '_org_crop'), fourcc, fps, (x2-x1, y2-y1))
            org_video_frames, _, _, _ = read_video_frames(args.org_video_path)
            print('org frames:', len(org_video_frames), ' wav2lip:', len(frames))
        else:
            print('org video file no exist:', args.org_video_path)
    else:
        out_crop_cap = None

    # print(x1, y1, x2, y2)
    # load data and inference
    for idx in range(0, len(frames), args.interval):
        wav2lip_imgs = frames[idx:idx+args.interval]
        if out_crop_cap is not None:
            org_frames = org_video_frames[idx:idx+args.interval]
        else:
            org_frames = wav2lip_imgs
        # x1, y1, x2, y2 = get_video_face_box(org_imgs)
        imgs = [cv2.cv2.resize(img[y1:y2, x1:x2], (512, 512)) for img in wav2lip_imgs]
        imgs = img2tensor(imgs, bgr2rgb=False, float32=True)
        imgs = torch.stack(imgs, dim=0)
        imgs = imgs / 255
        imgs = imgs.unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(imgs)
        # save imgs
        outputs = outputs.squeeze()
        outputs = list(outputs)
        for org_img, wav2lip_img, output, in zip(org_frames, wav2lip_imgs, outputs):
            output = tensor2img(output)
            output = cv2.resize(output, (x2-x1, y2-y1))
            if out_crop_cap is not None:
                out_tK_crop_cap.write(wav2lip_img[y1:y2, x1:x2][:, :, ::-1])
            wav2lip_img[y1:y2, x1:x2] = output[:, :, ::-1]
            out.write(wav2lip_img[:, :, ::-1])
            if out_crop_cap is not None:
                out_crop_cap.write(output)
                assert org_img.shape == wav2lip_img.shape
                out_org_crop_cap.write(org_img[y1:y2, x1:x2][:, :, ::-1])


    out.release()
    if out_crop_cap is not None:
        out_crop_cap.release()
        out_tK_crop_cap.release()
        out_org_crop_cap.release()


if __name__ == '__main__':
    main()
