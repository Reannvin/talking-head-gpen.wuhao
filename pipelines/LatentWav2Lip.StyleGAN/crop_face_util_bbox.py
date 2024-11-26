import sys

if sys.version_info[0] < 3 and sys.version_info[1] < 2:
    raise Exception("Must be using >= Python 3.2")

from os import listdir, path

# if not path.isfile('face_detection/detection/sfd/s3fd.pth'):
#     raise FileNotFoundError('Save the s3fd model to face_detection/detection/sfd/s3fd.pth \
#                             before running this script!')

from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import argparse, os, cv2, traceback, subprocess
from tqdm import tqdm
from glob import glob
# import audio
# from hparams import hparams as hp
import shlex
import torch
import pynvml
import face_detection

parser = argparse.ArgumentParser()

parser.add_argument('--ngpu', help='Number of GPUs across which to run in parallel', default=1, type=int)
parser.add_argument('--batch_size', help='Single GPU Face detection batch size', default=32, type=int)
parser.add_argument("--data_root", help="Root folder of the LRS2 dataset", required=True)
parser.add_argument("--preprocessed_root", help="Root folder of the preprocessed dataset", required=True)
parser.add_argument("--resize", help="Resize frames to 1/4th size", action='store_true')
parser.add_argument('--leftright_scale', help='Bbox left and right expansion coefficient', default=0, type=float)
parser.add_argument('--topbottom_scale', help='Bbox top and bottom expansion coefficient', default=0, type=float)
parser.add_argument('--file_num', help='Bbox bottom expansion coefficient', default=0, type=int)

args = parser.parse_args()

def get_free_gpu():
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    free_list = []
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)  # gpu利用率

        info = pynvml.nvmlDeviceGetMemoryInfo(handle)  # 显存使用情况
        free_list.append(info.free / 1024 ** 2)
    print(free_list)
    # 对列表中的元素和元素的下标进行排序（按照元素的值）
    sorted_pairs = sorted(enumerate(free_list), key=lambda x: x[1], reverse=True)

    # 提取排序后的下标
    sorted_indices = [index for index, value in sorted_pairs]
    
    return sorted_indices

gpu_idx = get_free_gpu()[0:args.ngpu]
print(f"欲选定GPU: {gpu_idx}")
try:
    gpu_idx.remove(5)    # 4090机器不用卡5
except Exception as e:
    pass
print(f"将使用GPU: {gpu_idx}")

try:
    detector = {gpu_id: face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False, device=f'cuda:{gpu_id}' if gpu_id >= 0 else 'cpu') for gpu_id in gpu_idx}
except Exception as e:
    detector = {gpu_id: face_detection.FaceAlignment(face_detection.LandmarksType.TWO_D, flip_input=False, device=f'cuda:{gpu_id}' if gpu_id >= 0 else 'cpu') for gpu_id in gpu_idx}


template = 'ffmpeg -loglevel panic -y -i {} -strict -2 {}'
# template2 = 'ffmpeg -hide_banner -loglevel panic -threads 1 -y -i {} -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 {}'
import os
import cv2
import numpy as np
import traceback




def process_and_save_cropped_faces(frames, preds, scale, fulldir, start_index, leftright_scale=0.05, topbottom_scale=0.1):
    """
    处理每个批次的帧并保存裁剪后的人脸图像。
    
    frames: 原始帧的列表
    preds: bbox 结果 x1 y1 x2 y2
    scale: 缩放比例，对于没有resize的情况为1
    fulldir: 保存图像的目录
    start_index: 帧的起始索引
    """
    i = start_index
    for j, bbox in enumerate(preds):
        i += 1
        if bbox is None:
            continue
        try:
            # 缩放关键点坐标
            x_min, y_min, x_max, y_max = bbox

            x_min *= scale
            y_min *= scale
            x_max *= scale
            y_max *= scale

            w = x_max - x_min
            h = y_max - y_min

            x_min = max(int(x_min - leftright_scale * w), 0)
            y_min = max(int(y_min - topbottom_scale * h), 0)
            x_max = min(int(x_max + leftright_scale * w), frames[j].shape[1])
            y_max = min(int(y_max + topbottom_scale * h), frames[j].shape[0])

            cropped_face = frames[j][y_min:y_max, x_min:x_max]
            cropped_face = cropped_face[:, :, ::-1]
            cv2.imwrite(os.path.join(fulldir, f'{i}.jpg'), cropped_face)
        except Exception as e:
            # print(f"文件 {fulldir} 帧号 {i} crop_face报错，详情: {traceback.format_exc()}")
            pass
    
    return i

def process_video_file(vfile, args, gpu_id):
    video_stream = cv2.VideoCapture(vfile)
    
    vidname = os.path.basename(vfile).split('.')[0]
    dirname = vfile.split('/')[-2]
    fulldir = os.path.join(args.preprocessed_root, dirname, vidname)
    os.makedirs(fulldir, exist_ok=True)

    i = -1
    while True:
        frames = []
        for _ in range(args.batch_size):
            still_reading, frame = video_stream.read()
            if not still_reading:
                break
            frame = frame[:, :, ::-1]
            frames.append(frame)
        
        if not frames:
            break

        if args.resize:
            # Resize frames to 1/4th size
            resized_frames = [cv2.resize(frame, (frame.shape[1] // 4, frame.shape[0] // 4)) for frame in frames]
            resized_frames = np.asarray(resized_frames)
            preds = detector[gpu_id].get_detections_for_batch(resized_frames)            
            i = process_and_save_cropped_faces(frames, preds, scale=4, fulldir=fulldir, start_index=i, leftright_scale=args.leftright_scale, topbottom_scale=args.topbottom_scale)
        else:
            resized_frames = np.asarray(frames)
            preds = detector[gpu_id].get_detections_for_batch(resized_frames)
            i = process_and_save_cropped_faces(frames, preds, scale=1, fulldir=fulldir, start_index=i, leftright_scale=args.leftright_scale, topbottom_scale=args.topbottom_scale)
                
    video_stream.release()



def process_audio_file(vfile, args):
    vidname = os.path.basename(vfile).split('.')[0]
    dirname = vfile.split('/')[-2]

    fulldir = path.join(args.preprocessed_root, dirname, vidname)
    os.makedirs(fulldir, exist_ok=True)

    wavpath = path.join(fulldir, 'audio.wav')

    # 使用 shlex.quote 对路径进行转义
    safe_vfile = shlex.quote(vfile)
    safe_wavpath = shlex.quote(wavpath)
    command = template.format(safe_vfile, safe_wavpath)
    # command = template.format(vfile, wavpath)
    print(command)
    subprocess.call(command, shell=True)

    
def mp_handler(job):
    vfile, args, gpu_id = job
    try:
        process_video_file(vfile, args, gpu_id)
    except KeyboardInterrupt:
        exit(0)
    except:
        traceback.print_exc()
        
def main(args):
    print('Started processing for {} with {} GPUs'.format(args.data_root, args.ngpu))

    filelist = glob(path.join(args.data_root, '*.mp4'))
    if args.file_num != 0:
        filelist = filelist[0:args.file_num]
    print("filelist:", filelist)

    jobs = [(vfile, args, gpu_idx[i%len(gpu_idx)]) for i, vfile in enumerate(filelist)]
    p = ThreadPoolExecutor(len(gpu_idx))
    futures = [p.submit(mp_handler, j) for j in jobs]
    _ = [r.result() for r in tqdm(as_completed(futures), total=len(futures))]

    print('Dumping audios...')

    for vfile in tqdm(filelist):
        try:
            process_audio_file(vfile, args)
        except KeyboardInterrupt:
            exit(0)
        except:
            traceback.print_exc()
            continue

if __name__ == '__main__':
    main(args)