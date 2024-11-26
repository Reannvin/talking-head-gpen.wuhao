import shutil

import cv2
import os
import os.path as osp
import glob
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing
from multiprocessing import Pool
import random
import torch
from glob import glob
from pathlib import Path


if sys.version_info[0] < 3 and sys.version_info[1] < 2:
    raise Exception("Must be using >= Python 3.2")

from os import listdir, path

if not path.isfile('face_detection/detection/sfd/s3fd.pth'):
    raise FileNotFoundError('Save the s3fd model to face_detection/detection/sfd/s3fd.pth \
							before running this script!')

import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import argparse, os, cv2, traceback, subprocess
from tqdm import tqdm
import audio
from hparams import hparams as hp
# from moviepy.editor import VideoFileClip
# from moviepy.audio.io.AudioFileClip import AudioFileClip
import librosa

import face_detection

parser = argparse.ArgumentParser()
parser.add_argument('--ngpu', help='Number of GPUs across which to run in parallel', default=7, type=int)
parser.add_argument('--batch_size', help='Single GPU Face detection batch size', default=128, type=int)
parser.add_argument("--data_root", help="Root folder of the LRS2 dataset", default='')
parser.add_argument("--preprocessed_root", help="Root folder of the preprocessed dataset", default='')
args = parser.parse_args()


def print_wav_sr():
    """
    打印指定目录下所有 .mp4 视频文件的帧率。
    """
    data_root = '/mnt/diskwei/dataset/head_talk/LRS2/lrs2_preprocessed'
    filelist = glob.glob(osp.join(data_root, '**/*.wav'), recursive=True)
    print(filelist[0])
    all_pfs = set()
    for video_file_path in tqdm(filelist):
        # video_clip = AudioFileClip(video_file_path)
        # fps = video_clip.fps
        _, fps = librosa.core.load(video_file_path, sr=None)
        all_pfs.add(fps)
        # video_clip.close()

    print('all wav sample_rate:', all_pfs)


def print_video_fps():
    """
    打印指定目录下所有 .mp4 视频文件的帧率。
    """
    data_root = '/mnt/diskwei/dataset/head_talk/3300+ID/数据集/fps25/'
    filelist = glob.glob(osp.join(data_root, '**/*.mp4'))
    print(filelist[0])
    all_pfs = set()
    all_sr = set()
    for video_file_path in tqdm(filelist):
        cap = cv2.VideoCapture(video_file_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        all_pfs.add(fps)
        cap.release()

    print('all fps:', all_pfs)


def statistic_face_arears_multi_process(args):
    data_root = '/data2/dataset/head_talk/LRS2/mvlrs_v1/main'
    print('Started processing for {} with {} GPUs'.format(data_root, args.ngpu))
    filelist = glob.glob(path.join(data_root, '*/*.mp4'))
    filelist = random.sample(filelist, 100)
    print('glob mp4 done')


    processes = []
    filelist_part_all = []
    device_all = []
    sub_step = 7000
    for i in range(0, 128):  # 创建三个进程，每个进程处理矩阵的一行
        start, end = sub_step*i, sub_step*(i+1)
        # device = torch.device(f'cuda:{i//1:d}')
        device = f'cuda:{i//1:d}'
        filelist_part = filelist[start:end]
        if len(filelist_part) == 0:
            continue
        print(f'start:{start} end:{end} actual len:{len(filelist_part)}, device:{device}')
        filelist_part_all.append(filelist_part)
        device_all.append(device)

    print(f'pool size:{len(filelist_part_all)}')
    pool = Pool(len(filelist_part_all))
    results = []
    for i in range(len(filelist_part_all)):
        re = pool.apply_async(statistic_face_arears_sub_process,
                                args=(filelist_part_all[i], args, device_all[i],
                                          ))
        
        results.append(re)

    face_areas = []
    for p in results:
        output = p.get()
        if len(output) == 0:
            continue
        face_areas.extend(output)

    face_areas = np.array(face_areas)
    hist, bins = np.histogram(face_areas, bins=1000)
    normalized_hist = hist /np.sum(hist)
    plt.bar(bins[:-1], normalized_hist)
    plt.title("face area statistics")
    plt.xlabel("face arae")
    plt.ylabel("num")
    plt.savefig('face_output.jpg')


# fa = [face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False,
                                   # device='cuda:{}'.format(id)) for id in range(args.ngpu)]
# print('fa N:', len(fa))
def statistic_face_arears_sub_process(vfiles, args, device):
    fa = face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False,
                                   device=device)

    face_areas = []
    for vfile in tqdm(vfiles):
        video_stream = cv2.VideoCapture(vfile)
        frames = []
        while 1:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break
            frames.append(frame)

        batches = [frames[i:i + args.batch_size] for i in range(0, len(frames), args.batch_size)]
        i = -1
        for fb in batches:
            preds = fa.get_detections_for_batch(np.asarray(fb))
            for j, f in enumerate(preds):
                i += 1
                if f is None:
                    continue
                x1, y1, x2, y2 = f
                face_areas.append(max(0, y2 - y1) * max(0, x2 - x1))

    return face_areas




def statistic_face_arears(args):
    data_root = '/data2/dataset/head_talk/LRS2/mvlrs_v1/main'
    print('Started processing for {} with {} GPUs'.format(data_root, args.ngpu))
    filelist = glob.glob(path.join(data_root, '*/*.mp4'))
    # filelist = random.sample(filelist, 100)

    print('glob mp4 done')

    jobs = [(vfile, args, i%args.ngpu) for i, vfile in enumerate(filelist)]
    p = ThreadPoolExecutor(args.ngpu)
    futures = [p.submit(mp_handler2, j) for j in jobs]
    print('num thr:', len(futures))
    print('atart face det')
    face_areas = []
    for r in tqdm(as_completed(futures), total=len(futures)):
        res = r.result()
        face_areas.extend(res)
    # plt.hist(face_areas, bins=100, density=True)
    face_areas = np.array(face_areas)
    hist, bins = np.histogram(face_areas, bins=1000)
    normalized_hist = hist /np.sum(hist)
    plt.bar(bins[:-1], normalized_hist)
    plt.title("face area statistics")
    plt.xlabel("face arae")
    plt.ylabel("num")
    plt.savefig('face_output.jpg')



def mp_handler2(job):
    vfile, args, gpu_id = job
    try:
        return process_video_file2(vfile, args, gpu_id)
    except KeyboardInterrupt:
        exit(0)
    except:
        traceback.print_exc()


def process_video_file2(vfile, args, gpu_id):
    video_stream = cv2.VideoCapture(vfile)
    frames = []
    while 1:
        still_reading, frame = video_stream.read()
        if not still_reading:
            video_stream.release()
            break
        frames.append(frame)

    batches = [frames[i:i + args.batch_size] for i in range(0, len(frames), args.batch_size)]
    i = -1
    face_areas = []
    for fb in batches:
        preds = fa[gpu_id].get_detections_for_batch(np.asarray(fb))
        for j, f in enumerate(preds):
            i += 1
            if f is None:
                continue
            x1, y1, x2, y2 = f
            face_areas.append(max(0, y2 - y1) * max(0, x2 - x1))

    return face_areas


def merge_videos():
    video_files = ["results/liuwei_authormodel.mp4",
                   "results/liuwei_wav2lip_author_syncnet223.mp4", ]
    video_order = ['author_model', 'wei_base_author_synnet']
    out_path = 'results/author_wei_base_author_synnet_cat.mp4'
    cap_list = [cv2.VideoCapture(f) for f in video_files]
    fps = cap_list[0].get(cv2.CAP_PROP_FPS)
    for i in range(1, len(cap_list)):
        assert fps == cap_list[i].get(cv2.CAP_PROP_FPS)

    width = int(cap_list[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_list[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_width = sum(width for cap in cap_list)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (total_width, height))
    total_frames = cap_list[0].get(cv2.CAP_PROP_FRAME_COUNT)
    # 定义帧 ID 计数器
    frame_id = 0
    while True:
        ret_list, frame_list = [], []
        for cap in cap_list:
            ret, frame = cap.read()
            ret_list.append(ret)
            frame_list.append(frame)
        # 终止条件
        if not all(ret_list):
            break
        # 水平合并帧
        merged_frame = np.hstack(frame_list)
        # 绘制帧 ID
        cv2.putText(merged_frame, f"Frame ID: {frame_id}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        cv2.putText(merged_frame, f"Video: {'/  '.join(video_order)}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        # 递增帧 ID
        frame_id += 1
        # 写入视频
        out.write(merged_frame)
        if frame_id % 100 == 0:
            print(f'{frame_id}/{total_frames}')

    # 释放资源
    for cap in cap_list:
        cap.release()
    out.release()


def video25fps():
    data_root = '/mnt/diskwei/dataset/head_talk/3300+ID/数据集/所有素材'
    filelist = glob(path.join(data_root, '**/*.mp4'), recursive=True)
    save_root = Path('/mnt/diskwei/dataset/head_talk/3300+ID/数据集/fps25/')

    for vfile in tqdm(filelist):
        save_path =  save_root / vfile[len(data_root):].strip('/')
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # video_template = f'ffmpeg -loglevel warning -i {vfile} -vf "fps=25" -ar 16000  -c:v libx264 -ac 1  -c:a aac -b:a 128k {str(save_path)}'
        video_template = f'ffmpeg -loglevel warning -y -i {vfile} -vf "fps=25" -c:v libx264 {str(save_path)}'
        subprocess.call(video_template, shell=True)


def resizeImg96_multi_process():

    img_ls_path = '/mnt/diskwei/dataset/head_talk/3300+ID/数据集/id3300_fps25_preprocessed/1.txt'
    img_root = Path('/mnt/diskwei/dataset/head_talk/3300+ID/数据集/id3300_fps25_preprocessed/')
    save_root = Path('/mnt/diskwei/dataset/head_talk/3300+ID/数据集/id3300_fps25_preprocessed96')
    img_size = 96
    with open(img_ls_path) as f:
        img_rel_paths = f.read().strip().splitlines()

    filelist_part_all = []
    sub_step = 50_0000
    for i in range(0, 128):  # 创建三个进程，每个进程处理矩阵的一行
        start, end = sub_step * i, sub_step * (i + 1)
        filelist_part = img_rel_paths[start:end]
        if len(filelist_part) == 0:
            continue
        print(f'start:{start} end:{end} actual len:{len(filelist_part)}')
        filelist_part_all.append(filelist_part)

    print(f'pool size:{len(filelist_part_all)}')
    pool = Pool(len(filelist_part_all))
    results = []
    for i in range(len(filelist_part_all)):
        re = pool.apply_async(resizeImg96_subProcess,
                              args=(filelist_part_all[i], img_root, save_root, img_size
                                    ))

        results.append(re)

    for p in results:
        output = p.get()


def resizeImg96_subProcess(img_rel_paths, img_root, save_root, img_size):
    for img_rel_path in tqdm(img_rel_paths):
        img_path = img_root / img_rel_path
        save_path = save_root / img_rel_path
        save_path.parent.mkdir(exist_ok=True, parents=True)
        img = cv2.imread(str(img_path))
        img = cv2.resize(img, (img_size, img_size))
        cv2.imwrite(str(save_path), img)

def copy_audio():
    img_ls_path = '/mnt/diskwei/dataset/head_talk/3300+ID/数据集/id3300_fps25_preprocessed/1.txt'
    img_root = Path('/mnt/diskwei/dataset/head_talk/3300+ID/数据集/id3300_fps25_preprocessed/')
    save_root = Path('/mnt/diskwei/dataset/head_talk/3300+ID/数据集/id3300_fps25_preprocessed96')
    with open(img_ls_path) as f:
        img_rel_paths = f.read().strip().splitlines()
    for img_rel_path in tqdm(img_rel_paths):
        src_path = img_root / img_rel_path
        dst_path = save_root / img_rel_path
        shutil.copy(src_path, dst_path)

if __name__ == '__main__':
    # print_video_fps()
    # print_wav_sr()
    # merge_videos()
    # statistic_face_arears(args)
    # video25fps()
    # statistic_face_arears_multi_process(args)
    # resizeImg96_multi_process()
    copy_audio()

