import sys
import matplotlib.pyplot as plt
import multiprocessing
from multiprocessing import Pool
from tqdm import tqdm

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
from glob import glob
import audio
from hparams import hparams as hp

import face_detection

parser = argparse.ArgumentParser()

parser.add_argument('--ngpu', help='Number of GPUs across which to run in parallel', default=1, type=int)
parser.add_argument('--batch_size', help='Single GPU Face detection batch size', default=32, type=int)
parser.add_argument("--data_root", help="Root folder of the LRS2 dataset", required=True)
parser.add_argument("--preprocessed_root", help="Root folder of the preprocessed dataset", required=True)

args = parser.parse_args()

# fa = [face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False, 
                                    # device='cuda:{}'.format(id)) for id in range(args.ngpu)]

# template = 'ffmpeg -loglevel panic -y -i {} -strict -2 {}'
template = 'ffmpeg -loglevel panic -y -i {} -strict -2 -ac 1 -vn -acodec pcm_s16le --ar 16000 {}'
# template2 = 'ffmpeg -hide_banner -loglevel panic -threads 1 -y -i {} -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 {}'

import time 
def process_video_file(vfile, args, gpu_id):
    video_stream = cv2.VideoCapture(vfile)
    t1 = time.time()
    frames = []
    while 1:
        still_reading, frame = video_stream.read()
        if not still_reading:
            video_stream.release()
            break
        frames.append(frame)
    
    det_frames = []
    height, width = frames[0].shape[:2]
    scale = min(320/ height, 320/ width)
    new_height = int(height * scale)
    new_width = int(width * scale)
    for frame in frames:
        resized_image = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        det_frames.append(resized_image)

    print('read:', time.time() - t1)
    t1 = time.time()
    vidname = os.path.basename(vfile).split('.')[0]
    dirname = vfile.split('/')[-2]

    fulldir = path.join(args.preprocessed_root, dirname, vidname)
    os.makedirs(fulldir, exist_ok=True)

    # batches = [frames[i:i + args.batch_size] for i in range(0, len(frames), args.batch_size)]
    batches = [det_frames[i:i + args.batch_size] for i in range(0, len(det_frames), args.batch_size)]
    print('realdy data:', time.time() - t1)

    i = -1
    for fb in batches:
        t1 = time.time()
        preds = fa[gpu_id].get_detections_for_batch(np.asarray(fb))
        # print('det :', time.time() - t1)
        

        for j, f in enumerate(preds):
            i += 1
            if f is None:
                continue
            f = [int(val/scale) for val in f]
            x1, y1, x2, y2 = f
            # cv2.imwrite(path.join(fulldir, '{}.jpg'.format(i)), fb[j][y1:y2, x1:x2])
            cv2.imwrite(path.join(fulldir, '{}.jpg'.format(i)), frames[i][y1:y2, x1:x2])
            # print(i, len(frames))

    assert i == (len(frames)-1)

def process_audio_file(vfile, args):
    vidname = os.path.basename(vfile).split('.')[0]
    dirname = vfile.split('/')[-2]

    fulldir = path.join(args.preprocessed_root, dirname, vidname)
    os.makedirs(fulldir, exist_ok=True)

    wavpath = path.join(fulldir, 'audio.wav')

    command = template.format(vfile, wavpath)
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

    filelist = glob(path.join(args.data_root, '**/*.mp4'), recursive=True)

    jobs = [(vfile, args, i%args.ngpu) for i, vfile in enumerate(filelist)]
    p = ThreadPoolExecutor(args.ngpu)
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



def main_multi_process(args):
    print('Started processing for {} with {} GPUs'.format(args.data_root, args.ngpu))

    filelist = glob(path.join(args.data_root, '**/*.mp4'), recursive=True)

    # jobs = [(vfile, args, i%args.ngpu) for i, vfile in enumerate(filelist)]
    # p = ThreadPoolExecutor(args.ngpu)
    # futures = [p.submit(mp_handler, j) for j in jobs]
    # _ = [r.result() for r in tqdm(as_completed(futures), total=len(futures))]

    processes = []
    filelist_part_all = []
    device_all = []
    sub_step = 500
    for i in range(0, 128):  # 创建三个进程，每个进程处理矩阵的一行
        start, end = sub_step*i, sub_step*(i+1)
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
        re = pool.apply_async(process_video_file_sub_process,
                                args=(filelist_part_all[i], args, device_all[i],
                                          ))
        
        results.append(re)

    for p in results:
        output = p.get()


    print('Dumping audios...')

    for vfile in tqdm(filelist):
        try:
            process_audio_file(vfile, args)
        except KeyboardInterrupt:
            exit(0)
        except:
            traceback.print_exc()
            continue



def process_video_file_sub_process(vfiles, args, device):
    fa = face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False,
                                   device=device)

    for vfile in tqdm(vfiles):
        video_stream = cv2.VideoCapture(vfile)
        t1 = time.time()
        frames = []
        while 1:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break
            frames.append(frame)
        
        det_frames = []
        height, width = frames[0].shape[:2]
        scale = min(320/ height, 320/ width)
        new_height = int(height * scale)
        new_width = int(width * scale)
        for frame in frames:
            resized_image = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
            det_frames.append(resized_image)

        print('read:', time.time() - t1)
        t1 = time.time()
        vidname = os.path.basename(vfile).split('.')[0]
        dirname = vfile.split('/')[-2]

        fulldir = path.join(args.preprocessed_root, dirname, vidname)
        os.makedirs(fulldir, exist_ok=True)

        # batches = [frames[i:i + args.batch_size] for i in range(0, len(frames), args.batch_size)]
        batches = [det_frames[i:i + args.batch_size] for i in range(0, len(det_frames), args.batch_size)]
        print('realdy data:', time.time() - t1)

        i = -1
        for fb in batches:
            t1 = time.time()
            preds = fa.get_detections_for_batch(np.asarray(fb))
            

            for j, f in enumerate(preds):
                i += 1
                if f is None:
                    continue
                f = [int(val/scale) for val in f]
                x1, y1, x2, y2 = f
                # cv2.imwrite(path.join(fulldir, '{}.jpg'.format(i)), fb[j][y1:y2, x1:x2])
                cv2.imwrite(path.join(fulldir, '{}.jpg'.format(i)), frames[i][y1:y2, x1:x2])
                # print(i, len(frames))

        assert i == (len(frames)-1)




if __name__ == '__main__':
    # main(args)
    main_multi_process(args)
