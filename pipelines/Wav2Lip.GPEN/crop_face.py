import sys

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
import shlex
import face_detection

parser = argparse.ArgumentParser(description="you should use resize, resize_to 640, crop_down_ration 0.1, to accomplish face detection and crop")

parser.add_argument('--ngpu', help='Number of GPUs across which to run in parallel', default=1, type=int)
parser.add_argument('--batch_size', help='Single GPU Face detection batch size', default=32, type=int)
parser.add_argument("--data_root", help="Root folder of the LRS2 dataset")
parser.add_argument("--preprocessed_root", help="Root folder of the preprocessed dataset", required=True)
parser.add_argument("--resize", help="Resize frames to 1/4th size", action='store_true')
parser.add_argument("--resize_to", type=int, default=640, help="resize original image to this size, s3fd is no good for large image detection")
parser.add_argument("--crop_down_ratio", help="Move crop area downward by ratio of the bbox", default=0, type=float)
parser.add_argument("--extend_ratio", help="Extend the crop area by ratio", default=0, type=float)

args = parser.parse_args()

fa = [face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False, 
									device='cuda:{}'.format(id)) for id in range(args.ngpu)]

template = 'ffmpeg -loglevel panic -y -i {} -strict -2 {}'
# template2 = 'ffmpeg -hide_banner -loglevel panic -threads 1 -y -i {} -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 {}'
import os
import cv2
import numpy as np

def adjust_bounding_box(x1, y1, x2, y2, crop_down_ratio, extend_ratio, frame_shape):
    # move crop area downward by ratio of the bbox
    if crop_down_ratio > 0:
        y1 = int(y1 + (y2 - y1) * crop_down_ratio)
        y2 = int(y2 + (y2 - y1) * crop_down_ratio)
    
    # extend the crop area by extend_ratio
    if extend_ratio > 0:
        width = x2 - x1
        height = y2 - y1
        x1 = max(0, x1 - int(width * extend_ratio))
        y1 = max(0, y1 - int(height * extend_ratio))
        x2 = min(frame_shape[1], x2 + int(width * extend_ratio))
        y2 = min(frame_shape[0], y2 + int(height * extend_ratio))
    
    return x1, y1, x2, y2

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
            frames.append(frame)
        
        if not frames:
            break

        if args.resize:
            resized_fb = []
            for frame in frames:
                h, w = frame.shape[:2]
                if h > w:
                    new_h = args.resize_to
                    new_w = int((w / h) * args.resize_to)
                else:
                    new_w = args.resize_to
                    new_h = int((h / w) * args.resize_to)
                resized_frame = cv2.resize(frame, (new_w, new_h))
                resized_fb.append(resized_frame)
            preds = fa[gpu_id].get_detections_for_batch(np.asarray(resized_fb))

            for j, f in enumerate(preds):
                i += 1
                if f is None:
                    continue

                scale_factor = args.resize_to / max(frames[j].shape[:2])
                x1, y1, x2, y2 = [int(coord / scale_factor) for coord in f]

                x1, y1, x2, y2 = adjust_bounding_box(
                    x1, y1, x2, y2, args.crop_down_ratio, args.extend_ratio, frames[j].shape)

                cv2.imwrite(os.path.join(fulldir, '{}.jpg'.format(i)), frames[j][y1:y2, x1:x2])
        else:
            preds = fa[gpu_id].get_detections_for_batch(np.asarray(frames))

            for j, f in enumerate(preds):
                i += 1
                if f is None:
                    continue

                x1, y1, x2, y2 = adjust_bounding_box(
                    f[0], f[1], f[2], f[3], args.crop_down_ratio, args.extend_ratio, frames[j].shape)

                cv2.imwrite(os.path.join(fulldir, '{}.jpg'.format(i)), frames[j][y1:y2, x1:x2])

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

if __name__ == '__main__':
	main(args)


