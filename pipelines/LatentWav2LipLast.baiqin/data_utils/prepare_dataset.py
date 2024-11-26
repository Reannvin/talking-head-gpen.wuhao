import subprocess
import numpy as np
import cv2
import pickle
import os
import json
from mmpose.apis import inference_topdown, init_model
from mmpose.structures import merge_data_samples
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
from tqdm import tqdm
from glob import glob
import sys
import gc

from skimage import transform as trans
# # Get current directory
# current_dir = os.path.dirname(os.path.abspath(__file__))
# # Get parent directory
# parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
# Add parent directory to sys.path
parent_dir="/data/wangbaiqin/project/MuseTalk"
sys.path.append(parent_dir)
from musetalk.utils.face_detection import FaceAlignment, LandmarksType

# Placeholder for insufficient bbox
coord_placeholder = (0.0, 0.0, 0.0, 0.0)

def resize_landmark(landmark, w, h, new_w, new_h):
    w_ratio = new_w / w
    h_ratio = new_h / h
    landmark_norm = landmark / [w, h]
    landmark_resized = landmark_norm * [new_w, new_h]
    return landmark_resized

def read_imgs(img_list):
    frames = []
    print('Reading images...')
    for img_path in tqdm(img_list):
        frame = cv2.imread(img_path)
        frames.append(frame)
    return frames

def read_img_from_video(video_path):
    video_stream = cv2.VideoCapture(video_path)
    frames = []
    while True:
        still_reading, frame = video_stream.read()
        if not still_reading:
            video_stream.release()
            break
        frames.append(frame)
    return frames

def get_bbox_range(model, fa, video_path, upperbondrange=0):
    frames = read_img_from_video(video_path)
    batch_size_fa = 1
    batches = [frames[i:i + batch_size_fa] for i in range(0, len(frames), batch_size_fa)]
    coords_list = []
    landmarks = []
    if upperbondrange != 0:
        print('Get key_landmark and face bounding boxes with the bbox_shift:', upperbondrange)
    else:
        print('Get key_landmark and face bounding boxes with the default value')
    average_range_minus = []
    average_range_plus = []
    for fb in tqdm(batches):
        results = inference_topdown(model, np.asarray(fb)[0])
        results = merge_data_samples(results)
        keypoints = results.pred_instances.keypoints
        face_land_mark = keypoints[0][23:91]
        face_land_mark = face_land_mark.astype(np.int32)
        
        bbox = fa.get_detections_for_batch(np.asarray(fb))
        
        for j, f in enumerate(bbox):
            if f is None:
                coords_list += [coord_placeholder]
                continue
            
            half_face_coord = face_land_mark[29]
            range_minus = (face_land_mark[30] - face_land_mark[29])[1]
            range_plus = (face_land_mark[29] - face_land_mark[28])[1]
            average_range_minus.append(range_minus)
            average_range_plus.append(range_plus)
            if upperbondrange != 0:
                half_face_coord[1] = upperbondrange + half_face_coord[1]

    text_range = f"Total frame:「{len(frames)}」 Manually adjust range: [ -{int(sum(average_range_minus) / len(average_range_minus))}~{int(sum(average_range_plus) / len(average_range_plus))} ], the current value: {upperbondrange}"
    return text_range
def get_bbox_only(fa ,video_path):
    frames = read_img_from_video(video_path)
    batch_size_fa = 16
    batches = [frames[i:i + batch_size_fa] for i in range(0, len(frames), batch_size_fa)]
    coords_list = []
    for fb in tqdm(batches):
        
        bbox = fa.get_detections_for_batch(np.asarray(fb))
        
        # adjust the bounding box refer to landmark
        # Add the bounding box to a tuple and append it to the coordinates list
        for j, f in enumerate(bbox):
            if f is None: # no face in the image
                coords_list += [coord_placeholder]
                continue
  
            coords_list += [f]
            w,h = f[2]-f[0], f[3]-f[1]
            # if w < 50 or h < 50:
            #     print("Error bbox:", f)

def get_landmark_only(model,  video_path):
    frames = read_img_from_video(video_path)
    
    landmarks = []
    for frame in frames:
        results = inference_topdown(model, frame)
        results = merge_data_samples(results)
        keypoints = results.pred_instances.keypoints
        face_land_mark = keypoints[0][23:91]
        face_land_mark = face_land_mark.astype(np.int32)
        landmarks.append(face_land_mark)
    return landmarks


def get_arc_face(model, video_path,image_size=256,save_dir=None):
    #frames = read_img_from_video(video_path)
    video_stream = cv2.VideoCapture(video_path)
    frame_count=video_stream.get(cv2.CAP_PROP_FRAME_COUNT)
    arcface_dst = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32)
    assert image_size%112==0 or image_size%128==0
    if image_size%112==0:
        ratio = float(image_size)/112.0
        diff_x = 0
    else:
        ratio = float(image_size)/128.0
        diff_x = 8.0*ratio
    dst = arcface_dst * ratio
    dst[:,0] += diff_x
    tform = trans.SimilarityTransform()
    
    landmarks=[]
    arc_face=[]
    warp_metric=[]
    i=0
    while i<frame_count:
        ret,frame=video_stream.read()
        if not ret:
            break

    #for i,frame in enumerate(tqdm(frames)):
        results = inference_topdown(model, frame)
        results = merge_data_samples(results)
        keypoints = results.pred_instances.keypoints
        face_land_mark = keypoints[0][23:91]
        face_land_mark = face_land_mark.astype(np.int32)
        left_eye=np.mean(face_land_mark[36:42],axis=0)
        right_eye=np.mean(face_land_mark[42:48],axis=0)
        nose=face_land_mark[30]
        left_mouth=face_land_mark[48]
        right_mouth=face_land_mark[54]
        lmk=np.array([left_eye,right_eye,nose,left_mouth,right_mouth])
        tform.estimate(lmk, dst)
        M = tform.params[0:2, :]
        warped = cv2.warpAffine(frame, M, (image_size, image_size), borderValue = 0.0)
        #landmark_warp=cv2.transform(np.array([face_land_mark]),M)[0]
        
        if save_dir is not None:
            try:
                cv2.imwrite(os.path.join(save_dir, "{}.jpg".format(i)), warped)
                np.save(os.path.join(save_dir, "{}_lmd.npy".format(i)), face_land_mark)
                np.save(os.path.join(save_dir, "{}_warpMetric.npy".format(i)), M)
            except Exception as e:
                print(e)
                print(f"Video {save_dir} frame {i} error")
                continue
        else:
            arc_face.append(warped)
            warp_metric.append(M)
            landmarks.append(face_land_mark)
        i+=1
    return warp_metric,landmarks,arc_face



def get_landmark_and_bbox(model, fa, video_path, upperbondrange=0,save_dir=None):
    frames = read_img_from_video(video_path)
    # if(len(frames)*frames[0].shape[0]*frames[0].shape[1]>800*1024*1024):
    #     print(f"Video {video_path} size too large, skip")
    #     return
    batch_size_fa = 1
    batches = [frames[i:i + batch_size_fa] for i in range(0, len(frames), batch_size_fa)]
    coords_list = []
    landmarks = []
    # if upperbondrange != 0:
    #     print('Get key_landmark and face bounding boxes with the bbox_shift:', upperbondrange)
    # else:
    #     print('Get key_landmark and face bounding boxes with the default value')
    average_range_minus = []
    average_range_plus = []
    for fb in batches:
        results = inference_topdown(model, np.asarray(fb)[0])
        results = merge_data_samples(results)
        keypoints = results.pred_instances.keypoints
        face_land_mark = keypoints[0][23:91]
        face_land_mark = face_land_mark.astype(np.int32)
        try:
            bbox = fa.get_detections_for_batch(np.asarray(fb))
            
            for j, f in enumerate(bbox):
                if f is None:
                    coords_list += [coord_placeholder]
                    continue
                
                half_face_coord = face_land_mark[29]
                range_minus = (face_land_mark[30] - face_land_mark[29])[1]
                range_plus = (face_land_mark[29] - face_land_mark[28])[1]
                average_range_minus.append(range_minus)
                average_range_plus.append(range_plus)
                if upperbondrange != 0:
                    half_face_coord[1] = upperbondrange + half_face_coord[1]
                half_face_dist = np.max(face_land_mark[:, 1]) - half_face_coord[1]
                upper_bond = half_face_coord[1] - half_face_dist
                
                f_landmark = (np.min(face_land_mark[:, 0]), int(upper_bond), np.max(face_land_mark[:, 0]), np.max(face_land_mark[:, 1]))
                x1, y1, x2, y2 = f_landmark
                
                if y2 - y1 <= 50 or x2 - x1 <= 50 or x1 < 0:
                    coords_list += [f]
                    w, h = f[2] - f[0], f[3] - f[1]
                    print("Error bbox:", f)
                    
                else:
                    coords_list += [f_landmark]
                
        except Exception as e:
            print(e)
            print(f"Video {video_path} error")
            return coords_list, frames
    
    return coords_list, frames

def get_landmark_and_bbox_new(model, fa, video_path, upperbondrange=0,save_dir=None):
    video_stream = cv2.VideoCapture(video_path)
    frame_count=video_stream.get(cv2.CAP_PROP_FRAME_COUNT)
    if len(glob(save_dir+"/*.jpg"))>frame_count:
        print(f"Video {video_path} already processed, skip")
        return
    landmarks=[]
    average_range_minus = []
    average_range_plus = []
    coords_list = []
    for i in tqdm(range(int(frame_count)),desc="video:{}".format(video_path)):
        ret,frame=video_stream.read()
        if not ret:
            break
        if os.path.exists(os.path.join(save_dir, "{}.jpg".format(i))):
            continue
    #for i,frame in enumerate(tqdm(frames)):
        results = inference_topdown(model, frame)
        results = merge_data_samples(results)
        keypoints = results.pred_instances.keypoints
        face_land_mark = keypoints[0][23:91]
        face_land_mark = face_land_mark.astype(np.int32)
        bbox = fa.get_detections_for_batch(np.asarray(frame.reshape(1,frame.shape[0],frame.shape[1],frame.shape[2])))[0]

        half_face_coord = face_land_mark[29]
        range_minus = (face_land_mark[30] - face_land_mark[29])[1]
        range_plus = (face_land_mark[29] - face_land_mark[28])[1]
        average_range_minus.append(range_minus)
        average_range_plus.append(range_plus)
        if upperbondrange != 0:
            half_face_coord[1] = upperbondrange + half_face_coord[1]
        half_face_dist = np.max(face_land_mark[:, 1]) - half_face_coord[1]
        upper_bond = half_face_coord[1] - half_face_dist
        
        f_landmark = (np.min(face_land_mark[:, 0]), int(upper_bond), np.max(face_land_mark[:, 0]), np.max(face_land_mark[:, 1]))
        x1, y1, x2, y2 = f_landmark
        
        if y2 - y1 <= 50 or x2 - x1 <= 50 or x1 < 0:
            coords_list += [bbox]
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            print("Error bbox:", bbox)
            
        else:
            coords_list += [f_landmark]
        save_frame=frame[coords_list[-1][1]:coords_list[-1][3],coords_list[-1][0]:coords_list[-1][2]]
        if save_dir is not None:
            try:
                cv2.imwrite(os.path.join(save_dir, "{}.jpg".format(i)), save_frame)
                np.save(os.path.join(save_dir, "{}_lmd.npy".format(i)), face_land_mark)
            except Exception as e:
                print(e)
                print(f"Video {save_dir} frame {i} error")
                continue
        else:
            landmarks.append(face_land_mark)

        
    return coords_list, landmarks
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12357'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def process_HDTF_video(rank, world_size, data_root, save_dir, upperbondrange=0):
    # dist.init_process_group("nccl", rank=rank, world_size=world_size)
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    
    device = "cuda:{}".format(rank)
    
    # Initialize the mmpose model
    config_file = '/data/wangbaiqin/project/MuseTalk/musetalk/utils/dwpose/rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py'
    checkpoint_file = '/data/wangbaiqin/project/MuseTalk/models/dwpose/dw-ll_ucoco_384.pth'
    model = init_model(config_file, checkpoint_file, device=device)
   # model = DDP(model, device_ids=[rank])
    
    # Initialize the face detection model
    fa = FaceAlignment(LandmarksType._2D, flip_input=False, device=device)
    
    os.makedirs(save_dir, exist_ok=True)
    video_list = os.listdir(data_root)
    video_list = [video for video in video_list if video.endswith('.mp4')]
    
    # Split the video list for this rank
    video_list = video_list[rank::world_size]
    pbar = tqdm(video_list)
    for video in pbar:
        pbar.set_description(f"Rank {rank} Processing video: {video}")
        pbar.update()
        # if video.startswith("video_bili_1"):
        #     continue
        try:
            video_path = os.path.join(data_root, video)
            save_path = os.path.join(save_dir, video.split(".")[0])
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            # else:
            #     img_num=len(os.listdir(save_path))
            #     if img_num>200:
            #        # print(f"Video {video} already processed, skip")
            #         continue
          #  print(f"Processing video: {video}")
            coords_list, landmarks = get_landmark_and_bbox_new(model, fa, video_path, upperbondrange,save_dir=save_path)
            #coords_list, frames = get_bbox_only(fa, video_path)
            # for i, (bbox, frame) in enumerate(zip(coords_list, frames)):
            #     try:
            #         if bbox == coord_placeholder:
            #             continue
            #         x1, y1, x2, y2 = bbox
            #         crop_frame = frame[y1:y2, x1:x2]
            #         cv2.imwrite(os.path.join(save_path, "{}.jpg".format(i)), crop_frame)
            #     except Exception as e:
            #         print(e)
            #         print(f"Video {video} frame {i} error")
            #         continue
        except Exception as e:
            print(e)
            print(f"Video {video} error")
        # del coords_list
        # del landmarks


    dist.destroy_process_group()

def process_3300_video(rank, world_size, data_root, save_dir, upperbondrange=0):
    # dist.init_process_group("nccl", rank=rank, world_size=world_size)
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    
    device = "cuda:{}".format(rank)
    
    # Initialize the mmpose model
    config_file = './musetalk/utils/dwpose/rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py'
    checkpoint_file = './models/dwpose/dw-ll_ucoco_384.pth'
    model = init_model(config_file, checkpoint_file, device=device)
   # model = DDP(model, device_ids=[rank])
    
    # Initialize the face detection model
    fa = FaceAlignment(LandmarksType._2D, flip_input=False, device=device)
    
    os.makedirs(save_dir, exist_ok=True)
    collect_list = os.listdir(data_root)
    video_list=[]
    for collect in collect_list:
        tmp_list= os.listdir(os.path.join(data_root,collect))
        for video in tmp_list:
            if video.endswith('.mp4'):
                video_list.append(os.path.join(collect,video))
    #video_list = [video for video in video_list if video.endswith('.mp4')]
    
    # Split the video list for this rank
    video_list = video_list[rank::world_size]
    
    for video in video_list:
        try:
            video_path = os.path.join(data_root, video)
            save_path = os.path.join(save_dir, video.split(".")[0])
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            else:
                img_num=len(os.listdir(save_path))
                if img_num>200:
                    print(f"Video {video} already processed, skip")
                    continue
            print(f"Processing video: {video}")
            coords_list, frames = get_landmark_and_bbox(model, fa, video_path, upperbondrange)
            #coords_list, frames = get_bbox_only(model, fa, video_path, upperbondrange)
            for i, (bbox, frame) in enumerate(zip(coords_list, frames)):
                try:
                    if bbox == coord_placeholder:
                        continue
                    x1, y1, x2, y2 = bbox
                    crop_frame = frame[y1:y2, x1:x2]
                    cv2.imwrite(os.path.join(save_path, "{}.jpg".format(i)), crop_frame)
                except Exception as e:
                    print(e)
                    print(f"Video {video} frame {i} error")
                    continue
        except Exception as e:
            print(e)
            print(f"Video {video} error")
            continue

    dist.destroy_process_group()

def get_bbox_and_landmark_from_image(image_path, model, fa,upperbondrange=0):
    frame = cv2.imread(image_path)
    results = inference_topdown(model, frame)
    results = merge_data_samples(results)
    keypoints = results.pred_instances.keypoints
    face_land_mark = keypoints[0][23:91]
    face_land_mark = face_land_mark.astype(np.int32)
    bbox = fa.get_detections_for_batch(np.asarray(frame.reshape(1,frame.shape[0],frame.shape[1],frame.shape[2])))[0]
    if bbox is None:
        bbox=coord_placeholder
    half_face_coord = face_land_mark[29]
    range_minus = (face_land_mark[30] - face_land_mark[29])[1]
    range_plus = (face_land_mark[29] - face_land_mark[28])[1]
    if upperbondrange != 0:
        half_face_coord[1] = upperbondrange + half_face_coord[1]
    half_face_dist = np.max(face_land_mark[:, 1]) - half_face_coord[1]
    upper_bond = half_face_coord[1] - half_face_dist

    f_landmark = (np.min(face_land_mark[:, 0]), int(upper_bond), np.max(face_land_mark[:, 0]), np.max(face_land_mark[:, 1]))
    x1, y1, x2, y2 = f_landmark
    if y2 - y1 <= 50 or x2 - x1 <= 50 or x1 < 0:
        x1, y1, x2, y2 = bbox
    if bbox!=coord_placeholder and bbox[1]<half_face_coord[1]:
        y1=bbox[1]
   # x1, y1, x2, y2 = f_landmark
    crop_frame=frame[y1:y2, x1:x2]
    face_land_mark-=np.array([x1,y1])
    return crop_frame, face_land_mark

def get_bbox_and_landmark_from_video(video_path, model, fa,upperbondrange=0):
    frames = read_img_from_video(video_path)
    batch_size_fa = 1
    batches = [frames[i:i + batch_size_fa] for i in range(0, len(frames), batch_size_fa)]
    crop_frames = []
    landmarks=[]
    for fb in tqdm(batches):
        try:
            results = inference_topdown(model, np.asarray(fb)[0])
            results = merge_data_samples(results)
            keypoints = results.pred_instances.keypoints
            face_land_mark = keypoints[0][23:91]
            face_land_mark = face_land_mark.astype(np.int32)
            bbox = fa.get_detections_for_batch(np.asarray(fb))[0]
            if bbox is None:
                bbox=coord_placeholder
            half_face_coord = face_land_mark[29]
        
            if upperbondrange != 0:
                half_face_coord[1] = upperbondrange + half_face_coord[1]
            half_face_dist = np.max(face_land_mark[:, 1]) - half_face_coord[1]
            upper_bond = half_face_coord[1] - half_face_dist

            f_landmark = (np.min(face_land_mark[:, 0]), int(upper_bond), np.max(face_land_mark[:, 0]), np.max(face_land_mark[:, 1]))
            x1, y1, x2, y2 = f_landmark
            if y2 - y1 <= 50 or x2 - x1 <= 50 or x1 < 0:
                x1, y1, x2, y2 = bbox
            if bbox!=coord_placeholder and bbox[1]<half_face_coord[1]:
                y1=bbox[1]
        # x1, y1, x2, y2 = f_landmark
            crop_frame=fb[0][y1:y2, x1:x2]
            face_land_mark-=np.array([x1,y1])
            crop_frames.append(crop_frame)
            landmarks.append(face_land_mark)
        except Exception as e:
            print(e)
            print(f"Video {video_path} error")
            continue
    return crop_frames, landmarks


def process_img_landmark(rank, world_size, data_root, save_dir,upperbondrange=0):
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    
    device = "cuda:{}".format(rank)
    img_list = os.listdir(data_root)
    img_list = [img for img in img_list if img.endswith('.jpg') or img.endswith('.png')]
    img_list = img_list[rank::world_size]
     # Initialize the mmpose model
    config_file = '/data/wangbaiqin/project/MuseTalk/musetalk/utils/dwpose/rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py'
    checkpoint_file = '/data/wangbaiqin/project/MuseTalk/models/dwpose/dw-ll_ucoco_384.pth'
    model = init_model(config_file, checkpoint_file, device=device)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
   # model = DDP(model, device_ids=[rank])
    # Initialize the face detection model
    fa = FaceAlignment(LandmarksType._2D, flip_input=False, device=device)
    for img in tqdm(img_list):
        img_path = os.path.join(data_root, img)
        lmk_save_path = os.path.join(save_dir, img.split(".")[0]+"_lmd.npy")
        img_save_path = os.path.join(save_dir, img.split(".")[0]+".jpg")
        if os.path.exists(lmk_save_path) and os.path.exists(img_save_path):
            continue
        try:
            crop_img,landmark=get_bbox_and_landmark_from_image(img_path, model, fa,upperbondrange)
            if crop_img is None:
                continue
            cv2.imwrite(img_save_path, crop_img)
            
            np.save(lmk_save_path, landmark)
        
        except Exception as e:
            print(e)
            print(f"Image {img} error")
            continue

def process_video_landmark(rank, world_size, data_root, save_dir, upperbondrange=0):
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    
    device = "cuda:{}".format(rank)
    # Initialize the mmpose model
    config_file = '/data/wangbaiqin/project/MuseTalk/musetalk/utils/dwpose/rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py'
    checkpoint_file = '/data/wangbaiqin/project/MuseTalk/models/dwpose/dw-ll_ucoco_384.pth'
    model = init_model(config_file, checkpoint_file, device=device)
    # model = DDP(model, device_ids=[rank])
    # Initialize the face detection model
    fa = FaceAlignment(LandmarksType._2D, flip_input=False, device=device)
    
    os.makedirs(save_dir, exist_ok=True)
    video_list = os.listdir(data_root)
    video_list = [video for video in video_list if video.endswith('.mp4')]
    
    # Split the video list for this rank
    video_list = video_list[rank::world_size]
    pbar = tqdm(video_list)
    for video in pbar:
        pbar.set_description(f"Rank {rank} Processing video: {video}")
        pbar.update()
        # if video.startswith("video_bili_1"):
        #     continue
       # try:
        video_path = os.path.join(data_root, video)
        save_path = os.path.join(save_dir, video.split(".")[0])
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        print(f"Processing video: {video}")
        crop_frames,landmarks = get_bbox_and_landmark_from_video(video_path,model, fa,  upperbondrange)
        #coords_list, frames = get_bbox_only(model, fa, video_path, upperbondrange)
        for i, (frame, landmark) in enumerate(zip(crop_frames, landmarks)):
            try:
                cv2.imwrite(os.path.join(save_path, "{}.jpg".format(i)), frame)
                np.save(os.path.join(save_path, "{}_lmd.npy".format(i)), landmark)
            except Exception:
                print(f"Video {video} frame {i} error")
                continue
        # except Exception as e:
        #     print(e)
        #     print(f"Video {video} error")
        #     continue




            
def process_HDTF_landmark_arcface(rank, world_size, data_root, save_dir,upperbondrange=0):
    # dist.init_process_group("nccl", rank=rank, world_size=world_size)
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    
    device = "cuda:{}".format(rank)
    
    # Initialize the mmpose model
    config_file = '/data/wangbaiqin/project/MuseTalk/musetalk/utils/dwpose/rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py'
    checkpoint_file = '/data/wangbaiqin/project/MuseTalk/models/dwpose/dw-ll_ucoco_384.pth'
    model = init_model(config_file, checkpoint_file, device=device)
   # model = DDP(model, device_ids=[rank])
    
    os.makedirs(save_dir, exist_ok=True)
    video_list = os.listdir(data_root)
    video_list = [video for video in video_list if video.endswith('.mp4')]
    
    # Split the video list for this rank
    video_list = video_list[rank::world_size]
    pbar = tqdm(video_list)
    for video in pbar:
        pbar.set_description(f"Rank {rank} Processing video: {video}")
        pbar.update()
        # if video.startswith("video_bili_1"):
        #     continue
        try:
            video_path = os.path.join(data_root, video)
            save_path = os.path.join(save_dir, video.split(".")[0])
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            else:
                img_num=len([image for image in os.listdir(save_path) if image.endswith('.jpg')])
                if img_num>200:
                   # print(f"Video {video} already processed, skip")
                    continue
            get_arc_face(model, video_path,save_dir=save_path)
          #  print(f"Processing video: {video}")
            #coords_list, frames = get_landmark_and_bbox(model, fa, video_path, upperbondrange)
            #coords_list, frames = get_bbox_only(fa, video_path)
            # metric_list,landmark_list,arc_faces = get_arc_face(model, video_path,save_dir=save_path)
            # for i, (M,landmark, frame) in enumerate(zip(metric_list,landmark_list, arc_faces)):
            #     try:
            #         cv2.imwrite(os.path.join(save_path, "{}.jpg".format(i)), frame)
            #         np.save(os.path.join(save_path, "{}_lmd.npy".format(i)), landmark)
            #         np.save(os.path.join(save_path, "{}_warpMetric.npy".format(i)), M)
            #     except Exception as e:
            #         print(e)
            #         print(f"Video {video} frame {i} error")
            #         continue
        except Exception as e:
            print(e)
            print(f"Video {video} error")
        # del landmark_list
        # del arc_faces
        # del metric_list
        gc.collect()
    #ffmpeg命令把文件夹下的图片合成视频
    #ffmpeg -f image2 -i %d.jpg -vcodec libx264 -b 800k test.mp4
    dist.destroy_process_group()







def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--video_path', type=str, help='Video path')
    parser.add_argument('--save_path', type=str, help='Save path')
    parser.add_argument('--upperbondrange', type=int, help='Upper bond range', default=0)
    parser.add_argument('--world_size', type=int, help='Number of GPUs to use',default=1)
    args = parser.parse_args()
    
    world_size = args.world_size
    mp.spawn(process_HDTF_video,
             args=(world_size, args.video_path, args.save_path, args.upperbondrange),
             nprocs=world_size,
             join=True)
    # mp.spawn(process_HDTF_landmark_arcface,
    #          args=(world_size, args.video_path, args.save_path),
    #          nprocs=world_size,
    #          join=True)

if __name__ == "__main__":
    main()
