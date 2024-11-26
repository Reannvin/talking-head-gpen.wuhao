import numpy as np
import cv2
import os
import sys
from tqdm import tqdm
import pandas as pd
from LPIPSmodels import util
import LPIPSmodels.dist_model as dm
import torch, face_detection
import torch
import argparse


def video_to_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

# List relevant video files in a directory
def list_videos_in_dir(directory):
    return [os.path.join(directory, f) for f in sorted(os.listdir(directory)) if f.endswith(('.mp4', '.avi', '.mov'))]

# PSNR calculation function
def psnr(img_true, img_pred):
    rms_error = np.sqrt(((img_true - img_pred) ** 2).mean())
    return 20 * np.log10(255.0 / rms_error)


def get_smoothened_boxes(boxes, T):
    for i in range(len(boxes)):
        if i + T > len(boxes):
            window = boxes[len(boxes) - T:]
        else:
            window = boxes[i : i + T]
        boxes[i] = np.mean(window, axis=0)
    return boxes

def get_extended_bbox(detector, images, leftright_scale=0.1, topbottom_scale=0.1):
    images = np.asarray(images)
    preds = detector.get_detections_for_batch(images)
    bbox_list = []
    for j, bbox in enumerate(preds):
        if bbox is None:
            continue
        try:
            x_min, y_min, x_max, y_max = bbox

            w = x_max - x_min
            h = y_max - y_min

            x_min = max(int(x_min - leftright_scale * w), 0)
            y_min = max(int(y_min - topbottom_scale * h), 0)
            x_max = min(int(x_max + leftright_scale * w), images[j].shape[1])
            y_max = min(int(y_max + topbottom_scale * h), images[j].shape[0])

            bbox_list.append((x_min, y_min, x_max, y_max))
        except Exception as e:
            bbox_list.append((0, 0, 0, 0))
    return bbox_list

def get_crop_down_bbox(detector, images, crop_down):
    images = np.asarray(images)
    preds = detector.get_detections_for_batch(images)
    bbox_list = []
    for j, bbox in enumerate(preds):
        if bbox is None:
            continue
        try:
            x_min, y_min, x_max, y_max = bbox

            w = x_max - x_min
            h = y_max - y_min

            y_min = min(int(y_min + crop_down * h), images[j].shape[0])
            y_max = min(int(y_max + crop_down * h), images[j].shape[0])

            bbox_list.append((x_min, y_min, x_max, y_max))
        except Exception as e:
            bbox_list.append((0, 0, 0, 0))
    return bbox_list

def face_detect(images, args, device):
    try:
        detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, 
                                                flip_input=False, device=device)
    except Exception as e:
        detector = face_detection.FaceAlignment(face_detection.LandmarksType.TWO_D, 
                                                flip_input=False, device=device)
    
    batch_size = args.face_det_batch_size
 
    resized_images = []
    if args.resize:
        for image in images:
            h, w = image.shape[:2]
            if h > w:
                new_h = args.resize_to
                new_w = int((w / h) * args.resize_to)
            else:
                new_w = args.resize_to
                new_h = int((h / w) * args.resize_to)
            resized_image = cv2.resize(image, (new_w, new_h))
            resized_images.append(resized_image)
    else:
        resized_images = images

    while 1:
        predictions = []
        try:
            if args.using_extended_bbox:
                for i in range(0, len(resized_images), batch_size):                    
                    predictions.extend(get_extended_bbox(detector, resized_images[i: i + batch_size], leftright_scale=args.leftright_scale, topbottom_scale=args.topbottom_scale))
            elif args.crop_down > 0:
                for i in range(0, len(resized_images), batch_size):                    
                    predictions.extend(get_crop_down_bbox(detector, resized_images[i: i + batch_size], crop_down=args.crop_down))
            else:        
                for i in range(0, len(resized_images), batch_size):
                    predictions.extend(detector.get_detections_for_batch(np.array(resized_images[i:i + batch_size])))
        except RuntimeError:
            if batch_size == 1: 
                raise RuntimeError('Image too big to run face detection on GPU. Please use the --resize_factor argument')
            batch_size //= 2
            print('Recovering from OOM error; New batch size: {}'.format(batch_size))
            continue
        break

    results = []
    pady1, pady2, padx1, padx2 = args.pads
    for rect, image in zip(predictions, images):
        if rect is None:
            cv2.imwrite(f'{args.temp_dir}/faulty_frame.jpg', image) # check this frame where the face was not detected.
            raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

        if args.resize:
            scale_factor = args.resize_to / max(image.shape[:2])
            rect = [int(coord / scale_factor) for coord in rect]  # Scale up the detection coordinates
        
        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)
        
        results.append([x1, y1, x2, y2])

    boxes = np.array(results)
    if not args.nosmooth: boxes = get_smoothened_boxes(boxes, T=5)
    results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

    del detector
    return results

def crop_8x8( img ):
    ori_h = img.shape[0]
    ori_w = img.shape[1]
    
    h = (ori_h//32) * 32
    w = (ori_w//32) * 32
    
    while(h > ori_h - 16):
        h = h - 32
    while(w > ori_w - 16):
        w = w - 32
    
    y = (ori_h - h) // 2
    x = (ori_w - w) // 2
    crop_img = img[y:y+h, x:x+w]
    return crop_img, y, x

def main():
    parser = argparse.ArgumentParser(description="Face Detection and Metrics Calculation")
    parser.add_argument('--fake', type=str, required=True, help='the path of results video directory')
    parser.add_argument('--real', type=str, required=True, help='the path of targets video directory')
    parser.add_argument('--face_det_batch_size', type=int, default=2, help='Batch size for face detection')
    parser.add_argument('--resize', action='store_true', help='Enables image resizing before processing')
    parser.add_argument('--resize_to', type=int, default=640, help='Resize images to this size if resizing is enabled')
    parser.add_argument('--using_extended_bbox', action='store_true', help='Use extended bounding box')
    parser.add_argument('--leftright_scale', type=float, default=0, help='Scale factor for left/right extension of bbox')
    parser.add_argument('--topbottom_scale', type=float, default=0, help='Scale factor for top/bottom extension of bbox')
    parser.add_argument('--crop_down', type=float, default=0.1, help='Crop down scaling factor for bounding box')
    parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0], help='Padding (top, bottom, left, right). Please adjust to include chin at least')
    parser.add_argument('--temp_dir', type=str, default='.', help='Temporary directory to save faulty frames')
    parser.add_argument('--nosmooth', default=False, action='store_true', help='Prevent smoothing face detections over a short temporal window')
    parser.add_argument('--device', type=str, default="0", help="Device to run the model on")
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

    model = dm.DistModel()
    model.initialize(model='net-lin', net='alex', use_gpu=True)

    cutfr = 2
    pd_dict ={}
    source_video = args.fake
    target_video = args.real
    
    source_frames = video_to_frames(source_video)
    target_frames = video_to_frames(target_video)
    align_frames = min(len(source_frames), len(target_frames))
    source_frames = source_frames[:align_frames]
    target_frames = target_frames[:align_frames]
    if len(source_frames) != len(target_frames):
        print(f"Frame count mismatch in {source_video} and {target_video}.")
    else:
        cropped_target_frames = face_detect(target_frames, args, device)
        psnr_list = []
        lpips_list = []
        tOF_list = []
        tlpips_list = []

        max_width = 0
        max_height = 0
  
        for i in range(cutfr, len(cropped_target_frames) - cutfr):
            _, target_coords = cropped_target_frames[i]
            y1, y2, x1, x2 = target_coords
            width = x2 - x1
            height = y2 - y1

            if width > max_width:
                max_width = width
            if height > max_height:
                max_height = height

        for i in range(cutfr, len(cropped_target_frames) - cutfr):
            target_face, target_coords = cropped_target_frames[i]
            # face use 0.6 * image
            y1, y2, x1, x2 = target_coords
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            new_x1 = max(center_x - max_width // 2, 0)
            new_y1 = max(center_y - max_height // 2, 0) 
            new_x2 = new_x1 + max_width
            new_y2 = new_y1 + max_height
            middle = int(new_y1 + 0.4 * max_height) 
            target_img = target_frames[i][middle:new_y2, new_x1:new_x2, ::-1]
            output_img = source_frames[i][middle:new_y2, new_x1:new_x2, ::-1]
            output_grey = cv2.cvtColor(output_img, cv2.COLOR_RGB2GRAY)
            target_grey = cv2.cvtColor(target_img, cv2.COLOR_RGB2GRAY)
            print(cutfr)
            if i > cutfr:
                print(pre_out_grey.shape,pre_tar_grey.shape, output_grey.shape, target_grey.shape)
                target_OF = cv2.calcOpticalFlowFarneback(pre_tar_grey, target_grey, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                output_OF = cv2.calcOpticalFlowFarneback(pre_out_grey, output_grey, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                target_OF, ofy, ofx = crop_8x8(target_OF)
                output_OF, ofy, ofx = crop_8x8(output_OF)
                OF_diff = np.absolute(target_OF - output_OF)
                    
                OF_diff = np.sqrt(np.sum(OF_diff * OF_diff, axis = -1)) 
                tOF_list.append(OF_diff.mean())
               
            
            pre_out_grey = output_grey
            pre_tar_grey = target_grey

            target_img, ofy, ofx = crop_8x8(target_img)
            output_img, ofy, ofx = crop_8x8(output_img)
            psnr_list.append(psnr(target_img, output_img))
            img0 = util.im2tensor(target_img)
            img1 = util.im2tensor(output_img)
            dist01 = model.forward(img0, img1)
            lpips_list.append(dist01[0])

            if (i > cutfr): # temporal metrics
                dist0t = model.forward(pre_img0, img0)
                dist1t = model.forward(pre_img1, img1)
               
                dist01t = np.absolute(dist0t - dist1t) * 100.0 ##########!!!!!
                tlpips_list.append( dist01t[0] )
              
            pre_img0 = img0
            pre_img1 = img1

    pd_dict["PSNR"] = np.array(psnr_list, dtype=np.float32)
    pd_dict["LPIPS"] = np.array(lpips_list, dtype=np.float32)
    pd_dict["tOF"] = np.array(tOF_list, dtype=np.float32)
    pd_dict["tLP"] = np.array(tlpips_list, dtype=np.float32)
    result_message = ""
    for num_data in pd_dict.keys():
        result_message += "%s, max %02.4f, min %02.4f, avg %02.4f\n" % (
                    num_data, np.max(pd_dict[num_data]), np.min(pd_dict[num_data]), np.mean(pd_dict[num_data])
                )
    print(result_message)

if __name__ == '__main__':
    main()