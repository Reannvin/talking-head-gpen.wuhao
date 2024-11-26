import cv2
import numpy as np
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm
import argparse

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

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

def face_detect(images, device, crop_down=0.1, batch_size=8):
    import face_detection
    try:
        detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, 
                                                flip_input=False, device=device)
    except Exception as e:
        detector = face_detection.FaceAlignment(face_detection.LandmarksType.TWO_D, 
                                                flip_input=False, device=device)
    
    resized_images = [cv2.resize(image, (image.shape[1] // 4, image.shape[0] // 4)) for image in images]

    while True:
        predictions = []
        try:
            for i in tqdm(range(0, len(resized_images), batch_size)):                    
                predictions.extend(get_crop_down_bbox(detector, resized_images[i: i + batch_size], crop_down=crop_down))
        except RuntimeError:
            if batch_size == 1: 
                raise RuntimeError('Image too big to run face detection on GPU. Please use a smaller batch size.')
            batch_size //= 2
            print(f'Recovering from OOM error; New batch size: {batch_size}')
            continue
        break

    results = []
    for rect, image in zip(predictions, images):
        if rect is None:
            cv2.imwrite('temp/faulty_frame.jpg', image)
            raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

        rect = [coord * 4 for coord in rect]
        
        y1 = max(0, rect[1])
        y2 = min(image.shape[0], rect[3])
        x1 = max(0, rect[0])
        x2 = min(image.shape[1], rect[2])
        results.append([x1, y1, x2, y2])
    results = [image[y1:y2, x1:x2] for image, (x1, y1, x2, y2) in zip(images, results)]

    del detector
    return results

def read_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert BGR to RGB
            frames.append(frame)
        else:
            break
    cap.release()
    return frames

def preprocess_frames(frames, target_size=(256, 256)):
    resized_frames = [cv2.resize(frame, target_size) for frame in frames]
    frames_array = np.array(resized_frames).astype(np.float32) / 255.0
    frames_array = np.transpose(frames_array, (0, 3, 1, 2)) # Change to [N, C, H, W]
    frames_tensor = torch.tensor(frames_array).to(DEVICE)
    return frames_tensor

def calculate_fid(video1_frames, video2_frames):
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(DEVICE)
    
    n = min(len(video1_frames), len(video2_frames))
    video1_frames, video2_frames = video1_frames[:n], video2_frames[:n]
    
    video1_frames = preprocess_frames(video1_frames)
    video2_frames = preprocess_frames(video2_frames)
    
    fid.update(video1_frames, real=True)
    fid.update(video2_frames, real=False)
    return fid.compute().item()

def main(video1_path, video2_path, crop_down, batch_size):
    video1_frames = read_video_frames(video1_path)
    video2_frames = read_video_frames(video2_path)
  
    video1_frames = face_detect(video1_frames, DEVICE, crop_down=crop_down, batch_size=batch_size)
    video2_frames = face_detect(video2_frames, DEVICE, crop_down=crop_down, batch_size=batch_size)

    average_fid = calculate_fid(video1_frames, video2_frames)
    
    print(f"Average FID: {average_fid}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate FID between two videos.')
    parser.add_argument('--video1_path', type=str, help='Path to the first video')
    parser.add_argument('--video2_path', type=str, help='Path to the second video')
    parser.add_argument('--crop_down', type=float, default=0.1, help='Crop down factor for face detection, default is 0.1')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for face detection, default is 8')
    
    args = parser.parse_args()
    
    print(f'Using {DEVICE} for inference.')
    main(args.video1_path, args.video2_path, args.crop_down, args.batch_size)
