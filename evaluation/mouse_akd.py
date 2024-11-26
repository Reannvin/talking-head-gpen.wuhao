import cv2
import mediapipe as mp
import numpy as np
import argparse
import torch
from tqdm import tqdm
from mmpose.apis import inference_topdown, init_model
from mmpose.structures import merge_data_samples

# Mediapipe face mesh constants
mp_face_mesh = mp.solutions.face_mesh
MOUTH_KEYPOINTS = list(range(61, 68)) + list(range(78, 88)) + list(range(95, 106)) + list(range(146, 164)) + list(range(178, 194))

def extract_keypoints_mediapipe(image, device):
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            keypoints = [(landmark.x * image.shape[1], landmark.y * image.shape[0]) for i, landmark in enumerate(face_landmarks.landmark) if i in MOUTH_KEYPOINTS]

            keypoints = torch.tensor(keypoints).to(device)
            return keypoints
    return None

def calculate_akd(keypoints1, keypoints2):
    if keypoints1 is None or keypoints2 is None:
        return None
    assert keypoints1.shape == keypoints2.shape, "number of keypoints are not equal"
    distances = torch.norm(keypoints1 - keypoints2, dim=1)
    return distances.mean().item()

def process_video_mediapipe(video_path, device):
    cap = cv2.VideoCapture(video_path)
    akd_values = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        keypoints = extract_keypoints_mediapipe(frame, device)
        if keypoints is not None:
            akd_values.append(keypoints)

    cap.release()
    return akd_values

def calculate_average_akd_mediapipe(video_path1, video_path2, device):
    akd_values1 = process_video_mediapipe(video_path1, device)
    akd_values2 = process_video_mediapipe(video_path2, device)
    assert len(akd_values1) == len(akd_values2), "frame length is not equal"

    akd_distances = []
    for kp1, kp2 in zip(akd_values1, akd_values2):
        akd = calculate_akd(kp1, kp2)
        if akd is not None:
            akd_distances.append(akd)

    return np.mean(akd_distances) if akd_distances else None

def get_bbox_by_landmark(model, frames):
    batch_size_fa = 1
    batches = [frames[i:i + batch_size_fa] for i in range(0, len(frames), batch_size_fa)]
    landmarks = []
    for fb in tqdm(batches):
        results = inference_topdown(model, np.asarray(fb)[0])
        results = merge_data_samples(results)
        keypoints = results.pred_instances.keypoints
        face_land_mark = keypoints[0][23:91]
        face_land_mark = face_land_mark.astype(np.int32)
        landmarks.append(face_land_mark)
    return landmarks

def extract_keypoints_mmpose(video_path, model):
    video_stream = cv2.VideoCapture(video_path)
    frames = []
    while True:
        still_reading, frame = video_stream.read()
        if not still_reading:
            video_stream.release()
            break
        frames.append(frame)

    return get_bbox_by_landmark(model, frames)

def calculate_average_distance(kps1, kps2):
    distances = []
    for kp1, kp2 in zip(kps1, kps2):
        # Assuming kp1 and kp2 are arrays of shape (n, 2)
        kp1_mouth = kp1[48:68]
        kp2_mouth = kp2[48:68]
        distance = np.linalg.norm(kp1_mouth - kp2_mouth, axis=1)
        distances.append(np.mean(distance))

    return np.mean(distances)

def main(args):
    device = f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    if args.method == 'mediapipe':
        video1 = args.video1
        video2 = args.video2
        average_akd = calculate_average_akd_mediapipe(video1, video2, device)
        if average_akd is not None:
            print(f"AKD : {average_akd}")
        else:
            print("Error: perhaps the length of frames are not equal")
    elif args.method == 'mmpose':
        config_file = args.config
        checkpoint_file = args.checkpoint
        model = init_model(config_file, checkpoint_file, device=device)
        generated_keypoints = extract_keypoints_mmpose(args.video1, model)
        real_keypoints = extract_keypoints_mmpose(args.video2, model)
        assert len(generated_keypoints) == len(real_keypoints), "The two videos should have the same number of frames for comparison."
        avg_distance = calculate_average_distance(generated_keypoints, real_keypoints)
        print(f'Average key points distance: {avg_distance:.2f}')
    else:
        print("Invalid method. Please choose 'mediapipe' or 'mmpose'.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculate average key points distance between two videos using AKD and other methods.")
    parser.add_argument('--video1', default='/data/fanshen/workspace/preprocessing/example1.mp4', type=str, help="video path 1")
    parser.add_argument('--video2', default='/data/fanshen/workspace/preprocessing/example1.mp4', type=str, help="video path 2")
    parser.add_argument('--method', choices=['mediapipe', 'mmpose'], required=True, help="method to use: 'mediapipe' or 'mmpose'")
    parser.add_argument('--gpu_id', type=int, default=5, help="GPU ID")
    parser.add_argument('--config', default='./models/dwpose/rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py', help='config file for the model (required for mmpose)')
    parser.add_argument('--checkpoint', default='./models/dwpose/dw-ll_ucoco_384.pth', help='checkpoint file for the  model (required for mmpose)')
    args = parser.parse_args()
    main(args)
