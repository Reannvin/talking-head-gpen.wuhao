import os
import cv2
import subprocess
from tqdm import tqdm

from yoloface.face_detector import YoloDetector

gpu_id=0
model = YoloDetector(target_size=None, device=f"cuda:{gpu_id}", min_face=90)

def get_face_bbox(input_video_path, target_file, gpu_id=0):
    cap = cv2.VideoCapture(input_video_path)
    flag, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    bboxes,points = model.predict(frame)
    x1, y1, x2, y2 = bboxes[0][0]

    cap.release()

    width = x2 - x1
    height = y2 - y1
    center_x = x1 + width // 2
    center_y = y1 + height // 2

    # Calculate new region dimensions (twice the bbox width and height)
    new_width = 2 * width
    new_height = 2 * height

    # Calculate new top-left corner ensuring it stays within the frame boundaries
    new_x1 = max(center_x - new_width // 2, 0)
    new_y1 = max(center_y - new_height // 2, 0)

    # Calculate new bottom-right corner ensuring it stays within the frame boundaries
    new_x2 = min(center_x + new_width // 2, frame.shape[1])
    new_y2 = min(center_y + new_height // 2, frame.shape[0])

    # Extract the region of interest
    face_region = frame[new_y1:new_y2, new_x1:new_x2]
    print(input_video_path, face_region.shape)

    # cv2.imshow("face_region", face_region)
    # cv2.waitKey()

    # out_width:out_height:x:y
    cmd = f"ffmpeg -i {input_video_path} -vf \"crop={new_width}:{new_height}:{new_x1}:{new_y1}\" -y {target_file}"
    try:
        # 使用 subprocess.run 并将 stdout 和 stderr 设置为 None
        process = subprocess.run(cmd, shell=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        error_msg = f"crop face 失败: {e.stderr}"
        print(error_msg)
        return False

if __name__ == "__main__":
    base_path = "D:/dataset/kehu_video/"
    target_path = "D:/dataset/kehu_video_face/"
    os.makedirs(target_path, exist_ok=True)
    for file in tqdm(os.listdir(base_path)):
        source_file = os.path.join(base_path, file)
        target_file = os.path.join(target_path, file[0:-4] + "_face.mp4")
        get_face_bbox(source_file, target_file)
