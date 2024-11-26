
from yoloface.face_detector import YoloDetector
from PIL import Image
import logging
def Yolo(video_path, gpu_id, frames, output_dir):
    model = YoloDetector(target_size=None, device=f"cuda:{gpu_id}", min_face=90)
    crop_images = []
    for i, frame in enumerate(frames):
        bboxes, points = model.predict(frame)
        if bboxes:
            x1, y1, x2, y2 = bboxes[0][0]
            x1, x2 = max(0, x1), min(frame.shape[1], x2)
            y1, y2 = max(0, y1), min(frame.shape[0], y2)
            crop_img = frame[y1:y2, x1:x2]
            crop_images.append(crop_img)
            img = Image.fromarray(crop_img)
            img.save(f"{output_dir}/{i}.jpg")
        else:
            logging.warning(f"未检测到脸部框架在帧 {i}")
    logging.info("YOLO 检测完成")
