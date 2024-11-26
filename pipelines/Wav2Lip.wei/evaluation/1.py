import cv2
import os
from tqdm import tqdm

# 指定要遍历的文件夹路径
folder_path = 'test_fid_syn_crop96_wav2lip_gpus'
# 指定保存裁剪后图片的文件夹路径
output_folder = 'test_fid_syn_crop96_wav2lip_gpus_half'

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历文件夹中的所有文件
for filename in tqdm(os.listdir(folder_path)):
    # 检查文件扩展名是否为 .jpg
    if filename.lower().endswith('.jpg'):
        # 构建完整的文件路径
        image_path = os.path.join(folder_path, filename)
        # 读取图片
        image = cv2.imread(image_path)

        # 假设我们想要图片的下半部分，这里我们取高度的一半
        height, width = image.shape[:2]
        cropped_image = image[int(height/2):, 0:width]

        # 构建输出文件路径
        output_path = os.path.join(output_folder, filename)

        # 保存裁剪后的图片
        cv2.imwrite(output_path, cropped_image)

