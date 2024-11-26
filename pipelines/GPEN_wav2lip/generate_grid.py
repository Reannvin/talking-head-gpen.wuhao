from PIL import Image
import os

def combine_images(input_folder, output_file):
    # 创建一个1024x1024的新图像
    new_image = Image.new('RGB', (1024, 1024))

    # 遍历16张图片
    for i in range(16):
        # 计算当前图片应该放置的位置
        row = i // 4
        col = i % 4

        # 打开图片
        img_path = os.path.join(input_folder, f'sample_{i}.png')
        img = Image.open(img_path)

        # 确保图片大小是256x256
        if img.size != (256, 256):
            img = img.resize((256, 256))

        # 将图片粘贴到新图像的正确位置
        new_image.paste(img, (col * 256, row * 256))

    # 保存新图像
    new_image.save(output_file)
    print(f"Combined image saved as {output_file}")

# 使用函数
input_folder = './training-run/202409011321/snapshots_070000'  # 替换为您的输入文件夹路径
output_file = './combined_image.png'  # 替换为您想要保存的输出文件路径

combine_images(input_folder, output_file)