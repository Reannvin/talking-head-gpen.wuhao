import os
import argparse
from PIL import Image
from tqdm import tqdm

def resize_image(input_path, output_path, max_size=100):
    with Image.open(input_path) as img:
        width, height = img.size
        if width > height:
            new_width = max_size
            new_height = int((max_size / width) * height)
        else:
            new_height = max_size
            new_width = int((max_size / height) * width)

        resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        resized_img.save(output_path)

def get_videos_list(root_dir):
    videos_list = []
    for person_id in sorted(os.listdir(root_dir)):
        person_path = os.path.join(root_dir, person_id)
        for video_id in sorted(os.listdir(person_path)):
            video_path = os.path.join(person_path, video_id)
            videos_list.append((video_path, person_id, video_id))
    return videos_list

def resize_dataset(root_dir, resized_dir, max_size=100):
    videos_list = get_videos_list(root_dir)
    for video_path, person_id, video_id in tqdm(videos_list):
        for img_filename in sorted([p for p in os.listdir(video_path) if p.endswith('.jpg')]):
            input_path = os.path.join(video_path, img_filename)
            output_path = os.path.join(resized_dir, person_id, video_id, img_filename)
            resize_image(input_path, output_path, max_size)

def main():
    parser = argparse.ArgumentParser(description="Resize images in a dataset to a maximum size.")
    parser.add_argument('--root_dir', type=str, required=True, help='Root directory of the dataset')
    parser.add_argument('--resized_dir', type=str, required=True, help='Directory to save the resized images')
    parser.add_argument('--max_size', type=int, default=100, help='Maximum size of the longest side of the image')

    args = parser.parse_args()

    resize_dataset(args.root_dir, args.resized_dir, args.max_size)

if __name__ == "__main__":
    main()
