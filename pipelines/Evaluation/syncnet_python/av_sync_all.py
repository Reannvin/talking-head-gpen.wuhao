import argparse
import os
import logging
from moviepy.editor import VideoFileClip, concatenate_videoclips, TextClip
import yaml
import shutil
import subprocess
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def sync_video(input_path, offset, output_path=None, fps=25):
    try:
        logging.info(f"Processing video: {input_path} with offset: {offset}")
        if output_path is None:
            filename = os.path.basename(input_path)
            base, ext = os.path.splitext(filename)
            output_path = f"{base}_synced{ext}"

        video = VideoFileClip(input_path)
        offset_seconds = offset / fps

        if offset_seconds < 0:
            first_frame = video.subclip(0, 1.0 / fps)
            padding_clip = first_frame.set_duration(-offset_seconds)
            synced_video = concatenate_videoclips([padding_clip, video])
        elif offset_seconds > 0:
            synced_video = video.subclip(offset_seconds)
        else:
            synced_video = video

        synced_video = synced_video.set_audio(video.audio)
        synced_video.write_videofile(output_path, codec="libx264", audio_codec="aac")
        
    except Exception as e:
        logging.error(f"Failed to process video {input_path}: {e}")

def create_labeled_video(original_video_path, synced_video_path, final_output_path, offset):
    original_label_command = [
        'ffmpeg',  '-y', '-i', original_video_path, '-vf',
        f"drawtext=text='Original':fontcolor=white:fontsize=70:x=(w-text_w)/2:y=50",
        '-c:a', 'copy', 'original_labeled.mp4'
    ]
    
    synced_label_command = [
        'ffmpeg', '-y', '-i', synced_video_path, '-vf',
        f"drawtext=text='Synced_{offset}':fontcolor=white:fontsize=70:x=(w-text_w)/2:y=50",
        '-c:a', 'copy', 'synced_labeled.mp4'
    ]
    
    concatenate_command = [
        'ffmpeg', '-y', '-i', 'original_labeled.mp4', '-i', 'synced_labeled.mp4', '-filter_complex',
        '[0:v][1:v]hstack=inputs=2[outv]', '-map', '[outv]', '-map', '0:a',
        '-c:v', 'libx264', '-strict', '-2', final_output_path
    ]
    
    subprocess.run(original_label_command)
    subprocess.run(synced_label_command)
    subprocess.run(concatenate_command)

def process_text_files(dataset_config, base_output_directory, fps=25, threshold=4):
    for data in dataset_config['datasets']:
        name = data['name']
        text_file_path = os.path.join(f'./filelists_{name}', data['split'] +'.txt')
        if not os.path.isfile(text_file_path):
            logging.warning(f"Text file {text_file_path} does not exist, skipping...")
            continue
        with open(text_file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    logging.warning(f"Skipping invalid line in file {text_file_path}: {line.strip()}")
                    continue

                dataset_name = data['name']
                video_id = parts[1]
                offset = float(parts[4])
                if abs(offset) >= threshold:
                    video_filename = f"{video_id}.mp4"
                    input_video_path = os.path.join(data['path'], video_filename)
                    synced_output_path = os.path.join(base_output_directory, dataset_name, f"{video_id}_synced_ori_{offset}.mp4")
                    final_output_path = os.path.join(base_output_directory, dataset_name, f"{video_id}.mp4")
                    if not os.path.isfile(input_video_path):
                        logging.warning(f"Input video file {input_video_path} does not exist, skipping...")
                        continue
                    os.makedirs(os.path.dirname(synced_output_path), exist_ok=True)
                   
                    sync_video(input_video_path, offset, synced_output_path, fps)
                    create_labeled_video(input_video_path, synced_output_path, final_output_path, offset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Batch process videos based on offset values from text files.')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file.')
    parser.add_argument('--base_output_directory', type=str, required=True, help='Base directory where output videos will be stored.')
    parser.add_argument('--fps', type=int, default=25, help='Frames per second for video processing (default is 25).')
    parser.add_argument('--threshold', type=float, default=5, help='Offset threshold for processing (default is 4 frames).')

    args = parser.parse_args()

    with open(args.config, 'r') as file:
        dataset_config = yaml.safe_load(file)
    process_text_files(dataset_config, args.base_output_directory, args.fps, args.threshold)
