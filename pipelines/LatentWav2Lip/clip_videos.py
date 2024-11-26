import subprocess
import os
import argparse

def split_video(source_dir, dest_dir, segment_length=10):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    for filename in os.listdir(source_dir):
        if filename.endswith(".mp4"):
            file_path = os.path.join(source_dir, filename)
            video_name = os.path.splitext(filename)[0]
            video_dest_dir = os.path.join(dest_dir, video_name)
            if not os.path.exists(video_dest_dir):
                os.makedirs(video_dest_dir)
            cmd = [
                'ffmpeg',
                '-i', file_path,
                '-map', '0:v',  # Include video stream
                '-map', '0:a?', # Optionally include audio stream if present
                '-c', 'copy', # Copy video and audio without re-encoding
                '-avoid_negative_ts', 'make_zero', # Try to fix negative timestamps
                '-segment_time', str(segment_length),
                '-f', 'segment',
                '-reset_timestamps', '1',
                os.path.join(video_dest_dir, '%05d.mp4')  # Save files as video_name/00001.mp4
            ]
            subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser(description='Split videos into 10-second segments.')
    parser.add_argument('--source_dir', type=str, help='Directory containing the source mp4 files', required=True)
    parser.add_argument('--dest_dir', type=str, help='Directory where the segmented videos will be stored', required=True)
    args = parser.parse_args()
    split_video(args.source_dir, args.dest_dir)

if __name__ == '__main__':
    main()
