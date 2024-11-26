import os
import numpy as np
import argparse
import torch

def get_videos_list(root_dir):
    videos_list = []
    for person_id in sorted(os.listdir(root_dir)):
        person_path = os.path.join(root_dir, person_id)
        # for video_id in sorted(os.listdir(person_path)):
            # video_path = os.path.join(person_path)
        videos_list.append((person_path, person_id))
    return videos_list

def split_whisper_files(data_root, syncnet_T):
    videos_list = get_videos_list(data_root)
    for video_path, person_id in videos_list:
        whisper_path = os.path.join(video_path, f'whisper.npy')
        if os.path.exists(whisper_path):
            whisper_list = torch.load(whisper_path)
            for i, array in enumerate(whisper_list):
                save_path = os.path.join(video_path, f'{i}.npy.')
                np.save(save_path, array)
            print(f'Split whisper.npy in {video_path} into {len(whisper_list)} files.')
        else:
            print(f'whisper.npy not found in {video_path}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split whisper.npy into separate files.')
    parser.add_argument('--data_root', type=str, required=True, help='Root directory of videos')
    parser.add_argument('--syncnet_T', type=int, default=5, help='Number of frames to extract from SyncNet')
    args = parser.parse_args()
    
    split_whisper_files(args.data_root, args.syncnet_T)
