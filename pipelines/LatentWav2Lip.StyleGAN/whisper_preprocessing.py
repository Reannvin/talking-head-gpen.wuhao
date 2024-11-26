import os
import librosa
import torch
from whisper.audio2feature import Audio2Feature
from tqdm import tqdm

def get_videos_list(root_dir):
    videos_list = []
    for person_id in sorted(os.listdir(root_dir)):
        person_path = os.path.join(root_dir, person_id)
        for video_id in sorted(os.listdir(person_path)):
            video_path = os.path.join(person_path, video_id)
            videos_list.append((video_path, person_id, video_id))
    return videos_list

def audio_process(audio_path, model_path, fps, syncnet_T):
    audio_processor = Audio2Feature(model_path=model_path)
    whisper_feature = audio_processor.audio2feat(audio_path)
    offset = syncnet_T // 2
    right_offset = offset if syncnet_T % 2 != 0 else offset - 1
    whisper_chunks = audio_processor.feature2chunks(feature_array=whisper_feature, fps=fps, audio_feat_length=[offset, right_offset])
    save_path = os.path.join(os.path.dirname(audio_path), f"whisper.npy.{syncnet_T}")
    torch.save(whisper_chunks, save_path)

def main(root_dir, model_path, fps, syncnet_T):
    videos_list = get_videos_list(root_dir)
    audio_paths = [(os.path.join(video_path, 'audio.wav'), model_path, fps) for video_path, _, _ in videos_list]

    for audio_path, model_path, fps in tqdm(audio_paths):
        audio_process(audio_path, model_path, fps, syncnet_T)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process audio embeddings using Whisper model")
    parser.add_argument('--root_dir', type=str, required=True, help='Root directory of videos')
    parser.add_argument('--model_path', type=str, default='tiny', help='Path to Whisper model')
    parser.add_argument('--fps', type=int, default=25, help='Frames per second for audio processing')
    parser.add_argument('--syncnet_T', type=int, default=5, help='Syncnet T value')

    args = parser.parse_args()
    main(args.root_dir, args.model_path, args.fps, args.syncnet_T)
