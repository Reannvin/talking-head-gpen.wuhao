import os
import torch
import librosa
import argparse
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

def get_videos_list(root_dir):
    videos_list = []
    for person_id in sorted(os.listdir(root_dir)):
        person_path = os.path.join(root_dir, person_id)
        for video_id in sorted(os.listdir(person_path)):
            video_path = os.path.join(person_path, video_id)
            videos_list.append((video_path, person_id, video_id))
    return videos_list

def load_audio(audio_file, sampling_rate=16000):
    audio, sampling_rate = librosa.load(audio_file, sr=sampling_rate)
    return audio, sampling_rate

def process_audio(audio, sampling_rate, processor, model, batch_size=1600000, min_length=16000):  # min_length 应根据模型要求调整
    # 分割音频为较小批次
    num_samples = audio.shape[0]
    embeddings = []

    # 处理每一小批音频数据
    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        batch_audio = audio[start:end]

        # 如果批次太小，增加填充
        if end - start < min_length:
            padding = min_length - (end - start)
            batch_audio = torch.nn.functional.pad(batch_audio, (0, padding), mode='constant', value=0)

        inputs = processor(batch_audio, return_tensors="pt", sampling_rate=sampling_rate)

        with torch.no_grad():
            outputs = model(**inputs)
            embeddings.append(outputs.last_hidden_state)

    # 将所有批次的嵌入向量连接起来
    embeddings = torch.cat(embeddings, dim=1)
    return embeddings

def process_videos(root_dir, device):
    processor = Wav2Vec2FeatureExtractor.from_pretrained("TencentGameMate/chinese-wav2vec2-base")
    model = Wav2Vec2Model.from_pretrained("TencentGameMate/chinese-wav2vec2-base").to(device)
    model = torch.nn.DataParallel(model)

    videos_list = get_videos_list(root_dir)
    for video_path, person_id, video_id in tqdm(videos_list):
        audio_file = os.path.join(video_path, "audio.wav")
        embeddings_file = os.path.join(video_path, "wav2vec2.pt")

        if os.path.exists(audio_file):
            audio, sampling_rate = load_audio(audio_file)
            audio = torch.tensor(audio).to(device)
            embeddings = process_audio(audio, sampling_rate, processor, model)
            torch.save(embeddings, embeddings_file)
        else:
            print(f"Audio file not found for {video_path}")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Convert audio files in a dataset to Wav2Vec2 embeddings")
    parser.add_argument("--data_root", type=str, required=True, help="Root directory of the dataset containing video folders")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use (cuda or cpu)")
    return parser.parse_args()

# Example usage
if __name__ == "__main__":
    args = parse_arguments()
    process_videos(args.data_root, args.device)
