from musetalk.whisper.audio2feature import Audio2Feature
from glob import glob
import os
from tqdm import tqdm
import numpy as np
import argparse
def audio_process(audio_path,save_path=None,fps=25):
    audio_processor = Audio2Feature(model_path="./models/whisper/tiny.pt")
    whisper_feature = audio_processor.audio2feat(audio_path)
    whisper_chunks = audio_processor.feature2chunks(feature_array=whisper_feature,fps=fps)
    # print("whisper_feature:",whisper_feature.shape)
    # print("whisper_chunks:",len(whisper_chunks))
    # print("whisper_chunks[0]:",whisper_chunks[0].shape)
    for i in range(len(whisper_chunks)):
        save_file=os.path.join(save_path,f"{i}.npy")
        np.save(save_file,whisper_chunks[i])
def audio_process_dirs(video_dir,audio_dir):
    os.makedirs(audio_dir,exist_ok=True)
    person_list=os.listdir(video_dir)
    for person in tqdm(person_list):
        audio_path=os.path.join(video_dir,person,"audio.wav")
        save_path=os.path.join(audio_dir,person)
        os.makedirs(save_path,exist_ok=True)
        audio_process(audio_path,save_path)
def audio_process_dirs_split(video_dir,audio_dir):
    os.makedirs(audio_dir,exist_ok=True)
    person_list=os.listdir(video_dir)
    for person in tqdm(person_list):
        save_path_parent=os.path.join(audio_dir,person)
        os.makedirs(save_path_parent,exist_ok=True)
        person_dir=os.path.join(video_dir,person)
        split_list=os.listdir(person_dir)
        for split in split_list:
            video_path=os.path.join(person_dir,split)
            audio_path=os.path.join(video_path,"audio.wav")
            save_path=os.path.join(save_path_parent,split)
            os.makedirs(save_path,exist_ok=True)
            audio_process(audio_path,save_path)

def get_dir_image_length(video_dir):
    image_list=glob(os.path.join(video_dir,"*.jpg|*.png"))
    return len(image_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir",type=str,default="/data/xuhao/datasets/hdtf_preprocessed/split_video_25fps")
    parser.add_argument("--audio_dir",type=str,default="/data/xuhao/datasets/hdtf_preprocessed/whisper_tiny_audios")
    args = parser.parse_args()
    audio_process_dirs_split(args.video_dir,args.audio_dir)