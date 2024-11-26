import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from musetalk.whisper.audio2feature import Audio2Feature
from glob import glob

from tqdm import tqdm
import numpy as np
import argparse
def audio_process(audio_path,save_path=None,fps=25):
    audio_processor = Audio2Feature(model_path="/data/fanshen/workspace/MuseTalk/musetalk/whisper/whisper/tiny.pt")
    whisper_feature = audio_processor.audio2feat(audio_path)
    whisper_chunks = audio_processor.feature2chunks(feature_array=whisper_feature,fps=fps)
    print("whisper_feature:",whisper_feature.shape)
    print("whisper_chunks:",len(whisper_chunks))
    print("whisper_chunks[0]:",whisper_chunks[0].shape)
    for i in range(len(whisper_chunks)):
        save_file=os.path.join(save_path,f"{i}.npy")
        np.save(save_file,whisper_chunks[i])
def audio_process_dirs(data_root,save_dir):
    os.makedirs(save_dir,exist_ok=True)
    person_list=os.listdir(data_root)
    for person in tqdm(person_list):
        person_path = os.path.join(data_root,person)
        for sub_cut in tqdm(os.listdir(person_path)):
            audio_path=os.path.join(data_root,person,sub_cut,"audio.wav")
            save_path=os.path.join(save_dir,person, sub_cut)
            os.makedirs(save_path,exist_ok=True)
            audio_process(audio_path,save_path)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",type=str,default="/data/xuhao/datasets/lrs2_preprocessed")
    parser.add_argument("--save_dir",type=str,default="/data/xuhao/datasets/lrs2_whisper_audio")
    args = parser.parse_args()
    audio_process_dirs(args.data_root,args.save_dir)