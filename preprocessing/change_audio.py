import subprocess
import os
import shutil
def extract_audio(video_file, audio_file):
    cmd = f'ffmpeg -loglevel panic -y -i {video_file} -strict -2 {audio_file}'
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
if __name__ == '__main__':
    input_path = '/data/fanshen/workspace/preprocessed_avspeech'
    for person in os.listdir(input_path):
        person_path = os.path.join(input_path, person)
        if 'audio.wav' in os.listdir(person_path):
            extract_audio(f'/data/fanshen/avspeech_fps25/{person}.mp4', os.path.join(person_path, 'audio.wav'))
            
           
                