import os
import subprocess
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from tqdm import tqdm
from utils.scenedetect import detect_scenes, split_video_by_scenes
from utils.histgram import detect_transitions

def get_personid_list(data_root):
    person_ids = []
    for person_id in os.listdir(data_root):
        person_ids.append(os.path.splitext(person_id)[0])
    return person_ids
        
def split_video(input_file, split_times, output_folder):
    video_split_path = []
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    name = os.path.basename(input_file).split('.mp4')[0]

    split_times = sorted([0] + split_times)

    for i in range(len(split_times) - 1):
        start_time = split_times[i]
        end_time = split_times[i + 1]
        part_output = os.path.join(output_folder, f"{name}_{i+1}.mp4")
        video_split_path.append(part_output)
        ffmpeg_command = [
            'ffmpeg','-y', '-i', input_file, '-ss', str(start_time), '-to', str(end_time), '-c:v', 'libx264', '-c:a', 'aac', '-y', part_output
        ]
        process = subprocess.Popen(' '.join(ffmpeg_command), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            print(f"文件 {input_file} 处理失败，详情: {str(stderr.decode('utf-8'))}")
    final_start_time = split_times[-1]
    final_part_output = os.path.join(output_folder, f"{name}_{len(split_times)}.mp4")
    video_split_path.append(final_part_output)
    final_ffmpeg_command = [
        'ffmpeg', '-y', '-i', input_file, '-ss', str(final_start_time), '-c:v', 'libx264', '-c:a', 'aac', '-y', final_part_output
    ]
    process = subprocess.Popen(' '.join(final_ffmpeg_command), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(f"文件 {input_file} 处理失败，详情: {str(stderr.decode('utf-8'))}")
    return video_split_path

def process_video(person_id, data_root, temp_path, scene_detect, ignore):
    video_file = os.path.join(data_root, person_id + '.mp4')
   
    if not scene_detect:
        sence_list = []
        transitions = []
    else:
        sence_list = detect_scenes(video_file)
        transitions = detect_transitions(video_file, threshold=0.5, fps=25)
    
    if len(sence_list) == 0 and transitions == []:
        return {video_file: [{"person_id": person_id}]}
    else:
        if ignore:
            pass
        else:
            sence_split_path = os.path.join(temp_path, person_id)
            if len(sence_list) > 0:
                os.makedirs(sence_split_path, exist_ok=True)
                split_path = split_video_by_scenes(video_file, sence_list, sence_split_path)
            else:
                split_path = split_video(video_file, transitions, sence_split_path)
            return {p: [{"person_id": person_id}] for p in split_path}

def gen_file_list(data_root, temp_path, person_ids, scene_detect, num_workers, ignore):
    info_dict = defaultdict(list)
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_video, person_id, data_root, temp_path, scene_detect, ignore) for person_id in person_ids]
    
        for future in tqdm(futures):
            result = future.result()
            for k, v in result.items():
                info_dict[k].extend(v)
    return info_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='data preprocessing')
    parser.add_argument("--data_root", default='/data/fanshen/avspeech_fps25', help="input root path", type=str)
    parser.add_argument("--temp_root", default='tmp', help="your temp root path", type=str)
    parser.add_argument("--num_workers", default=8, help="workers to use", type=int)
    parser.add_argument("--scene_detect", type=bool, default=True, help="Enable scene detection true or disable False")
    parser.add_argument("--ignore", type=bool, default=True, help="whether to ignore the video which is not consistendy for all the frames")
    args = parser.parse_args()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    person_ids = get_personid_list(args.data_root)
    temp_path = os.path.join(current_dir, args.temp_root)
    os.makedirs(temp_path, exist_ok=True)
    info_dict = gen_file_list(args.data_root, temp_path, person_ids, args.scene_detect, args.num_workers, args.ignore)
    with open('videosence.txt', 'w') as v:
        for video_file, info in tqdm(info_dict.items()):
            person_id = info[0].get("person_id")
            v.writelines(f"'{video_file} {person_id}'\n")
