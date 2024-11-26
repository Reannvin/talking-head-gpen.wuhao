import os
import cv2
import shutil
import argparse
import subprocess
import concurrent.futures

import sys
sys.path.append(os.getcwd())


def split_video(input_file, output_folder, interval=5):
    folder_name = os.path.splitext(os.path.basename(input_file))[0]
    output_path = os.path.join(output_folder, folder_name)
    os.makedirs(output_path, exist_ok=True)

    cap = cv2.VideoCapture(input_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = total_frames / fps

    num_segments = int(duration / interval) + 1

    for i in range(num_segments):
        start_time = i * interval
        end_time = min((i + 1) * interval, duration)
        segment_file = os.path.join(output_path, f'{folder_name}_{i:05d}.mp4')

        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(segment_file, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

        while True:
            ret, frame = cap.read()
            if not ret or cap.get(cv2.CAP_PROP_POS_FRAMES) > end_frame:
                break
            out.write(frame)

        out.release()

    cap.release()

def split_audio(input_file, output_folder, interval=5):
    folder_name = os.path.splitext(os.path.basename(input_file))[0]
    output_path = os.path.join(output_folder, folder_name)
    os.makedirs(output_path, exist_ok=True)

    # 使用 FFmpeg 分割音频
    command = f'ffmpeg -i "{input_file}" -vn -ac 1 -ar 16000 -f segment -segment_time {interval} "{output_path}/{folder_name}_%05d.wav" -y'
    subprocess.call(command, shell=True)

def merge_video_audio(video_folder, audio_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    # 遍历视频文件夹中的所有分段视频
    for video_file in os.listdir(video_folder):
        video_name = os.path.splitext(video_file)[0]
        video_path = os.path.join(video_folder, video_file)
        audio_path = os.path.join(audio_folder, f'{video_name}.wav')
        output_path = os.path.join(output_folder, f'{video_name}.mp4')

        # 使用 FFmpeg 合并视频和音频
        command = f'ffmpeg -i "{video_path}" -i "{audio_path}" -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 "{output_path}" -y'
        subprocess.call(command, shell=True)

    # 删除中间文件夹
    # shutil.rmtree(video_folder)
    # shutil.rmtree(audio_folder)

    # 删除合并后的视频文件夹中的所有视频文件
    # video_files = glob.glob(os.path.join(output_folder, '*.mp4'))
    # for video_file in video_files:
    #     os.remove(video_file)

def convert_videos(input_folder, output_folder):
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 遍历输入文件夹中的所有文件
    for file_name in os.listdir(input_folder):
        input_file = os.path.join(input_folder, file_name)
        output_file = os.path.join(output_folder, file_name)

        # 使用 FFmpeg 进行帧率转换
        command = f"ffmpeg -i {input_file} -r 25 -b:v 20M {output_file} -y"
        subprocess.call(command, shell=True)

def process_video_files(input_folder, output_folder, num_threads=16, task=""):
    # 遍历输入文件夹中的所有文件
    input_files = [filename for filename in os.listdir(input_folder) if filename.endswith(".mp4")]

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        # 提交任务给线程池
        futures = []
        for filename in input_files:
            file_path = os.path.join(input_folder, filename)
            video_output_folder = os.path.join(output_folder, os.path.splitext(filename)[0])
            audio_output_folder = os.path.join(output_folder, os.path.splitext(filename)[0], 'audio')
            os.makedirs(audio_output_folder, exist_ok=True)
            futures.append(executor.submit(split_video, file_path, video_output_folder))
            futures.append(executor.submit(split_audio, file_path, audio_output_folder))

        # 等待任务完成
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"处理视频时出现错误: {str(e)}")
               
    # 合并视频和音频
    idx = 0
    total_files = len(input_files)
    for filename in input_files:
        idx += 1
        video_output_folder = os.path.join(output_folder, os.path.splitext(filename)[0],os.path.splitext(filename)[0])
        audio_output_folder = os.path.join(output_folder, os.path.splitext(filename)[0], 'audio',os.path.splitext(filename)[0])
        merged_output_folder = os.path.join(output_folder, os.path.splitext(filename)[0],'merged')
        merge_video_audio(video_output_folder, audio_output_folder, merged_output_folder)
        audio_folder = os.path.join(output_folder, os.path.splitext(filename)[0], 'audio')
        output_folder1 = os.path.join(output_folder, os.path.splitext(filename)[0])

        convert_videos(merged_output_folder, output_folder1)
        # merge_path =os.path.join(output_folder, os.path.splitext(filename)[0],'merged')
        # 删除中间文件夹
        shutil.rmtree(video_output_folder)
        shutil.rmtree(audio_folder)
        shutil.rmtree(merged_output_folder)
        
if __name__ == '__main__':
    # 示例用法
    parser = argparse.ArgumentParser(description='split video to every 5s video')

    parser.add_argument("--input_path", help="your input path", type=str)
    parser.add_argument("--output_path", help="your output path", type=str)
    parser.add_argument("--task", help="task_name", type=str, required=False, default="")

    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    process_video_files(args.input_path, args.output_path, num_threads=8, task=args.task)
