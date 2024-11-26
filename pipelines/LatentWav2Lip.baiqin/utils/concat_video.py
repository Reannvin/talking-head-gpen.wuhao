import cv2
import numpy as np
from tqdm import tqdm
import subprocess
import os

def horizontally_concatenate_videos(video1_path, video2_path, output_path):
    # 读取视频1
    video1 = cv2.VideoCapture(video1_path)
    fps = int(video1.get(cv2.CAP_PROP_FPS))
    width = int(video1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video1.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 读取视频2
    video2 = cv2.VideoCapture(video2_path)

    # 创建写入视频的对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))

    while True:
        ret1, frame1 = video1.read()
        ret2, frame2 = video2.read()
        text1=video1_path.split('/')[-1].split('.')[0]
        cv2.putText(frame1, text1, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        text2 = video2_path.split('/')[-1].split('.')[0]
        cv2.putText(frame2, text2, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        # 如果两个视频中有一个到达末尾，则停止拼接
        if not ret1 or not ret2:
            break

        # 将两个帧横向拼接
        concatenated_frame = cv2.hconcat([frame1, frame2])

        # 将拼接后的帧写入输出视频文件
        output_video.write(concatenated_frame)

    # 释放资源
    video1.release()
    video2.release()
    output_video.release()
    cv2.destroyAllWindows()

def space_concatenate_videos(video1_path, video2_path, output_path):
    # 读取视频1
    video1 = cv2.VideoCapture(video1_path)
    fps = int(video1.get(cv2.CAP_PROP_FPS))
    width = int(video1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video1.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 读取视频2
    video2 = cv2.VideoCapture(video2_path)

    # 创建写入视频的对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(output_path, fourcc, fps, (width , height*2))

    while True:
        ret1, frame1 = video1.read()
        ret2, frame2 = video2.read()

        # 如果两个视频中有一个到达末尾，则停止拼接
        if not ret1 or not ret2:
            break

        # 将两个帧横向拼接
        concatenated_frame = cv2.vconcat([frame1, frame2])

        # 将拼接后的帧写入输出视频文件
        output_video.write(concatenated_frame)

    # 释放资源
    video1.release()
    video2.release()
    output_video.release()
    cv2.destroyAllWindows()

def concat_grid(input_paths, output_path, rows, cols):
    # Open the input videos
    assert len(input_paths) == rows * cols
    videos = [cv2.VideoCapture(path) for path in input_paths]
        
    video_names = [path.split('/')[-1].split('.')[0] for path in input_paths]
    # Get the width and height of the first video
    width = int(videos[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(videos[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps=int(videos[0].get(cv2.CAP_PROP_FPS))
    for video in videos:
        assert int(video.get(cv2.CAP_PROP_FRAME_WIDTH)) == width
        assert int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)) == height
        assert int(video.get(cv2.CAP_PROP_FPS)) == fps
    # Create an output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width * cols, height * rows))
    video_length = int(videos[0].get(cv2.CAP_PROP_FRAME_COUNT))
    # Read and concatenate frames
    for _ in tqdm(range(video_length)):
        frames = []
        for i, video in enumerate(videos):
            ret, frame = video.read()
            if not ret:
                break
            frame=cv2.putText(frame, video_names[i], (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)
            frames.append(frame)

        if len(frames) < len(videos):
            break

        # Create a black canvas for padding
        canvas = np.zeros((height * rows, width * cols, 3), dtype=np.uint8)
        #print(canvas.shape)
        # Concatenate frames horizontally
        for i, frame in enumerate(frames):
            row = i // cols
            col = i % cols
            start_x = col * width
            start_y = row * height
            end_x = start_x + width
            end_y = start_y + height
           # print(start_x,end_x,start_y,end_y)
            canvas[start_y:end_y, start_x:end_x,:] = frame
        out.write(canvas)

    # Release the video objects and writer
    for video in videos:
        video.release()
    out.release()
    cv2.destroyAllWindows()

def add_audio(video_path,audio_path,output_path):
    if os.path.exists(output_path):
        command=['rm',output_path]
        subprocess.run(command,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    command=['ffmpeg', '-i', video_path, '-i', audio_path, '-c:v', 'copy', '-c:a', 'aac', '-strict', 'experimental', output_path]
    #subprocess.run(command,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    subprocess.run(command)
    command=['rm',video_path]
    subprocess.run(command,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)



# 示例使用
# video_list=['results/GroundTruth.mp4','results/lrs2_example1.mp4','results/3300IDHDTF_example1.mp4',
#             'results/lrs2_latent_example1.mp4','results/3300ID_latent_example1_1.mp4','results/3300ID_latent_example1_2.mp4']
video_list=['results/GroundTruth.mp4','results/MuseTalk_example1.mp4','results/3300IDHDTF_example1.mp4','results/3300ID_latent_imageMask_example1.mp4']
audio_file='data/example/audio/example1.wav'
tmp_path='results/tmp.mp4'
output_path='results/output2.mp4'
rows=1
cols=4
concat_grid(video_list,tmp_path,rows,cols)
add_audio(tmp_path,audio_file,output_path)

