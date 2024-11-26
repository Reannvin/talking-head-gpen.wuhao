import argparse
import os
from moviepy.editor import VideoFileClip, concatenate_videoclips

def sync_video(input_path, offset, output_path=None, fps=25):
    # 如果没有指定 output_path，则在当前目录下使用输入文件名加 "_synced"
    if output_path is None:
        filename = os.path.basename(input_path)  # 获取文件名
        base, ext = os.path.splitext(filename)
        output_path = f"{base}_synced{ext}"  # 当前目录下的默认输出文件名

    # 读取视频
    video = VideoFileClip(input_path)
    
    # 将帧数转换为时间（秒）
    offset_seconds = offset / fps

    if offset_seconds < 0:
        # 视频需要延迟，往视频开头插入静态帧
        first_frame = video.subclip(0, 1.0 / fps)  # 提取第一帧
        padding_clip = first_frame.set_duration(-offset_seconds)  # 延长第一帧的时间
        synced_video = concatenate_videoclips([padding_clip, video])
    elif offset_seconds > 0:
        # 视频需要提前，去掉开头的部分
        synced_video = video.subclip(offset_seconds)
    else:
        # 没有偏移，直接使用原视频
        synced_video = video

    # 输出同步后的文件
    synced_video = synced_video.set_audio(video.audio)  # 保持原音频
    synced_video.write_videofile(output_path, codec="libx264", audio_codec="aac")


if __name__ == "__main__":
    # 设置 argparse 解析命令行参数
    parser = argparse.ArgumentParser(description='Sync video by shifting frames based on frame offset.')
    
    parser.add_argument('--input', type=str, required=True, help='Path to the input video file.')
    parser.add_argument('--offset', type=int, required=True, help='Number of frames to offset the video (negative for delay, positive for advance).')
    parser.add_argument('--output', type=str, default=None, help='Path to the output synced video file (default: current directory with _synced).')
    parser.add_argument('--fps', type=int, default=25, help='Frames per second of the video (default: 25).')

    args = parser.parse_args()

    # 调用同步函数
    sync_video(args.input, args.offset, args.output, args.fps)
