import os
import subprocess
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector, ThresholdDetector, AdaptiveDetector, ThresholdDetector


def detect_scenes(video_path):
    # 创建视频管理器对象
    video_manager = VideoManager([video_path])

    # 创建场景管理器对象
    scene_manager = SceneManager()

    # 添加内容检测器到场景管理器
    scene_manager.add_detector(ContentDetector())

    # 启动视频管理器并获取视频信息
    video_manager.set_downscale_factor()
    video_manager.start()

    # 进行场景检测
    scene_manager.detect_scenes(frame_source=video_manager)

    # 获取检测到的场景列表
    scene_list = scene_manager.get_scene_list()

    # 打印每个场景的开始和结束时间
    for i, scene in enumerate(scene_list):
        print(f"场景 {i + 1}: 开始时间 {scene[0].get_timecode()} - 结束时间 {scene[1].get_timecode()}")

    # 释放视频管理器资源
    video_manager.release()
    return scene_list

def split_video_by_scenes(video_path, scene_list, output_dir):
    # scene_list = detect_scenes(video_path)
    ouput_path = []
    for i, scene in enumerate(scene_list):
        start_time = scene[0].get_timecode()
        end_time = scene[1].get_timecode()
        output_file = os.path.join(output_dir, f'{i + 1}.mp4')
        ouput_path.append(output_file)
        ffmpeg_command = [
            'ffmpeg',
            '-i', video_path,
            '-ss', start_time,
            '-to', end_time,
            '-c:v', 'libx264',  # 使用 H.264 编码
            '-preset', 'medium',  # 设置编码器参数
            '-crf', '18',  # 设置质量
            '-g', '15',    # 设置关键帧间隔
            '-c:a', 'copy',  # 复制音频流
            output_file,
            '-y'  # 覆盖输出文件（如果存在）
        ]
        
        # 输出ffmpeg命令，便于调试
        # print(' '.join(ffmpeg_command))
        
        # 调用ffmpeg命令分割视频
        subprocess.run(ffmpeg_command, check=True)
        return  ouput_path  


if __name__ == '__main__':
    # 调用函数检测视频中的场景切换
    video_path = './testdata/input/1.mp4'
    scene_list = detect_scenes(video_path)
    split_video_by_scenes(video_path=video_path, scene_list=scene_list, output_dir="./testdata/output/")