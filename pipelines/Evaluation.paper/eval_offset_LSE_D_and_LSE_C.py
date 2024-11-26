import argparse
import subprocess
import os

def run_command(command):
    """执行命令并捕获异常"""
    try:
        print(f"Running: {' '.join(command)}")
        subprocess.run(command, check=True)
        print("Command executed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Command failed with error: {e}")

def get_reference_from_videofile(videofile):
    """根据视频文件路径生成默认的参考名（去掉扩展名）"""
    return os.path.splitext(os.path.basename(videofile))[0]

def main():
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="Run pipeline and syncnet in sequence.")
    
    parser.add_argument("--videofile", required=True, help="Path to the video file.")
    parser.add_argument("--reference", help="Temporary reference name, defaults to the video file name without extension.")
    parser.add_argument("--data_dir", default="temp_data", help="Directory for temporary data (default: temp_data).")
    
    args = parser.parse_args()

    # 如果没有指定 --reference，则使用视频文件名作为默认值
    reference = args.reference or get_reference_from_videofile(args.videofile)

    # 构建两个命令
    pipeline_command = [
        "python", "run_pipeline.py",
        "--videofile", args.videofile,
        "--reference", reference,
        "--data_dir", args.data_dir,
        # "--crop_scale", "0.0",
        # "--facedet_scale", "0.0"
    ]

    syncnet_command = [
        "python", "run_syncnet.py",
        "--videofile", args.videofile,
        "--reference", reference,
        "--data_dir", args.data_dir,
        # "--crop_scale", "0.0",
        # "--facedet_scale", "0.0"
    ]

    # 顺序执行命令
    run_command(pipeline_command)
    run_command(syncnet_command)

if __name__ == "__main__":
    main()
