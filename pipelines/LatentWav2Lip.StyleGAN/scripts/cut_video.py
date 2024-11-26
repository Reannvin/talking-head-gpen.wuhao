from moviepy.editor import VideoFileClip

def cut_video(input_path, output_path, duration_seconds):

    video = VideoFileClip(input_path)

    clipped_video = video.subclip(0, duration_seconds)

    clipped_video.write_videofile(output_path, codec='libx264', audio_codec='aac')

if __name__ == "__main__":
    input_video_path = "/data/wuhao/media/videos_25fps/ziyan/方怡0606.mp4"
    output_video_path = "/data/wuhao/media/videos_25fps/fangyi.mp4" 
    duration_seconds = 30

    cut_video(input_video_path, output_video_path, duration_seconds)