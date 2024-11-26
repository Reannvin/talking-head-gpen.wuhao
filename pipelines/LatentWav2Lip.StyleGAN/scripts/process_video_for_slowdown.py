import os
from moviepy.editor import VideoFileClip,vfx

def process_videos(input_folder, output_folder):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.endswith(".mp4"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            with VideoFileClip(input_path) as video:

                slowed_video = video.fx(vfx.speedx, 0.5)
  
                slowed_video.write_videofile(output_path, codec="libx264", audio_codec="aac")
                
            print(f"Processed {filename} and saved to {output_folder}")

if __name__ == "__main__":
    input_folder = "/data/wuhao/media/videos_25fps/"
    output_folder = "/data/wuhao/media/slow_down/"
    process_videos(input_folder, output_folder)