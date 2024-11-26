from moviepy.editor import VideoFileClip, concatenate_videoclips, vfx

input_file = "/data/wuhao/media/videos_25fps/chenyi/test_vid/chenyi_1.mp4"
output_file = "/data/wuhao/media/videos_25fps/chenyi_inference.mp4"

clip = VideoFileClip(input_file)

clip_A = clip.subclip(0, 5)

clip_B = clip_A.fx(vfx.time_mirror)

final_clip = concatenate_videoclips([clip_A, clip_B, clip_A])

# 保存最终视频
final_clip.write_videofile(output_file, codec="libx264")

print(f"输出文件已生成: {output_file}")