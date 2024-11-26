ffmpeg \
-i /data/wuhao/code/talking-head/pipelines/LatentWav2Lip.Reann/results/jianfeng_newpt-1146000-steps.mp4 \
-i /data/wuhao/code/talking-head/pipelines/LatentWav2Lip.OnTheFly/results/eleven/0718/jianfeng-cn-1020000-steps.mp4 \
-filter_complex "\
[0:v]scale=iw:ih,drawtext=text='HDTF+shensi262-8Kstep':x=10:y=10:fontsize=48:fontcolor=white@0.8[v0]; \
[1:v]scale=iw:ih,drawtext=text='HDTF+shensi100-8Kstep':x=10:y=10:fontsize=48:fontcolor=white@0.8[v1]; \
[v0][v1]hstack=inputs=2[v]" \
-map "[v]" \
-map 0:a \
-c:v libx264 -crf 23 -preset fast \
-c:a aac -b:a 192k \
../results/test.mp4