# 依赖  
pip install face-alignment
# 推理
python inference/inference_basicvsrpp_deploy.py \
--model_path 视频恢复模型路径  --input_path 低质视频路径 \
--save_path 保存结果路径 \
--save_crop --org_video_path   wav2lip原始输入视频   
save_crop:保存wav2lip原始输入视频开关，为False时，org_video_path 可以不填写

        

