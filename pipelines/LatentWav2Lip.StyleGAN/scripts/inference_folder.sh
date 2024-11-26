checkpoint_folder="/data/wuhao/checkpoint/test/jianfeng/"
face="/data/wuhao/media/videos_25fps/jianfeng.mp4"
audio="/data/wuhao/media/audios/chinese.wav"
outfile_base="./results/jianfeng_newpt"
mask_ratio=0.6
crop_down=0.1

for ckpt_path in ${checkpoint_folder}*step=*.ckpt; do
    step=$(echo $ckpt_path | grep -oP '(?<=step=)\d+')
    outfile="${outfile_base}-${step}-steps.mp4"
    python unet_inference_v3_underconstruction.py \
        --checkpoint_path $ckpt_path \
        --face $face \
        --audio $audio \
        --outfile $outfile \
        --mask_ratio $mask_ratio \
        --crop_down $crop_down \
        --resize \
        --resize_to 640 \
        --face_parsing
    echo "Inference Down."
done