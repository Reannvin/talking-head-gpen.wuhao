export CUDA_VISIBLE_DEVICES=5
python inference.py \
    --checkpoint_path /data/wuhao/code/GPEN_wav2lip/training-run/202409041654/checkpoints/checkpoint_70000.pt\
    --face /data/wuhao/media/videos_25fps/liuwei30.mp4 \
    --audio /data/wuhao/media/audios/english.wav \
    --outfile ./infer_result_liuwei_sml1loss.mp4 \
    --mask_ratio 0.6 \
    --crop_down 0.1 \
    --resize \
    --resize_to 640 \
    --ema \
    --face_parsing