python unet_inference_v3_underconstruction.py \
    --checkpoint_path /data/wuhao/code/talking-head/pipelines/LatentWav2Lip.Discriminator/image_wav2lip/016ygd83/checkpoints/wav2lip-i-qianjie-s=256-t=5-epoch=1209-step=1215000-train_loss=-0.022-val_loss=0.210-val_sync_loss=0.000.ckpt \
    --face /data/wuhao/media/videos_25fps/qianjie.mp4 \
    --audio /data/wuhao/media/audios/english.wav \
    --outfile ./results/qianjie-test.mp4 \
    --mask_ratio 0.6 \
    --crop_down 0.1 \
    --resize \
    --resize_to 640 \
    --face_parsing