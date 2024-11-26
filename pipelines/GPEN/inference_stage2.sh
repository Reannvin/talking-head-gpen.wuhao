export CUDA_VISIBLE_DEVICES=3
python inference_stage2.py \
    --checkpoint_path /data/fanshen/workspace/style_gan/talking-head/pipelines/GPEN/stage2_val_3/ckeckpoint/010000.pth \
    --face /data/wuhao/media/videos_25fps/liuwei30.mp4 \
    --audio  /data/wuhao/media/audios/english.wav \
    --outfile ./results/GPEN10wstep-en.mp4 \
    --face_parsing \
    --mask_ratio 0.6 \
    --crop_down 0.1 \
    --resize \
    --image_size 256 \
    --face_det_batch_size 4 \
    --resize_to 640
