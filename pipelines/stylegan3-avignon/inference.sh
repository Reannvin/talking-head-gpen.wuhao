python inference.py \
    --checkpoint_path /data/wuhao/code/stylegan3-avignon/training-runs/00004-stylegan3-t--gpus1-batch16-gamma2/network-snapshot-001800.pkl \
    --face /data/wuhao/media/videos_25fps/adam.mp4 \
    --audio /data/wuhao/media/audios/chinese.wav \
    --outfile ./infer_res_2.mp4 \
    --mask_ratio 0.6 \
    --crop_down 0.1 \
    --resize \
    --resize_to 640 \
    --face_parsing