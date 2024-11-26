CUDA_VISIBLE_DEVICES=2 python unet_wav2lip_lightning_unc.py \
        --dataset_config /data/wuhao/code/talking-head/pipelines/LatentWav2Lip.StyleGAN/data/dataset_config.yaml \
        --overfit \
        --batch_size 2 \
        --accu_grad 2 \
        --syncnet_T 5 \
        --mask_ratio 0.6 \
        --enable_stylegan_loss
        # --wandb
        # --wandb_name qianjie-gan-loss-with-d_loss
        # --ckpt /data/wuhao/checkpoint/base_model_1205000.ckpt 
