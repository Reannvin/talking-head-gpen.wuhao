export CUDA_VISIBLE_DEVICES=1
python gpen_base_training.py \
             --batch 8 \
             --path /mnt/wangbaiqin/dataset/FFHQ/ \
             --n_sample 16 \
             --pretrain ./training-run/202408311457/checkpoints/checkpoint_20000.pt \
             --save_ckpt_freq 10000 \
             --save_sample_freq 5000