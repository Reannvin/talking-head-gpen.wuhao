export CUDA_VISIBLE_DEVICES=0
python gpen_w2l_s2.py \
        --data_root /mnt/hdtf_btm_move/ \
        --batch 1 \
        --save_ckpt_freq 10000 \
        --save_sample_freq 5000 \
        --syncnet_T 5 \
        --stage_1_pt ./training-run/202409011321/checkpoints/checkpoint_330000.pt \
        --enable_sync_loss 
        # --concat_condition \
        # --wandb
