export CUDA_VISIBLE_DEVICES=3
python inference_stage1.py \
    --checkpoint_path /data/fanshen/workspace/style_gan/talking-head/pipelines/GPEN/stage2_val_3/ckeckpoint/010000.pth \
    --output ./results/ \
