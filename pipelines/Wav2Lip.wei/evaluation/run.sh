export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES=3

python gen_videos_from_filelist_torch.py --filelist test_filelists/lrs2.txt \
--results_dir test_lr2V2_wav2lip_gpus \
--real_root test_fid_real_crop96_wav2lip_gpus \
--syn_root test_fid_syn_crop96_wav2lip_gpus \
--data_root /mnt/diskwei/dataset/head_talk/LRS2/mvlrs_v1/main/ \
--checkpoint_path ../checkpoints/wav2lip_GPUs/checkpoint_step000174000.pth



