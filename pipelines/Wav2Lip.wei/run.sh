#export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES=0,1,2,3
#export CUDA_VISIBLE_DEVICES=4

tname=`date +%Y-%m-%d-%H-%M-%S.txt`
model_name=lipsync_expert_GPUs_id3300_lr4
#model_name=wav2lip_GPUs
if [ ! -d logs/${model_name} ]
then
    echo logs/${model_name}
    mkdir -p logs/${model_name}
fi

find ./ -path ./logs -prune  -o -path ./result -prune  -o  \( -name "*.py" -o -name "*.py.bk"  -o -name "*.sh" -o -name "*.yaml" \)  | xargs tar -czf  logs/${model_name}/${model_name}_${tname}.tar.gz  --exclude="./logs" --exclude="./result"
echo "tar *.py *.sh done"



#推理
#python inference.py --checkpoint_path checkpoints/wav2lip.pth \
	#--face /data2/weijinghuan/head_talk/Wav2Lip/data/shensi/liuwei_part1.mp4 \
	#--audio /data2/weijinghuan/head_talk/Wav2Lip/data/shensi/liuwei_part1.mp4 \
	#--outfile /data2/weijinghuan/head_talk/Wav2Lip/results/liuwei_authormodel.mp4 \
	#--resize_factor 1

	#--audio /data2/weijinghuan/head_talk/Wav2Lip/data/LRS2/5535423430009926848/00001.mp4 \
	#--audio /data2/weijinghuan/head_talk/Wav2Lip/data/LRS2/5535423430009926848/00001.mp4 \

#视频预处理
#python preprocess.py --data_root /mnt/diskwei/dataset/head_talk/3300+ID/数据集/fps25 \
		#--preprocessed_root /mnt/diskwei/dataset/head_talk/3300+ID/数据集/id3000_fps25_preprocessed/ \
		#--ngpu 4 --batch_size  32
# syncnet训练
#python color_syncnet_train.py --data_root /mnt/diskwei/dataset/head_talk/LRS2/lrs2_preprocessed/ --checkpoint_dir checkpoints/lipsync_expert
# GPUs syncnet训练
python -m torch.distributed.launch --master_port=12345  --nproc_per_node=4 --use_env color_syncnet_train_torch_gpus.py \
	--data_root /mnt/diskwei/dataset/head_talk/3300+ID/数据集/id3300_fps25_preprocessed96/ \
	--log_dir runs/${model_name} \
	--checkpoint_dir checkpoints/${model_name}

	#--data_root  /mnt/diskwei/dataset/head_talk/LRS2/lrs2_preprocessed/ \
# no visual quanlity gan wav2lip训练
#python wav2lip_train.py --data_root /mnt/diskwei/dataset/head_talk/LRS2/lrs2_preprocessed/ \
	   #--checkpoint_dir checkpoints/${model_name} \
	#--syncnet_checkpoint_path checkpoints/lipsync_expert_GPUs/checkpoint_step000090000.pth
#python -m torch.distributed.launch --master_port=12345  --nproc_per_node=4 --use_env wav2lip_train_torch_gpus.py \
	   #--data_root /mnt/diskwei/dataset/head_talk/LRS2/lrs2_preprocessed/ \
	#--checkpoint_dir checkpoints/${model_name} \
	#--log_dir runs/${model_name} \
	#--syncnet_checkpoint_path checkpoints/lipsync_expert_GPUs/checkpoint_step000090000.pth


# add visual quanlity gan wav2lip训练
#python hq_wav2lip_train.py --data_root /mnt/diskwei/dataset/head_talk/LRS2/lrs2_preprocessed/ \
	   #--checkpoint_dir checkpoints/wav2lip_gan_author_syncnet  \
	#--syncnet_checkpoint_path checkpoints/lipsync_expert.pth





