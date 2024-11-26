#export CUDA_VISIBLE_DEVICES=0,1,2,3

tname=`date +%Y-%m-%d-%H-%M-%S.txt`

model_name='BasicVS_vfhq_VFR'
#model_name='BasicVS_vfhq_VFR_perce_W2'

if [ ! -d logs/${model_name} ]
then
    echo logs/${model_name}
    mkdir -p logs/${model_name}
fi

find ./ -path ./logs -prune  -o -path ./result -prune  -o  \( -name "*.py" -o -name "*.py.bk"  -o -name "*.sh" -o -name "*.yaml" -o -name "*.yml" \)  | xargs tar -czf  logs/${model_name}/${model_name}_${tname}.tar.gz  --exclude="./logs" --exclude="./result"
echo "tar *.py *.sh done"
# train
# 超分训练
#CUDA_VISIBLE_DEVICES=0,1,2,3 \
#./scripts/dist_train.sh 4 options/train/VFHQ/BasicVSR.yml

#视频恢复训练
CUDA_VISIBLE_DEVICES=0,1,2,3 \
./scripts/dist_train.sh 4 options/train/VFHQ/BasicVSRVFR.yml


# 评测
#PYTHONPATH="./:${PYTHONPATH}" \
	#CUDA_VISIBLE_DEVICES=7 \
	#python basicsr/test.py -opt options/test/VFHQ/BasicVSR.yml


