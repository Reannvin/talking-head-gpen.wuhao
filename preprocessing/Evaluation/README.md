# 算法效果定量评估

## 0. 安装与准备
```
cd syncnet_python
pip install -r requirements.txt
sh download_model.sh
```

## 1. 推理生成结果视频

```
python unet_gen_videos_from_filelist.py --filelist test_filelists/lrs2.txt --results_dir /path/to/result --data_root /data/xuhao/datasets/lrs2_original --checkpoint_path /path/to/checkpoint --unet_config /path/to/unet/config
```

## 2. 计算推理结果的定量指标

```
cd syncnet_python/
python calculate_scores_LRS.py --data_root /path/to/result --tmp_dir /path/to/temp/dir
```

The output should be something like:

```
Average Confidence: 4.891456213864413
Average Minimum Distance: 8.73389814116738
```