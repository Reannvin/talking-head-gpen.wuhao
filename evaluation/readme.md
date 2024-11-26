## 1, install requirements  
```sh
pip install pytorch_fid
pip install mmpose(require you to install mmcv mmpose, mmdet, etc.)
```
## 2, download dwpose model
```sh
sh ./dwpose/download_models.sh
```
## 3, caculation
### video defination(FID)
```sh
python --video1 /path/to/video1.mp4 --video2 /path/to/video2.mp4  --gpu_id 0
```
### mouse landmarks AKD (support method mediapipe/mmpose)
```sh
python mouse_akd.py --method mmpose  --video1 /path/to/video1.mp4 --video2 /path/to/video2.mp4 --gpu_id 0
```
