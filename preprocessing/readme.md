## 1, install requirements  
```sh
pip install -r requirement.txt 
mim install mmengine  
mim install "mmcv>=2.0.1" 
mim install "mmdet>=3.1.0" 
mim install "mmpose>=1.1.0" 
```

## 2, 

### 1 change fps to 25 
```sh
python video-fps.py --data_root /path/to/dataset --output_root /path/to/savedir/
```
### 2 crop video to small video, run code crop video if the video have long diration or high resolution,(base on 2 times of bbox area)
```sh
python utils/cropUtil.py --src_dir /path/to/dataset --des_dir /path/to/savedir/
```
### 3 sence detect,return file list, 
#### if need sence detect to generate filelist, if you want to ignore base case:
```sh
python sence_detect.py --data_root /path/to/dataset --ignore True
```
#### else, run:
```sh
python sence_detect.py --data_root /path/to/dataset --ignore True --temp_dir  /path/to/sence split dir/
```
### 4, eval to remove some video not-sync
```sh
python eval_syncnet.py --lse_d thres of distance --lse_c thres of confidence --filelist /path/to/filelist.txt --output /output/filelist.txt
```
### 5, crop face mmpose, yolo
```sh
python face_detect.py --filelist /path/to/filelist.txt --method method to detect --gpu_id gpu to use --output /path/to/save root dor
```
### 6, whisper preprocess
```sh
python whisper_preprocessing.py --root_dir /path/to/audio root dir
```
### 7, split whisper
```sh
python split_whisper.py ----data_root /path/to/whipser npy root dir
```
