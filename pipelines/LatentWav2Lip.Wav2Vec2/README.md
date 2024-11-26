# Latent Wav2Lip with Wav2Vec2 as Audio Encoder
当前代码用 [Wav2Vec2](https://huggingface.co/facebook/wav2vec2-base-960h) 替换了原 Syncnet 和 Wav2Lip 代码中的 Audio Encoder，期望得到更好的训练结果。

# 代码说明

## 1. 数据预处理
视频图像部分数据预处理，参考 LatentWav2Lip。音频部分：
```
python audio_preprocess.py --data_root /path/to/latent/dataset
```
该脚本会使用 Wav2Vec2 获取每个视频文件夹内的 audio.wav 对应的 audio embedding，并存储到 wav2vec2.pt。

## 2. syncnet 训练
```
python wav2vec2_syncnet_lightning.py --data_root /path/to/latent/dataset
```


## 3. wav2lip 训练
```
python wav2vec2_wav2lip_lightning.py --data_root /path/to/latent/dataset --syncnet /path/to/syncnet/checkpoint
```

## 4. wav2lip 推理
```
python wav2vec2_inference.py --checkpoint_path /path/to/wav2lip/checkpoint --face /path/to/face/video --audio /path/to/speech/audio
```