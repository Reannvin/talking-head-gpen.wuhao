# UNet Latent Wav2Lip, uses Wav2Vec2 as Audio Encoder
当前代码用 [Stable Diffusion v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5) 中的UNet作为模型定义，并加载了初始化权重，然后用[Wav2Vec2](https://huggingface.co/facebook/wav2vec2-base-960h) 替换了原 Syncnet 和 Wav2Lip 代码中的 Audio Encoder。并将得到的 audio embedding 通过 cross attention 加到了UNet中。由于在 3090 上 24G 显存不足，训练使用了 deepspeed_stage_3。并将 syncnet_T 改成了 2 ，单卡 batch size 改成了 1 。单卡显存消耗在20G左右。

# 代码说明

## 1. 数据预处理
视频图像部分数据预处理，参考 LatentWav2Lip。音频部分：
```
python audio_preprocess.py --data_root /path/to/latent/dataset
```
该脚本会使用 Wav2Vec2 获取每个视频文件夹内的 audio.wav 对应的 audio embedding，并存储到 wav2vec2.pt。

## 2. 提取 unet
```
python unet_loader.py --extract
```
默认会从 runwayml/stable-diffusion-v1-5 提取 unet，并存储到 stable-diffusion-v1-5 文件夹。

## 3. syncnet 训练
```
python unet_syncnet_lightning.py --data_root /path/to/latent/dataset
```

## 4.1 wav2lip 训练，用 stable diffusion v1-5 的 unet，859M 参数
```
python unet_wav2lip_lightning.py --data_root /path/to/latent/dataset --syncnet /path/to/syncnet/checkpoint
```

## 4.2 wav2lip 训练，用自定义的 unet，80M 参数
```
python unet_wav2lip_lightning.py --data_root /path/to/latent/dataset --syncnet /path/to/syncnet/checkpoint --unet_ckpt customized_unet/
```

## 5. 合并得到 ckpt
```
cd experiments/latent_wav2lip_experiment/version_xxx/checkpoints/wav2lip-u-xxxx.ckpt/
python zero_to_fp32.py . /path/to/unet/checkpoint
```
该脚本会将 deepspeed 的多 shard 文件合并成一个 pytorch 格式的 ckpt 。

## 4. wav2lip 推理
```
python unet_inference.py --checkpoint_path /path/to/wav2lip/checkpoint --face /path/to/face/video --audio /path/to/speech/audio
```
