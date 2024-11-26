# UNet V2 Latent Wav2Lip, uses Wav2Vec2 as Audio Encoder
当前代码用 [Stable Diffusion v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5) 中的UNet，略作修改后作为模型定义，然后用[Wav2Vec2](https://huggingface.co/facebook/wav2vec2-base-960h) 替换了原 Syncnet 和 Wav2Lip 代码中的 Audio Encoder。并将得到的 audio embedding 通过 cross attention 加到了UNet中。由于在 3090 上 24G 显存不足，训练使用了 deepspeed_stage_2。

# 代码说明

## 1. 数据预处理
视频图像部分数据预处理，参考 LatentWav2Lip。音频部分：
```
python audio_preprocess.py --data_root /path/to/latent/dataset
```
该脚本会使用 Wav2Vec2 获取每个视频文件夹内的 audio.wav 对应的 audio embedding，并存储到 wav2vec2.pt。

## 2. 数据集说明
当前支持的数据集有：lrs2，hdtf，3300,3300_and_hdtf, all。其中all包含所有3个数据集。

每个支持的数据集有单独的 filelists，名符合命名规则 filelists_{dataset_name}。

因此在syncnet和unet训练的时候，增加了一个 --dataset_name 参数，有效的取值就是上面5个数据集。

## 2. 提取 unet
如果想使用 sd-v1-5 的 859M 参数的 UNet 来训练的话，可以按以下步骤操作：
```
python unet_loader.py --extract
```
默认会从 runwayml/stable-diffusion-v1-5 提取 unet，并存储到 stable-diffusion-v1-5 文件夹。

## 3. syncnet 训练
使用原版image encoder的u版本：
```
python unet_syncnet_lightning.py --data_root /path/to/latent/dataset --dataset_name name
```

使用更多参数的image encoder的xl版本：
```
python unet_syncnet_lightning.py --data_root /path/to/latent/dataset --dataset_name name --xl
```

## 4.1 wav2lip 训练，用自定义的 unet，80M 参数
如果第3步使用了 --xl 训练syncnet，则用 --syncnet_xl 代替 --syncnet 。

```
python unet_wav2lip_lightning.py --data_root /path/to/latent/dataset --dataset_name name --syncnet /path/to/syncnet/checkpoint --unet_ckpt customized_unet/
```

## 4.2 wav2lip 训练，用自定义的 unet，24M 参数
```
python unet_wav2lip_lightning.py --data_root /path/to/latent/dataset --dataset_name name --syncnet /path/to/syncnet/checkpoint --unet_ckpt customized_unet_small/
```

## 4.3 wav2lip 训练，用自定义的 unet，140M 参数
```
python unet_wav2lip_lightning.py --data_root /path/to/latent/dataset --dataset_name name --syncnet /path/to/syncnet/checkpoint --unet_ckpt customized_unet_large/
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
# 不使用syncnet时的训练
## 1、数据预处理
```
python python crop_face_util.py --ngpu 2 --batch_size 8 --data_root /path/to/your/data_root/ --preprocessed_root /path/to/your/preprocessed_dir/ --resize --topbottom_scale 0.05
```

## 2、生成资源文件
```
cp -r filelist_shensi/ filelist_shensi2/
```
之后在filelist_shensi2/目录下执行如下命令：
```
python gen_file_list.py --root_dir /path/to/your/root_dir/
```
完成后可见main.txt中为所有的数据，train.txt、val.txt、test.txt分别为训练、验证、测试集，可通过修改上述文件手动调整数据集。


如下步骤中，如果想省事，可直接使用现成的环境，运行如下命令：
```
source /data/xuhao/talking-head/pipelines/LatentWav2Lip.UNet.V2/.venv/bin/activate
```

## 3、进行whisper embedding
```
python whisper_preprocessing.py --root_dir /path/to/your/root_dir/
```

## 4、拆分whisper embedding
```
python split_whisper.py --data_root /path/to/your/root_dir/
```

## 5、启动训练
```
CUDA_VISIBLE_DEVICES=1,4 python unet_wav2lip_lightning.py --data_root /path/to/your/data_root/ --dataset_name your_dataset_name --overfit --batch_size 2 --accu_grad 2 --full_image_loss --syncnet_T 5
```
其中，使用--overfit参数，将使用main.txt中全部数据进行训练。
