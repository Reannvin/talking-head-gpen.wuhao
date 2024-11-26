# Draft training codes

We provde the draft training codes here. Unfortunately, data preprocessing code is still being reorganized.

## Setup

We trained our model on an NVIDIA A100 with `batch size=8, gradient_accumulation_steps=4` for 20w+ steps. Using multiple GPUs should accelerate the training.

## Data preprocessing
 You could refer the inference codes which [crop the face images](https://github.com/TMElyralab/MuseTalk/blob/main/scripts/inference.py#L79) and [extract audio features](https://github.com/TMElyralab/MuseTalk/blob/main/scripts/inference.py#L69).

Finally, the data should be organized as follows:
```
./data/
├── images
│     └──RD_Radio10_000
│         └── 0.png
│         └── 1.png
│         └── xxx.png
│     └──RD_Radio11_000
│         └── 0.png
│         └── 1.png
│         └── xxx.png
├── audios
│     └──RD_Radio10_000
│         └── 0.npy
│         └── 1.npy
│         └── xxx.npy
│     └──RD_Radio11_000
│         └── 0.npy
│         └── 1.npy
│         └── xxx.npy
```

## Training <br>
Simply run after preparing the preprocessed data <br>
1，prepocessing data<br>
''' <br>
cd /train_codes/utils <br>
python preprocess.py --data_root /dataset/ --save_dir /audio_save_dir/<br>
2, copy filelist which you decide to train from previous Unet version filelists or cd to filelists<br>
'''<br>
python gen_file_list --data_root /dataset_path/<br>
'''<br>
3, simple train<br>
'''<br>
accelerate launch train.py --mixed_precision="fp16" --unet_config_file="./musetalk.json" --pretrained_model_name_or_path=’stabilityai/sd-vae-ft-mse’ --data_root="/data/xuhao/datasets/lrs2_preprocessed" --train_batch_size=8 --gradient_accumulation_steps=1 --gradient_checkpointing --max_train_steps=200000 --learning_rate=5e-05 --max_grad_norm=1 --lr_scheduler="cosine" --lr_warmup_steps=0 --output_dir='train' --val_out_dir='val' --testing_speed --checkpointing_steps=1000 --validation_steps=1000 --reconstruction --resume_from_checkpoint="latest" --use_audio_length_left=2 --use_audio_length_right=2 --whisper_model_type="tiny" --audio_root /data/xuhao/datasets/lrs2_whisper_audio --wandb<br>
'''<br>
Now only support gradient_accumulation_steps = 1
## TODO
- [ ] release data preprocessing codes
- [ ] release some novel designs in training (after technical report)
