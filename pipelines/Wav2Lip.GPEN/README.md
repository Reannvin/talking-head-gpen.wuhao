# Wav2Lip.GPEN Training and Inference

This repository contains code for training and inference of a GPEN-based wav2lip model, focusing on syncing lip movements to audio input in video sequences.

## Table of Contents
- [Introduction](#introduction)
- [Requirements](#requirements)
- [Setup](#setup)
- [Training](#training)
- [Inference](#inference)
- [Configuration](#configuration)
- [License](#license)

## Introduction

The project implements a lip-syncing model using Wav2Lip for high-quality audio-driven talking faces. The training process leverages distributed PyTorch, and the inference supports fine-grained face parsing and various augmentations to improve output quality.

## Requirements

To get started, you will need the following dependencies:

- Python 3.7+
- PyTorch 1.8+
- CUDA 10.2+ (with GPU support)
- Other dependencies specified in `requirements.txt`

To install the necessary Python libraries, run:

```bash
pip install -r requirements.txt
```

## Setup

Before running the training or inference scripts, ensure that you have the following:

1. Correct dataset paths and configuration in `configs/dataset_config_liuwei.yaml`.
2. Pre-trained model checkpoints for training and inference, available in the `experiments` directory.

## Training

To train the model, use the following command:

```bash
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
--nproc_per_node=2 --master_port=4322 train_wav2lip.py \
--image_size 256 \
--channel_multiplier 2 \
--narrow 1 \
--batch 4 \
--ckpt experiments/1e98f495-137b-49aa-b324-e96e4bfa1625/ckpts/500000.pth \
--l1_weight 100.0 \
--data_aug_image \
--drop_ref \
--dataset_config configs/dataset_config_liuwei.yaml \
--wandb
```

### Explanation of Key Arguments:
- `--image_size`: Set the input image resolution for the model.
- `--channel_multiplier`: Adjust the number of channels in the model.
- `--narrow`: Control the model size by narrowing the layers.
- `--batch`: Batch size for training.
- `--ckpt`: Path to the pre-trained model checkpoint to resume training from.
- `--l1_weight`: Weight for the L1 loss.
- `--data_aug_image`: Apply data augmentation to images.
- `--drop_ref`: Drop the reference image during training to improve generalization.
- `--dataset_config`: Path to the dataset configuration file.
- `--wandb`: Enable logging to Weights and Biases (WandB).

## Inference

For inference, use the following command:

```bash
python inference.py \
--resize \
--face_parsing \
--face liuwei_30s.mp4 \
--audio source.wav \
--ckpt experiments/1e98f495-137b-49aa-b324-e96e4bfa1625/ckpts/340000.pth \
--ema \
--mask_ratio 0.6 \
--crop_down 0.1
```

### Explanation of Key Arguments:
- `--resize`: Resize the input images before processing.
- `--face_parsing`: Use face parsing for better facial region segmentation.
- `--face`: Path to the input video file.
- `--audio`: Path to the audio file to sync with the video.
- `--ckpt`: Model checkpoint path for inference.
- `--ema`: Enable Exponential Moving Average for smoother results.
- `--mask_ratio`: Define the mask ratio for face region to augment.
- `--crop_down`: Crop the input video from the bottom to improve focus on the face.

## Configuration

The dataset configuration and other model hyperparameters can be adjusted in the YAML configuration files located in the `configs/` directory. Ensure that the paths and settings in `configs/dataset_config_liuwei.yaml` are correctly set before training or inference.
