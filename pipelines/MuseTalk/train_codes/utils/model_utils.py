import torch.nn.functional as F
import torch
import torch.nn as nn
import time
import math
from .utils import decode_latents, preprocess_img_tensor
import os
from PIL import Image
from typing import Any, Dict, List, Optional, Tuple, Union
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
)
from torch import Tensor, nn
import logging
import json

RESIZED_IMG = 256

class PositionalEncoding(nn.Module):
    """
    Transformer 中的位置编码（positional encoding）
    """
    def __init__(self, d_model=384, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        b, seq_len, d_model = x.size()
        pe = self.pe[:, :seq_len, :]
        #print(b, seq_len, d_model)
        x = x + pe.to(x.device)
        return x

def validation(vae: torch.nn.Module,
               vae_fp32: torch.nn.Module,
      unet:torch.nn.Module, 
      unet_config,
      weight_dtype: torch.dtype,
      epoch: int,
      global_step: int,
      val_data_loader,
      output_dir,
      whisper_model_type,
      UNet2DConditionModel=UNet2DConditionModel,
      accelerator=None
     ):
    
     # Get the validation pipeline
    unet_copy = UNet2DConditionModel(**unet_config)
    
    unet_copy.load_state_dict(unet.state_dict())
    unet_copy.to(vae.device).to(dtype=weight_dtype)
    unet_copy.eval()
    
    if whisper_model_type == "tiny":
        pe = PositionalEncoding(d_model=384)
    elif whisper_model_type == "largeV2":
        pe = PositionalEncoding(d_model=1280)
    elif whisper_model_type == "tiny-conv":
        pe = PositionalEncoding(d_model=384)
        print(f" whisper_model_type: {whisper_model_type} Validation does not need PE")
    else:
        print(f"not support whisper_model_type {whisper_model_type}")
    pe.to(vae.device, dtype=weight_dtype)
    
    start = time.time()
    with torch.no_grad():
        val_loss_list, val_loss_lip_list, val_loss_latents_list = [], [], []
        for step, (ref_image, image, masked_image, masks, audio_feature) in enumerate(val_data_loader):

            masks = masks.unsqueeze(1).unsqueeze(1).to(vae.device)
            ref_image = preprocess_img_tensor(ref_image).to(vae.device)
            image = preprocess_img_tensor(image).to(vae.device)
            masked_image = preprocess_img_tensor(masked_image).to(vae.device)

             # Convert images to latent space 
            latents = vae.encode(image.to(dtype=weight_dtype)).latent_dist.sample() # init image
            latents = latents * vae.config.scaling_factor
            # Convert masked images to latent space
            masked_latents = vae.encode(
                masked_image.reshape(image.shape).to(dtype=weight_dtype)  # masked image
            ).latent_dist.sample()
            masked_latents = masked_latents * vae.config.scaling_factor
            # Convert ref images to latent space
            ref_latents = vae.encode(
                ref_image.reshape(image.shape).to(dtype=weight_dtype)  # ref image
            ).latent_dist.sample()
            ref_latents = ref_latents * vae.config.scaling_factor

            mask = torch.stack(
                [
                    torch.nn.functional.interpolate(mask, size=(mask.shape[-1] // 8, mask.shape[-1] // 8))
                    for mask in masks
                ]
            )
            mask = mask.reshape(-1, 1, mask.shape[-1], mask.shape[-1])
            bsz = latents.shape[0]
            timesteps = torch.tensor([0], device=latents.device)

            if unet_config['in_channels'] == 9:
                latent_model_input = torch.cat([mask, masked_latents, ref_latents], dim=1)
            else:
                latent_model_input = torch.cat([masked_latents, ref_latents], dim=1)
            audio_feature=audio_feature.to(dtype=weight_dtype)
            image_pred = unet_copy(latent_model_input, timesteps, encoder_hidden_states = audio_feature).sample
            
            image_pred_img = (1 / vae_fp32.config.scaling_factor) * image_pred.to(dtype=torch.float32)
            image_pred_img = vae_fp32.decode(image_pred_img).sample

            # Mask the top half of the image and calculate the loss only for the lower half of the image.
            image_pred_img = image_pred_img[:, :, image_pred_img.shape[2]//2:, :]
            image = image[:, :, image.shape[2]//2:, :]    
            val_loss_lip = F.l1_loss(image_pred_img.float(), image.float(), reduction="mean") # the loss of the decoded images
            val_loss_latents = F.l1_loss(image_pred.float(), latents.float(), reduction="mean") # the loss of the latents
        
            val_loss = 2.0*val_loss_lip + val_loss_latents # add some weight to balance the loss

            val_loss_lip_list.append(val_loss_lip.detach().item())
            val_loss_latents_list.append(val_loss_latents.detach().item())
            val_loss_list.append(val_loss.detach().item())

            image = Image.new('RGB', (RESIZED_IMG*4, RESIZED_IMG))
            image.paste(decode_latents(vae_fp32,masked_latents), (0, 0))
            image.paste(decode_latents(vae_fp32, ref_latents), (RESIZED_IMG, 0))
            image.paste(decode_latents(vae_fp32, latents), (RESIZED_IMG*2, 0))
            image.paste(decode_latents(vae_fp32, image_pred), (RESIZED_IMG*3, 0))
            
            val_img_dir = f"images/{output_dir}/{global_step}"
            if not os.path.exists(val_img_dir):
                os.makedirs(val_img_dir)
            image.save('{0}/val_epoch_{1}_{2}_image.png'.format(val_img_dir, global_step,step))

           # print("valtion in step:{0}, time:{1}".format(step,time.time()-start))
            
            if accelerator is not None:
                logs = {"val_loss_lip": val_loss_lip.detach().item(),
                        "val_loss_latents": val_loss_latents.detach().item(),
                        "val_loss": val_loss.detach().item()} 
#             accelerator.log(logs, step=global_step)

                accelerator.log(
                    {
                        "val_loss/loss": logs["val_loss"],
                        "val_loss/lip_loss": logs["val_loss_lip"],
                        "val_loss/latents_loss": logs["val_loss_latents"],
                    },
                    step=global_step,
                )
        print("valtion_done in epoch:{0}, time:{1}".format(epoch,time.time()-start))
        if accelerator is not None:
            print("val_loss_lip:",sum(val_loss_lip_list)/len(val_loss_lip_list))
            print("val_loss_latents:",sum(val_loss_latents_list)/len(val_loss_latents_list))
            print("val_loss:",sum(val_loss_list)/len(val_loss_list))
    
