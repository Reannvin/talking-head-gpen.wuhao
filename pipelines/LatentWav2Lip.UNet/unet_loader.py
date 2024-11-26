import torch
from pytorch_lightning import LightningModule
from diffusers import UNet2DConditionModel
from diffusers import StableDiffusionPipeline
import json
import argparse
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LatentWav2LipUNet(LightningModule):
    def __init__(self, h):
        super().__init__()
        self.save_hyperparameters(h)

def extract_unet(ckpt_path, pipeline_name):
    # pipe = StableDiffusionPipeline.from_pretrained(pipeline_name, torch_dtype=torch.float16)
    unet = UNet2DConditionModel.from_pretrained(pipeline_name, subfolder="unet", torch_dtype=torch.float16)
        
    # make checkpoints folder if not exists
    import os
    os.makedirs(ckpt_path, exist_ok=True)
    
    # save unet config and weights
    unet.save_pretrained(ckpt_path)
    
    return unet

def load_unet(ckpt_path):
    # make sure if unet weights and config exist
    if not os.path.exists(f"{ckpt_path}/config.json"):
        raise ValueError(f"UNet weights or config not found in {ckpt_path}")
    
    unet = UNet2DConditionModel.from_pretrained(ckpt_path, torch_dtype=torch.float16)
    return unet

def validate_pipeline(pipeline_name):
    pipe = StableDiffusionPipeline.from_pretrained(pipeline_name, torch_dtype=torch.float16).to(device)
    prompt = "a photo of an astronaut riding a horse on mars"
    image = pipe(prompt).images[0]  
    return image
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pipe', type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument('--load', action="store_true", help="load unet model")
    parser.add_argument('--extract', action="store_true", help="extract unet model")
    parser.add_argument('--validate', action="store_true", help="validate pipeline")
    parser.add_argument('--ckpt', type=str, default="sd-v1-5-unet")
    args = parser.parse_args()
    
    if args.extract:
        print("Extracting UNet model")
        unet = extract_unet(ckpt_path=args.ckpt, pipeline_name=args.pipe)
        print(unet)
        print(f"UNet model saved in {args.ckpt}")
    elif args.load:
        print("Loading UNet model")
        unet = load_unet(ckpt_path=args.ckpt)
        print(unet)
        print(f"UNet model loaded from {args.ckpt}")
    elif args.validate:
        print("Validating pipeline")
        image = validate_pipeline(args.pipe)
        image.save("validate.png")
        print("Validation image saved as validate.png")
    else:
        print("Please specify an action to perform")
        print("python unet_loader.py --extract")
        print("python unet_loader.py --load")
        print("python unet_loader.py --validate")
    
