
import numpy as np
import cv2, os, argparse 
from tqdm import tqdm
import torch
from face_model.gpen_model import FullGenerator 
parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')

parser.add_argument('--checkpoint_path', type=str, 
                    help='Name of saved checkpoint to load weights from', required=True)

parser.add_argument('--output', type=str, help='Video path to save result. See default for an e.g.', 
                                default='./results/stage1')

parser.add_argument('--image_size', type=int, default=256)
parser.add_argument('--size', type=int, default=256)
parser.add_argument('--channel_multiplier', type=int, default=2)
parser.add_argument('--narrow', type=float, default=1.0)
args = parser.parse_args()
    
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} for inference stage 1.'.format(device))

def load_model(args):
    print("Load model from: {}".format(args.checkpoint_path))
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    args.latent = 512
    args.n_mlp = 8
    generator = FullGenerator(args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier, narrow=args.narrow, device=device, stage=1)
    generator.load_state_dict(checkpoint['g_ema'])
    return generator.eval()

def main():
    os.makedirs(args.output, exist_ok= True)
    model = load_model(args).to(device)
    print ("Model loaded")
    for i in tqdm(range(5)):
        sample_z = torch.randn(6, args.image_size, args.image_size)
        sample_z = sample_z.to(device)
        with torch.no_grad():
            pred_image, _ = model(inputs=sample_z.unsqueeze(0), audio=torch.empty((1, 0, 0)))
            pred_image = pred_image[:, [2, 1, 0]]
            pred_image = (pred_image + 1) / 2
        pred = pred_image.clamp(0, 1).cpu().numpy().transpose(0, 2, 3, 1) * 255.
        pred = pred.astype(np.uint8)
        cv2.imwrite(f"{args.output}/{i}.jpg", pred[0])
if __name__ == '__main__':
    main()
