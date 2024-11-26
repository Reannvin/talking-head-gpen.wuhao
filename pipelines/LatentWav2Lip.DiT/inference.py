from torch.utils.data import Dataset
import os
import cv2
from PIL import Image
import torch, torchvision
import face_detection
from whisper.audio2feature import Audio2Feature
import numpy as np
from tqdm import tqdm
from helpers import encode_frames, decode_frames
import math
from einops import rearrange


class AudioVisualDatasetForInference(Dataset):
    def __init__(self, args):
        self.args = args
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((args.image_size, args.image_size)),
            torchvision.transforms.ToTensor()
        ])
        
        self.face_detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False, device=self.args.device)
        
        self.video_frames, self.face_images, self.face_bboxes = self.extract_face_sequences(self.args.face_path)
        self.video_height, self.video_width = self.video_frames[0].shape[:2]
        
        # Double the video_frames, face_sequences, face_bboxes, [1..N] -> [1..N, N..1]
        self.video_frames += self.video_frames[::-1]
        self.face_images += self.face_images[::-1]
        self.face_bboxes += self.face_bboxes[::-1]
        
        self.audio_sequences = self.extract_audio_sequences(self.args.audio_path, self.args.fps)
    
    def extract_face_sequences(self, face_path):
        # Load the video frames
        video_frames = self.get_video_frames(face_path)
        face_images = []
        face_bboxes = []
        
        # Process frames in batches
        print(f"Extracting face sequences from {len(video_frames)} frames")
        for i in tqdm(range(0, len(video_frames), self.args.face_detection_batch_size)):
            batch_frames = video_frames[i:i+self.args.face_detection_batch_size]
            
            # Call face_detection function on the batch (replace with your actual face detection method)
            batch_face_bboxes = self.detect_faces_batch(batch_frames)
            face_bboxes.extend(batch_face_bboxes)
            
            for frame, bbox in zip(batch_frames, batch_face_bboxes):
                x1, y1, x2, y2 = bbox
                                
                # Crop the face from the frame
                face_img = frame[y1:y2, x1:x2]
                
                # Convert to PIL image
                face_pil = Image.fromarray(face_img)
                
                # Apply necessary transformations
                face_transformed = self.transforms(face_pil)
                
                # Append to the list
                face_images.append(face_transformed)
        
        print(f"Extracted {len(face_images)} face images")        
        return video_frames, face_images, face_bboxes
    
    def get_video_frames(self, video_path):
        print(f"Loading video frames from {video_path}")
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError("Error opening video file")
        
        # set tqdm for progress bar
        progress_bar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            progress_bar.update(1)
        
        cap.release()
        return frames

    def detect_faces_batch(self, batch_frames):
        resized_frames = self.resize_frames_for_face_detection(batch_frames, self.args.image_size_for_face_detection)
        resized_frames_np = np.array(resized_frames)
        resized_faces = self.face_detector.get_detections_for_batch(resized_frames_np)
        faces = []
        for resized in resized_faces:
            # scale the face bounding box back to the original frame size
            scale_factor = self.args.image_size_for_face_detection / max(batch_frames[0].shape[:2])
            original = [int(coord / scale_factor) for coord in resized]  # Scale up the detection coordinates
            
            # move downward according to the crop-down ratio if needed
            if self.args.crop_down > 0:
                x1, y1, x2, y2 = original
                h = y2 - y1
                video_height = batch_frames[0].shape[0]
                y1 = min(y1 + int(h * self.args.crop_down), video_height)
                y2 = min(y2 + int(h * self.args.crop_down), video_height)
                original = [x1, y1, x2, y2]
            
            faces.append(original)
        return faces
        
    def resize_frames_for_face_detection(self, batch_frames, resize_to):
        resized_frames = []
        for frame in batch_frames:
            h, w = frame.shape[:2]
            if h > w:
                new_h = resize_to
                new_w = int((w / h) * resize_to)
            else:
                new_w = resize_to
                new_h = int((h / w) * resize_to)
            resized_frame = cv2.resize(frame, (new_w, new_h))
            resized_frames.append(resized_frame)
        return resized_frames
    
    def extract_audio_sequences(self, audio_path, fps):
        print(f"Extracting audio sequences from {audio_path}")
        whisper_processor = Audio2Feature(model_path=self.args.whisper_model)
        whisper_feature = whisper_processor.audio2feat(audio_path)
        audio_sequences = whisper_processor.feature2chunks(feature_array=whisper_feature,fps=fps)
        audio_sequences = [torch.tensor(audio_chunk).unsqueeze(0).float() for audio_chunk in audio_sequences]
        print(f"Extracted {len(audio_sequences)} audio sequences")
        return audio_sequences
            
    def __len__(self):
        return math.ceil((len(self.audio_sequences) - 1) / (self.args.syncnet_T - self.args.overlap))

    def __getitem__(self, idx):
        T = self.args.syncnet_T
        overlap_start = idx * (T - self.args.overlap) # overlap_start = 0, 3, 6, 9 ... for T = 4 
        
        face_idx_start = overlap_start % len(self.video_frames)
        face_idx_end = (face_idx_start + T) % len(self.video_frames)
        
        video_frames = self.video_frames[face_idx_start:face_idx_end]
        face_images = self.face_images[face_idx_start:face_idx_end]
        face_bboxes = self.face_bboxes[face_idx_start:face_idx_end]
        
        # if there's no enough audio sequences, repeat the last audio sequence
        audio_idx_start = overlap_start
        audio_idx_end = min(audio_idx_start + T, len(self.audio_sequences))
        if audio_idx_end - audio_idx_start < T:
            last_audio_sequence = self.audio_sequences[-1]
            remaining = T - (audio_idx_end - audio_idx_start)
            audio_sequences = self.audio_sequences[audio_idx_start:audio_idx_end] + [last_audio_sequence] * remaining
        else:
            audio_sequences = self.audio_sequences[audio_idx_start:audio_idx_end]
                
        # convert video_frames, face_images, face_bboxes to tensors, no encode
        video_frames = torch.tensor(np.array(video_frames)).permute(0, 3, 1, 2).div(255).float()
        face_images = torch.stack(face_images)        
        face_bboxes = torch.tensor(face_bboxes)
        audio_sequences = torch.stack(audio_sequences)
        return video_frames, face_images, face_bboxes, audio_sequences

from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from models import DiT_models

def load_models(args):
    latent_size = args.image_size // 8
    dit = DiT_models[args.model](
        input_size=latent_size,
        in_channels=4, # ref image + masked image + latent z
        out_channels=4,
        audio_embedding_dim=384,
        temporal_attention=args.temporal_attention,
    ).to(args.device)
    
    # load a custom DiT checkpoint from train.py:
    state_dict = torch.load(args.ckpt, weights_only=False)
    dit.load_state_dict(state_dict['ema' if args.ema else 'model']) # should use ema instead of model?
    dit.eval()  # important!
    
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}", local_files_only=True).to(args.device)
    return dit, diffusion, vae

def apply_mask(image, mask_ratio):
        # Apply a mask to the lower part of the image based on the mask_ratio
        mask_height = int(image.shape[3] * mask_ratio)
        image[:, :, :, -mask_height:, :] = 0.  # Setting masked pixels to 0
        return image
    
def generate_talking_faces(dit, diffusion, vae, face_images, audio_sequences, args):
    # Prepare image conditioning:
    masked = apply_mask(face_images.clone(), args.mask_ratio)
    i = torch.cat([masked, face_images], 2).permute(0, 2, 1, 3, 4).to(args.device)
    i = encode_frames(i, vae)
    i = i.view(-1, i.shape[2], i.shape[3], i.shape[4])    
    ii = torch.cat([i, i], 0)
    
    # Prepare audio conditioning:
    audio_sequences = audio_sequences.to(args.device)
    a = audio_sequences.view(-1, audio_sequences.shape[3], audio_sequences.shape[4])
    a_null = torch.zeros_like(a, device=args.device)
    aa = torch.cat([a, a_null], 0)
    
    # Prepare latent z:
    z = torch.randn(i.shape[0], 4, i.shape[2], i.shape[3], device=args.device)
    zz = torch.cat([z, z], 0)
    
    # Sample images:
    model_kwargs = dict(i=ii, a=aa, cfg_scale=args.cfg_scale)
    samples = diffusion.p_sample_loop(
        dit.forward_with_cfg, zz.shape, zz, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=args.device
    )
    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    samples = samples.view(-1, args.syncnet_T, samples.shape[1], samples.shape[2], samples.shape[3])
    samples = decode_frames(samples, vae)
    return samples.add(1).div(2).clamp(0, 1)

def generate_temporal_mask(syncnet_T, overlap):
    """
    Generate a mask based on the syncnet_T and overlap values.
    
    Args:
        syncnet_T (int): The total number of frames.
        overlap (int): The number of overlapping frames.
    
    Returns:
        List[int]: A mask of length syncnet_T where the first 'overlap' elements are 1, and the rest are 0.
    """
    if overlap > syncnet_T:
        raise ValueError("Overlap cannot be greater than syncnet_T")

    mask = [1] * overlap + [0] * (syncnet_T - overlap)
    return torch.tensor(mask)

def generate_talking_faces_with_rg(dit, diffusion, vae, overlaps, face_images, audio_sequences, args):
    # Prepare image conditioning:
    masked = apply_mask(face_images.clone(), args.mask_ratio)
    i = torch.cat([masked, face_images], 2).permute(0, 2, 1, 3, 4).to(args.device)
    i = encode_frames(i, vae)
    i = i.view(-1, i.shape[2], i.shape[3], i.shape[4])    
    ii = torch.cat([i, i], 0) if args.cfg else i
    
    # Prepare audio conditioning:
    audio_sequences = audio_sequences.to(args.device)
    a = audio_sequences.view(-1, audio_sequences.shape[3], audio_sequences.shape[4])
    a_null = torch.zeros_like(a, device=args.device)
    aa = torch.cat([a, a_null], 0) if args.cfg else a
    
    # Prepare latent z:
    z = torch.randn(i.shape[0], 4, i.shape[2], i.shape[3], device=args.device)
    # Add the last 'overlap' frames from the last batch to the current batch
    if overlaps is not None:
        overlaps = overlaps.permute(0, 2, 1, 3, 4)
        overlaps = encode_frames(overlaps, vae)
        z = rearrange(z, '(b f) c h w -> b f c h w', f=args.syncnet_T)
        z[:, :args.overlap] = overlaps
        z = rearrange(z, 'b f c h w -> (b f) c h w')
    # Duplicate the latent z for the cfg
    zz = torch.cat([z, z], 0) if args.cfg else z
    
    # Sample images:
    model_kwargs = dict(i=ii, a=aa)
    if args.cfg:
        model_kwargs['cfg_scale'] = args.cfg_scale
    
    if overlaps is not None:
        # Generate the sequence with overlaps
        samples = diffusion.p_sample_with_rg_loop(
            dit.forward_with_cfg if args.cfg else dit.forward,
            zz.shape, 
            zz, 
            clip_denoised=False, 
            model_kwargs=model_kwargs, 
            progress=True, 
            device=args.device, 
            syncnet_T=args.syncnet_T,
            mask=generate_temporal_mask(args.syncnet_T, args.overlap),
            rg_scale=args.rg_scale,
            use_grad=args.use_rg_grad,
        )
    else:
        # No overlaps, generate the whole sequence
        samples = diffusion.p_sample_loop(
            dit.forward_with_cfg if args.cfg else dit.forward,
            zz.shape, 
            zz, 
            clip_denoised=False, 
            model_kwargs=model_kwargs, 
            progress=True, 
            device=args.device
        )
    
    if args.cfg:
        samples, _ = samples.chunk(2, dim=0) # Remove null class samples
    samples = samples.view(-1, args.syncnet_T, samples.shape[1], samples.shape[2], samples.shape[3])
    samples = decode_frames(samples, vae)
    return samples.add(1).div(2).clamp(0, 1)
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument('--syncnet-T', type=int, choices=[1, 5, 10, 25], default=5)
    parser.add_argument('--face-path', type=str, required=True)
    parser.add_argument('--audio-path', type=str, required=True)
    parser.add_argument('--fps', type=int, default=25)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--face-detection-batch-size', type=int, default=32)
    parser.add_argument('--image-size-for-face-detection', type=int, default=640)
    parser.add_argument('--dry-run', action='store_true', help="If set, the script will only paste back the face images to the video frames and save them")
    parser.add_argument('--output-path', type=str, default='output')
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-S/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--cfg", action='store_true')
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mask-ratio", type=float, default=0.6)
    parser.add_argument("--crop-down", type=float, default=0.1)
    parser.add_argument("--whisper-model", type=str, default='tiny')
    parser.add_argument("--overlap", type=int, default=0)
    parser.add_argument('--temporal-attention', action='store_true')
    parser.add_argument('--use-rg', action='store_true')
    parser.add_argument('--use-rg-grad', action='store_true')
    parser.add_argument('--rg-scale', type=float, default=5.0)
    parser.add_argument('--ema', action='store_true')
    
    args = parser.parse_args()
    
    if args.use_rg or args.use_rg_grad:
        assert args.overlap > 0, "Overlap must be greater than 0 when using RG"
        args.batch_size = 1  # RG requires batch_size=1 due to its auto-regressive nature
    
    torch.manual_seed(args.seed)
    
    dit, diffusion, vae = load_models(args)
    print("Models loaded")
    
    dataset = AudioVisualDatasetForInference(args)    
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    print("Generating talking face sequences")
    os.makedirs('output', exist_ok=True)
    output_video = cv2.VideoWriter(f"{args.output_path}/temp.mp4", cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (dataset.video_width, dataset.video_height))
    
    last_batch_generated_faces = None
    for batch in tqdm(data_loader):
        video_frames, face_images, face_bboxes, audio_sequences = batch
        
        if args.dry_run:
            generated_faces = face_images
        else:
            if args.use_rg or args.use_rg_grad:
                overlaps = last_batch_generated_faces[:, -args.overlap:] if last_batch_generated_faces is not None else None
                generated_faces = generate_talking_faces_with_rg(dit, diffusion, vae, overlaps, face_images, audio_sequences, args)
            else:
                with torch.no_grad():
                    generated_faces = generate_talking_faces(dit, diffusion, vae, face_images, audio_sequences, args)
        
        last_batch_generated_faces = generated_faces
        
        # paste face images back to the video frames, according to the face bounding box, and make a video
        batch_size = video_frames.shape[0]
        T = video_frames.shape[1]
        
        for i in range(batch_size):
            for j in range(T - args.overlap):
                x1, y1, x2, y2 = face_bboxes[i, j]
                
                face_img = generated_faces[i, j]
                face_img = torch.nn.functional.interpolate(face_img.unsqueeze(0), (y2-y1, x2-x1), mode='bilinear', align_corners=False).squeeze(0)
                
                video_frame = video_frames[i, j]
                video_frame[:, y1:y2, x1:x2] = face_img
                
                output_video.write((video_frame.permute(1, 2, 0).cpu().numpy()[..., ::-1] * 255).astype(np.uint8))
            
    output_video.release()
    
    # add audio to the video
    os.system(f"ffmpeg -i {args.output_path}/temp.mp4 -i {args.audio_path} -c:v copy -c:a aac -strict experimental {args.output_path}/talking-face.mp4")
    os.remove(f"{args.output_path}/temp.mp4")