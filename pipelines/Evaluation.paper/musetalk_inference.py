from os import listdir, path
import numpy as np
import librosa
import scipy, cv2, os, sys, argparse #, audio
import json, subprocess, random, string
from tqdm import tqdm
from glob import glob
import torch, face_detection
import platform
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from whisper.audio2feature import Audio2Feature
from diffusers import AutoencoderKL, UNet2DConditionModel
from face_parsing import FaceParser, BiSeNet
from PIL import Image
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from data_utils.blending import get_image
from mmpose.apis import inference_topdown, init_model
from mmpose.structures import merge_data_samples
from musetalk.unet import UNet,PositionalEncoding
from face_detection import FaceAlignment,LandmarksType
parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Musetalk models')

parser.add_argument('--checkpoint_path', type=str, 
                    help='Name of saved checkpoint to load weights from', required=True)


parser.add_argument('--face', type=str, 
                    help='Filepath of video/image that contains faces to use', required=True)
parser.add_argument('--audio', type=str, 
                    help='Filepath of video/audio file to use as raw audio source', required=True)
parser.add_argument('--outfile', type=str, help='Video path to save result. See default for an e.g.', 
                                default='results/result_voice.mp4')

parser.add_argument('--static', type=bool, 
                    help='If True, then use only first video frame for inference', default=False)
parser.add_argument('--fps', type=float, help='Can be specified only if input is a static image (default: 25)', 
                    default=25., required=False)

parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0], 
                    help='Padding (top, bottom, left, right). Please adjust to include chin at least')

parser.add_argument('--face_det_batch_size', type=int, 
                    help='Batch size for face detection', default=8)
parser.add_argument('--wav2lip_batch_size', type=int, help='Batch size for Wav2Lip model(s)', default=4)

parser.add_argument('--resize_factor', default=1, type=int, 
            help='Reduce the resolution by this factor. Sometimes, best results are obtained at 480p or 720p')
parser.add_argument('--vae_path', type=str, help='Path to the VAE model', default='stabilityai/sd-vae-ft-mse')
parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1], 
                    help='Crop video to a smaller region (top, bottom, left, right). Applied after resize_factor and rotate arg. ' 
                    'Useful if multiple face present. -1 implies the value will be auto-inferred based on height, width')

parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1], 
                    help='Specify a constant bounding box for the face. Use only as a last resort if the face is not detected.'
                    'Also, might work only if the face is not moving around much. Syntax: (top, bottom, left, right).')

parser.add_argument('--rotate', default=False, action='store_true',
                    help='Sometimes videos taken from a phone can be flipped 90deg. If true, will flip video right by 90deg.'
                    'Use if you get a flipped result, despite feeding a normal looking video')

parser.add_argument('--nosmooth', default=False, action='store_true',
                    help='Prevent smoothing face detections over a short temporal window')

parser.add_argument('--unet_config', type=str, help='Path to the unet config file', default='musetalk')
parser.add_argument('--wav2vec2', default=False, action='store_true', help='Use Wav2Vec2 embeddings instead of Whisper embeddings')
parser.add_argument('--image_size', type=int, default=256, help='Image size for the VAE model.')
parser.add_argument('--resize', action='store_true', help='Resize the input image to 1/4 of the original size.')
parser.add_argument("--resize_to", type=int, default=640, help="resize original image to this size, s3fd is no good for large image detection")
parser.add_argument('--face_parsing', action='store_true', help='Enable face parsing.')
parser.add_argument('--using_extended_bbox', action='store_true', help='using extended crop.')
parser.add_argument('--leftright_scale', help='Bbox left and right expansion coefficient', default=0.1, type=float)
parser.add_argument('--topbottom_scale', help='Bbox bottom expansion coefficient', default=0.1, type=float)
parser.add_argument('--lora_rank', type=int, default=64, help='Rank of Lora Matrix')
parser.add_argument('--lora_alpha', type=int, default=64, help='Alpha of Lora Matrix')
parser.add_argument('--lora_ckpt', type=str, help='Path to the lora checkpoint to load the model from')
parser.add_argument('--ema', action='store_true', help='Use EMA weights for inference')
parser.add_argument('--mask_ratio', type=float, default=0.5, help='Mask ratio for the image')
parser.add_argument('--crop_down', type=float, default=0, help='Crop down the image')
parser.add_argument('--cfg', action='store_true', help='Enable Classifier Free Guidance')
parser.add_argument('--guidance_scale', type=float, default=7.5, help='Guidance Scale')
parser.add_argument('--refine_config', type=str, help='Path to the refine config file', default='customized_unet_v4_large')
parser.add_argument('--refine_ckpt', type=str, help='Path to the refine checkpoint to load the model from')
parser.add_argument('--temp_dir', type=str, default='temp', help='temp dir for inference')
parser.add_argument('--coord_placeholder', type=tuple, default=(0.0,0.0,0.0,0.0), help='coord placeholder for the face')
args = parser.parse_args()

if os.path.isfile(args.face) and args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
    args.static = True

def get_smoothened_boxes(boxes, T):
    for i in range(len(boxes)):
        if i + T > len(boxes):
            window = boxes[len(boxes) - T:]
        else:
            window = boxes[i : i + T]
        boxes[i] = np.mean(window, axis=0)
    return boxes




def face_parsing(images, face_parsing_batch):

    pil_images = []
    for i in images:
        # 将BGR格式的frame转换为RGB格式
        frame_rgb = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
            
        # 转换为PIL Image
        pil_image = Image.fromarray(frame_rgb)

        # 添加到PIL Image列表
        pil_images.append(pil_image)

    face_parser = FaceParser()
    face_masks = []
    for i in range(0, len(pil_images), face_parsing_batch):
        start_frame = i
        end_frame = min(i + face_parsing_batch, len(pil_images))
        masks = face_parser.parse(pil_images[start_frame:end_frame])
        face_masks.extend(masks)
   
    return face_masks

def datagen(frames, audios):
    # double the frames, from [0, N) to [0, N) + (N, 0]
    frames = frames + frames[::-1]
    
    img_batch, audio_batch, frame_batch, coords_batch, mask_batch = [], [], [], [], []

    if args.box[0] == -1:    # 这里可改为从数据预处理获取bbox
        if not args.static:
            face_det_results = get_landmark_and_bbox(frames) # BGR2RGB for CNN face detection
        else:
            face_det_results = get_landmark_and_bbox([frames[0]])
    else:
        print('Using the specified bounding box instead of face detection...')
        y1, y2, x1, x2 = args.box
        face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]
    # print(face_det_results)
    for i, a in enumerate(audios):
        idx = 0 if args.static else i%len(frames)
        frame_to_save = frames[idx].copy()
        face, coords = face_det_results[idx].copy()

        face = cv2.resize(face, (args.image_size, args.image_size))
            
        img_batch.append(face)
        audio_batch.append(a)
        frame_batch.append(frame_to_save)
        coords_batch.append(coords)

        if len(img_batch) >= args.wav2lip_batch_size:

            # if args.face_parsing:
            #     face_masks = face_parsing(img_batch, args.wav2lip_batch_size)
            #     mask_batch = np.asarray(face_masks)

            img_batch, audio_batch = np.asarray(img_batch), torch.stack(audio_batch)

            img_masked = img_batch.copy()
            
            # mask lower part of the image, according to mask_ratio
            mask_height = int(args.image_size * args.mask_ratio)
            img_masked[:, args.image_size - mask_height:] = 0.

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.

            yield img_batch, audio_batch, frame_batch, coords_batch, mask_batch
            img_batch, audio_batch, frame_batch, coords_batch, mask_batch = [], [], [], [], []

    if len(img_batch) > 0:

        # if args.face_parsing:
        #         face_masks = face_parsing(img_batch, args.wav2lip_batch_size)
        #         mask_batch = np.asarray(face_masks)

        img_batch, audio_batch = np.asarray(img_batch), torch.stack(audio_batch)
  
        if args.face_parsing:
            mask_batch = np.asarray(mask_batch)

        img_masked = img_batch.copy()
        
        # mask lower part of the image, according to mask_ratio
        mask_height = int(args.image_size * args.mask_ratio)
        img_masked[:, args.image_size - mask_height:] = 0.

        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.

        yield img_batch, audio_batch, frame_batch, coords_batch, mask_batch

audio_step_size = 10
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} for inference.'.format(device))
config_file = './musetalk/dwpose/rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py'
checkpoint_file = './checkpoints/dw-ll_ucoco_384.pth'
model = init_model(config_file, checkpoint_file, device=device)
fa = FaceAlignment(LandmarksType._2D, flip_input=False,device=device)
def get_landmark_and_bbox(frames, upperbondrange=0):
    batch_size_fa = 4
    batches = [frames[i:i + batch_size_fa] for i in range(0, len(frames), batch_size_fa)]
    results_list = []  # This will hold lists of [cropped_face, coords] for each frame

    if upperbondrange != 0:
        print('get key_landmark and face bounding boxes with the bbox_shift:', upperbondrange)
    else:
        print('get key_landmark and face bounding boxes with the default value')
    
    average_range_minus = []
    average_range_plus = []

    for fb in tqdm(batches):
        results = inference_topdown(model, np.asarray(fb)[0])
        results = merge_data_samples(results)
        keypoints = results.pred_instances.keypoints
        face_land_mark = keypoints[0][23:91].astype(np.int32)
        
        # Get bounding boxes by face detection
        bbox_list = fa.get_detections_for_batch(np.asarray(fb))
        
        for j, bbox in enumerate(bbox_list):
            if bbox is None:  # No face in the image
                results_list.append([None, None])
                continue

            half_face_coord = face_land_mark[29]
            range_minus = (face_land_mark[30] - face_land_mark[29])[1]
            range_plus = (face_land_mark[29] - face_land_mark[28])[1]
            average_range_minus.append(range_minus)
            average_range_plus.append(range_plus)
            if upperbondrange != 0:
                half_face_coord[1] = upperbondrange + half_face_coord[1]

            half_face_dist = np.max(face_land_mark[:, 1]) - half_face_coord[1]
            upper_bond = half_face_coord[1] - half_face_dist

            f_landmark = (np.min(face_land_mark[:, 0]), int(upper_bond), np.max(face_land_mark[:, 0]), np.max(face_land_mark[:, 1]))
            x1, y1, x2, y2 = f_landmark

            if y2 - y1 <= 0 or x2 - x1 <= 0 or x1 < 0:  # If the landmark bbox is not suitable, reuse the bbox
                coords = bbox
                cropped_face = None
                print("Error bbox:", bbox)
            else:
                coords = f_landmark
                cropped_face = fb[j][y1:y2, x1:x2] if y1 < y2 and x1 < x2 else None

            results_list.append([cropped_face, coords])
                
    print("********************************************bbox_shift parameter adjustment**********************************************************")
    print(f"Total frame:「{len(frames)}」 Manually adjust range : [ -{int(sum(average_range_minus) / len(average_range_minus))}~{int(sum(average_range_plus) / len(average_range_plus))} ] , the current value: {upperbondrange}")
    print("*************************************************************************************************************************************")

    return results_list

    
def _load(checkpoint_path):
    if device == 'cuda:':
        checkpoint = torch.load(checkpoint_path,map_location=torch.device('cuda'))
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint

def add_lora(unet, lora_config, lora_ckpt=None):
        unet.add_adapter(lora_config)
        
        if lora_ckpt is not None:
            print(f"Loading Lora checkpoint from {lora_ckpt}")
            lora_state_dict = torch.load(args.lora_ckpt)['state_dict']
            new_s = {}
            for k, v in lora_state_dict.items():
                if k.startswith("unet."):
                    new_s[k.replace('unet.', '')] = v
            unet.load_state_dict(new_s, strict=False)
            
def load_model(args):
    # unet_config = UNet2DConditionModel.load_config(f"{args.unet_config}/config.json")
    # unet = UNet2DConditionModel.from_config(unet_config)
    unet = UNet(unet_config=f"{args.unet_config}/config.json",
                model_path =args.checkpoint_path)
    print("Load model from: {}".format(args.checkpoint_path))
    
    # s = checkpoint
    # if 'state_dict' in s:
    #     s = s['state_dict']
  
    # new_s = {}
    # for k, v in s.items():
    #     if k.startswith("unet."):
    #         new_s[k.replace('unet.', '')] = v
    # unet.load_state_dict(new_s)
    
    # Apply LoRA fine-tuning if enabled
    if args.lora_ckpt is not None:
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=[
                "to_q",
                "to_k",
                "to_v",
                "to_out.0",
                "proj_in",
                "proj_out",
                "ff.net.0.proj",
                "ff.net.2",
                "conv1",
                "conv2",
                "conv_shortcut",
                "downsamplers.0.conv",
                "upsamplers.0.conv",
                "time_emb_proj",
            ],
        )
        
        add_lora(unet, lora_config, args.lora_ckpt)
        
    # unet = unet.to(device)
    
    # Load EMA if enabled
    if args.ema and 'ema' in checkpoint:
        from diffusers.training_utils import EMAModel
        ema = EMAModel(unet.parameters())
        ema.load_state_dict(checkpoint['ema'])
        ema.to(device)
        ema.copy_to(unet.parameters())
        print("Loaded EMA from checkpoint")
    
    return unet

def load_refiner(args):
    refine_config = UNet2DConditionModel.load_config(f"{args.refine_config}/config.json")
    refine = UNet2DConditionModel.from_config(refine_config)
    
    print("Load refiner from: {}".format(args.refine_ckpt))
    checkpoint = torch.load(args.refine_ckpt, map_location=device)
    
    s = checkpoint
    if 'state_dict' in s:
        s = s['state_dict']
  
    new_s = {}
    for k, v in s.items():
        if k.startswith("refine."):
            new_s[k.replace('refine.', '')] = v
    refine.load_state_dict(new_s)
    
    refine = refine.to(device)
    
    # Load EMA if enabled
    if args.ema and 'ema' in checkpoint:
        from diffusers.training_utils import EMAModel
        ema = EMAModel(refine.parameters())
        ema.load_state_dict(checkpoint['ema'])
        ema.to(device)
        ema.copy_to(refine.parameters())
        print("Loaded EMA from checkpoint")
    
    return refine.eval()

def to_latent(vae, img):
    rescaled_image = 2 * img - 1
    masked_image = rescaled_image[:, :3]
    reference_image = rescaled_image[:, 3:]
    
    with torch.no_grad():
        upper_half_latent = vae.encode(masked_image).latent_dist.sample()
        reference_latent = vae.encode(reference_image).latent_dist.sample()
    
    l = torch.cat([upper_half_latent, reference_latent], dim=1)
    scaling_factor = vae.config.scaling_factor
    l = l * scaling_factor
    return l
 
 
def from_latent(vae, latent):
    # scale back the latent
    scaling_factor = vae.config.scaling_factor
    latent = latent / scaling_factor
    
    # latent N x 4 x H x W
    with torch.no_grad():
        image = vae.decode(latent).sample
    
    # rescale image to [0, 1]
    rescaled_image = (image + 1) / 2
    return rescaled_image

def reshape_face_sequences(tensor):
    """
    Reshape and concatenate a tensor assuming a format similar to [64, 8, 5, 96, 96],
    but dimensions are taken from the input tensor to increase flexibility.

    Parameters:
        tensor (torch.Tensor): A tensor with dimensions [batch_size, channels, groups, height, width].
    
    Returns:
        torch.Tensor: A reshaped tensor with dimensions [batch_size * groups, channels, height, width].
    """
    # 获取输入tensor的维度
    batch_size, channels, groups, height, width = tensor.shape
            
    # Reshape the tensor
    reshaped_tensor = tensor.permute(0, 2, 1, 3, 4).contiguous().view(batch_size * groups, channels, height, width)        
    return reshaped_tensor

def inverse_reshape_face_sequences(tensor):
    """
    Inverse operation for the reshape_face_sequences function, reconstructing the original tensor
    from a reshaped format of [batch_size, channels * groups, height, width].
    
    Parameters:
        tensor (torch.Tensor): A tensor with dimensions [batch_size * groups, channels, height, width].
    
    Returns:
        torch.Tensor: A tensor with dimensions [batch_size, channels, groups, height, width].
    """
    total_batch_size, channels, height, width = tensor.shape
    groups = 1
    batch_size = total_batch_size // groups
    
    # check if the total batch size is divisible by the number of groups
    if total_batch_size % groups != 0:
        raise ValueError("Total batch size is not divisible by the number of groups.")
    
    # Reshape the tensor to its original dimensions
    original_shape_tensor = tensor.view(batch_size, groups, channels, height, width).permute(0, 2, 1, 3, 4)        
    return original_shape_tensor

def reshape_audio_sequences(tensor):
    """
    Reshape a tensor from [batch_size, dim1, dim2, dim3, features] to [batch_size * dim1, dim2 * dim3, features].
    
    Parameters:
        tensor (torch.Tensor): A tensor with dimensions [batch_size, dim1, dim2, dim3, features].
    
    Returns:
        torch.Tensor: A reshaped tensor with dimensions [batch_size * dim1, dim2 * dim3, features].
    """
    batch_size, dim1, dim2, dim3, features = tensor.shape
    
    # Reshape the tensor
    reshaped_tensor = tensor.view(batch_size * dim1, dim2 * dim3, features)
    
    # print(f"Reshaped tensor shape: {reshaped_tensor.shape}")
    return reshaped_tensor

def main():
    if not os.path.isfile(args.face):
        raise ValueError('--face argument must be a valid path to video/image file')

    elif args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
        full_frames = [cv2.imread(args.face)]
        fps = args.fps

    else:
        video_stream = cv2.VideoCapture(args.face)
        fps = video_stream.get(cv2.CAP_PROP_FPS)

        print('Reading video frames...')

        full_frames = []
        while 1:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break
            if args.resize_factor > 1:
                frame = cv2.resize(frame, (frame.shape[1]//args.resize_factor, frame.shape[0]//args.resize_factor))

            if args.rotate:
                frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)

            y1, y2, x1, x2 = args.crop
            if x2 == -1: x2 = frame.shape[1]
            if y2 == -1: y2 = frame.shape[0]

            frame = frame[y1:y2, x1:x2]

            full_frames.append(frame)

    print ("Number of frames available for inference: "+str(len(full_frames)))

    if not args.audio.endswith('.wav'):
        print('Extracting raw audio...')
        command = 'ffmpeg -y -i {} -strict -2 {}'.format(args.audio, f'{args.temp_dir}/temp.wav')

        subprocess.call(command, shell=True)
        args.audio = f'{args.temp_dir}/temp.wav'


 
    if not args.wav2vec2:
        whisper_processor = Audio2Feature(model_path='tiny')
        whisper_feature = whisper_processor.audio2feat(args.audio)
        audio_chunks = whisper_processor.feature2chunks(feature_array=whisper_feature,fps=fps)
        audio_chunks = [torch.tensor(audio_chunk).unsqueeze(0).float() for audio_chunk in audio_chunks]
    else:
        wav = librosa.load(args.audio, sr=16000)[0]
        wav = torch.tensor(wav).to(device)
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h", local_files_only=True)
        wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h", local_files_only=True).to(device)
        inputs = processor(wav, return_tensors="pt", sampling_rate=16000).to(device)
        with torch.no_grad():
            outputs = wav2vec2(**inputs)
        audio_embeddings = outputs.last_hidden_state
    
        if np.isnan(audio_embeddings.cpu().reshape(-1)).sum() > 0:
            raise ValueError('Audio contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

        audio_chunks = []
        audio_idx_multiplier = 50./fps 
        i = 0
        while 1:
            start_idx = int(i * audio_idx_multiplier)
            if start_idx + audio_step_size > len(audio_embeddings[0]):
                audio_chunks.append(audio_embeddings[:, len(audio_embeddings[0]) - audio_step_size:])
                break
            audio_chunks.append(audio_embeddings[:, start_idx : start_idx + audio_step_size])
            i += 1

    print("Length of audio chunks: {}".format(len(audio_chunks)))
    print("Shape of audio chunks: {}".format(audio_chunks[0].shape))

    full_frames = full_frames[:len(audio_chunks)]

    batch_size = args.wav2lip_batch_size
    zero_timestep = torch.zeros([])
    gen = datagen(full_frames.copy(), audio_chunks)
    index = 0
    for i, (img_batch, audio_batch, frames, coords, masks) in enumerate(tqdm(gen, 
                                            total=int(np.ceil(float(len(audio_chunks))/batch_size)))):
        if i == 0:
            unet = load_model(args)
            print ("Model loaded")
            
            if args.refine_ckpt is not None:
                refiner = load_refiner(args)
                print ("Refiner loaded")

            frame_h, frame_w = full_frames[0].shape[:-1]
            out = cv2.VideoWriter(f'{args.temp_dir}/result.avi', 
                                    cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))

        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
        audio_batch = audio_batch.to(device)

        #torch.cuda.empty_cache()
        # load vae
        vae = AutoencoderKL.from_pretrained(args.vae_path, local_files_only=True).to(device)
        pe = PositionalEncoding(d_model=384)
        image_latent = to_latent(vae, img_batch)

        # assert shape of image_latent be batch_size x 8 x 96 x 96
        # assert image_latent.size(0) == img_batch.size(0) and image_latent.size(1) == 8 and image_latent.size(2) == 96 and image_latent.size(3) == 96 

        with torch.no_grad():
            latent_face_sequences = reshape_face_sequences(image_latent.unsqueeze(2))
            audio_sequences = reshape_audio_sequences(audio_batch.unsqueeze(2))
            audio_sequences = pe(audio_sequences)      
            if args.cfg:
                # use classifier free guidance
                unconditional_audio = torch.zeros_like(audio_sequences).to(device)
                combined_audio = torch.cat([unconditional_audio, audio_sequences])
                latent_face_sequences = torch.cat([latent_face_sequences] * 2)
                pred = unet.model(latent_face_sequences, timestep=zero_timestep, encoder_hidden_states=combined_audio).sample
                uncond_pred, cond_pred = pred.chunk(2)
                g_latent = uncond_pred + args.guidance_scale * (cond_pred - uncond_pred)
            elif args.refine_ckpt is not None:
                g_base = unet.model(latent_face_sequences, timestep=zero_timestep, encoder_hidden_states=audio_sequences).sample
                # replace first part of latent_face_sequences with the base model prediction
                latent_face_sequences[:, :4] = g_base
                g_delta = refiner(latent_face_sequences, timestep=zero_timestep, encoder_hidden_states=audio_sequences).sample
                g_latent = g_base + g_delta
            else:
                g_latent = unet.model(latent_face_sequences, timestep=zero_timestep, encoder_hidden_states=audio_sequences).sample
            
            g_latent = inverse_reshape_face_sequences(g_latent)
            g_latent = g_latent.squeeze(2)

        # assert shape of pred_latent be batch_size x 4 x 96 x 96
        # assert pred_latent.size(0) == img_batch.size(0) and pred_latent.size(1) == 4 and pred_latent.size(2) == 96 and pred_latent.size(3) == 96
  
        pred_image = from_latent(vae, g_latent)
  
        pred = pred_image.clamp(0, 1).cpu().numpy().transpose(0, 2, 3, 1) * 255.
        pred = pred.astype(np.uint8)
        cv2.imwrite(f'pred.png', pred[0])
        if args.face_parsing:
            # for p, f, c, m in zip(pred, frames, coords, masks):
            for p, f, c in zip(pred, frames, coords):
                y1, y2, x1, x2 = c
                p_resized = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))

                # mask_resized = cv2.resize(m, (x2 - x1, y2 - y1))

                # # Ensure mask is binary (0 or 1)
                # mask_resized = (mask_resized > 0).astype(np.uint8) * 255

                # # Apply Gaussian Blur to the mask to smooth edges
                # kernel_size = mask_resized.shape[0] // 10 // 2 * 2 + 1
                # mask_blurred = cv2.GaussianBlur(mask_resized, (kernel_size, kernel_size), 0)

                # # Convert mask to float and normalize to range 0 to 1
                # mask_blurred = mask_blurred.astype(np.float32) / 255.

                # # extend mask_blurred from 1 to 3 channels
                # mask_blurred = np.stack([mask_blurred] * 3, axis=-1)

                # # Extract the region of interest from the original frame and prediction
                # roi_f = f[y1:y2, x1:x2].astype(np.float32)
                # roi_p = p_resized.astype(np.float32)

                # # Perform alpha blending
                # blended = roi_f * (1 - mask_blurred) + roi_p * mask_blurred

                # # Convert the blended result back to uint8
                # blended = blended.astype(np.uint8)

                # # Place the blended region back into the frame
                # f[y1:y2, x1:x2] = blended

                # out.write(f)

                c=x1,y1,x2,y2
                new_frame = get_image(f,p_resized,c)
                out.write(new_frame)
        else:
            for p, f, c in zip(pred, frames, coords):
                x1, y1, x2, y2 = c
                p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
                f[y1:y2, x1:x2] = p
                out.write(f)


    out.release()

    command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(args.audio, f'{args.temp_dir}/result.avi', args.outfile)
    subprocess.call(command, shell=platform.system() != 'Windows')

if __name__ == '__main__':
    main()

