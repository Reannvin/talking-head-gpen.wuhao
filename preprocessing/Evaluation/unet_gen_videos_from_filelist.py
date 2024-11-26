import numpy as np
import cv2, os, argparse
import subprocess
from tqdm import tqdm
import torch
import face_detection
from whisper.audio2feature import Audio2Feature
from diffusers import AutoencoderKL, UNet2DConditionModel

parser = argparse.ArgumentParser(description='Code to generate results for test filelists')

parser.add_argument('--filelist', type=str, 
					help='Filepath of filelist file to read', required=True)
parser.add_argument('--results_dir', type=str, help='Folder to save all results into', 
									required=True)
parser.add_argument('--data_root', type=str, required=True)
parser.add_argument('--checkpoint_path', type=str, 
					help='Name of saved checkpoint to load weights from', required=True)

parser.add_argument('--pads', nargs='+', type=int, default=[0, 0, 0, 0], 
					help='Padding (top, bottom, left, right)')
parser.add_argument('--face_det_batch_size', type=int, 
					help='Single GPU batch size for face detection', default=4)
parser.add_argument('--wav2lip_batch_size', type=int, help='Batch size for Wav2Lip', default=4)

parser.add_argument('--unet_config', type=str, help='Path to the unet config file', default='customized_unet_v4')
parser.add_argument('--vae_path', type=str, help='Path to the VAE model', default='stabilityai/sd-vae-ft-mse')

# parser.add_argument('--resize_factor', default=1, type=int)

args = parser.parse_args()
args.img_size = 768

def get_smoothened_boxes(boxes, T):
	for i in range(len(boxes)):
		if i + T > len(boxes):
			window = boxes[len(boxes) - T:]
		else:
			window = boxes[i : i + T]
		boxes[i] = np.mean(window, axis=0)
	return boxes

def face_detect(images):
	batch_size = args.face_det_batch_size
	
	while 1:
		predictions = []
		try:
			for i in range(0, len(images), batch_size):
				predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
		except RuntimeError:
			if batch_size == 1:
				raise RuntimeError('Image too big to run face detection on GPU')
			batch_size //= 2
			args.face_det_batch_size = batch_size
			print('Recovering from OOM error; New batch size: {}'.format(batch_size))
			continue
		break

	results = []
	pady1, pady2, padx1, padx2 = args.pads
	for rect, image in zip(predictions, images):
		if rect is None:
			raise ValueError('Face not detected!')

		y1 = max(0, rect[1] - pady1)
		y2 = min(image.shape[0], rect[3] + pady2)
		x1 = max(0, rect[0] - padx1)
		x2 = min(image.shape[1], rect[2] + padx2)
		
		results.append([x1, y1, x2, y2])

	boxes = get_smoothened_boxes(np.array(results), T=5)
	results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2), True] for image, (x1, y1, x2, y2) in zip(images, boxes)]

	return results 

def datagen(frames, face_det_results, mels):
	img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

	for i, m in enumerate(mels):
		if i >= len(frames): raise ValueError('Equal or less lengths only')

		frame_to_save = frames[i].copy()
		face, coords, valid_frame = face_det_results[i].copy()
		if not valid_frame:
			continue

		face = cv2.resize(face, (args.img_size, args.img_size))
			
		img_batch.append(face)
		mel_batch.append(m)
		frame_batch.append(frame_to_save)
		coords_batch.append(coords)

		if len(img_batch) >= args.wav2lip_batch_size:
			img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

			img_masked = img_batch.copy()
			img_masked[:, args.img_size//2:] = 0

			img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.

			yield img_batch, mel_batch, frame_batch, coords_batch
			img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

	if len(img_batch) > 0:
		img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

		img_masked = img_batch.copy()
		img_masked[:, args.img_size//2:] = 0

		img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.

		yield img_batch, mel_batch, frame_batch, coords_batch

fps = 25
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} for inference.'.format(device))

detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, 
											flip_input=False, device=device)

def load_model(path):
	unet_config = UNet2DConditionModel.load_config(f"{args.unet_config}/config.json")
	unet = UNet2DConditionModel.from_config(unet_config)
    
	print("Load checkpoint from: {}".format(path))
	#checkpoint = _load(path)
	s = torch.load(path, map_location=device)

	# if state_dict is in s
	if 'state_dict' in s:
		s = s['state_dict']
  
	new_s = {}
	for k, v in s.items():
		if k.startswith("unet."):
			new_s[k.replace('unet.', '')] = v
	unet.load_state_dict(new_s)

	unet = unet.to(device)
	return unet.eval()

model = load_model(args.checkpoint_path)

def to_latent(vae, img):
    resized_image = torch.nn.functional.interpolate(img, size=(args.img_size, args.img_size), mode='bilinear')
    rescaled_image = 2 * resized_image - 1
    masked_image = rescaled_image[:, :3]
    
    upper_half_image = masked_image[:, :, :masked_image.size(2) // 2]
    lower_half_zeros = torch.zeros_like(upper_half_image)
    stitch_upper_half_image = torch.cat([upper_half_image, lower_half_zeros], dim=2)
    
    reference_image = rescaled_image[:, 3:]
    
    with torch.no_grad():
        upper_half_latent = vae.encode(stitch_upper_half_image).latent_dist.sample()
        reference_latent = vae.encode(reference_image).latent_dist.sample()
    
    l = torch.cat([upper_half_latent, reference_latent], dim=1)
    scaling_factor = vae.config.scaling_factor
    l = l * scaling_factor
    return l
 
 
def from_latent(vae, latent):
    scaling_factor = vae.config.scaling_factor
    latent = latent / scaling_factor
    
    with torch.no_grad():
        image = vae.decode(latent).sample
    
    rescaled_image = (image + 1) / 2
    return rescaled_image

def reshape_face_sequences(tensor):
	batch_size, channels, groups, height, width = tensor.shape
	reshaped_tensor = tensor.permute(0, 2, 1, 3, 4).contiguous().view(batch_size * groups, channels, height, width)        
	return reshaped_tensor

def inverse_reshape_face_sequences(tensor):
	total_batch_size, channels, height, width = tensor.shape
	groups = 1
	batch_size = total_batch_size // groups
	
	if total_batch_size % groups != 0:
		raise ValueError("Total batch size is not divisible by the number of groups.")
	
	original_shape_tensor = tensor.view(batch_size, groups, channels, height, width).permute(0, 2, 1, 3, 4)        
	return original_shape_tensor

def reshape_audio_sequences(tensor):
	batch_size, dim1, dim2, dim3, features = tensor.shape
	reshaped_tensor = tensor.view(batch_size * dim1, dim2 * dim3, features)
	return reshaped_tensor

def main():
	assert args.data_root is not None
	data_root = args.data_root

	if not os.path.isdir(args.results_dir): os.makedirs(args.results_dir)

	with open(args.filelist, 'r') as filelist:
		lines = filelist.readlines()

	zero_timestep = torch.zeros([])
	whisper_processor = Audio2Feature(model_path='tiny')
	vae = AutoencoderKL.from_pretrained(args.vae_path, local_files_only=True).to(device)

	for idx, line in enumerate(tqdm(lines)):
		audio_src, video = line.strip().split()

		audio_src = os.path.join(data_root, audio_src) + '.mp4'
		video = os.path.join(data_root, video) + '.mp4'

		command = 'ffmpeg -loglevel panic -y -i {} -strict -2 {}'.format(audio_src, 'temp/temp.wav')
		subprocess.call(command, shell=True)
		temp_audio = 'temp/temp.wav'

		whisper_feature = whisper_processor.audio2feat(temp_audio)
		audio_chunks = whisper_processor.feature2chunks(feature_array=whisper_feature,fps=fps)
		audio_chunks = [torch.tensor(audio_chunk).unsqueeze(0).float() for audio_chunk in audio_chunks]
	
		video_stream = cv2.VideoCapture(video)
			
		full_frames = []
		while 1:
			still_reading, frame = video_stream.read()
			if not still_reading or len(full_frames) > len(audio_chunks):
				video_stream.release()
				break
			full_frames.append(frame)

		if len(full_frames) < len(audio_chunks):
			continue

		full_frames = full_frames[:len(audio_chunks)]

		try:
			face_det_results = face_detect(full_frames.copy())
		except ValueError as e:
			continue

		gen = datagen(full_frames.copy(), face_det_results, audio_chunks)

		for i, (img_batch, audio_batch, frames, coords) in enumerate(gen):
			if i == 0:
				frame_h, frame_w = full_frames[0].shape[:-1]
				out = cv2.VideoWriter('temp/result.avi', 
								cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))

			img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
			audio_batch = torch.FloatTensor(audio_batch).to(device)
   
			image_latent = to_latent(vae, img_batch)

			with torch.no_grad():
				image_latent = reshape_face_sequences(image_latent.unsqueeze(2))
				audio_batch = reshape_audio_sequences(audio_batch.unsqueeze(2))
				pred_latent = model(image_latent, timestep=zero_timestep, encoder_hidden_states=audio_batch).sample
				pred_latent = inverse_reshape_face_sequences(pred_latent)
				pred_latent = pred_latent.squeeze(2)
					
			pred_image = from_latent(vae, pred_latent)
			pred = pred_image.clamp(0, 1).cpu().numpy().transpose(0, 2, 3, 1) * 255.
			
			for pl, f, c in zip(pred, frames, coords):
				y1, y2, x1, x2 = c
				pl = cv2.resize(pl.astype(np.uint8), (x2 - x1, y2 - y1))
				f[y1:y2, x1:x2] = pl
				out.write(f)

		out.release()

		vid = os.path.join(args.results_dir, '{}.mp4'.format(idx))

		command = 'ffmpeg -loglevel panic -y -i {} -i {} -strict -2 -q:v 1 {}'.format(temp_audio, 
								'temp/result.avi', vid)
		subprocess.call(command, shell=True)

if __name__ == '__main__':
	main()
