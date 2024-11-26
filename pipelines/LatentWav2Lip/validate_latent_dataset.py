import argparse
import os
from tqdm import tqdm
import torch
import torchaudio
from diffusers import AutoencoderKL
from PIL import Image
from torchvision import transforms
import audio
from hparams import hparams
from models import SyncNet_color as SyncNet

use_cuda = torch.cuda.is_available()
print('use_cuda: {}'.format(use_cuda))

def _load(checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint

def load_checkpoint(path, model, optimizer, reset_optimizer=False, overwrite_global_states=True):
    global global_step
    global global_epoch

    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    if overwrite_global_states:
        global_step = checkpoint["global_step"]
        global_epoch = checkpoint["global_epoch"]

    return model

# define image transform
image_size = 768

class RescaleTransform:
    """将图像像素值从[0, 1]缩放到[-1, 1]的转换"""
    def __call__(self, tensor):
        return (tensor * 2.0) - 1.0
    
image_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    RescaleTransform(),  # 使用自定义的转换类替代transforms.Lambda
])

def get_videos_list(root_dir):
    videos_list = []
    for person_id in sorted(os.listdir(root_dir)):
        person_path = os.path.join(root_dir, person_id)
        for video_id in sorted(os.listdir(person_path)):
            video_path = os.path.join(person_path, video_id)
            videos_list.append((video_path, person_id, video_id))
    return videos_list

def validate(vae, syncnet, image_root, latent_root):
    image_video_list = get_videos_list(image_root)
    latent_video_list = get_videos_list(latent_root)
    
    # Print out size of the two lists
    print(f"Image video list size: {len(image_video_list)}")
    print(f"Latent video list size: {len(latent_video_list)}")
    
    # Check every video folder for latent dataset
    checked_latent_video_count = 0
    for video_path, person_id, video_id in tqdm(latent_video_list):
        latent_file = os.path.join(video_path, "latent.pt")
        audio_file = os.path.join(video_path, "audio.wav")
        
        # 1. to make sure latent.pt and audio.wav exist.
        if not os.path.exists(latent_file) or not os.path.exists(audio_file):
            print(f"Missing latent.pt or audio.wav for video: {video_path}")
            continue
        
        # 2. to make sure sample rate of audio.wav is 16000
        _, sample_rate = torchaudio.load(audio_file)
        if sample_rate != 16000:
            print(f"Audio.wav sample rate is not 16000 for video: {video_path}")
            continue
        
        # 3. to make sure latent.pt includes keys of "frame_ids", "full_image", "upper_half" and "lower_half"
        latent_dict = torch.load(latent_file)
        if ["frame_ids", "full_image", "upper_half", "lower_half"] != list(latent_dict.keys()):
            print(f"Keys of latent.pt is not correct for video: {video_path}")
            continue
        
        # 4. to make sure frame_ids is sorted
        frame_ids = latent_dict["frame_ids"]
        sorted_frame_ids = sorted(frame_ids)
        if frame_ids != sorted_frame_ids:
            print(f"frame_ids is not sorted for video: {video_path}")
            continue
        
        # 5. to make sure frame_ids is sorted and continuous
        frame_ids = latent_dict["frame_ids"]
        for i in range(len(frame_ids)):
            if frame_ids[i] != i:
                print(f"frame_ids is not continuous for video: {video_path} in frame {i}")
                break
            
        # 6. to make sure full_image: [N, 4, 96, 96], upper_half: [N, 4, 48, 96] and lower_half: [N, 4, 48, 96] are all tensors in correct shape
        full_image_latent = latent_dict["full_image"]
        upper_half_latent = latent_dict["upper_half"]
        lower_half_latent = latent_dict["lower_half"]
        
        if full_image_latent.shape != (len(frame_ids), 4, 96, 96):
            print(f"full_image shape is not correct for video: {video_path}")
            continue
        
        if upper_half_latent.shape != (len(frame_ids), 4, 48, 96):
            print(f"upper_half shape is not correct for video: {video_path}")
            continue
        
        if lower_half_latent.shape != (len(frame_ids), 4, 48, 96):
            print(f"lower_half shape is not correct for video: {video_path}")
            continue
        
        # 7. to decode latent and compare with the image with L1 loss in batches
        # original_image_paths = [path for path in os.listdir(os.path.join(image_root, person_id, video_id)) if path.endswith(".jpg")]
        # sorted_original_image_path = sorted(original_image_paths, key=lambda x: int(os.path.basename(x).split(".")[0]))
        # if (len(sorted_original_image_path) - 1) != int(os.path.basename(sorted_original_image_path[-1]).split(".")[0]):
        #     # only check for hight risk videos, eg: missing frames
        #     print(f"Original image paths are missing frames for video: {video_path}, checking loss...")
        loss_full, loss_upper, loss_lower, loss_sync = decode_and_compare(vae, syncnet, image_root, person_id, video_id, latent_dict)
        print(f"Loss full: {loss_full}, Loss upper: {loss_upper}, Loss lower: {loss_lower}, Loss sync: {loss_sync}")
        
        checked_latent_video_count += 1
    print(f"Checked latent video count: {checked_latent_video_count}")

def load_original_image_batch(image_root, person_id, video_id, frame_ids_slice):
    image_paths = [os.path.join(image_root, person_id, video_id, f"{frame_id.item()}.jpg") for frame_id in frame_ids_slice]
    images = [Image.open(image_path).convert("RGB") for image_path in image_paths]
    images = [image_transform(image) for image in images]
    images = torch.stack(images)
    return images

syncnet_T = 5
syncnet_mel_step_size = 16

def load_audio_to_mel(audio_path):
    wav = audio.load_wav(audio_path, hparams.sample_rate)
    orig_mel = audio.melspectrogram(wav).T
    # print(f"orig_mel shape: {orig_mel.shape}")
    return orig_mel

def crop_audio_window(spec, start_frame):
    start_idx = int(80. * (start_frame / float(hparams.fps)))
    end_idx = start_idx + syncnet_mel_step_size
    return spec[start_idx : end_idx, :]

def compute_audio_video_sync(video_path, syncnet, mel, frame_ids, lower_half_decoded, i):    
    if lower_half_decoded.size(0) < syncnet_T:
        return None

    # check if missing frames
    for offset in range(0, syncnet_T):
        if frame_ids[i + offset] != frame_ids[i] + offset:
            print(f"Missing frames in video: {video_path}, frame index: {frame_ids[i + offset]}")
            return None
    
    # audio
    start_frame = frame_ids[i]    
    mel_crop = crop_audio_window(mel, start_frame)
    mel_crop = torch.FloatTensor(mel_crop).to(device)
    mel_crop = mel_crop.T.unsqueeze(0).unsqueeze(0)

    # video
    video_crop = lower_half_decoded[:syncnet_T]
    
    # save resized_video_crop image
    # stacked_video_crop = torch.cat([video_crop[i] for i in range(syncnet_T)], dim=2).permute(1, 2, 0)
    # print(stacked_video_crop.shape)
    # stacked_video_crop = ((stacked_video_crop + 1) / 2 * 255).byte().cpu().numpy()
    # os.makedirs(video_path, exist_ok=True)
    # Image.fromarray(stacked_video_crop).save(f"{video_path}/resized_video_crop_{i}.jpg")
    
    video_crop = video_crop.view(-1, video_crop.size(2), video_crop.size(3)).unsqueeze(0)
    resized_video_crop = torch.nn.functional.interpolate(video_crop, size=(48, 96), mode='bilinear', align_corners=False)
    resized_video_crop = resized_video_crop.add(1).div(2).clamp(0, 1)
    
    with torch.no_grad():
        audio_emb, video_emb = syncnet(mel_crop, resized_video_crop)
    
    sync_loss = torch.nn.functional.cosine_embedding_loss(audio_emb, video_emb, torch.ones(1).float().to(device))
    return sync_loss
    
def decode_and_compare(vae, syncnet, image_root, person_id, video_id, latent_dict, batch_size=syncnet_T):
    mel = load_audio_to_mel(os.path.join(image_root, person_id, video_id, "audio.wav"))
    frame_ids = latent_dict["frame_ids"]
    full_image_latent = latent_dict["full_image"]
    upper_half_latent = latent_dict["upper_half"]
    lower_half_latent = latent_dict["lower_half"]
    
    # decode and compare with original images in batches
    l1_loss_full_list = []
    l1_loss_upper_list = []
    l1_loss_lower_list = []
    sync_loss_list = []
    
    for i in range(0, len(frame_ids), batch_size):
        full_image_latent_batch = full_image_latent[i:i+batch_size].to(device)
        upper_half_latent_batch = upper_half_latent[i:i+batch_size].to(device)
        lower_half_latent_batch = lower_half_latent[i:i+batch_size].to(device)
        
        # video
        with torch.no_grad():
            full_image_decoded = vae.decode(full_image_latent_batch).sample
            upper_half_decoded = vae.decode(upper_half_latent_batch).sample
            lower_half_decoded = vae.decode(lower_half_latent_batch).sample
        
        # Compare the decoded image with the image in the image dataset
        original_image_batch = load_original_image_batch(image_root, person_id, video_id, frame_ids[i:i+batch_size]).to(device)
        
        # audio
        sync_loss = compute_audio_video_sync(f"{person_id}/{video_id}", syncnet, mel, frame_ids, lower_half_decoded, i)
        if sync_loss is not None:
            sync_loss_list.append(sync_loss)
            
        # Compute L1 losses
        l1_loss_full = torch.nn.functional.l1_loss(full_image_decoded, original_image_batch)
        l1_loss_upper = torch.nn.functional.l1_loss(upper_half_decoded, original_image_batch[:, :, :image_size // 2])
        l1_loss_lower = torch.nn.functional.l1_loss(lower_half_decoded, original_image_batch[:, :, image_size // 2:])
        
        l1_loss_full_list.append(l1_loss_full)
        l1_loss_upper_list.append(l1_loss_upper)
        l1_loss_lower_list.append(l1_loss_lower)
        
    
    l1_loss_full = torch.stack(l1_loss_full_list).mean()
    l1_loss_upper = torch.stack(l1_loss_upper_list).mean()
    l1_loss_lower = torch.stack(l1_loss_lower_list).mean()
    sync_loss = torch.stack(sync_loss_list).mean()
    
    return l1_loss_full.item(), l1_loss_upper.item(), l1_loss_lower.item(), sync_loss.item()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Code to validate latent dataset')
    parser.add_argument("--latent_root", help="Root folder of the latent LRS2 dataset", required=True)
    parser.add_argument("--image_root", help="Root folder of the image LRS2 dataset", required=True)
    parser.add_argument("--syncnet_checkpoint_path", help="Path to the SyncNet checkpoint", required=True)
    args = parser.parse_args()
    
    # Initialize the models
    model_name = 'stabilityai/sd-vae-ft-mse'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = AutoencoderKL.from_pretrained(model_name, local_files_only=True).to(device)
    syncnet = SyncNet().to(device)
    load_checkpoint(args.syncnet_checkpoint_path, syncnet, None, reset_optimizer=True, overwrite_global_states=False)
    syncnet.eval()
    
    validate(vae, syncnet, args.image_root, args.latent_root)