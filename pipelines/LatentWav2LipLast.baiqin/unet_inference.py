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
from PIL import Image
from data_utils.blending import get_image
from mmpose.apis import inference_topdown, init_model
from mmpose.structures import merge_data_samples
import torchvision.transforms as transforms
from safetensors.torch import load_file
from skimage import transform as trans

parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')

parser.add_argument('--ckpt', type=str, 
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
parser.add_argument('--wav2lip_batch_size', type=int, help='Batch size for Wav2Lip model(s)', default=8)

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

parser.add_argument('--unet_config', type=str, help='Path to the unet config file', default='unet_config/customized_unet_v4_large')
parser.add_argument('--wav2vec2', default=False, action='store_true', help='Use Wav2Vec2 embeddings instead of Whisper embeddings')
parser.add_argument('--image_size', type=int, default=256, help='Image size for the VAE model.')
parser.add_argument('--resize', action='store_true', help='Resize the input image to 1/4 of the original size.')
parser.add_argument('--noparse', action='store_true', help='Enable face parsing.')
parser.add_argument('--acc', action='store_true', help='audio encoded with acc when combine with video')
parser.add_argument('--eval', action='store_true', help='evaluate on ckpt dir')
parser.add_argument('--save_sample', action='store_true', help='save sample image')
parser.add_argument('--crop_landmark', action='store_true', help='Enable crop by landmark.')
parser.add_argument('--arc_face', action='store_true', help='Enable crop by landmark.')
parser.add_argument('--bottom_scale', help='Bbox bottom expansion coefficient', default=0, type=float)
parser.add_argument('--sf3d_up', help='upperbound with sf3d',action='store_true')
parser.add_argument('--lmk_pad', help='upperbound range when landmark crop', default=0, type=float)
parser.add_argument('--lmk_ratio', help='upperbound range when landmark crop', default=0, type=float)
parser.add_argument('--double_detect', help='double face detect', action='store_true')
parser.add_argument('--ref', action='store_true', help='use static reference image')
parser.add_argument('--gs_blur', action='store_true', help='gs blur when mask lower half')
parser.add_argument('--mmpose_config_file', type=str, default='/data/wangbaiqin/project/MuseTalk/musetalk/utils/dwpose/rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py', help='your mmpose_config_file')
parser.add_argument('--mmpose_checkpoint_file', type=str, default='/data/wangbaiqin/project/MuseTalk/models/dwpose/dw-ll_ucoco_384.pth', help='your mmpose_checkpoint_file')
args = parser.parse_args()

if os.path.isfile(args.face) and args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
    args.static = True

def get_smoothened_boxes(boxes, T):
    new_boxes = []
    for i in range(len(boxes)):
        min_dix = max(0, i-T)
        max_dix = min(len(boxes), i+T)
        window = boxes[min_dix:max_dix]
        smooth_bbox = np.mean(window, axis=0).astype(np.int32)
        new_boxes.append(smooth_bbox)
    return new_boxes

def get_smooth_landmark(landmarks,T):
    new_landmarks=[]
    for i in range(len(landmarks)):
        min_dix = max(0,i-T)
        max_dix = min(len(landmarks),i+T)
        window = landmarks[min_dix:max_dix]
        smooth_landmark = np.mean(window, axis=0).astype(np.int32)
        new_landmarks.append(smooth_landmark)
    return new_landmarks

def get_bbox_by_landmark(model, fa, frames, upperbondrange =0,lmk_ratio=0):
    coord_placeholder = (0.0,0.0,0.0,0.0)
    height, width, _ = frames[0].shape

    # frames = read_imgs(img_list)
    batch_size_fa = 1
    batches = [frames[i:i + batch_size_fa] for i in range(0, len(frames), batch_size_fa)]
    coords_list = []
    landmarks = []
    # if upperbondrange != 0:
    #     print('get key_landmark and face bounding boxes with the bbox_shift:',upperbondrange)
    # else:
    #     print('get key_landmark and face bounding boxes with the default value')
    average_range_minus = []
    average_range_plus = []
    for fb in batches:
        results = inference_topdown(model, np.asarray(fb)[0])
        results = merge_data_samples(results)
        keypoints = results.pred_instances.keypoints
        face_land_mark= keypoints[0][23:91]
        face_land_mark = face_land_mark.astype(np.int32)

        # get bounding boxes by face detetion
        bbox = fa.get_detections_for_batch(np.asarray(fb))

        # adjust the bounding box refer to landmark
        # Add the bounding box to a tuple and append it to the coordinates list
        for j, f in enumerate(bbox):
            if f is None: # no face in the image
                coords_list += [coord_placeholder]
                continue

            half_face_coord =  face_land_mark[29]#np.mean([face_land_mark[28], face_land_mark[29]], axis=0)
            range_minus = (face_land_mark[30]- face_land_mark[29])[1]
            range_plus = (face_land_mark[29]- face_land_mark[28])[1]
            average_range_minus.append(range_minus)
            average_range_plus.append(range_plus)
            if upperbondrange != 0:
                half_face_coord[1] = upperbondrange+half_face_coord[1] #手动调整  + 向下（偏29）  - 向上（偏28）
            half_face_dist = np.max(face_land_mark[:,1]) - half_face_coord[1]
            upper_bond = half_face_coord[1]-half_face_dist
            if lmk_ratio!=0:
                height=np.max(face_land_mark[:,1])-upper_bond
                upper_bond+=lmk_ratio*height
            if args.sf3d_up:
                upper_bond = f[1]
            f_landmark = (np.min(face_land_mark[:, 0]),int(upper_bond),np.max(face_land_mark[:, 0]),np.max(face_land_mark[:,1]))
            x1, y1, x2, y2 = f_landmark

            if y2-y1<=0 or x2-x1<=0 or x1<0 or y1<0: # if the landmark bbox is not suitable, reuse the bbox
                coords_list += [f]
                w,h = f[2]-f[0], f[3]-f[1]
                print("error bbox:",f)
            else:
                coords_list += [f_landmark]

    return coords_list 


def get_bbox_and_landmark(model, fa, frames, upperbondrange =0):
    coord_placeholder = (0.0,0.0,0.0,0.0)
    height, width, _ = frames[0].shape

    # frames = read_imgs(img_list)
    batch_size_fa = 1
    batches = [frames[i:i + batch_size_fa] for i in range(0, len(frames), batch_size_fa)]
    coords_list = []
    landmarks = []

    average_range_minus = []
    average_range_plus = []
    for fb in tqdm(batches,desc='get bbox and landmark'):
        results = inference_topdown(model, np.asarray(fb)[0])
        results = merge_data_samples(results)
        keypoints = results.pred_instances.keypoints
        face_land_mark= keypoints[0][23:91]
        face_land_mark = face_land_mark.astype(np.int32)
        landmarks.append(face_land_mark)
        # get bounding boxes by face detetion
        bbox = fa.get_detections_for_batch(np.asarray(fb))

        # adjust the bounding box refer to landmark
        # Add the bounding box to a tuple and append it to the coordinates list
        for j, f in enumerate(bbox):
            if f is None: # no face in the image
                coords_list += [coord_placeholder]
                continue

            half_face_coord =  face_land_mark[29]#np.mean([face_land_mark[28], face_land_mark[29]], axis=0)
            range_minus = (face_land_mark[30]- face_land_mark[29])[1]
            range_plus = (face_land_mark[29]- face_land_mark[28])[1]
            average_range_minus.append(range_minus)
            average_range_plus.append(range_plus)
            if upperbondrange != 0:
                half_face_coord[1] = upperbondrange+half_face_coord[1] #手动调整  + 向下（偏29）  - 向上（偏28）
            half_face_dist = np.max(face_land_mark[:,1]) - half_face_coord[1]
            upper_bond = half_face_coord[1]-half_face_dist

            f_landmark = (np.min(face_land_mark[:, 0]),int(upper_bond),np.max(face_land_mark[:, 0]),np.max(face_land_mark[:,1]))
            x1, y1, x2, y2 = f_landmark

            if y2-y1<=0 or x2-x1<=0 or x1<0 or y1<0: # if the landmark bbox is not suitable, reuse the bbox
                coords_list += [f]
                w,h = f[2]-f[0], f[3]-f[1]
                print("error bbox:",f)
            else:
                coords_list += [f_landmark]

    return coords_list,landmarks


def get_arc_face( frames,upperbondrange =0,image_size=256,save_dir=None):
    model = init_model(args.mmpose_config_file, args.mmpose_checkpoint_file, device=device)
    fa = face_detection.FaceAlignment(face_detection.LandmarksType._2D, 
                                            flip_input=False, device=device)
    arcface_dst = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32)
    assert image_size%112==0 or image_size%128==0
    if image_size%112==0:
        ratio = float(image_size)/112.0
        diff_x = 0
    else:
        ratio = float(image_size)/128.0
        diff_x = 8.0*ratio
    dst = arcface_dst * ratio
    dst[:,0] += diff_x
    tform = trans.SimilarityTransform()
    arc_face=[]
    warp_metric=[]
    coords_list,landmarks=get_bbox_and_landmark(model, fa, frames, upperbondrange =upperbondrange)
    landmarks=get_smooth_landmark(landmarks,2)
    coords_list=get_smoothened_boxes(coords_list,2)
    for idx,fb in enumerate(tqdm(frames,desc='get warping image')):
        left_eye=np.mean(landmarks[idx][36:42],axis=0)
        right_eye=np.mean(landmarks[idx][42:48],axis=0)
        nose=landmarks[idx][30]
        left_mouth=landmarks[idx][48]
        right_mouth=landmarks[idx][54]
        lmk=np.array([left_eye,right_eye,nose,left_mouth,right_mouth])
        tform.estimate(lmk, dst)
        M = tform.params[0:2, :]
        warped = cv2.warpAffine(fb, M, (image_size, image_size), borderValue = 0.0)
  
        arc_face.append(warped)
        warp_metric.append(M)
       # landmarks.append(face_land_mark)
      #  frames.append(frame)
    return arc_face,coords_list,warp_metric #,frames

def unwarp( arc_face,M,frame):
	#unwarp
	white=np.ones_like(arc_face)*255
	inv_M = cv2.invertAffineTransform(M)
	unwarp_frame=cv2.warpAffine(arc_face, inv_M, (frame.shape[1], frame.shape[0]), borderValue = 0.0)
	white_unwarp_frame=cv2.warpAffine(white, inv_M, (frame.shape[1], frame.shape[0]), borderValue = 0.0)
	mask=np.zeros_like(frame)
	mask[white_unwarp_frame==255]=1
	gen_frame=frame.copy()
	gen_frame[mask==1]=unwarp_frame[mask==1]
	return gen_frame

def face_detect(images):
    detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, 
                                            flip_input=False, device=device)

    batch_size = args.face_det_batch_size
 
    resized_images = []
    if args.resize:
        resized_images = [cv2.resize(image, (image.shape[1] // 4, image.shape[0] // 4)) for image in images]
    else:
        resized_images = images

    while 1:
        predictions = []
        try:
            if args.crop_landmark:
                model = init_model(args.mmpose_config_file, args.mmpose_checkpoint_file, device=device)
                for i in tqdm(range(0, len(resized_images), batch_size)):
                    predictions.extend(get_bbox_by_landmark(model, detector, resized_images[i: i + batch_size],upperbondrange=args.lmk_pad,lmk_ratio=args.lmk_ratio))
            else:
                for i in tqdm(range(0, len(resized_images), batch_size)):
                    predictions.extend(detector.get_detections_for_batch(np.array(resized_images[i:i + batch_size])))
        except RuntimeError:
            if batch_size == 1: 
                raise RuntimeError('Image too big to run face detection on GPU. Please use the --resize_factor argument')
            batch_size //= 2
            print('Recovering from OOM error; New batch size: {}'.format(batch_size))
            continue
        break

    results = []
    pady1, pady2, padx1, padx2 = args.pads
    for rect, image in zip(predictions, images):
        if rect is None:
            cv2.imwrite('temp/faulty_frame.jpg', image) # check this frame where the face was not detected.
            raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

        if args.resize:
            rect = [coord * 4 for coord in rect]  # Scale up the detection coordinates by 4
        
        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)
        
        results.append([x1, y1, x2, y2])

    boxes = np.array(results)
    if not args.nosmooth and not args.crop_landmark: boxes = get_smoothened_boxes(boxes, T=2)
    results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

    del detector
    #torch.cuda.empty_cache()
    return results 


def datagen(frames, audios):
    # double the frames, from [0, N) to [0, N) + (N, 0]
    frames = frames + frames[::-1]
    
    img_batch, audio_batch, frame_batch, coords_batch, metric_batch = [], [], [], [], []

    if args.arc_face:
        if not args.static:
            arc_face,coords_list,warp_metric=get_arc_face(frames,image_size=args.image_size)
        else:
            arc_face,coords_list,warp_metric=get_arc_face([frames[0]],image_size=args.image_size)
        face_det_results=[[f, c] for f,c in zip(arc_face,coords_list)]
    elif args.box[0] == -1:
        if not args.static:
            face_det_results = face_detect(frames) # BGR2RGB for CNN face detection
        else:
            face_det_results = face_detect([frames[0]])
    else:
        print('Using the specified bounding box instead of face detection...')
        y1, y2, x1, x2 = args.box
        face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]
    if args.double_detect:
        crop_pad_frame=[]
        ori_frames=frames
        centers=[]
        max_w=max([x2-x1 for img,(y1, y2, x1, x2) in face_det_results])
        max_h=max([y2-y1 for img,(y1, y2, x1, x2) in face_det_results])
        max_pad=int((max(max_w,max_h)//2)*1.2)
        print('max_pad:',max_pad)
        for i in range(len(frames)):
            y1, y2, x1, x2 =face_det_results[i][1]
            center=(y1+y2)//2,(x1+x2)//2
            centers.append(center)
            crop_pad_frame.append(frames[i][center[0]-max_pad:center[0]+max_pad,center[1]-max_pad:center[1]+max_pad])
        del face_det_results
        frames=crop_pad_frame
        if not args.static:
            face_det_results = face_detect(crop_pad_frame) # BGR2RGB for CNN face detection
            for k in range(len(face_det_results)):
                y1, y2, x1, x2 = face_det_results[k][1]
                y1+=centers[k][0]-max_pad
                y2+=centers[k][0]-max_pad
                x1+=centers[k][1]-max_pad
                x2+=centers[k][1]-max_pad
                face_det_results[k][1]=(y1, y2, x1, x2)
              #  print('face_det_results:',(y1, y2, x1, x2))
        else:
            face_det_results = face_detect([crop_pad_frame[0]])
        print("face detect result:",face_det_results[0][1])
        frames=ori_frames

    for i, a in enumerate(audios):
        idx = 0 if args.static else i%len(frames)
        frame_to_save = frames[idx].copy()
        face, coords = face_det_results[idx].copy()

        face = cv2.resize(face, (args.image_size, args.image_size))
        if args.unet_config=="unet_config/musetalk":
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        img_batch.append(face)
        audio_batch.append(a)
        frame_batch.append(frame_to_save)
        coords_batch.append(coords)
        
        if args.arc_face:
            metric_batch.append(warp_metric[idx])

        if len(img_batch) >= args.wav2lip_batch_size:
                        
            

            img_batch, audio_batch = np.asarray(img_batch), torch.stack(audio_batch)

            img_masked = img_batch.copy()
            if args.gs_blur:
                img_masked[:, args.image_size//2:] = np.random.normal(0, 30, img_masked[:, args.image_size//2:].shape)
            else:
                img_masked[:, args.image_size//2:] = 0
            #print(img_batch.shape)
            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.

            yield img_batch, audio_batch, frame_batch, coords_batch, metric_batch
            img_batch, audio_batch, frame_batch, coords_batch, metric_batch = [], [], [], [], []

    if len(img_batch) > 0:

        

        img_batch, audio_batch = np.asarray(img_batch), torch.stack(audio_batch)


        img_masked = img_batch.copy()

        if args.gs_blur:
            img_masked[:, args.image_size//2:] = np.random.normal(0, 30, img_masked[:, args.image_size//2:].shape)
        else:
            img_masked[:, args.image_size//2:] = 0

        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.

        yield img_batch, audio_batch, frame_batch, coords_batch, metric_batch

audio_step_size = 10
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} for inference.'.format(device))

def _load(checkpoint_path):
    if device == 'cuda:':
        checkpoint = torch.load(checkpoint_path,map_location=torch.device('cuda'))
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint

def load_model(path):
    unet_config = UNet2DConditionModel.load_config(f"{args.unet_config}/config.json")
    unet = UNet2DConditionModel.from_config(unet_config)
    
    print("Load checkpoint from: {}".format(path))
    #checkpoint = _load(path)
    if path.endswith('.safetensors'):
        new_s = load_file(path)
    else:
        s = torch.load(path, map_location=device)

    # if state_dict is in s
        if 'state_dict' in s:
            s = s['state_dict']
       # print(s)
        new_s = {}
        for k, v in s.items():
            if k.startswith("unet."):
                new_s[k.replace('unet.', '')] = v
    unet.load_state_dict(new_s)

    unet = unet.to(device)
    return unet.eval()

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
    audio_name = args.audio.split('/')[-1].split('.')[0]
    video_name=args.face.split('/')[-1].split('.')[0]
    ckpt_dir=os.path.dirname(args.ckpt)
    if args.outfile=='results/result_voice.mp4' and args.eval:
        #ckpt_dataset_name=args.ckpt.split('-')[2]
        os.makedirs(f'{ckpt_dir}/temp', exist_ok=True)
        args.outfile=f'{ckpt_dir}/{video_name}_{audio_name}.mp4'
        print(f'outfile: {args.outfile}')
    if not args.audio.endswith('.wav'):
        print('Extracting raw audio...')
        #audio_name=args.audio.split('/')[-1].split('.')[0]
        command = 'ffmpeg -y -i {} -strict -2 {}'.format(args.audio, f'{ckpt_dir}/temp/{audio_name}.wav')

        subprocess.call(command, shell=True)
        args.audio = f'{ckpt_dir}/temp/{audio_name}.wav'


 
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
    for i, (img_batch, audio_batch, frames, coords, metric) in enumerate(tqdm(gen, 
                                            total=int(np.ceil(float(len(audio_chunks))/batch_size)))):
        if i == 0:
            model = load_model(args.ckpt)
            print ("Model loaded")

            frame_h, frame_w = full_frames[0].shape[:-1]
            out = cv2.VideoWriter(f'{ckpt_dir}/temp/result.avi', 
                                    cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))

        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
        audio_batch = audio_batch.to(device)

        #torch.cuda.empty_cache()
        # load vae
        vae = AutoencoderKL.from_pretrained(args.vae_path, local_files_only=True).to(device)

        image_latent = to_latent(vae, img_batch)

        if i == 0:
            first_frame_latent = image_latent[7]
        if args.ref:
            for j in range(img_batch.size(0)):
                image_latent[j][4:,:]=first_frame_latent[4:,:]
        #print(image_latent.size())
        # assert shape of image_latent be batch_size x 8 x 96 x 96
        # assert image_latent.size(0) == img_batch.size(0) and image_latent.size(1) == 8 and image_latent.size(2) == 96 and image_latent.size(3) == 96 

        with torch.no_grad():
            image_latent = reshape_face_sequences(image_latent.unsqueeze(2))
            audio_batch = reshape_audio_sequences(audio_batch.unsqueeze(2))
            pred_latent = model(image_latent, timestep=zero_timestep, encoder_hidden_states=audio_batch).sample
            pred_latent = inverse_reshape_face_sequences(pred_latent)
            pred_latent = pred_latent.squeeze(2)

        # assert shape of pred_latent be batch_size x 4 x 96 x 96
        # assert pred_latent.size(0) == img_batch.size(0) and pred_latent.size(1) == 4 and pred_latent.size(2) == 96 and pred_latent.size(3) == 96
  
        pred_image = from_latent(vae, pred_latent)
  
        pred = pred_image.clamp(0, 1).cpu().numpy().transpose(0, 2, 3, 1) * 255.
        if args.save_sample:
            dirname=f"{ckpt_dir}/temp/sample_{args.outfile.split('.')[0].split('/')[-1]}"
            os.makedirs(dirname, exist_ok=True)
            img_batch = img_batch.cpu().numpy().transpose(0, 2, 3, 1) * 255.
            ref_img = img_batch[:, :, :, :3]
            mask_img=img_batch[:, :, :, 3:]
            concat_batch = np.concatenate((ref_img, mask_img,pred), axis=2)
            for j, concat in enumerate(concat_batch):
                cv2.imwrite(os.path.join(dirname, f"{i}_{j}.jpg"), concat)
        
        for idx,(p, f, c) in enumerate(zip(pred, frames, coords)):
            if args.unet_config=="unet_config/musetalk":
                p=cv2.cvtColor(p, cv2.COLOR_RGB2BGR)
            if  args.arc_face:
                x1,y1,x2,y2=c
                p=unwarp(p,metric[idx],f)
                p=p[y1: y2, x1:x2]
            else:
                y1, y2, x1, x2 = c
                p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
            #print(c)
            if args.noparse:
                f[y1:y2, x1:x2] = p
                out.write(f)
            else:
                c=x1,y1,x2,y2
                new_frame=get_image(f,p,c)
                #print(new_frame.shape)
                out.write(new_frame)

    out.release()

    if args.acc:
        command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(args.audio, f'{ckpt_dir}/temp/result.avi', args.outfile)
    else:
        command="ffmpeg -y -v warning -i {} -i {} -hide_banner -strict -2 -q:v 1 -c:a libvorbis {}".format(args.audio,f'{ckpt_dir}/temp/result.avi',args.outfile)
    
    subprocess.call(command, shell=platform.system() != 'Windows')

    print("output video saved at: {}".format(args.outfile))
if __name__ == '__main__':
    main()
