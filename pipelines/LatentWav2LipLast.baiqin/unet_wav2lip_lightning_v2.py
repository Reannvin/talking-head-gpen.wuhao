from os.path import dirname, join, basename, isfile
from glob import glob
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os, random, argparse
from hparams import hparams, get_image_list
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import RichProgressBar, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from lr_scheduler import LambdaLinearScheduler
import cv2
from diffusers import AutoencoderKL, UNet2DConditionModel
from models import SyncNet_image_256,PerceptualLoss,define_D,GANLoss,ResNetArcFace,Discriminator
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from piq import  psnr, ssim, FID
from piq.feature_extractors import InceptionV3

class CombinedDataset(Dataset):
    def __init__(self, base_dataset, fine_tune_dataset=None, base_ratio=0.5):
        self.base_dataset = base_dataset
        self.fine_tune_dataset = fine_tune_dataset
        self.base_ratio = base_ratio
        self.base_length = len(base_dataset)
        self.fine_tune_length = len(fine_tune_dataset) if fine_tune_dataset is not None else 0
        
    def __len__(self):
        if self.fine_tune_dataset is None:
            return self.base_length
        return self.base_length + self.fine_tune_length

    def __getitem__(self, idx):
        if self.fine_tune_dataset is None or random.random() < self.base_ratio:
            base_idx = random.randint(0, self.base_length - 1)
            return self.base_dataset[base_idx]
        else:
            fine_tune_idx = random.randint(0, self.fine_tune_length - 1)
            return self.fine_tune_dataset[fine_tune_idx]


class AudioVisualDataset(Dataset):
    def __init__(self, data_root, dataset_name, split, args, audio_root=None, dataset_size=512000):
        self.all_videos = get_image_list(data_root, dataset_name, split)
        self.dataset_size = dataset_size
        self.syncnet_audio_size = 10 * args.syncnet_T if not args.wav2vec2 else 2 * args.syncnet_T
        self.frame_audio_size = 10 * 5 if not args.wav2vec2 else 2 * 5 # always use syncnet_T=5 for each frame
        self.args = args
        self.data_root = data_root
        self.audio_root = audio_root

        
    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])

    def get_window(self, start_frame):
        start_id = self.get_frame_id(start_frame)
        vidname = dirname(start_frame)

        window_fnames = []
        for frame_id in range(start_id, start_id + self.args.syncnet_T):
            frame = join(vidname, '{}.jpg'.format(frame_id))
            if not isfile(frame):
                return None
            window_fnames.append(frame)
        return window_fnames

    def read_window(self, window_fnames):
        if window_fnames is None: return None
        window = []

        for fname in window_fnames:
            img = cv2.imread(fname)
            if img is None:
                return None
            try:
                #获得一个随机数0——0.15
                r = random.random() * self.args.up_ratio
                img_new_height_start=int(img.shape[0]*r)
                img=img[img_new_height_start:,:,:]
                img = cv2.resize(img, (hparams.image_size, hparams.image_size))
            except Exception as e:
                return None

            window.append(img)

        return window
    
    def read_mask_window(self, window_fnames):
        if window_fnames is None: return None
        window = []
        
        for fname in window_fnames:
            try:

                lmd_path=fname.replace(".jpg","_lmd.npy")
                if not os.path.exists(lmd_path):
                    print("read_window has error lmd_path not exist:",lmd_path)
                    return None
                lmd=np.load(lmd_path)
                half_face_coord = lmd[29]
                half_face_dist = np.max(lmd[:, 1]) - half_face_coord[1]
                upper_bond = half_face_coord[1] - half_face_dist
                    
                coord = (np.min(lmd[:, 0]), int(upper_bond), np.max(lmd[:, 0]), np.max(lmd[:, 1]))
                mask_img=np.zeros((coord[3]-coord[1],coord[2]-coord[0],3),dtype=np.uint8)
                mouth_lmd=lmd[48:68]
                coord_mouth = (np.min(mouth_lmd[:, 0]), np.min(mouth_lmd[:, 1]), np.max(mouth_lmd[:, 0]), np.max(mouth_lmd[:, 1]))
                coord_mouth=(coord_mouth[0]-coord[0]-5,coord_mouth[1]-coord[1]-5,coord_mouth[2]-coord[0]+5,coord_mouth[3]-coord[1]+5)
                #计算当img mask到hparams.image_size时，mouth的坐标
                #coord_mouth=(int(coord_mouth[0]/(coord[2]-coord[0])*hparams.image_size),int(coord_mouth[1]/(coord[3]-coord[1])*hparams.image_size),int(coord_mouth[2]/(coord[2]-coord[0])*hparams.image_size),int(coord_mouth[3]/(coord[3]-coord[1])*hparams.image_size))
                mask_img=cv2.rectangle(mask_img, (coord_mouth[0], coord_mouth[1]), (coord_mouth[2], coord_mouth[3]), 255, -1)
                mask_img=cv2.resize(mask_img, (hparams.image_size, hparams.image_size))
            except Exception as e:
                return None
            window.append(mask_img)

        return window

    def get_whisper_embedding(self, vidname, frame_id, syncnet_T):
        try:
            whisper_file = f"{frame_id}.npy" if syncnet_T == 5 else f"{frame_id}.npy.{syncnet_T}.npy"
            audio_path = join(vidname, whisper_file)
            audio_embedding = np.load(audio_path)
            audio_embedding = torch.from_numpy(audio_embedding)
        except:
            print(f"Error loading {audio_path}")
            audio_embedding = None
        return audio_embedding
    
    def get_whisper_segmented_audios(self, vidname, frame_id):
        audios = []
        offset = self.args.syncnet_T // 2
        start_frame_num = frame_id 
        if start_frame_num - offset < 0: return None
        for i in range(start_frame_num, start_frame_num + self.args.syncnet_T):
            m = self.get_whisper_embedding(vidname, i - offset, syncnet_T=5) # always use syncnet_T=5 for each frame
            if m is None or m.shape[0] != self.frame_audio_size:
                return None
            audios.append(m)

        audios = torch.stack(audios)
        return audios
    
    def crop_audio_window(self, audio_embeddings, start_frame, syncnet_T):
        if type(start_frame) == int:
            start_frame_num = start_frame
        else:
            start_frame_num = self.get_frame_id(start_frame) # 0-indexing ---> 1-indexing
        start_idx = int(50. * (start_frame_num / float(hparams.fps)))
        end_idx = start_idx + 2 * syncnet_T
        return audio_embeddings[start_idx : end_idx]
    
    def get_segmented_audios(self, audio_embeddings, start_frame):
        audios = []
        offset = self.args.syncnet_T // 2
        start_frame_num = self.get_frame_id(start_frame) + 1
        if start_frame_num - offset < 0: return None
        for i in range(start_frame_num, start_frame_num + self.args.syncnet_T):
            m = self.crop_audio_window(audio_embeddings, i - offset, syncnet_T=5) # always use syncnet_T=5 for each frame
            if m.shape[0] != self.frame_audio_size:
                return None
            audios.append(m)

        audios = torch.stack(audios)
        return audios

    def prepare_window(self, window):
        # 3 x T x H x W
        x = np.asarray(window) / 255.
        x = np.transpose(x, (3, 0, 1, 2))

        return x

    def __len__(self):
        #return min(self.dataset_size,len(self.all_videos)) # len(self.all_videos)
        return self.dataset_size

    def __getitem__(self, idx):
        while 1:
            idx = random.randint(0, len(self.all_videos) - 1)
            vidname = self.all_videos[idx]
            img_names = list(glob(join(vidname, '*.jpg')))
            img_names = [img for img in img_names if 'mask' not in img]
            if len(img_names) <= 30:#3 * self.args.syncnet_T:
                # print(f"Video {vidname} has less frames than required")
                continue
           # print(f"Video {vidname} has {len(img_names)} frames")
            img_name = random.choice(img_names)
            wrong_img_name = random.choice(img_names)
            # Ensure wrong_img_name is at least syncnet_T frames away from img_name
            while abs(self.get_frame_id(img_name) - self.get_frame_id(wrong_img_name)) < 10:#self.args.syncnet_T:
                wrong_img_name = random.choice(img_names)
            
            window_fnames = self.get_window(img_name)
            wrong_window_fnames = self.get_window(wrong_img_name)
            if window_fnames is None or wrong_window_fnames is None:
                continue

            window = self.read_window(window_fnames)
            

            if window is None:
                print(f"Window is None for {vidname}")
                continue
            if self.args.mouth_mask:
                mask_window = self.read_mask_window(window_fnames)
                if mask_window is None:
                    continue
            wrong_window = self.read_window(wrong_window_fnames)
            if wrong_window is None:
                # print(f"Wrong Window is None for {vidname}")
                continue
            
            if self.audio_root:
                # switch fro data_root to audio_root
                vidname = vidname.replace(self.data_root, self.audio_root)
            
            if not args.wav2vec2:
                # load syncnet_T frames of audio embeddings for syncnet loss
                if self.args.syncnet:
                    audio_cropped = self.get_whisper_embedding(vidname, self.get_frame_id(img_name), syncnet_T=self.args.syncnet_T)
                    
                # always load for each frame of 5 frames of audio embeddings for cross attention
                indiv_audios = self.get_whisper_segmented_audios(vidname, self.get_frame_id(img_name))
            else:
                # load audio embedding from file wav2vec2.pt
                audio_path = join(vidname, "wav2vec2.pt")
                audio_embeddings = torch.load(audio_path, map_location='cpu')[0]
                
                # load syncnet_T frames of audio embeddings for syncnet loss
                if self.args.syncnet:
                    audio_cropped = self.crop_audio_window(audio_embeddings.clone(), img_name, syncnet_T=self.args.syncnet_T)
                # always load for each frame of 5 frames of audio embeddings for cross attention
                indiv_audios = self.get_segmented_audios(audio_embeddings.clone(), img_name)
            if self.args.syncnet and (audio_cropped is None or audio_cropped.shape[0] != self.syncnet_audio_size): continue
            if indiv_audios is None: continue

            window = self.prepare_window(window)
            y = window.copy()
            window[:, :, window.shape[2]//2:] = 0.
            
            wrong_window = self.prepare_window(wrong_window)
            x = np.concatenate([window, wrong_window], axis=0)
            x = torch.FloatTensor(x)
            
            if self.args.syncnet:
                audio_cropped = audio_cropped.unsqueeze(0).float()
            
            indiv_audios = indiv_audios.unsqueeze(1).float()
            y = torch.FloatTensor(y)

           # return_tuple=()
            if self.args.mouth_mask:
                mask_window = self.prepare_window(mask_window)
                mask_window = torch.FloatTensor(mask_window)
            
            if self.args.syncnet and self.args.mouth_mask:
                return x, indiv_audios, audio_cropped, y, mask_window
            elif self.args.syncnet:
                return x, indiv_audios, audio_cropped, y
            elif self.args.mouth_mask:
                return x, indiv_audios, y, mask_window
            else:
                return x, indiv_audios, y
            



class LatentWav2Lip(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        unet_config = UNet2DConditionModel.load_config(f"{self.hparams.unet_config}/config.json")
        self.unet = UNet2DConditionModel.from_config(unet_config)
        self.gan_loss_start_epoch=10
        if self.hparams.gan_loss_wt>0:
            if self.hparams.disc_type=='stylegan2':
                self.disc=Discriminator(self.hparams.image_size, channel_multiplier=2)
                if self.hparams.disc_ckpt:
                    self.load_disc(self.hparams.disc_ckpt)
            else:
                self.disc=define_D(input_nc=3, ndf=64, n_layers_D=3, norm='instance', use_sigmoid=False, num_D=2, getIntermFeat=True)
                self.criterionGAN=GANLoss(use_lsgan=True, tensor=torch.cuda.FloatTensor)
                self.criterionFeat = torch.nn.L1Loss()
            
            # self.criterionGAN=GANLoss(use_lsgan=True, tensor=torch.cuda.FloatTensor)
            # self.criterionFeat = torch.nn.L1Loss()
        self.unet.train()
        self.zero_timestep = torch.zeros([])
        
        if self.hparams.syncnet:
            self.syncnet = self.load_syncnet()
            self.syncnet.eval()  # Ensure syncnet is always in eval mode
        
        self.vae = self.load_vae('stabilityai/sd-vae-ft-mse')
        self.vae.eval()  # Ensure vae is always in eval mode
        if self.hparams.id_loss_wt>0:
            self.face_id= self.load_face_recognize_model()
            self.face_id.eval()
        # 为了只加载 Wav2Lip 的参数，我们需要将 strict_loading 设置为 False
        self.strict_loading = False
        
        # 定义损失函数
        self.recon_loss = nn.L1Loss() if not self.hparams.l2_loss else nn.MSELoss()
        self.log_loss = nn.BCELoss()
        #self.lpips_loss = LearnedPerceptualImagePatchSimilarity(net_type='vgg')
        if self.hparams.lpips_type=='vgg19':
            self.lpips_loss=PerceptualLoss(network='vgg19',layers=['relu_1_1', 'relu_2_1', 'relu_3_1', 'relu_4_1', 'relu_5_1'],num_scales=2,instance_normalized=True).eval()
        else:
            self.lpips_loss=LearnedPerceptualImagePatchSimilarity(net_type='vgg').eval()
        

        self.automatic_optimization = False if self.hparams.gan_loss_wt>0 else True

        self.psnr_list, self.ssim_list, self.fid_list = [], [], []

    def load_syncnet(self):
        syncnet = SyncNet_image_256(not self.hparams.wav2vec2, self.hparams.syncnet_T)
        ckpt = torch.load(self.hparams.syncnet)
        new_state_dict = {k[len("model."):] if k.startswith("model.") else k: v for k, v in ckpt['state_dict'].items()}
        syncnet.load_state_dict(new_state_dict)

        # 冻结 Syncnet 的所有参数
        for param in syncnet.parameters():
            param.requires_grad = False
        return syncnet
    
    def load_vae(self, model_name):
        vae = AutoencoderKL.from_pretrained(model_name, local_files_only=True)
        
        # 冻结 VAE 的所有参数
        for param in vae.parameters():
            param.requires_grad = False
        return vae
    def load_disc(self, model_name):
        ckpt = torch.load(model_name)
   
        self.disc.load_state_dict(ckpt["d"])

    
    def load_face_recognize_model(self, model_name='models/arcface_resnet18/arcface_resnet18.pth'):
        arcface_res=ResNetArcFace('IRBlock', [2, 2, 2, 2], use_se=False)
        s=torch.load(model_name)
        new_s={k[len("module."):] if k.startswith("module.") else k: v for k, v in torch.load(model_name).items()}
        arcface_res.load_state_dict(new_s)
        for p in arcface_res.parameters():
            p.requires_grad = False
        return arcface_res
    
    def load_unet(self, unet_ckpt):
        try:
            self.unet.from_pretrained(unet_ckpt)
            print(f"Loaded U-Net from {unet_ckpt}")
        except Exception as e:
            print("Train UNet from scratch")
        self.unet.train()
        return self.unet
    
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        # 从模型的状态字典中删除 VAE 和 Syncnet 的参数
        # 传递额外的参数给父类的 state_dict
        original_state_dict = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        # 过滤掉包含 "vae" 和 "syncnet" 的键
        filtered_state_dict = {k: v for k, v in original_state_dict.items() if "vae" not in k and "syncnet" not in k and "face_id" not in k}
        return filtered_state_dict
    
    def cosine_loss(self, a, v, y):
        d = nn.functional.cosine_similarity(a, v)
        # Disable autocast for unsafe operations
        with torch.cuda.amp.autocast(enabled=False):
            loss = self.log_loss(d.unsqueeze(1), y)
        return loss
    
    def get_sync_loss(self, mel, g):
        # B, 4 * T, H, W
        g = torch.cat([g[:, :, i] for i in range(self.hparams.syncnet_T)], dim=1)
        
        # if image size is not [128, 256], resize the image to [128, 256]
        if g.size(2) != 128 or g.size(3) != 256:
            g = nn.functional.interpolate(g, size=(128, 256), mode='bilinear')
        
        a, v = self.syncnet(mel, g)
        y = torch.ones(g.size(0), 1).to(self.device)
        loss = self.cosine_loss(a, v, y)
        return loss
    
    def get_lpips_loss(self, g, gt):
        if len(g.shape)==3 or len(gt.shape)==3:
            g=g.unsqueeze(0)
            gt=gt.unsqueeze(0)
        # Expected both input arguments to be normalized tensors with shape [N, 3, H, W].
        if len(g.shape) != 4 or len(gt.shape) != 4 or g.shape[1] != 3 or gt.shape[1] != 3:
            # reshape the tensors to [N, 3, H, W]
            g = self.reshape_gt_image_for_vae(g)
            gt = self.reshape_gt_image_for_vae(gt)

        # loss = self.lpips_loss(g, gt)
        #将[8,3,3,128,256]转成[24,3,128,256]

        if self.hparams.lpips_type=='vgg19':    
            # g = self.reshape_gt_image_for_vae(g)#.clamp(0,1)
            # gt = self.reshape_gt_image_for_vae(gt)#.clamp(0,1)
            
            loss=0.05*self.lpips_loss(g, gt, use_style_loss=(self.hparams.no_style_loss is not True),
                                                weight_style_to_perceptual=250).mean()
        else:
            g = g.clamp(0,1)
            gt = gt.clamp(0,1)
            loss = self.lpips_loss(g, gt)
        
        return loss
    
    def g_nonsaturating_loss(self,fake_pred):
        loss = nn.functional.softplus(-fake_pred).mean()
        return loss
    
    def d_logistic_loss(self,real_pred, fake_pred):
        real_loss = nn.functional.softplus(-real_pred)
        fake_loss = nn.functional.softplus(fake_pred)
        return real_loss.mean() + fake_loss.mean()
    
    def gray_resize_for_identity(self, out, size=128):
        out_gray = (0.2989 * out[:, 0, :, :] + 0.5870 * out[:, 1, :, :] + 0.1140 * out[:, 2, :, :])
        out_gray = out_gray.unsqueeze(1)
        out_gray = nn.functional.interpolate(out_gray, (size, size), mode='bilinear', align_corners=False)
        return out_gray
    def get_faceid_loss(self,x,g,gt):
        #ref=x[:,3:,:,:,:]
        g = self.reshape_gt_image_for_vae(g).clamp(0,1)*255
        gt = self.reshape_gt_image_for_vae(gt).clamp(0,1)*255
        #ref=self.reshape_gt_image_for_vae(ref).clamp(0,1)*255
        g_gray=self.gray_resize_for_identity(g)
        gt_gray=self.gray_resize_for_identity(gt)
      #  ref_gray=self.gray_resize_for_identity(ref)
        g_id=self.face_id(g_gray)
        gt_id=self.face_id(gt_gray)
       # ref_id=self.face_id(ref_gray)
      #  face_id_loss=(self.recon_loss(g_id,gt_id)+self.recon_loss(g_id,ref_id))*0.5
        face_id_loss=self.recon_loss(g_id,gt_id)
        return face_id_loss
    
    def inverse_reshape_face_sequences(self, tensor):
        """
        Inverse operation for the reshape_face_sequences function, reconstructing the original tensor
        from a reshaped format of [batch_size, channels * groups, height, width].
        
        Parameters:
            tensor (torch.Tensor): A tensor with dimensions [batch_size * groups, channels, height, width].
        
        Returns:
            torch.Tensor: A tensor with dimensions [batch_size, channels, groups, height, width].
        """
        total_batch_size, channels, height, width = tensor.shape
        groups = self.hparams.syncnet_T
        batch_size = total_batch_size // groups
        
        # check if the total batch size is divisible by the number of groups
        if total_batch_size % groups != 0:
            raise ValueError("Total batch size is not divisible by the number of groups.")
        
        # Reshape the tensor to its original dimensions
        original_shape_tensor = tensor.view(batch_size, groups, channels, height, width).permute(0, 2, 1, 3, 4)        
        return original_shape_tensor
    
    def reshape_face_sequences_for_vae(self, tensor): # [8, 6, 5, 768, 768] -> [80, 3, 768, 768]
        batch_size, double_channels, groups, height, width = tensor.shape
        reshaped_tensor = tensor.permute(0, 2, 1, 3, 4).contiguous().view(batch_size * groups * 2, double_channels // 2, height, width)
        return reshaped_tensor
    
    def reshape_gt_image_for_vae(self, tensor): # # [8, 3, 5, 768, 768] -> [40, 3, 768, 768]
        batch_size, channels, groups, height, width = tensor.shape
        reshaped_tensor = tensor.permute(0, 2, 1, 3, 4).contiguous().view(batch_size * groups, channels, height, width)
        return reshaped_tensor
    
    def reshape_audio_sequences_for_unet(self, tensor):
        batch_size, dim1, dim2, dim3, features = tensor.shape
        reshaped_tensor = tensor.view(batch_size * dim1, dim2 * dim3, features)
        return reshaped_tensor
    
    def reshape_latent_faces_for_unet(self, tensor):
        # [batch_size * 2, channels, height, width] -> [batch_size, channels * 2, height, width]
        batch_size_times_2, channels, height, width = tensor.shape
        batch_size = batch_size_times_2 // 2
        reshaped_tensor = tensor.view(batch_size, channels * 2, height, width)
        return reshaped_tensor

    def encode_with_vae(self, face_sequences):
        # face_sequences are images of [0, 1] range
        face_sequences = face_sequences * 2. - 1.
        
        latent_face_sequences = self.vae.encode(face_sequences).latent_dist.sample()
        
        # scale the latent space to have unit variance when training unet
        scaling_factor = self.vae.config.scaling_factor
        latent_face_sequences = latent_face_sequences * scaling_factor
        return latent_face_sequences
    
    def decode_with_vae(self, latent_face_sequences):
        # scale the latent from unet when decoding with vae
        scaling_factor = self.vae.config.scaling_factor
        latent_face_sequences = latent_face_sequences / scaling_factor
        
        image_face_sequences = self.vae.decode(latent_face_sequences).sample
        
        # convert the image to [0, 1] range
        image_face_sequences = (image_face_sequences + 1.) / 2.
        return image_face_sequences
    
    def forward(self, audio_sequences, face_sequences, with_image=True):
        face_sequences = self.reshape_face_sequences_for_vae(face_sequences)
        latent_face_sequences = self.encode_with_vae(face_sequences)

        latent_face_sequences = self.reshape_latent_faces_for_unet(latent_face_sequences)
        audio_sequences = self.reshape_audio_sequences_for_unet(audio_sequences) 
        
        g_latent = self.unet(latent_face_sequences, timestep=self.zero_timestep, encoder_hidden_states=audio_sequences).sample    
        
        if with_image:
            g_image = self.decode_with_vae(g_latent)
            g_image = self.inverse_reshape_face_sequences(g_image)
        else:
            g_image = None
        
        g_latent = self.inverse_reshape_face_sequences(g_latent)
        return g_latent, g_image
    
    def training_step(self, batch, batch_idx):
        if self.hparams.syncnet and self.hparams.mouth_mask:
            x, indiv_audios, audio_cropped, gt,mask = batch
        elif self.hparams.mouth_mask:
            x, indiv_audios, gt,mask = batch
        elif self.hparams.syncnet:
            x, indiv_audios, audio_cropped, gt = batch
        else:
            x, indiv_audios, gt = batch
        
        # dropout reference frames if enabled
        if self.hparams.dropout_ref:
            # to drop the ref frames, based on dropout_ref_prob
            need_dropout = random.random() < self.hparams.dropout_ref_prob
            # print(f"need_dropout: {need_dropout}")
            if need_dropout:
                upper_half, ref = x.chunk(2, dim=1)
                # print(f"upper_half: {upper_half.shape}, ref: {ref.shape}")
                # upper_half: torch.Size([8, 4, 5, 96, 96]), ref: torch.Size([8, 4, 5, 96, 96])
                x = torch.cat([upper_half, torch.zeros_like(ref)], dim=1)

        if self.hparams.gan_loss_wt>0 and self.current_epoch>self.gan_loss_start_epoch:
            optimizer_g,optimizer_d=self.optimizers()
            lr_scheduler_g,lr_scheduler_d=self.lr_schedulers()
            if self.hparams.disc_type=='stylegan2':
                self.unet.eval()
                self.disc.train()
                g_latent, g_image = self(indiv_audios, x, not self.hparams.no_image_loss)
                with torch.cuda.amp.autocast(enabled=False):
                    
                    fake_pred = self.disc(g_image.to(torch.float32))
                    real_pred = self.disc(gt.to(torch.float32))
                    disc_loss = self.d_logistic_loss(real_pred, fake_pred)
                    optimizer_d.zero_grad()
                    self.manual_backward(disc_loss)
                    self.clip_gradients(optimizer_d, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
                    optimizer_d.step()
                    lr_scheduler_d.step()
                self.disc.eval()
                self.unet.train()
            

        g_latent, g_image = self(indiv_audios, x, not self.hparams.no_image_loss)




        if not self.hparams.no_image_loss:
            if self.hparams.full_image_loss:
                image_recon_loss = self.recon_loss(g_image, gt)
                lpips_loss =  0 if self.hparams.lpips_weight == 0. else self.get_lpips_loss(g_image, gt)
            else:
                image_recon_loss = self.recon_loss(g_image[:, :, :, g_image.size(3) // 2:, :], gt[:, :, :, gt.size(3) // 2:, :])
                lpips_loss = 0 if self.hparams.lpips_weight == 0. else self.get_lpips_loss(g_image[:, :, :, g_image.size(3) // 2:, :], gt[:, :, :, gt.size(3) // 2:, :])
            if self.hparams.mouth_mask:
                #只计算mouth部分的loss时加权
              
                image_recon_loss=image_recon_loss+3*self.recon_loss(g_image*mask,gt*mask)
                #lpips_loss=lpips_loss+3*self.get_lpips_loss(g_image*mask,gt*mask)

                
        else:
            image_recon_loss = 0.
            lpips_loss = 0.
        # if self.hparams.lpips_weight == 0.:
        #     lpips_loss = 0.
        image_loss = self.hparams.lpips_weight * lpips_loss + image_recon_loss
        
        gt_for_vae = self.reshape_gt_image_for_vae(gt)        
        gt_latent = self.encode_with_vae(gt_for_vae)
        gt_latent = self.inverse_reshape_face_sequences(gt_latent)
        latent_recon_loss = self.recon_loss(g_latent, gt_latent)
        
        if self.hparams.syncnet and self.hparams.syncnet_wt > 0.:
            sync_loss = self.get_sync_loss(audio_cropped, g_image[:, :, :, g_image.size(3) // 2:, :])
        else:
            sync_loss = 0.  
        if self.hparams.id_loss_wt>0 and self.current_epoch>self.gan_loss_start_epoch:    
            face_id_loss=self.get_faceid_loss(x,g_image,gt)
        else:
            face_id_loss=0


        if self.hparams.disc_type=='stylegan2' and self.hparams.gan_loss_wt>0 and self.current_epoch>self.gan_loss_start_epoch:
            with torch.cuda.amp.autocast(enabled=False):
                fake_pred = self.disc(g_image.to(torch.float32))
                gan_loss = self.g_nonsaturating_loss(fake_pred)


        elif self.hparams.gan_loss_wt>0 and self.current_epoch>self.gan_loss_start_epoch:
            # if self.hparams.disc_type=='stylegan2':
            #     #gan_loss=
            #     pass
            # else:
            # discriminator
            g_image=self.reshape_gt_image_for_vae(g_image)
            pred_fake = self.disc.forward(g_image.detach())
            loss_D_fake = self.criterionGAN(pred_fake, False)
            # Real Detection and Loss
            pred_real = self.disc.forward(gt_for_vae .clone().detach())
            loss_D_real = self.criterionGAN(pred_real, True)
            loss_D = (loss_D_fake + loss_D_real).mean() * 0.5
            #
            # GAN loss
            pred_fake = self.disc.forward(g_image)
            loss_G_GAN = self.criterionGAN(pred_fake, True).mean()
            n_layers_D = 3
            num_D = 2
            # GAN feature matching loss
            loss_G_GAN_Feat = 0
            feat_weights = 4.0 / (n_layers_D + 1)
            D_weights = 1.0 / num_D
            for i in range(num_D):
                for j in range(len(pred_fake[i]) - 1):
                    loss_G_GAN_Feat += D_weights * feat_weights * \
                                    self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach()).mean() * 2.5
            gan_loss=(0.25*loss_G_GAN + loss_G_GAN_Feat)
            disc_loss=self.hparams.gan_loss_wt*loss_D
        else:
            gan_loss=0
            disc_loss=0
        
        recon_loss = latent_recon_loss + self.hparams.image_loss_wt * image_loss+self.hparams.gan_loss_wt*gan_loss+self.hparams.id_loss_wt*face_id_loss
        loss = self.hparams.syncnet_wt * sync_loss + (1 - self.hparams.syncnet_wt) * recon_loss
        if self.hparams.gan_loss_wt>0 and self.current_epoch>self.gan_loss_start_epoch :
            if self.hparams.disc_type!='stylegan2':
                optimizer_d.zero_grad()
                self.manual_backward(disc_loss)
                self.clip_gradients(optimizer_d, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
                optimizer_d.step()
                lr_scheduler_d.step()

            optimizer_g.zero_grad()
            self.manual_backward(loss)
            self.clip_gradients(optimizer_g, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
            optimizer_g.step()
            lr_scheduler_g.step()
        
           


        self.log('train_loss', loss, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log('train_image_recon_loss', image_recon_loss, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log('train_lpips_loss', lpips_loss, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log('train_latent_recon_loss', latent_recon_loss, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log('train_sync_loss', sync_loss, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log('train_id_loss', face_id_loss, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log('train_gan_loss', gan_loss, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log('train_disc_loss', disc_loss, prog_bar=True, sync_dist=True, on_epoch=True)
        return loss

    def evaluation_metric(self,  g, gt): #B,C,T,H,W
        # refs = x[:, 3:, :, :, :]
        # inps = x[:, :3, :, :, :]
        # #reshape the tensors to [B*T, C, H, W]
        # refs = (refs.view(-1, refs.size(3), refs.size(4), refs.size(1)).clamp(0, 1) * 255).byte().cpu().numpy().astype(np.uint8)
        # inps = (inps.view(-1, inps.size(3), inps.size(4), inps.size(1)).clamp(0, 1) * 255).byte().cpu().numpy().astype(np.uint8)
        psnr_values = []
        ssim_values = []
        g= g.view(-1, g.size(1), g.size(3), g.size(4)).clamp(0, 1) 
        gt = gt.view(-1, gt.size(1), gt.size(3), gt.size(4)).clamp(0, 1) 
        FID_batch_size = 1024
        #计算g和gt之间的PSNR,SSIM,FID
        psnr_score = psnr(g, gt, reduction='none')
        psnr_values.extend([e.item() for e in psnr_score])
        ssim_score = ssim(g, gt, data_range=1, reduction='none')
        ssim_values.extend([e.item() for e in ssim_score])

        fid_metric = FID()
        B_mul_T = g.size(0)
        total_images = torch.cat((gt, g), 0)
        if len(total_images) > FID_batch_size:
            total_images = torch.split(total_images, FID_batch_size, 0)
        else:
            total_images = [total_images]
        feature_extractor = InceptionV3()
        total_feats = []
        for sub_images in total_images:
            sub_images = sub_images.cuda()
            feats = fid_metric.compute_feats([
                {'images': sub_images},
            ], feature_extractor=feature_extractor)
            feats = feats.detach()
            total_feats.append(feats)
        total_feats = torch.cat(total_feats, 0)
        gt_feat, pd_feat = torch.split(total_feats, (B_mul_T, B_mul_T), 0)

        gt_feats = gt_feat.cuda()
        pd_feats = pd_feat.cuda()
        
        fid = fid_metric.compute_metric(pd_feats, gt_feats).item()
        

        return np.asarray(psnr_values).mean(), np.asarray(ssim_values).mean(), fid

    def validation_step(self, batch, batch_idx):
        if self.hparams.syncnet:
            x, indiv_audios, audio_cropped, gt = batch
        else:
            x, indiv_audios, gt = batch
        
        g_latent, g_image = self(indiv_audios, x, not self.hparams.no_image_loss or self.hparams.sample_images!=0)        
        
        if not self.hparams.no_image_loss:
            if self.hparams.full_image_loss:
                image_recon_loss = self.recon_loss(g_image, gt)
                lpips_loss =   0 if self.hparams.lpips_weight == 0. else self.get_lpips_loss(g_image, gt)
            else:
                image_recon_loss = self.recon_loss(g_image[:, :, :, g_image.size(3) // 2:, :], gt[:, :, :, gt.size(3) // 2:, :])
                lpips_loss = 0 if self.hparams.lpips_weight == 0. else  self.get_lpips_loss(g_image[:, :, :, g_image.size(3) // 2:, :], gt[:, :, :, gt.size(3) // 2:, :])
        else:
            image_recon_loss = 0.
            lpips_loss = 0.
        image_loss = self.hparams.lpips_weight * lpips_loss + image_recon_loss
        
        gt_for_vae = self.reshape_gt_image_for_vae(gt)        
        gt_latent = self.encode_with_vae(gt_for_vae)
        gt_latent = self.inverse_reshape_face_sequences(gt_latent)
        latent_recon_loss = self.recon_loss(g_latent, gt_latent)
        
        if self.hparams.syncnet:
            sync_loss = self.get_sync_loss(audio_cropped, g_image[:, :, :, g_image.size(3) // 2:, :])
        else:
            sync_loss = 0.
        
        recon_loss = latent_recon_loss + self.hparams.image_loss_wt * image_loss
        val_loss = self.hparams.syncnet_wt * sync_loss + (1 - self.hparams.syncnet_wt) * recon_loss
        
        if self.hparams.sample_images!=0 and batch_idx == 0 and self.current_epoch%self.hparams.sample_images==0:
            for i in range(min(16, x.size(0))):
                self.save_sample_images(x[i:i+1], g_image[i:i+1], gt[i:i+1], self.global_step, self.trainer.checkpoint_callback.dirpath,i)
        
        if self.hparams.eval!=0 and self.current_epoch%self.hparams.eval==0:
            psnr, ssim, fid = self.evaluation_metric(g_image, gt)
            self.log('val_psnr', psnr, prog_bar=True, sync_dist=True, on_epoch=True)
            self.log('val_ssim', ssim, prog_bar=True, sync_dist=True, on_epoch=True)
            self.log('val_fid', fid, prog_bar=True, sync_dist=True, on_epoch=True)
            self.psnr_list.append(psnr)
            self.ssim_list.append(ssim)
            self.fid_list.append(fid)
          #  self.save_sample_images(x[:1], g_image[:1], gt[:1], self.global_step, self.trainer.checkpoint_callback.dirpath)
        
        self.log('val_loss', val_loss, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log('val_image_recon_loss', image_recon_loss, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log('val_lpips_loss', lpips_loss, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log('val_latent_recon_loss', latent_recon_loss, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log('val_sync_loss', sync_loss, prog_bar=True, sync_dist=True, on_epoch=True)
        return val_loss

    def save_sample_images(self, x, g, gt, global_step, checkpoint_dir,idx):
        refs = x[:, 3:, :, :, :]
        inps = x[:, :3, :, :, :]
        
        sample_image_dir = join(os.path.dirname(checkpoint_dir), "sample_images")
        os.makedirs(sample_image_dir, exist_ok=True)
        
        folder = join(sample_image_dir, "samples_step_{:09d}".format(global_step))
        os.makedirs(folder, exist_ok=True)
        
        refs = (refs.squeeze(0).clamp(0, 1) * 255).byte().cpu().numpy().transpose(1, 2, 3, 0).astype(np.uint8)
        inps = (inps.squeeze(0).clamp(0, 1) * 255).byte().cpu().numpy().transpose(1, 2, 3, 0).astype(np.uint8)
        g = (g.squeeze(0).clamp(0, 1) * 255).byte().cpu().numpy().transpose(1, 2, 3, 0).astype(np.uint8)
        gt = (gt.squeeze(0).clamp(0, 1) * 255).byte().cpu().numpy().transpose(1, 2, 3, 0).astype(np.uint8)
            
        collage = np.concatenate((refs, inps, g, gt), axis=2)
        # print(f"collage: {collage.shape}")
        
        for t, c in enumerate(collage[:1]):
            # print(f"batch_idx: {t}, c: {c.shape}")
            cv2.imwrite('{}/{}_{}.jpg'.format(folder, idx, t), c)

    def on_validation_epoch_end(self) -> None:
        if self.hparams.syncnet:
            val_sync_loss = self.trainer.logged_metrics['val_sync_loss']
            if val_sync_loss < .75:
                print(f"Syncnet loss {val_sync_loss} is less than 0.75, setting syncnet_wt to {self.hparams.sync_loss_weight}")
                self.hparams.syncnet_wt = self.hparams.sync_loss_weight
        if self.hparams.eval!=0 and self.current_epoch%self.hparams.eval==0:
            print(f"PSNR: {np.mean(self.psnr_list)}, SSIM: {np.mean(self.ssim_list)}, FID: {np.mean(self.fid_list)}")
            self.psnr_list, self.ssim_list, self.fid_list = [], [], []
            
    def configure_optimizers(self):
        # optimizer = FusedAdam(self.unet.parameters(), lr=1e-4)
        if self.hparams.gan_loss_wt==0:
            optimizer = torch.optim.AdamW(self.unet.parameters(), lr=1e-4)
        
            # 设置 LambdaLinearScheduler 参数
            warm_up_steps = [10000]  # 预热步数
            f_min = [1.0]  # 最小学习率
            f_max = [1.0]  # 最大学习率
            f_start = [1.e-6]  # 开始学习率
            cycle_lengths = [10000000000000]  # 周期长度
            
            # 创建 LambdaLinearScheduler 实例
            scheduler = LambdaLinearScheduler(
                warm_up_steps=warm_up_steps,
                f_min=f_min,
                f_max=f_max,
                f_start=f_start,
                cycle_lengths=cycle_lengths,
                # verbosity_interval=1000  # 每1000步打印一次学习率信息
            )
            
            # 使用 LambdaLR 包装自定义的调度器
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler.schedule)
            
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": lr_scheduler, "interval": "step", "frequency": 1}}
        else:
            optimizer_g = torch.optim.AdamW(self.unet.parameters(), lr=1e-4, betas=(0.5, 0.999))
            optimizer_d = torch.optim.AdamW(self.disc.parameters(), lr=1e-4, betas=(0.5, 0.999))
            
            warm_up_steps = [10000]  # 预热步数
            f_min = [1.0]  # 最小学习率
            f_max = [1.0]  # 最大学习率
            f_start = [1.e-6]  # 开始学习率
            cycle_lengths = [10000000000000]  # 周期长度
            
            # 创建 LambdaLinearScheduler 实例
            scheduler = LambdaLinearScheduler(
                warm_up_steps=warm_up_steps,
                f_min=f_min,
                f_max=f_max,
                f_start=f_start,
                cycle_lengths=cycle_lengths,
                # verbosity_interval=1000  # 每1000步打印一次学习率信息
            )
            lr_scheduler_g = torch.optim.lr_scheduler.LambdaLR(optimizer_g, lr_lambda=scheduler.schedule)
            lr_scheduler_d=torch.optim.lr_scheduler.LambdaLR(optimizer_d, lr_lambda=scheduler.schedule)

            return [optimizer_g, optimizer_d],[lr_scheduler_g,lr_scheduler_d]


    def train_dataloader(self):
        train_dataset = AudioVisualDataset(self.hparams.data_root, 
                                           audio_root=self.hparams.audio_root if self.hparams.audio_root else None, 
                                           split='train' if not self.hparams.overfit else 'main', 
                                           dataset_name=self.hparams.dataset_name, 
                                           args=self.hparams, 
                                           dataset_size=self.hparams.dataset_size,
) 
        if self.hparams.ft_dataset:
            fine_tune_dataset = AudioVisualDataset(self.hparams.ft_root, 
                                                   split='train' if not self.hparams.overfit else 'main', 
                                                   audio_root=self.hparams.ft_audio_root if self.hparams.ft_audio_root else None,
                                                   dataset_name=self.hparams.ft_dataset, 
                                                   args=self.hparams, 
                                                   dataset_size=self.hparams.dataset_size,
                                                  )
            train_dataset = CombinedDataset(train_dataset, fine_tune_dataset,base_ratio=0.3)
        return DataLoader(train_dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers)

    def val_dataloader(self):
        test_dataset = AudioVisualDataset(self.hparams.data_root, 
                                          audio_root=self.hparams.audio_root if self.hparams.audio_root else None, 
                                          split='val', 
                                          dataset_name=self.hparams.dataset_name, 
                                          args=self.hparams, 
                                          dataset_size=self.hparams.dataset_size // 20,
                                          )
        if self.hparams.ft_dataset:
            fine_tune_dataset = AudioVisualDataset(self.hparams.ft_root, 
                                                   audio_root=self.hparams.ft_audio_root if self.hparams.ft_audio_root else None,
                                                   split='val', 
                                                   dataset_name=self.hparams.ft_dataset, 
                                                   args=self.hparams, 
                                                   dataset_size=self.hparams.dataset_size // 20,
                                                   )
            test_dataset = CombinedDataset(test_dataset, fine_tune_dataset)
        return DataLoader(test_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)

def print_training_info(args):
    print("\nTraining Configuration:")
    print(f"Dataset Name: {args.dataset_name}")
    print(f"Data Root: {args.data_root}")
    print(f"Audio Root: {args.audio_root}")
    print(f"Clip Loss Enabled: {args.clip_loss}")
    print(f"U-Net Config File: {args.unet_config}")
    print(f"Checkpoint Path: {args.ckpt}")
    print(f"Sample Images Enabled: {args.sample_images}")
    print(f"WandB Logging Enabled: {args.wandb}")
    print(f"Overfit Mode Enabled: {args.overfit}")
    print(f"Dropout on Reference Frames Enabled: {args.dropout_ref}")
    print(f"Wav2Vec2 Embeddings Enabled: {args.wav2vec2}")
    print(f"Gradient Accumulation Steps: {args.accu_grad}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Sync Loss Weight: {args.sync_loss_weight}")
    print(f"Fine-Tune Dataset: {args.ft_dataset}")
    print(f"Fine-Tune Dataset Root: {args.ft_root}")
    print(f"Fine-Tune Audio Root: {args.ft_audio_root}")
    print(f"L2 Loss Enabled: {args.l2_loss}")
    print(f"Image Size: {args.image_size}")
    print(f"Image Loss Weight: {args.image_loss_wt}")
    print(f"Syncnet Checkpoint: {args.syncnet}")
    print(f"Syncnet T: {args.syncnet_T}")
    print(f"Dataset Size: {args.dataset_size}")
    print(f"No Image Loss: {args.no_image_loss}")
    print(f"LPIPS Loss Weight: {args.lpips_weight}")
    print(f"Full Image Loss Enabled: {args.full_image_loss}")
    print("\nStarting training...\n")
    
if __name__ == "__main__":
    # Set the matrix multiplication precision
    # Use 'medium' for better performance with acceptable precision loss
    torch.set_float32_matmul_precision('medium')

    parser = argparse.ArgumentParser(description='Code to train latent wav2lip with lightning')
    parser.add_argument("--data_root", help="Root folder of the preprocessed LRS2 dataset", default='/data/wangbaiqin/dataset/all_images')
    parser.add_argument('--audio_root', type=str, help='Root folder of the preprocessed audio dataset',default='/data/wangbaiqin/dataset/all_audios')
    parser.add_argument("--dataset_name", help="Name of the dataset to use, eg: 3300, hdtf, 3300_and_hdtf", required=True)
    parser.add_argument('--clip_loss', action='store_true', help='Enable clip loss.')
    parser.add_argument('--unet_config', type=str, help='Path to the unet config file', default='unet_config/customized_unet_v4_large')
    parser.add_argument('--ckpt', type=str, help='Path to the checkpoint to load the model from')
    parser.add_argument('--sample_images',type=int, default=5, help='epoch to save sample image')
    parser.add_argument('--wandb', action='store_true', help='Enable wandb logging')
    parser.add_argument('--overfit', action='store_true', help='Enable overfitting mode to focus training on training loss.')
    parser.add_argument('--dropout_ref', action='store_true', help='Enable dropout on the reference frames.')
    parser.add_argument('--wav2vec2', action='store_true', help='Use wav2vec2 embeddings')
    parser.add_argument('--accu_grad', type=int, default=2, help='Number of gradient accumulation steps')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--sync_loss_weight', type=float, default=0.01, help='Weight for sync loss')
    parser.add_argument('--ft_dataset', type=str, help='Fine-tune dataset name')
    parser.add_argument('--ft_root', type=str, help='Root folder of the fine-tune dataset')
    parser.add_argument('--ft_audio_root', type=str, help='Root folder `of the fine-tune audio dataset')
    parser.add_argument('--l2_loss', action='store_true', help='Enable L2 loss')
    parser.add_argument('--image_size', type=int, default=256, help='Image size')
    parser.add_argument('--image_loss_wt', type=float, default=1.0, help='Weight for image reconstruction loss')
    parser.add_argument('--gan_loss_wt', type=float, default=0.5, help='Weight for image reconstruction loss')
    parser.add_argument('--syncnet', type=str, help='Path to the syncnet checkpoint to load the model from')
    parser.add_argument('--syncnet_T', type=int, default=1, help='Number of frames to consider for syncnet loss')
    parser.add_argument('--dataset_size', type=int, default=8000, help='Size of the dataset')
    parser.add_argument('--no_image_loss', action='store_true', help='Disable image loss')
    parser.add_argument('--lpips_type',  type=str, help='type of lpips loss',default='vgg19')
    parser.add_argument('--no_style_loss', action='store_true', help='not enable style loss when use vgg19')
    parser.add_argument('--lpips_weight', type=float, default=1, help='Weight for LPIPS loss')
    parser.add_argument('--full_image_loss', action='store_true', help='Enable full image loss')
    parser.add_argument('--mouth_mask', action='store_true', help='Enable full image loss')
    parser.add_argument('--up_ratio', type=float, default=0, help='Up ratio when loading images')
    parser.add_argument('--gs_blur', action='store_true', help='Enable Gaussian blur when mask image')
    parser.add_argument('--id_loss_wt', type=float, default=0.001, help='Weight for ID loss')
    parser.add_argument('--eval',  type=int, default=8, help='epoch to evaluate')
    parser.add_argument('--disc_type',  type=str, default="stylegan2", help='disc type')
    parser.add_argument('--unet_ckpt',type=str,help="unet to initial")
    parser.add_argument('--disc_ckpt',type=str,help="disc checkpoint to initial")
    args = parser.parse_args()
    
    if args.ft_dataset and not args.ft_root:
        raise RuntimeError("Please specify the root folder of the fine-tune dataset by --ft_root")
    elif args.ft_root and not args.ft_dataset:
        raise RuntimeError("Please specify the fine-tune dataset name by --ft_dataset")
    
    # Print the training information
    print_training_info(args)
    
    # Convert hparams instance to a dictionary
    hparams_dict = hparams.data

    # Update hparams with args
    hparams_dict.update(vars(args))

    # Create an instance of LatentWav2Lip with merged parameters
    model = LatentWav2Lip(hparams_dict)
    
    # Load the UNet model if a checkpoint is not provided
    if not model.hparams.ckpt and args.unet_ckpt:
        model.load_unet(args.unet_ckpt)
    elif not model.hparams.ckpt:
        model.load_unet(model.hparams.unet_config)

    # Checkpoint callback to save the model periodically
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        # put image size and syncnet_T into ckpt name
        filename='wav2lip-i-' + args.dataset_name + '-S=' + str(args.image_size) + '-T=' + str(args.syncnet_T) + '-{epoch}-{step}-{train_loss:.3f}-{val_loss:.3f}-{val_sync_loss:.3f}',
        save_top_k=3,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )
    checkpoint_callback_latest = pl.callbacks.ModelCheckpoint(
        # put image size and syncnet_T into ckpt name
        filename='wav2lip-i-' + args.dataset_name + '-S=' + str(args.image_size) + '-T=' + str(args.syncnet_T) + '-{epoch}-{step}-{train_loss:.3f}-{val_loss:.3f}-{val_sync_loss:.3f}',
        save_top_k=1,
        verbose=True,
        monitor='epoch',
        mode='max'
    )
    # 设置日志目录和实验名称
    if args.wandb:
        logger = WandbLogger(project='image_wav2lip')
    else:
        logger = TensorBoardLogger('experiments', name='image_wav2lip_experiment')

    callbacks = [checkpoint_callback,checkpoint_callback_latest, RichProgressBar(), LearningRateMonitor(logging_interval='step')]

    # Include EarlyStopping if overfitting is enabled
    # if args.overfit:
    #     early_stopping_callback = EarlyStopping(monitor='train_loss', min_delta=0.0001, patience=100, verbose=True, mode='min', stopping_threshold=0.1)
    #     callbacks.append(early_stopping_callback)
        
    # Trainer setup for multi-GPU training
    trainer = Trainer(
        logger=logger,
        max_epochs=model.hparams.nepochs,
        strategy=DDPStrategy(find_unused_parameters=True), 
        precision='16-mixed',
        #accumulate_grad_batches=model.hparams.accu_grad,
        #gradient_clip_val=0.5,
        callbacks=callbacks
    )
    if args.gan_loss_wt>0:
        trainer = Trainer(
        logger=logger,
        max_epochs=model.hparams.nepochs,
        strategy=DDPStrategy(find_unused_parameters=True), 
        precision='16-mixed',

        callbacks=callbacks
    )
    else:
       trainer = Trainer(
        logger=logger,
        max_epochs=model.hparams.nepochs,
        strategy=DDPStrategy(find_unused_parameters=True), 
        precision='16-mixed',
        accumulate_grad_batches=model.hparams.accu_grad,
        gradient_clip_val=0.5,
        callbacks=callbacks
    ) 

    trainer.fit(model, ckpt_path=model.hparams.ckpt if model.hparams.ckpt else None)

