import torch
import numpy as np
from os.path import dirname, join, basename, isfile
from glob import glob
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os, random, argparse
from models import audio
from models.hparams import hparams 
import cv2



def get_image_list(data_root, dataset_name, split):
    filelist = []

    with open('filelists/{}/{}.txt'.format(dataset_name, split)) as f:
        for line in f:
            line = line.strip()
            if ' ' in line: line = line.split()[0]
            filelist.append(os.path.join(data_root, line))

    return filelist

class AudioVisualDataset(Dataset):
    def __init__(self, data_root, dataset_name, split, args, audio_root=None, dataset_size=512000,up_ratio=0,gs_blur=False,transform=None):
        self.all_videos = get_image_list(data_root, dataset_name, split)
        self.dataset_size = dataset_size
        self.syncnet_audio_size = 10 * args.syncnet_T if not args.wav2vec2 else 2 * args.syncnet_T
        self.frame_audio_size = 10 * 5 if not args.wav2vec2 else 2 * 5 # always use syncnet_T=5 for each frame
        self.args = args
        self.data_root = data_root
        self.audio_root = audio_root
        self.up_ratio=up_ratio
        self.transform=transform
        self.gs_blur=gs_blur
        self.image_size=256
        self.syncnet_mel_step_size = 10
        self.ref_num=args.ref_num
        self.drop_ref_prob=args.drop_ref_prob
        if os.path.exists('filelists/{}/offset.txt'.format(dataset_name)):
            self.offset_file='filelists/{}/offset.txt'.format(dataset_name)
            self.offset=self.read_offset(self.offset_file)
        else:
            self.offset=None
        #self.refnum=args.refnum

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
    def get_window_from_list(self,frame_list):
        window_fnames = []
        for frame in frame_list:
            frame_id=self.get_frame_id(frame)
            frame=join(dirname(frame),'{}.jpg'.format(frame_id))
            if not isfile(frame):
                return None
            window_fnames.append(frame)
        return window_fnames

        return 
    def read_window(self, window_fnames):
        if window_fnames is None: return None
        window = []
        for fname in window_fnames:
            img = cv2.imread(fname)
            if img is None:
                return None
            try:
                #获得一个随机数0——0.15
                r = random.random() * self.up_ratio
                img_new_height_start=int(img.shape[0]*r)
                img=img[img_new_height_start:,:,:]
                img = cv2.resize(img, (self.image_size, self.image_size))
            except Exception as e:
                return None

            window.append(img)
        
        s=random.random()
        if s>0.66:
            for win in window:
                win=np.flip(win,1)

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
        start_idx = int(50. * (start_frame_num / float(25)))
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
        #x=x*2-1
      #  x = np.asarray(window)
        x = np.transpose(x, (3, 0, 1, 2))

        return x

    def read_offset(self, offset_file):
        offset={}
        with open(offset_file,'r') as f:
            for line in f:
                line=line.strip()
                if not line:
                    continue
                content=line.split()
                video_name=content[1].split('/')[-1]
                offset[video_name]=int(float(content[4]))
        return offset

    def __len__(self):
        #return min(self.dataset_size,len(self.all_videos)) # len(self.all_videos)
        return self.dataset_size

    def __getitem__(self, idx):
        while 1:
            idx = random.randint(0, len(self.all_videos) - 1)
            vidname = self.all_videos[idx]
           # print(vidname)
            img_names = list(glob(join(vidname, '*.jpg')))
            img_names = [img for img in img_names if 'mask' not in img]
            if len(img_names) <= 30:#3 * self.args.syncnet_T:
                # print(f"Video {vidname} has less frames than required")
                continue
            
            img_name = random.choice(img_names)
            wrong_img_names=[]
            for i in range(self.ref_num):
                wrong_img_name = random.choice(img_names)
                # Ensure wrong_img_name is at least syncnet_T frames away from img_name
                while abs(self.get_frame_id(img_name) - self.get_frame_id(wrong_img_name)) < 10:
                    wrong_img_name = random.choice(img_names)
                wrong_img_names.append(wrong_img_name)
            # wrong_img_name = random.choice(img_names)
            # # Ensure wrong_img_name is at least syncnet_T frames away from img_name
            # while abs(self.get_frame_id(img_name) - self.get_frame_id(wrong_img_name)) < 10:#self.args.syncnet_T:
            #     wrong_img_name = random.choice(img_names)

            window_fnames = self.get_window(img_name)
            wrong_window_fnames = self.get_window_from_list(wrong_img_names)
            if window_fnames is None or wrong_window_fnames is None:
                continue

            window = self.read_window(window_fnames)
            if window is None:
                # print(f"Window is None for {vidname}")
                continue

            wrong_window = self.read_window(wrong_window_fnames)
            if wrong_window is None:
                # print(f"Wrong Window is None for {vidname}")
                continue
            
            wrong_window=np.concatenate(wrong_window,axis=2).reshape(1,256,256,-1)
            #wrong window重复5次
            wrong_window=np.repeat(wrong_window,self.args.syncnet_T,axis=0)
            # try:
            #     wavpath = join(vidname, "audio.wav")
            #     wav = audio.load_wav(wavpath, hparams.sample_rate)

            #     orig_mel = audio.melspectrogram(wav).T
            # except Exception as e:
            #     print(e)
            #     continue

            # mel = self.crop_audio_window(orig_mel.copy(), img_name,self.args.syncnet_T )
            
            # if (mel.shape[0] != self.syncnet_mel_step_size):
            #     print(vidname,mel.shape)
            #     continue

            # indiv_mels = self.get_segmented_mels(orig_mel.copy(), img_name)
            # #print(vidname)
            # if indiv_mels is None: continue
            # mel = torch.FloatTensor(mel.T).unsqueeze(0)
            # indiv_mels = torch.FloatTensor(indiv_mels).unsqueeze(1)


            if self.audio_root:
                # switch fro data_root to audio_root
                vidname = vidname.replace(self.data_root, self.audio_root)
            #将一个list str拼接起来为一个str
          
            vidname_tmp=vidname.split('/')[-1].split('_')[0:-1]
            vidname_offset=vidname_tmp[0]
            for i in range(1,len(vidname_tmp)):
                vidname_offset=vidname_offset+'_'+vidname_tmp[i]
            #print(vidname_offset)
            if self.offset and self.offset.get(vidname_offset):
                offsets=self.offset[vidname_offset]                
            else:
                offsets=0

            if not self.args.wav2vec2:
                # load syncnet_T frames of audio embeddings for syncnet loss
                
                if self.args.syncnet:
                    audio_cropped = self.get_whisper_embedding(vidname, self.get_frame_id(img_name)-offsets, syncnet_T=self.args.syncnet_T)
                    
                # always load for each frame of 5 frames of audio embeddings for cross attention
                indiv_audios = self.get_whisper_segmented_audios(vidname, self.get_frame_id(img_name)-offsets)
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
            if self.args.syncnet:
                audio_cropped = audio_cropped.unsqueeze(0).float()
            
            indiv_audios = indiv_audios.unsqueeze(1).float()


            window = self.prepare_window(window)
            y = window.copy()
            #window[:, :, window.shape[2]//2:] = 0.
            #下半部分替换为0——1之间的随机数
            if self.gs_blur:
                #window[:,:,window.shape[2]//2:]=np.random.randint(0,256,(window.shape[0],window.shape[1],window.shape[2]//2,window.shape[3]))
                window[:,:,window.shape[2]//2:]=np.random.rand(window.shape[0],window.shape[1],window.shape[2]//2,window.shape[3])
            else:
                window[:, :, window.shape[2]//2:] = 0.
            wrong_window = self.prepare_window(wrong_window)
            if self.drop_ref_prob>0:
                if random.random()<self.drop_ref_prob:
                    wrong_window=np.zeros_like(wrong_window)
            if self.ref_num!=0:
                x = np.concatenate([window, wrong_window], axis=0)
            else:
                x = window
            #if self.transform is None:    
            x = torch.FloatTensor(x)
            y = torch.FloatTensor(y)
            # else:
            #     x=self.transform(x)
            #     y=self.transform(y)
           # print(vidname)
            # if not self.args.syncnet:
            #     return x, indiv_mels, y
            
            # return x, indiv_mels, mel, y
            
            
            # don't return audio_cropped if syncnet is not enabled
            if not self.args.syncnet:
                return x, indiv_audios, y
            
            return x, indiv_audios, audio_cropped, y
        
def save_sample_images(self, x, g, gt, global_step, checkpoint_dir,idx):
        refs = x[:, 3:, :, :, :]
        inps = x[:, :3, :, :, :]
        
        sample_image_dir = join(os.path.dirname(checkpoint_dir), "sample")
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