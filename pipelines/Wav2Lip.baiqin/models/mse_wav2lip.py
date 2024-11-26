import torch
from torch import nn
from torch.nn import functional as F
import math
import os
from .conv import Conv2dTranspose, Conv2d, nonorm_Conv2d
from diffusers import AutoencoderKL

class Wav2Lip_Latent(nn.Module):
    def __init__(self,alreadyEncode=False):
        super(Wav2Lip_Latent, self).__init__()

     #   print(mse.encoder)
        self.vae=AutoencoderKL.from_pretrained(os.path.join(os.curdir,"models/SD_MSE"),use_safetensors=True)
        self.alreadyEncode=alreadyEncode
        for p in self.vae.parameters():
            p.requires_grad = False 
        self.face_encoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(8, 8, kernel_size=7, stride=1, padding=3)), # 8,64,64

            nn.Sequential(Conv2d(8, 16, kernel_size=5, stride=2, padding=2), # 16,32,32
            Conv2d(16, 16, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(16, 16, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # 32,16,16
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(32, 64, kernel_size=3, stride=2, padding=1),                 # B, 32, 16, 16 -> B, 64, 8, 8
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),  # B, 64, 8, 8 -> B, 64, 8, 8
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True)),  # B, 64, 8, 8 -> B, 64, 8, 8

            nn.Sequential(Conv2d(64, 128, kernel_size=3, stride=2, padding=1),                # B, 64, 8, 8 -> B, 128, 4, 4
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),# B, 128, 4, 4 -> B, 128, 4, 4
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True)),# B, 128, 4, 4 -> B, 128, 4, 4

            nn.Sequential(Conv2d(128, 256, kernel_size=3, stride=2, padding=1),               # B, 128, 4, 4 -> B, 256, 2, 2
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),# B, 256, 2, 2 -> B, 256, 2, 2
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True)),# B, 256, 2, 2 -> B, 256, 2, 2

            nn.Sequential(Conv2d(256, 512, kernel_size=2, stride=1, padding=0),               # B, 256, 2, 2 -> B, 512, 1, 1
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0))                # B, 512, 1, 1 -> B, 512, 1, 1
            ])
        
           

        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)

        self.face_decoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(512, 512, kernel_size=1, stride=1, padding=0)),

            nn.Sequential(Conv2dTranspose(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True)), # 2, 2

            nn.Sequential(Conv2dTranspose(768, 384, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True),), # 4, 4

            nn.Sequential(Conv2dTranspose(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),), # 8, 8

            nn.Sequential(Conv2dTranspose(320, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),), # 16, 16

            nn.Sequential(Conv2dTranspose(160, 64, kernel_size=3, stride=2, padding=1, output_padding=1), 
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),), # 32, 32

            nn.Sequential(Conv2dTranspose(80, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),),]) # 64,64

        self.output_block = nn.Sequential(Conv2d(40, 16, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(16, 4, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()) 

    def forward(self, audio_sequences, face_sequences):
        # audio_sequences = (B, T, 1, 80, 16)
        B = audio_sequences.size(0)

        input_dim_size = len(face_sequences.size())
        if input_dim_size > 4:
            audio_sequences = torch.cat([audio_sequences[:, i] for i in range(audio_sequences.size(1))], dim=0)
            face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)
            # audio_sequences = audio_sequences[:,0,:,:]
            # face_sequences = face_sequences[:,:,0,:,:]
        # print(face_sequences.shape)
        # print(audio_sequences.shape)
        audio_embedding = self.audio_encoder(audio_sequences) # B, 512, 1, 1
    #    print("audio embedding shape:",audio_embedding.shape)
        feats = []
       # print(face_sequences.shape)
        face_sequences1=face_sequences[:,:3,:,:]
        face_sequences2=face_sequences[:,3:,:,:]
        with torch.no_grad():
            face1=self.vae.encode(face_sequences1).latent_dist.sample()
            face2=self.vae.encode(face_sequences2).latent_dist.sample()
        face_sequences=torch.cat((face1,face2),dim=1)
        x = face_sequences
        
        print("input shape:",x.shape)
        for f in self.face_encoder_blocks:
            x = f(x)
           # print("encoder x:",x.shape)
            feats.append(x)

        x = audio_embedding
        for f in self.face_decoder_blocks:
          #  print("decoder x:",x.shape)
            x = f(x)
            try:
                x = torch.cat((x, feats[-1]), dim=1)
            except Exception as e:
                print(x.size())
                print(feats[-1].size())
                raise e
            
            feats.pop()

        x = self.output_block(x)
        
        x=self.vae.decode(x).sample
        #x=self.vae.decode(x).sample
       # print("out latent shape:",x.shape)
        if input_dim_size > 4:
            x = torch.split(x, B, dim=0) # [(B, C, H, W)]
           # x=self.vae.decode(x).sample
            outputs = torch.stack(x, dim=2) # (B, C, T, H, W)

        else:
           # outputs=self.vae.decode(x).sample
            outputs = x
            
        return outputs


if __name__=="__main__":
    device='cuda:0'
   # mse=AutoencoderKL.from_pretrained('./SD_MSE',use_safetensors=True,torch_dtype=torch.float16).to(device)
    model=Wav2Lip_Latent().to(device).to(torch.float16)
    model.eval()
    x1=torch.randint(0,256,(1,6,512,512)).to(device).to(torch.float16)
    x2=torch.randint(0,256,(1,3,512,512)).to(device).to(torch.float16)
    audio=torch.randint(0,256,(1,1,80,16)).to(device).to(torch.float16)
    output=model(audio,x1)
    print(output.shape)