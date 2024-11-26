#!/usr/bin/python
#-*- coding: utf-8 -*-
# Video 25 FPS, Audio 16000HZ

import torch
import numpy
import time, pdb, argparse, subprocess, os, math, glob
import cv2
import python_speech_features

from scipy import signal
from scipy.io import wavfile
from SyncNetModel import *
from shutil import rmtree
from tqdm import tqdm

import time, pdb, argparse, subprocess
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

# ==================== Get OFFSET ====================

def calc_pdist(feat1, feat2, vshift=10):
    
    win_size = vshift*2+1

    feat2p = torch.nn.functional.pad(feat2,(0,0,vshift,vshift))

    dists = []

    for i in range(0,len(feat1)):

        dists.append(torch.nn.functional.pairwise_distance(feat1[[i],:].repeat(win_size, 1), feat2p[i:i+win_size,:]))

    return dists

# ==================== MAIN DEF ====================
def crop_and_resize(img, output_size=(224, 224)):
    # 获取图像的高度和宽度
    h, w, _ = img.shape

    # 设置裁剪宽度为图像的原始宽度
    crop_width = w
    # 从底部开始裁剪，裁剪高度为宽度
    crop_height = min(h, crop_width)

    # 计算裁剪的起始点和结束点
    start_y = h - crop_height  # 从图像底部开始裁剪
    end_y = h
    start_x = 0
    end_x = crop_width

    # 裁剪图像
    cropped_img = img[start_y:end_y, start_x:end_x]

    # 将裁剪后的图像缩放到 224x224
    resized_img = cv2.resize(cropped_img, output_size)

    return resized_img

class SyncNetInstance(torch.nn.Module):

    def __init__(self, dropout = 0, num_layers_in_fc_layers = 1024):
        super(SyncNetInstance, self).__init__();

        self.__S__ = S(num_layers_in_fc_layers = num_layers_in_fc_layers).cuda();

    def evaluate(self, opt, flist, audio_path, frame_rate=25):

        self.__S__.eval();

        # ========== ==========
        # Convert files
        # ========== ==========

        # if os.path.exists(os.path.join(opt.tmp_dir,opt.reference)):
        #   rmtree(os.path.join(opt.tmp_dir,opt.reference))

        # os.makedirs(os.path.join(opt.tmp_dir,opt.reference))

        # command = ("ffmpeg -y -i %s -threads 1 -f image2 %s" % (videofile,os.path.join(opt.tmp_dir,opt.reference,'%06d.jpg')))
        # output = subprocess.call(command, shell=True, stdout=None)

        # command = ("ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 %s" % (videofile,os.path.join(opt.tmp_dir,opt.reference,'audio.wav')))
        # output = subprocess.call(command, shell=True, stdout=None)
        
        # ========== ==========
        # Load video 
        # ========== ==========

        images = []
        
        first_img_num = int(os.path.basename(flist[0]).split('.')[0])
        last_img_num = int(os.path.basename(flist[-1]).split('.')[0])
        for fname in tqdm(flist, desc='imread'):
            img = cv2.imread(fname)
            img = crop_and_resize(img, output_size=(224, 224))
            # cv2.imwrite('out.jpg', img)
            # return
            images.append(img)

        im = numpy.stack(images,axis=3)
        im = numpy.expand_dims(im,axis=0)
        im = numpy.transpose(im,(0,3,4,1,2))

        imtv = torch.autograd.Variable(torch.from_numpy(im.astype(float)).float())

        sample_rate, audio = wavfile.read(audio_path)
        assert sample_rate == 16_000, f'{sample_rate}!=16_000'
        start_time = first_img_num / frame_rate
        end_time = last_img_num / frame_rate
        print(audio.shape)
        audio = load_audio_segment(audio, sample_rate, start_time, end_time)
        if audio.size == 0:
            return None, None, None

        mfcc = zip(*python_speech_features.mfcc(audio,sample_rate))
        mfcc = numpy.stack([numpy.array(i) for i in mfcc])

        cc = numpy.expand_dims(numpy.expand_dims(mfcc,axis=0),axis=0)
        cct = torch.autograd.Variable(torch.from_numpy(cc.astype(float)).float())

        # ========== ==========
        # Check audio and video input length
        # ========== ==========

        if (float(len(audio))/16000) != (float(len(images))/25) :
            print("WARNING: Audio (%.4fs) and video (%.4fs) lengths are different."%(float(len(audio))/16000,float(len(images))/25))

        min_length = min(len(images),math.floor(len(audio)/640))
        
        # ========== ==========
        # Generate video and audio feats
        # ========== ==========

        lastframe = min_length-5
        im_feat = []
        cc_feat = []

        tS = time.time()
        for i in tqdm(range(0,lastframe,opt.batch_size), desc='feat'):
            
            im_batch = [ imtv[:,:,vframe:vframe+5,:,:] for vframe in range(i,min(lastframe,i+opt.batch_size)) ]
            im_in = torch.cat(im_batch,0)
            im_out  = self.__S__.forward_lip(im_in.cuda());
            im_feat.append(im_out.data.cpu())

            cc_batch = [ cct[:,:,:,vframe*4:vframe*4+20] for vframe in range(i,min(lastframe,i+opt.batch_size)) ]
            cc_in = torch.cat(cc_batch,0)
            cc_out  = self.__S__.forward_aud(cc_in.cuda())
            cc_feat.append(cc_out.data.cpu())

        if len(im_feat) == 0 or len(cc_feat) == 0:
            return None, None, None
        im_feat = torch.cat(im_feat,0)
        cc_feat = torch.cat(cc_feat,0)

        # ========== ==========
        # Compute offset
        # ========== ==========
            
        print('Compute time %.3f sec.' % (time.time()-tS))

        dists = calc_pdist(im_feat,cc_feat,vshift=opt.vshift)
        mdist = torch.mean(torch.stack(dists,1),1)

        minval, minidx = torch.min(mdist,0)

        offset = opt.vshift-minidx
        conf   = torch.median(mdist) - minval

        fdist   = numpy.stack([dist[minidx].numpy() for dist in dists])
        # fdist   = numpy.pad(fdist, (3,3), 'constant', constant_values=15)
        fconf   = torch.median(mdist).numpy() - fdist
        fconfm  = signal.medfilt(fconf,kernel_size=9)
        
        numpy.set_printoptions(formatter={'float': '{: 0.3f}'.format})
        print('Framewise conf: ')
        print(fconfm)
        print('AV offset: \t%d \nMin dist: \t%.3f\nConfidence: \t%.3f' % (offset,minval,conf))

        dists_npy = numpy.array([ dist.numpy() for dist in dists ])
        # return offset.numpy(), conf.numpy(), dists_npy
        return offset,minval,conf

    def extract_feature(self, opt, videofile):

        self.__S__.eval();
        
        # ========== ==========
        # Load video 
        # ========== ==========
        cap = cv2.VideoCapture(videofile)

        frame_num = 1;
        images = []
        while frame_num:
            frame_num += 1
            ret, image = cap.read()
            if ret == 0:
                break

            images.append(image)

        im = numpy.stack(images,axis=3)
        im = numpy.expand_dims(im,axis=0)
        im = numpy.transpose(im,(0,3,4,1,2))

        imtv = torch.autograd.Variable(torch.from_numpy(im.astype(float)).float())
        
        # ========== ==========
        # Generate video feats
        # ========== ==========

        lastframe = len(images)-4
        im_feat = []

        tS = time.time()
        for i in range(0,lastframe,opt.batch_size):
            
            im_batch = [ imtv[:,:,vframe:vframe+5,:,:] for vframe in range(i,min(lastframe,i+opt.batch_size)) ]
            im_in = torch.cat(im_batch,0)
            im_out  = self.__S__.forward_lipfeat(im_in.cuda());
            im_feat.append(im_out.data.cpu())

        im_feat = torch.cat(im_feat,0)

        # ========== ==========
        # Compute offset
        # ========== ==========
            
        print('Compute time %.3f sec.' % (time.time()-tS))

        return im_feat


    def loadParameters(self, path):
        loaded_state = torch.load(path, map_location=lambda storage, loc: storage);

        self_state = self.__S__.state_dict();

        for name, param in loaded_state.items():

            self_state[name].copy_(param);



parser = argparse.ArgumentParser(description = "SyncNet");

parser.add_argument('--initial_model', type=str, default="data/syncnet_v2.model", help='');
parser.add_argument('--batch_size', type=int, default='20', help='');
parser.add_argument('--vshift', type=int, default='15', help='');
# parser.add_argument('--videofile', type=str, default="data/example.avi", help='');
# parser.add_argument('--tmp_dir', type=str, default="data/work/pytmp", help='');
# parser.add_argument('--reference', type=str, default="demo", help='');

opt = parser.parse_args();
# 配置参数
img_root = '/mnt/diskwei/dataset/head_talk/preprocessed_shensi262_btm_move/'  # 图像根目录
audio_root = '/mnt/diskwei/dataset/head_talk/preprocessed_shensi262_btm_move/'  # 音频根目录
main_list_path = '/data2/weijinghuan/head_talk/1/talking-head/pipelines/LatentWav2Lip.OnTheFly/filelists_shensi_10/main.txt'  # 视频列表文件路径
frame_rate = 25  # 每秒视频帧数
audio_sample_rate = 16000  # 音频采样率
save_path = 'output/shensi_10.txt'
data_name = 'shensi_10'



def load_image_sequences(img_dir):
    # 获取文件名，并按数字顺序排序
    img_files = glob.glob(os.path.join(img_dir, '*.jpg'))
    img_files.sort(key=lambda f: int(os.path.basename(f).split('.')[0]))

    # 检查并找出连续的图像序列
    sequences = []
    current_sequence = []

    last_num = None
    for img_file in img_files:
        img_num = int(os.path.basename(img_file).split('.')[0])

        # 如果当前序列为空，或图像编号连续，加入当前序列
        if last_num is None or img_num == last_num + 1:
            current_sequence.append(img_file)
        else:
            # 保存当前序列
            sequences.append(current_sequence)
            current_sequence = [img_file]

        last_num = img_num

    # 添加最后一个序列
    if current_sequence:
        sequences.append(current_sequence)

    return sequences


def load_audio_segment(audio, sr, start_time, end_time):
    # 计算音频起止点的样本索引
    start_sample = int(start_time * audio_sample_rate)
    end_sample = int(end_time * audio_sample_rate)

    return audio[start_sample:end_sample]

def process_video_list(main_list_path, opt):
    out_fd = open(save_path, 'w')
    s = SyncNetInstance();
    s.loadParameters(opt.initial_model);
    print("Model %s loaded." % opt.initial_model);

    # 读取 main_list.txt
    with open(main_list_path, 'r') as f:
        video_paths = f.readlines()

    # 遍历每个视频路径
    for video_path in tqdm(video_paths):
        video_path = video_path.strip()
        print(video_path)

        # 图像目录和音频路径
        img_dir = os.path.join(img_root, video_path)
        audio_path = os.path.join(audio_root, video_path, 'audio_16k.wav')

        if not os.path.exists(img_dir) or not os.path.exists(audio_path):
            print(f"Warning: Missing image or audio for {video_path}")
            continue

        # 加载连续的图像序列
        sequences = load_image_sequences(img_dir)
        # 处理每段连续的图像序列
        for sequence in sequences:
            first_img_num = os.path.basename(sequence[0])
            last_img_num = os.path.basename(sequence[-1])
            offset, minval, conf = s.evaluate(opt, sequence, audio_path, frame_rate=25)
            if offset is None:
                continue
            out_fd.write(f'{data_name} {video_path} {first_img_num:s} {last_img_num} {offset:.5f} {minval:.5f} {conf:.5f}\n')
            out_fd.flush()

    out_fd.close()

def audio2SR16K():
    # 读取 main_list.txt
    main_list_path = '/data2/weijinghuan/head_talk/1/talking-head/pipelines/LatentWav2Lip.OnTheFly/filelists_shensi_10/main.txt'
    audio_root = '/mnt/diskwei/dataset/head_talk/preprocessed_shensi262_btm_move'
    with open(main_list_path, 'r') as f:
        video_paths = f.readlines()

    # 遍历每个视频路径
    for video_path in tqdm(video_paths):
        video_path = video_path.strip()

        audio_path = os.path.join(audio_root, video_path, 'audio.wav')
        save_audio_path = audio_path.replace('.wav', '_16k.wav')
        command = ("ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 %s" % (audio_path, save_audio_path))
        output = subprocess.call(command, shell=True, stdout=None)


process_video_list(main_list_path, opt)
# audio2SR16K()
