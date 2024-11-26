import os
from tqdm import tqdm
from glob import glob
import shutil
import numpy as np
import argparse
import random


def shuffle(source_dir,dest_dir, images_per_folder=150,end_with='.jpg',offset=0):
    images = [img for img in glob(os.path.join(source_dir, '*'+end_with))]
    print(f'Found {len(images)} images')
    num_folders = (len(images) + images_per_folder - 1) // images_per_folder
    #images.sort(key=lambda x:int(x.split("/")[-1].split(".")[0]))
    for i in range(num_folders):
        folder_name = os.path.join(dest_dir, source_dir.split('/')[-1]+"_{:02d}".format(i))
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        start_index = i * images_per_folder-offset
        end_index = start_index + images_per_folder
       # current_images = images[start_index:end_index]
        for j in range(start_index,end_index):
            image_name=os.path.join(source_dir,str(j)+end_with)
            if not os.path.exists(image_name):
                continue
            save_name=os.path.join(folder_name,str(j-start_index)+end_with)
            #用软链接的方式copy
            os.symlink(image_name, save_name)
            #shutil.copy(image_name, save_name)
def read_offset(offset_file):
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

def shuffle_dir(data_dir,dest_dir,images_per_folder=150,end_with='.jpg'):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    folders = os.listdir(data_dir)
    offset_dict=read_offset('/data/wangbaiqin/dataset/offset/filelists_hdtf/info.txt')
    for i,folder in tqdm(enumerate(folders)):
        print(f'Processing folder: ',folder)
        folder_path = os.path.join(data_dir,folder)
        if not os.path.isdir(folder_path):
            continue
        offset=offset_dict[folder]
        if offset is None:
            print(f'offset is None for {folder}')
            continue
        shuffle(folder_path,dest_dir,images_per_folder,end_with,offset)
        # if i>2:
        #     break
def shuffle_back():
    pass

def gen_file(data_root,txt_dir,split_ratio=0.1):
    if not os.path.exists(txt_dir):
        os.makedirs(txt_dir)
    folders = os.listdir(data_root)
    last_dir=data_root.split('/')[-1]
    #随机打乱folders
    random.shuffle(folders)
    print(len(folders))
    #read from txt
    # folders=[]
    # with open(data_root,'r') as f:
    #     for line in f:
    #         folders.append(line.strip())
    #split to val and train
    #val_num = int(len(folders)*split_ratio)
    val_num=3
    val_folders = folders[:val_num]
    train_folders = folders[val_num:]
    print(f'val_num:{val_num},train_num:{len(train_folders)}')
    with open(os.path.join(txt_dir,'val.txt'),'w') as f:
        for folder in val_folders:
            f.write(last_dir+'/'+folder+'\n')
    with open(os.path.join(txt_dir,'train.txt'),'w') as f:
        for folder in train_folders:
            f.write(last_dir+'/'+folder+'\n')
    with open(os.path.join(txt_dir,'main.txt'),'w') as f:
        for folder in folders:
            f.write(last_dir+'/'+folder+'\n')
    print('Done')

def gen_file2(data_root,txt_dir,split_ratio=0.1):
    if not os.path.exists(txt_dir):
        os.makedirs(txt_dir)
    videos = os.listdir(data_root)
    folders=[]
    for video in videos:
        video_path = os.path.join(data_root,video)
        if not os.path.isdir(video_path):
            continue
        persons = os.listdir(video_path)
        for person in persons:
            person_path = os.path.join(video_path,person)
            if not os.path.isdir(person_path):
                continue
            folders.append(os.path.join(video,person))
        
    
    #split to val and train
    #val_num = int(len(folders)*split_ratio)
    val_num=128
    val_folders = folders[:val_num]
    train_folders = folders[val_num:]
    print(f'val_num:{val_num},train_num:{len(train_folders)}')
    with open(os.path.join(txt_dir,'split3300_val.txt'),'w') as f:
        for folder in val_folders:
            f.write(folder+'\n')
    with open(os.path.join(txt_dir,'split3300_train.txt'),'w') as f:
        for folder in train_folders:
            f.write(folder+'\n')
    with open(os.path.join(txt_dir,'split3300_all.txt'),'w') as f:
        for folder in folders:
            f.write(folder+'\n')
    print('Done')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/data/wangbaiqin/dataset/audios')
    parser.add_argument('--dest_dir', type=str, default='/data/wangbaiqin/dataset/audios_shuffle')
    parser.add_argument('--images_per_folder', type=int, default=150)
    parser.add_argument('--end_with', type=str, default='.jpg')

    args = parser.parse_args()

    #gen_file(args.data_root,args.dest_dir)
    #gen_file2(args.data_root,args.dest_dir)
    #shuffle_dir(args.data_root,args.dest_dir,args.images_per_folder,args.end_with)
    shuffle(args.data_root,args.dest_dir,args.images_per_folder,args.end_with)
    
    #shuffle_dir('/data/wangbaiqin/dataset/audios','/data/wangbaiqin/dataset/HDTF_whisper_shuffle',images_per_folder=150,end_with='.npy')
    #gen_file('/data/wangbaiqin/dataset/3300ID_whisper/bili3000_5','filelists/3300_bili_shuffle')
    #gen_file2('/data/wangbaiqin/dataset/3300ID_whisper','filelists/3300_shuffle')
    #shuffle('/data/wangbaiqin/dataset/audios/0','/data/wangbaiqin/dataset/audios_shuffle',images_per_folder=150,end_with='.npy')