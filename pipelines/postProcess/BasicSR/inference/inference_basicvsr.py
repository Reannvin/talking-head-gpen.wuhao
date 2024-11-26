import argparse
import cv2
import glob
import os
import shutil
import torch

from basicsr.archs.basicvsr_arch import BasicVSR
from basicsr.data.data_util import read_img_seq
from basicsr.utils.img_util import tensor2img


def inference(imgs, imgnames, model, save_path):
    with torch.no_grad():
        outputs = model(imgs)
    # save imgs
    outputs = outputs.squeeze()
    outputs = list(outputs)
    for output, imgname in zip(outputs, imgnames):
        output = tensor2img(output)
        cv2.imwrite(os.path.join(save_path, f'{imgname}_BasicVSR.png'), output)


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model_path', type=str, default='experiments/pretrained_models/BasicVSR_REDS4.pth')
    # parser.add_argument('--model_path', type=str, default='experiments/pretrained_models/Blind-Setting-X4/BasicVSR/net_g_300000.pth')
    # parser.add_argument('--model_path', type=str, default='experiments/6004_BasicVSRGAN_6000300kft_realpaper_degradationorder2_vfhq_b2n7/models/net_g_5000.pth')
    parser.add_argument('--model_path', type=str, default='experiments/BasicVS_vfhq_finetuneV2/models/net_g_25000.pth')
    parser.add_argument(
        '--input_path', type=str, default='datasets/test/Test_LR/Clip+Y-Mi5-dACwA+P0+C2+F26876-27075', help='input test image folder')
        # '--input_path', type=str, default='/data2/weijinghuan/superRestoration/GFPGAN/results/Clip+2QWD9XCVyDg+P0+C0+F5808-5970_128', help='input test image folder')
        # '--input_path', type=str, default='/data2/weijinghuan/superRestoration/GFPGAN/results/test_liuwei1_128', help='input test image folder')
        # '--input_path', type=str, default='datasets/REDS4/sharp_bicubic/000', help='input test image folder')
    parser.add_argument('--save_path', type=str, default='results/BasicVSR_fineV2_I7', help='save image path')
    # parser.add_argument('--interval', type=int, default=15, help='interval size')
    parser.add_argument('--interval', type=int, default=7, help='interval size')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set up model
    model = BasicVSR(num_feat=64, num_block=30)
    print(args.model_path)
    msg = model.load_state_dict(torch.load(args.model_path)['params'], strict=True)
    print(msg)
    model.eval()
    model = model.to(device)

    os.makedirs(args.save_path, exist_ok=True)

    # extract images from video format files
    input_path = args.input_path
    use_ffmpeg = False
    if not os.path.isdir(input_path):
        use_ffmpeg = True
        video_name = os.path.splitext(os.path.split(args.input_path)[-1])[0]
        input_path = os.path.join('./BasicVSR_tmp', video_name)
        os.makedirs(os.path.join('./BasicVSR_tmp', video_name), exist_ok=True)
        os.system(f'ffmpeg -i {args.input_path} -qscale:v 1 -qmin 1 -qmax 1 -vsync 0  {input_path} /frame%08d.png')

    # load data and inference
    imgs_list = sorted(glob.glob(os.path.join(input_path, '*')))
    num_imgs = len(imgs_list)
    if len(imgs_list) <= args.interval:  # too many images may cause CUDA out of memory
        imgs, imgnames = read_img_seq(imgs_list, return_imgname=True)
        imgs = imgs.unsqueeze(0).to(device)
        inference(imgs, imgnames, model, args.save_path)
    else:
        for idx in range(0, num_imgs, args.interval):
            interval = min(args.interval, num_imgs - idx)
            imgs, imgnames = read_img_seq(imgs_list[idx:idx + interval], return_imgname=True)
            imgs = imgs.unsqueeze(0).to(device)
            print(imgs.shape)
            inference(imgs, imgnames, model, args.save_path)

    # delete ffmpeg output images
    if use_ffmpeg:
        shutil.rmtree(input_path)


if __name__ == '__main__':
    main()
