import numpy
import subprocess
import cv2
import os
import argparse
import shutil
import random
#不输出warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
current_workspace=os.path.dirname(os.path.abspath(__file__))
random_num=random.randint(0,10000)
def run_eval(file, sync_path):
    # ret_list = []
    # for file in filelist:
    os.chdir(sync_path)
    tmp_dir=f'tmp_dir{random_num}'
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    
    # x1, y1, x2, y2 = box
    ret_dict = {}
    cmd1 = f"python run_pipeline.py --videofile {file} --reference wav2lip --data_dir {tmp_dir}"    #  --x1 {x1} --x2 {x2} --y1 {y1} --y2 {y2}
    # print("cmd1:", cmd1)
    cmd2 = f"python calculate_scores_real_videos.py --videofile {file} --reference wav2lip --data_dir {tmp_dir}"
    # print("cmd2:", cmd2)

    process1 = subprocess.Popen(cmd1, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process1.communicate()
    if process1.returncode != 0:
        print(f"文件 {file} 处理失败，详情: {str(stderr.decode('utf-8'))}")
    # else:
    #     print(f"文件 {file} run_pipeline returncode: {process1.returncode}")
    
    process2 = subprocess.Popen(cmd2, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout2, stderr2 = process2.communicate()
    if process2.returncode != 0:
        print(f"文件 {file} eval失败，详情: {str(stderr2.decode('utf-8'))}")
    # else:
    #     print(f"文件 {file} calc_score returncode: {process2.returncode}")

    output = stdout2.decode('utf-8')
    # TODO 为什么返回值不一定是2
    output = output.split()
    # print(output)
    if len(output) == 2:
        dist, conf = output[0], output[1]
    else:
        dist, conf = 0, 0
    shutil.rmtree(tmp_dir)
    return float(dist), float(conf)


def run_inference(args,video_path,audio_path,sync_path='../Evaluation/syncnet_python'):
    ckpt_dir=os.path.dirname(args.ckpt)
    video_name=os.path.basename(video_path).split('.')[0]
    audio_name=os.path.basename(audio_path).split('.')[0]
    outfile=f'{ckpt_dir}/{video_name}_{audio_name}.mp4'
    script_name=args.script
    #os.chdir(current_workspace)
    if args.crop_landmark:
        command=['python',script_name,'--ckpt',args.ckpt,'--face',video_path,'--audio',audio_path,'--crop_landmark','--eval']
    elif args.arcface:
        command=['python',script_name,'--ckpt',args.ckpt,'--face',video_path,'--audio',audio_path,'--arc_face','--eval']
    else:
        command=['python',script_name,'--ckpt',args.ckpt,'--face',video_path,'--audio',audio_path,'--eval']
    if args.acc:
        command.append('--acc')
    if args.gs_blur:
        command.append('--gs_blur')
    if args.save_sample:
        command.append('--save_sample')
    if args.sf3d_up:
        command.append('--sf3d_up')
    if args.warp_mouth:
        command.append('--warp_mouth')
    if args.ref_num:
        command.append('--ref_num')
        command.append(f'{args.ref_num}')
    subprocess.run(command)
    if args.sync:
        abs_outfile=os.path.abspath(outfile)
        dist, conf =run_eval(abs_outfile,sync_path)
        os.chdir(current_workspace)
        new_outfile=f'{ckpt_dir}/{video_name}_{audio_name}_dis_{dist}_conf_{conf}.mp4'
        os.rename(outfile,new_outfile)
        print(f"outfile {new_outfile} sync distance: {dist} confidence: {conf} ")
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--video',type=str,help='video path')
    # parser.add_argument('--audio',type=str,help='audio path')
    video_list = ['data/video/example1.mp4','data/video/wbq.jpg','data/video/zhuxiangyu_black.mp4','data/video/girl.mp4']
    audio_list =['data/audio/english.wav','data/audio/newyear.wav','data/audio/example1.wav','data/audio/liuwei_30s.mp3']
    # video_list = ['data/video/sample.mp4','data/video/sample.mp4']
    # audio_list =['data/audio/english.wav','data/audio/liuwei_30s.mp3']
    parser.add_argument('--ckpt',type=str,help='checkpoint path')
    parser.add_argument('--crop_landmark',action='store_true',help='crop landmark')
    parser.add_argument('--arcface',action='store_true',help='arcface')
    parser.add_argument('--acc',action='store_true',help='arcface')
    parser.add_argument('--sync',action='store_true',help='evaluate sync')
    parser.add_argument('--sync_path',type=str,help='syncnet path',default='../Evaluation/syncnet_python')
    parser.add_argument('--gs_blur',action='store_true',help='gs_blur')
    parser.add_argument('--save_sample',action='store_true',help='save_sample')
    parser.add_argument('--ref_num',type=int,help='reference number',default=1)
    parser.add_argument('--sf3d_up',action='store_true',help='upper bound sf3d')
    parser.add_argument('--warp_mouth',action='store_true',help='warp mouth')
    parser.add_argument('--script',type=str,help='script to run inference',default='styleSync_inference.py')
    args = parser.parse_args()
    for idx,(video,audio) in enumerate(zip(video_list,audio_list)):
        print("Run inference for video {} and audio {}".format(video,audio))
        run_inference(args,video,audio)



