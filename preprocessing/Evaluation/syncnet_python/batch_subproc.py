import os
import shutil
import subprocess

def run_eval(file, i):
    # ret_list = []
    # for file in filelist:
    if os.path.exists("tmp_dir"):
        shutil.rmtree("tmp_dir")
    
    # x1, y1, x2, y2 = box
    ret_dict = {}
    cmd1 = f"python ./Evaluation/syncnet_python/run_pipeline.py --videofile {file} --reference wav2lip_{i} --data_dir tmp_dir{i}"    #  --x1 {x1} --x2 {x2} --y1 {y1} --y2 {y2}
    # print("cmd1:", cmd1)
    cmd2 = f"python ./Evaluation/syncnet_python/calculate_scores_real_videos.py --videofile {file} --reference wav2lip_{i} --data_dir tmp_dir{i}"
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
    return float(dist), float(conf)


# if __name__ == '__main__':
#     filelist = []
#     base_path = "/home/renxiaotian/workspace/data_process/testdata/input/test_video"
#     for file in os.listdir(base_path):
#         filelist.append(os.path.join(base_path, file))
    
#     # filelist.append(("/home/renxiaotian/workspace/data_process/testdata/input/test_video/UlO6WovFvz4_278.044433-290.123167.mp4", [236, 245, 563, 547]))
#     output = run_eval(filelist=filelist[:2])
#     print(output)