# from Evaluation.syncnet_python.batch_subproc import *
# from tqdm import tqdm
# import argparse
# import json
# def eval(filelist, lse_d, lse_c):
#     person_ids = []
#     video_files = []
#     result = []
#     with open(filelist, 'r') as f:
#         lines = f.readlines()
#         for line in lines:
#             video_files.append(line.split(' ')[0])
#             person_ids.append(line.split(' ')[-1])
#     print("********* evaluation ************")
#     for video_file,person_id in tqdm(zip(video_files,person_ids)):
#         # print(f"Processing... {video_file}")
#         dists, conf = run_eval(video_file)
#         person_id =  person_id.strip()  
#         # print(person_id)
#         if dists < lse_d and conf > lse_c:  
#             result.append(f"{video_file} {person_id} {conf} {dists}")
#         else:
#             continue
#     return result
# if __name__ == '__main__':       
#     parser = argparse.ArgumentParser(description='data preprocessing')
#     parser.add_argument("--lse_d", default=8,help="index:distance", type=float)
#     parser.add_argument("--lse_c", default=6,help="index: confidence", type=float)
#     parser.add_argument("--filelist", default = 'filelist.txt',help="file list gennerate by sence detect", type=str)
#     parser.add_argument("--output", default = "file_list_eval.txt",help="output JSON file", type=str)
#     args = parser.parse_args()
#     result = eval(args.filelist, args.lse_d, args.lse_c)
#     with open(args.output, 'w') as f:
#         for line in result:
#             f.write(f"{line}\n")
    
from Evaluation.syncnet_python.batch_subproc import run_eval
from tqdm import tqdm
import argparse
import json
import multiprocessing
import logging
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(filename='eval.log', level=logging.INFO, format=LOG_FORMAT)

def eval_video(args):
    video_file, person_id, lse_d, lse_c, pid = args
    dists, conf = run_eval(video_file, pid)
    result = None
    if dists < lse_d and conf > lse_c:
        result = f"qualified:{video_file} {person_id.strip()} {conf} {dists}"
    else:
        result = f"not_qualified:{video_file} {person_id.strip()} {conf} {dists}"
    logging.info(result)
    return result

def eval(filelist, lse_d, lse_c):
    person_ids = []
    video_files = []
    result = []
    with open(filelist, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split(' ')
            video_files.append(parts[0])
            person_ids.append(parts[-1])

    logging.info("********* evaluation ************")

    num_workers = 4  # Number of worker processes

    # Preparing arguments for parallel processing
    args_list = [(video_files[i], person_ids[i], lse_d, lse_c, f'{i % num_workers}') for i in range(len(video_files))]

    # Using multiprocessing to parallelize the evaluation with specified workers
    with multiprocessing.Pool(processes=num_workers) as pool:
        for output in tqdm(pool.imap(eval_video, args_list), total=len(args_list)):
            if output:
                result.append(output)
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='data preprocessing')
    parser.add_argument("--lse_d", default=7.5, help="index: distance", type=float)
    parser.add_argument("--lse_c", default=6.5, help="index: confidence", type=float)
    parser.add_argument("--filelist", default='filelist.txt', help="file list generated by scene detect", type=str)
    parser.add_argument("--output", default="file_list_eval.txt", help="output JSON file", type=str)
    args = parser.parse_args()
    
    result = eval(args.filelist, args.lse_d, args.lse_c)
    logging.info(f"write to file {args.output}")
    with open(args.output, 'w') as f:
        for line in result:
            f.write(f"{line}\n")
