import os
import shutil
import argparse
from multiprocessing import Pool

def copy_file_with_duration(file_data):
    file_path, dest_dir, duration, duration_thres = file_data
    file_name = os.path.basename(file_path)
    dest_path = os.path.join(dest_dir, file_name)
    if duration >= duration_thres and not os.path.exists(dest_path):
        shutil.copy(file_path, dest_path)
        print(f"Copied {file_path} to {dest_path}")
    else:
        if os.path.exists(dest_path):
            print(f"file {dest_path} exist, skipping")
        else:
            print(f"Duration: {file_path} (expected gt {duration_thres}, found {duration})")

def process_file(input_file, dest_dir, duration_thres, num_workers):
    with open(input_file, 'r') as file:
        lines = file.readlines()
    
    file_data_list = []
    for line in lines:   
        parts = line.strip().split()
        print(parts)
        file_path = parts[0]
        print(parts[-1])
        file_duration = float(parts[-1])
        print(file_path,file_duration)
        file_data_list.append((file_path, dest_dir, file_duration, duration_thres))

    with Pool(num_workers) as pool:
        pool.map(copy_file_with_duration, file_data_list)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Copy video files if the duration matches.")
    parser.add_argument("--input_filelist",default='/data/fanshen/workspace/preprocessing/avspeech_info.txt', help="Path to the input text file")
    parser.add_argument("--dest_dir", default='/data/fanshen/avspeech', help="Destination directory to copy the files")
    parser.add_argument("--duration_thres", type=float,default=10.0, help="Duration to match against each file's duration, change to 0 to copy all video")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of worker processes to use for copying files")
    args = parser.parse_args()
    if not os.path.exists(args.dest_dir):
        os.makedirs(args.dest_dir)

    process_file(args.input_filelist, args.dest_dir, args.duration_thres,  args.num_workers)
