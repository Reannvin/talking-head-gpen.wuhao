import os
import argparse
import random

def find_subdirs(root_dir):
    """
    Find all subdirectories under root_dir that match the pattern and write their paths to target_file.
    Paths are normalized to ensure consistent behavior regardless of trailing slashes in the input.

    :param root_dir: The root directory to start the search from.
    """
  
    
    videos = []
    for f in  os.listdir(root_dir):
        # Normalize and calculate the difference in hierarchy levels
        # if f.endswith('.jpg') :
            videos.append(os.path.splitext(f)[0])
        
    # shuffle videos
    random.shuffle(videos)
    
    # val file    
    with open('val.txt', 'w') as f:
        for video in videos[:5000]:
            f.write(video + '\n')
    
    # val file    
    with open('test.txt', 'w') as f:
        for video in videos[2000:5000]:
            f.write(video + '\n')
    
    # val file    
    with open('train.txt', 'w') as f:
        for video in videos[5000:]:
            f.write(video + '\n')
    with open('main.txt', 'w') as f:
        for video in videos[:]:
            f.write(video + '\n')
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Find subdirectories matching a specific pattern and write their simplified paths to a file, with an optional limit on the number of paths.')
    parser.add_argument('--root_dir', type=str, required=True, help='The root directory to start the search from.')

    args = parser.parse_args()

    find_subdirs(args.root_dir)
