import os
import argparse
import random

def find_subdirs(root_dir):
    """
    Find all subdirectories under root_dir that match the pattern and write their paths to target_file.
    Paths are normalized to ensure consistent behavior regardless of trailing slashes in the input.

    :param root_dir: The root directory to start the search from.
    """
    # Normalize the root directory path to ensure consistent behavior
    root_dir = os.path.normpath(root_dir)
    
    videos = []
    for root, dirs, _ in os.walk(root_dir, followlinks=True):
        # Normalize and calculate the difference in hierarchy levels
        normalized_root = os.path.normpath(root)
        if normalized_root.count(os.sep) - root_dir.count(os.sep) == 2:
            # Extract the last two parts of the path
            path_parts = normalized_root.split(os.sep)[-2:]
            simplified_path = os.sep.join(path_parts)
            videos.append(simplified_path)
        
    # shuffle videos
    random.shuffle(videos)
    
    # val file    
    with open('val.txt', 'w') as f:
        for video in videos[:25]:
            f.write(video + '\n')
    
    # val file    
    with open('test.txt', 'w') as f:
        for video in videos[25:50]:
            f.write(video + '\n')
    
    # val file    
    with open('train.txt', 'w') as f:
        for video in videos[50:]:
            f.write(video + '\n')

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Find subdirectories matching a specific pattern and write their simplified paths to a file, with an optional limit on the number of paths.')
    parser.add_argument('--root_dir', type=str, required=True, help='The root directory to start the search from.')

    args = parser.parse_args()

    find_subdirs(args.root_dir)
