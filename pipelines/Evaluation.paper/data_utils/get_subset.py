import os
import argparse

def create_symlinks(source_dir, target_dir, limit=100):
    # 检查目标目录是否存在，不存在则创建
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    # 获取所有 .mp4 文件路径
    files = []
    for root, _, filenames in os.walk(source_dir):
        for filename in filenames:
            if filename.endswith('.mp4') and not filename.startswith('.'):
                files.append(os.path.join(root, filename))
    
    # 仅取前100个 .mp4 文件
    files = files[:limit]
    
    created_links = 0

    # 创建软链接
    for file_path in files:
        file_name = os.path.basename(file_path)
        link_name = os.path.join(target_dir, file_name)
        
        # 检查软链接是否已存在
        if os.path.exists(link_name):
            print(f"Symlink already exists: {link_name}")
            continue
        
        try:
            # 创建软链接
            os.symlink(os.path.abspath(file_path), link_name)
            print(f"Created symlink: {os.path.abspath(file_path)} -> {link_name}")
            created_links += 1
        except OSError as e:
            print(f"Failed to create symlink for {file_path}: {e}")

    print(f"Total symlinks created: {created_links}")

def main():
    parser = argparse.ArgumentParser(description="Create symbolic links for the first 100 .mp4 files in a directory.")
    parser.add_argument('--input', type=str, required=True, help="The source directory containing the files.")
    parser.add_argument('--output', type=str, required=True, help="The target directory where the symlinks will be created.")
    parser.add_argument('--limit', type=int, default=100, help="The number of files to process (default is 100).")
    
    args = parser.parse_args()
    
    create_symlinks(args.input, args.output, args.limit)

if __name__ == "__main__":
    main()
