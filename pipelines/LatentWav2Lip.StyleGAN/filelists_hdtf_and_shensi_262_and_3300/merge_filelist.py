import random
import argparse

def merge_and_shuffle_files(file1, file2, output_file):
    # 读取两个文件的内容
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()

    # 去掉每一行末尾的换行符
    lines1 = [line.strip() for line in lines1]
    lines2 = [line.strip() for line in lines2]

    # 计算文件的行数
    len1 = len(lines1)
    len2 = len(lines2)
    max_len = max(len1, len2)

    # 确保最终文件一半内容来自file1，一半内容来自file2
    half1 = lines1.copy()
    half2 = lines2.copy()

    while len(half1) < max_len:
        half1.append(random.choice(lines1))
    while len(half2) < max_len:
        half2.append(random.choice(lines2))

    # 合并两个文件的内容
    merged_lines = []
    for line1, line2 in zip(half1, half2):
        merged_lines.append(line1)
        merged_lines.append(line2)

    # 随机打乱合并后的内容
    random.shuffle(merged_lines)

    # 将结果写入输出文件
    with open(output_file, 'w') as out_file:
        for line in merged_lines:
            out_file.write(line + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merge two files and shuffle the lines randomly.')
    parser.add_argument('file1', type=str, help='First input file')
    parser.add_argument('file2', type=str, help='Second input file')
    parser.add_argument('output_file', type=str, help='Output file')

    args = parser.parse_args()
    merge_and_shuffle_files(args.file1, args.file2, args.output_file)
