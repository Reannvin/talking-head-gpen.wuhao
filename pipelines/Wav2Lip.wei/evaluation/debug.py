import os
author_file_path = 'test_filelists/lrs2.txt'
org_data_test_path = '../filelists/test.txt'
org_data_val_path = '../filelists/val.txt'

with open(org_data_test_path) as f:
    test_lines = f.read().strip().splitlines()

test_lines = [line.split()[0] for line in test_lines]
print('test line:', test_lines[0])

with open(org_data_val_path) as f:
    val_lines = f.read().strip().splitlines()

print('val line:', val_lines[0])

with open(author_file_path) as f:
    author_lines = f.read().strip().splitlines()

author_lines = [line.split()[1]  for line in author_lines]
print('author:', author_lines[0])


print(f'test ls len:{len(test_lines)} set len:{len(set(test_lines))}')
print(f'val ls len:{len(val_lines)} set len:{len(set(val_lines))}')
print(f'authort ls len:{len(author_lines)} set len:{len(set(author_lines))}')

print(f'author inter test:', len(set(test_lines).intersection(set(author_lines))))
print(f'author inter val:', len(set(val_lines).intersection(set(author_lines))))
