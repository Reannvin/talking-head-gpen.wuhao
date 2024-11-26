#!/bin/bash

REPO_ID="yzd-v/DWPose"
FILENAME="dw-ll_ucoco_384.pth"
SAVE_DIRECTORY="./dwpose"
TEMP_SCRIPT=$(mktemp)
export HF_ENDPOINT=https://hf-mirror.com
mkdir -p "$SAVE_DIRECTORY"
cat << EOF > "$TEMP_SCRIPT"
from huggingface_hub import snapshot_download

repo_id = "$REPO_ID"
filename = "$FILENAME"
save_directory = "$SAVE_DIRECTORY"

file_path = snapshot_download(repo_id=repo_id, allow_patterns=filename,  local_dir=save_directory)
print(f'File downloaded and saved to {file_path}')
EOF
python "$TEMP_SCRIPT"

rm "$TEMP_SCRIPT"

