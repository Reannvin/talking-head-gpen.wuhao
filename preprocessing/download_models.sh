#!/bin/bash
mkdir models
# wget https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt -O models/tiny.pt
# wget http://www.robots.ox.ac.uk/~vgg/software/lipsync/data/syncnet_v2.model -O ./models/syncnet_v2.model
# wget https://www.robots.ox.ac.uk/~vgg/software/lipsync/data/sfd_face.pth -O ./models/sfd_face.pth
# wget https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth -O ./models/s3fd-619a316812.pth
# wget https://huggingface.co/yzd-v/DWPose/blob/main/dw-ll_ucoco_384.pth  -O ./models/dwpose/dw-ll_ucoco_384.pth

REPO_ID="yzd-v/DWPose"
FILENAME="dw-ll_ucoco_384.pth"
SAVE_DIRECTORY="./models/dwpose"
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

