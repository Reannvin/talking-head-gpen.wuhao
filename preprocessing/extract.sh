#!/bin/bash

SOURCE_DIR="/path/to/source_directory"
DEST_DIR="/path/to/new_directory"

mkdir -p "$DEST_DIR"
find "$SOURCE_DIR" -type f \( -name '*.zip' -o -name '*.tar' -o -name '*.tar.gz' \) -exec sh -c '
  for file; do
    dest_subdir="$DEST_DIR/$(basename "${file%.*}")"
    mkdir -p "$dest_subdir"
    
    case "$file" in
      *.zip)
        unzip -o "$file" -d "$dest_subdir" || mv "$file" "$DEST_DIR/unzip_err"
        ;;
      *.tar)
        tar -xf "$file" -C "$dest_subdir" || mv "$file" "$DEST_DIR/unzip_err"
        ;;
      *.tar.gz)
        tar -zxf "$file" -C "$dest_subdir" || mv "$file" "$DEST_DIR/unzip_err"
        ;;
      *)
        echo "未知文件类型: $file"
        ;;
    esac
  done
' sh {} +

echo "FINISHED"
