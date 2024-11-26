#!/bin/bash
# rm all_scores.txt

# $eachfile = "$1"
if [ -f "$1" ]; then
   echo "Processing file: $1" >> all_scores.txt 
   python run_pipeline.py --videofile "$1" --reference wav2lip --data_dir tmp_dir
   python calculate_scores_real_videos.py --videofile "$1" --reference wav2lip --data_dir tmp_dir >> all_scores.txt
fi

