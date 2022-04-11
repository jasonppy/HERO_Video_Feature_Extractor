#!/bin/sh
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tf2
export CUDA_VISIBLE_DEVICES=0
python extract.py --num_decoding_thread 4 --model_version ViT-B/32 # default things: 3 sec per frame within a clip, ffmpeg sampling 1 frame per sec., if no scene segmentation, 1 frame per sec.