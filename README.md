# Dockerized Video Feature Extraction for HERO

This repo aims at providing feature extraction code for video data in [HERO Paper](https://arxiv.org/abs/2005.00200) (EMNLP 2020).
For official pre-training and finetuning code on various of datasets, please refer to [HERO Github Repo](https://github.com/linjieli222/HERO).

Some code in this repo are copied/modified from opensource implementations made available by
[PyTorch](https://github.com/pytorch/pytorch),
[Dataflow](https://github.com/tensorpack/dataflow),
[SlowFast](https://github.com/facebookresearch/SlowFast),
[HowTo100M Feature Extractor](https://github.com/antoine77340/video_feature_extractor),
[S3D_HowTo100M](https://github.com/antoine77340/S3D_HowTo100M)
and [CLIP](https://github.com/openai/CLIP).

## Update
We added support on two other models: [S3D_HowTo100M](https://github.com/antoine77340/S3D_HowTo100M)
and [CLIP](https://github.com/openai/CLIP), which are used in VALUE baselines ([[paper]](), [[website]](https://value-leaderboard.github.io/)).

## Requirements

We provide Docker image for easier reproduction. Please install the following:
  - [nvidia driver](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#package-manager-installation) (418+), 
  - [Docker](https://docs.docker.com/install/linux/docker-ce/ubuntu/) (19.03+), 
  - [nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-docker#quickstart).

Our scripts require the user to have the [docker group membership](https://docs.docker.com/install/linux/linux-postinstall/)
so that docker commands can be run without sudo.
We only support Linux with NVIDIA GPUs. We test on Ubuntu 18.04 and V100 cards.

## Quick Start

### Launch Docker Container

```bash
# docker image should be automatically pulled
CUDA_VISIBLE_DEVICES=0 source launch_container.sh $PATH_TO_STORAGE/raw_video_dir $PATH_TO_STORAGE/feature_output_dir
```
The launch script respects $CUDA_VISIBLE_DEVICES environment variable.
We suggest to launch seperate containers to launch parallel feature extraction processes,
as the feature extraction script is intended to be run on ONE single GPU only.

Note that the source code is mounted into the container under `/src` instead 
of built into the image so that user modification will be reflected without
re-building the image. (Data folders are mounted into the container separately
for flexibility on folder structures.
Specifically, `$PATH_TO_STORAGE/raw_video_dir` is mounted to `/video` and `$PATH_TO_STORAGE/feature_output_dir` is mounted to `/output`.)

### SlowFast Feature Extraction

```bash
cd /src/slowfast
```
1. Generate a csv file with input and output files
```bash
python extract_feature/gather_video_paths.py
```
By defult, all video files under ``/video`` directory will be collected,
and the output folder is set to be ``/output/slowfast_features``.
The csv file is written to ``/output/csv/slowfast_info.csv`` with the following format:
```bash
video_path,feature_path
/video/video1.mp4, /output/slowfast_features/video1.npz
/video/video2.webm, /output/slowfast_features/video2.npz
...
```

2. Extract features
```bash
python extract_feature/extract.py --dataflow --csv /output/csv/slowfast_info.csv \
    --batch_size 45 --num_decoding_thread 4 --clip_len 2\
    TEST.CHECKPOINT_FILE_PATH /models/SLOWFAST_8x8_R50.pkl
```
This command will extract 3D SlowFast video features for videos listed in `/output/csv/slowfast_info.csv`
and save them as npz files to `/output/slowfast_features`.
* `--num_decoding_thread`: how many parallel cpu thread are used for the decoding of the videos
* `--clip_len (in seconds)`: 1 feature per `clip_len` seconds, we set it to `3/2` or `2` in our paper
* `--dataflow`: enables faster parallel video decoding and pre-processing with [Dataflow](https://github.com/tensorpack/dataflow), remove to disable and use PyTorch dataloader instead


We use the pre-trained SlowFast model on Kinetics: [SLOWFAST_8X8_R50.pkl](https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/kinetics400/SLOWFAST_8x8_R50.pkl).
The checkpoint is already downloaded under `/models` directory in our provided docker image.
If you wish to use other SlowFast models, you can download them from [SlowFast Model Zoo](https://github.com/facebookresearch/SlowFast/blob/master/MODEL_ZOO.md). Note that you will need to set the corresponding config file through `--cfg`.

### 2D ResNet Feature Extraction

```bash
cd /src/resnets
```
1. Generate a csv file with input and output files
```bash
python gather_video_paths.py
```
By defult, all video files under ``/video`` directory will be collected,
and the output folder is set to be ``/output/resnet_features``.
The csv file is written to ``/output/csv/resnet_info.csv`` with the following format:
```bash
video_path,feature_path
/video/video1.mp4, /output/resnet_features/video1.npz
/video/video2.webm, /output/resnet_features/video2.npz
...
```

2. Extract features
```bash
python extract.py --csv /output/csv/resnet_info.csv --num_decoding_thread 4 --clip_len 2
```
This command will extract 2D ResNet features for videos listed in `/output/csv/resnet_info.csv`
and save them as npz files to `/output/resnet_features`.
* `--num_decoding_thread`: how many parallel cpu thread are used for the decoding of the videos
* `--clip_len (in seconds)`: 1 feature per `clip_len` seconds, we set it to `3/2` or `2` in our paper.

The model used to extract 2D features is the pytorch model zoo ResNet-152 pretrained on ImageNet, which will be downloaded on the fly.

This script is copied and modified from [HowTo100M Feature Extractor](https://github.com/antoine77340/video_feature_extractor).
It also supports feature extraction from a pre-trained 3D ResNext-101 model, which is not fully tested in our current release.
Plese follow the original repo if you would like to use their 3D feature extraction pipeline.


### MIL-NCE pre-trained S3D features 

```bash
cd /src/mil-nce
```
1. Generate a csv file with input and output files
```bash
python gather_video_paths.py
```
By defult, all video files under ``/video`` directory will be collected,
and the output folder is set to be ``/output/mil-nce_features``.
The csv file is written to ``/output/csv/mil-nce_info.csv`` with the following format:
```bash
video_path,feature_path
/video/video1.mp4, /output/resnet_features/video1.npz
/video/video2.webm, /output/resnet_features/video2.npz
...
```

2. Extract features
```bash
python extract.py --dataflow --csv /output/csv/mil-nce_info.csv  --batch_size 45 --num_decoding_thread 4 
```
This command will extract S3D features for videos listed in `/output/csv/mil-nce_info.csv`
and save them as npz files to `/output/mil-nce_features`.
* `--num_decoding_thread`: how many parallel cpu thread are used for the decoding of the videos
* `--clip_len (in seconds)`: 1 feature per `clip_len` seconds, we set it to `3/2` for VALUE baselines.

The model used to extract S3D features is pre-trained on HowTo100M videos, refer to [the original paper](https://arxiv.org/abs/1912.06430) for more details.
The checkpoint is already downloaded under `/models` directory in our provided docker image.

This script is copied and modified from [S3D_HowTo100M](https://github.com/antoine77340/S3D_HowTo100M).


## Image-text pre-trained CLIP features 
Note that the docker image is different from the one used for the above three features. 
```bash
cd ./clip

# docker image should be automatically pulled
CUDA_VISIBLE_DEVICES=0 source launch_container.sh "/saltpool0/data/pyp/vqhighlight/"
```

1. Generate a csv file with input and output files
```bash
python gather_video_paths.py
```
By defult, all video files under ``/video`` directory will be collected,
and the default output folder is set to be ``/output/clip-vit_features``.
The csv file is written to ``/output/csv/clip-vit_info.csv`` with the following format:
```bash
video_path,feature_path
/video/video1.mp4, /output/clip-vit_features/video1.npz
/video/video2.webm, /output/clip-vit_features/video2.npz
...
```

2. Extract features
```bash
python extract.py --csv /output/csv/clip-vit_info.csv --num_decoding_thread 4 --model_version ViT-B/32 
```
This command will extract CLIP features for videos listed in `/output/csv/clip-vit_info.csv`
and save them as npz files to `/output/clip-vit_features`.
* `--num_decoding_thread`: how many parallel cpu thread are used for the decoding of the videos
* `--clip_len (in seconds)`: 1 feature per `clip_len` seconds, we set it to `3/2` for VALUE baselines.
* `--model_version`: You can switch between `RN50x4` and `ViT-B/32 `, other models are not tested.

The model used to extract CLIP features is pre-trained on large-scale image-text pairs, refer to [the original paper](https://arxiv.org/abs/2103.00020) for more details. The checkpoint will be downloaded on the fly.


## Citation

If you find this code useful for your research, please consider citing:
```
@inproceedings{li2020hero,
  title={HERO: Hierarchical Encoder for Video+ Language Omni-representation Pre-training},
  author={Li, Linjie and Chen, Yen-Chun and Cheng, Yu and Gan, Zhe and Yu, Licheng and Liu, Jingjing},
  booktitle={EMNLP},
  year={2020}
}

```

## License

MIT
