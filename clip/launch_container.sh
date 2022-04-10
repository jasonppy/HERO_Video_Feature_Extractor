# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

mountpath=$1

if [ -z $CUDA_VISIBLE_DEVICES ]; then
    CUDA_VISIBLE_DEVICES='all'
fi

docker run --gpus '"'device=$CUDA_VISIBLE_DEVICES'"' --ipc=host --network=host --rm -it \
    --mount src=$(pwd),dst=/src,type=bind \
    --mount src=$mountpath,dst=/vqhighlight,type=bind \
    -e NVIDIA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    -w /src linjieli222/hero-video-feature-extractor:clip \
    bash -c "bash" \
