#!/bin/bash
# source /home/bingxing2/apps/package/pytorch/1.13.1+cu116_cp38/env.sh
# conda activate moe-gs





source /home/bingxing2/apps/package/pytorch/1.13.1+cu116_cp38/env.sh
export PATH="/home/bingxing2/ailab/gaoyuanyuan_p/miniconda3/bin:$PATH"

source ~/.bashrc


export PATH="/home/bingxing2/ailab/gaoyuanyuan_p/miniconda3/bin:$PATH"

conda init

conda activate moe-gs



# export CUDA_VISIBLE_DEVICES=0
# pip install submodules/diff-gaussian-rasterization
python train.py --eval -s data/nyc -m outputs/nyc-a100_octree001_thred02 --resolution 2 --densify_grad_threshold 0.0002
# python render.py -m outputs/nyc-a100_octree0005_thred004
# python metrics.py -m output/london-a100_octree0002_thred04

# export NCCL_ALGO=Ring
# export NCCL_MAX_NCHANNELS=16
# export NCCL_MIN_NCHANNELS=16export NCCL_DEBUG=INFO
# export NCCL_TOPO_FILE=/home/bingxing2/apps/nccl/conf/dump.xml

# export NCCL_IB_HCA=mlx5_0,mlx5_2
# export NCCL_IB_GID_INDEX=3
# export NCCL_IB_TIMEOUT=23
# export NCCL_IB_RETRY_CNT=7