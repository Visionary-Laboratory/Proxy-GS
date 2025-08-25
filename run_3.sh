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
python train.py --eval -s data/london -m outputs/london-a100_octree0005_thred004 --resolution 2 --densify_grad_threshold 0.00004
# python render.py -m output/london-a100_octree0002_thred04
# python metrics.py -m output/london-a100_octree0002_thred04

