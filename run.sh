#!/bin/bash
# source /home/bingxing2/apps/package/pytorch/1.13.1+cu116_cp38/env.sh
# conda activate moe-gs





# source /home/bingxing2/apps/package/pytorch/1.13.1+cu116_cp38/env.sh

module load compilers/cuda/12.1 cudnn/8.8.1.3_cuda12.x compilers/gcc/9.3.0

export PATH="/home/bingxing2/ailab/gaoyuanyuan_p/miniconda3/bin:$PATH"

source ~/.bashrc


export PATH="/home/bingxing2/ailab/gaoyuanyuan_p/miniconda3/bin:$PATH"

conda init

conda activate py310_torch210_cuda118
python train.py --eval -s data/small_city/street/pose/block_E -m ouputs/Block-E-Octree --ply_path data/small_city/aerial/small_city_pointcloud/point_cloud_ds20/aerial/Block_E.ply  --use_wandb

DATA_DIR="gs_datasets"
OUTPUT_DIR="output"
RES=2

# 获取所有子文件夹
dirs=($(ls -d ${DATA_DIR}/*/))

# GPU 数量
NUM_GPUS=16

for i in "${!dirs[@]}"; do
    dataset_name=$(basename ${dirs[$i]})   # 获取文件夹名（不带路径）
    gpu_id=$((i % NUM_GPUS))              # 轮流分配 GPU

    echo ">>> Processing ${dataset_name} on GPU ${gpu_id}"

    # 后台运行（& 表示后台执行）
    CUDA_VISIBLE_DEVICES=${gpu_id} \
    python train.py --eval -s ${DATA_DIR}/${dataset_name} -m ${OUTPUT_DIR}/${dataset_name}-a100_octree_002 --resolution ${RES} \
    && CUDA_VISIBLE_DEVICES=${gpu_id} \
    python render.py -m ${OUTPUT_DIR}/${dataset_name}-a100_octree_002 \
    && CUDA_VISIBLE_DEVICES=${gpu_id} \
    python metrics.py -m ${OUTPUT_DIR}/${dataset_name}-a100_octree_002 &

    # 控制并发数
    if (( (i+1) % NUM_GPUS == 0 )); then
        wait   # 等待这批任务完成，再开新一批
    fi
done

wait  # 等待所有任务结束
echo ">>> All datasets processed!"


# export NCCL_ALGO=Ring
# export NCCL_MAX_NCHANNELS=16
# export NCCL_MIN_NCHANNELS=16export NCCL_DEBUG=INFO
# export NCCL_TOPO_FILE=/home/bingxing2/apps/nccl/conf/dump.xml

# export NCCL_IB_HCA=mlx5_0,mlx5_2
# export NCCL_IB_GID_INDEX=3
# export NCCL_IB_TIMEOUT=23
# export NCCL_IB_RETRY_CNT=7