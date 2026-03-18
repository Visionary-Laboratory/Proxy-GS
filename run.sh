#!/bin/bash
# source /home/bingxing2/apps/package/pytorch/1.13.1+cu116_cp38/env.sh
# conda activate moe-gs





source /home/bingxing2/apps/package/pytorch/1.13.1+cu116_cp38/env.sh
export PATH="/home/bingxing2/ailab/gaoyuanyuan_p/miniconda3/bin:$PATH"

source ~/.bashrc


export PATH="/home/bingxing2/ailab/gaoyuanyuan_p/miniconda3/bin:$PATH"

conda init

conda activate proxy-gs

#  python render.py -m ouputs/Block-E-Horizon --skip_test --ply_path data/small_city/aerial/small_city_pointcloud/point_cloud_ds20/aerial/Block_A.ply 
#  python train.py --eval -s data/small_city/street/pose/block_D_1_1 -m ouputs/Block-D_1_1-Horizon-ours \
#  --ply_path data/small_city/aerial/small_city_pointcloud/point_cloud_ds20/aerial/Block_D.ply \
#  --ply_mesh ouputs/Block-D_1_1-Horizon-ours/mesh/tsdf_fusion_post.ply --use_wandb
# rm -rf submodules/diff-gaussian-rasterization/build/
# pip install submodules/diff-gaussian-rasterization

python train.py --eval -s data/small_city/street/pose/block_E -m ouputs_cvpr/block_E_moutain020 \
 --ply_path datasets/small_city/aerial/small_city_pointcloud/point_cloud_ds20/aerial/Block_E.ply  \
 --ply_mesh cvpr/block_E_moutain020.ply \
 --use_wandb

python render_real.py -m ouputs_cvpr/block_E_moutain020 --skip_train --ply_path  datasets/small_city/aerial/small_city_pointcloud/point_cloud_ds20/aerial/Block_E.ply \
 --ply_mesh cvpr/block_E_moutain020.ply -s data/small_city/street/pose/block_E




python metrics.py  -m ouputs_cvpr/block_E_moutain020 -s data/small_city/street/pose/block_E
# python render_real.py -m ouputs/hier_smallcity_ours_base_1_04_wo_densify_tao01 --skip_train --ply_path  data/hirarchy/small_city/camera_calibration/rectified/sparse/0/points3D.ply \
#  --ply_mesh data/hirarchy/small_city/camera_calibration/rectified/tsdf_fusion_post_smallcity.ply -s data/hirarchy/small_city/camera_calibration/rectified




# python metrics.py  -m ouputs/hier_smallcity_ours_base_1_04_wo_densify_tao01 -s data/hirarchy/small_city/camera_calibration/rectified/masks
#  python train.py --eval -s data/berlin -m ouputs/cuhk_ours_1_0005_porxy_3_6_wodensfy --resolution 4 \
#  --base_layer -1 --ply_path data/cuhk/hku_1.ply  --use_wandb --ply_mesh data/cuhk/hku_1.ply
# CUDA_VISIBLE_DEVICES=1 python train.py --eval -s data/small_city/street/pose/block_D_1_1 -m ouputs/Block-D_1_1-Horizon-ours \
#     --ply_path data/small_city/aerial/small_city_pointcloud/point_cloud_ds20/aerial/Block_D.ply \
#     --ply_mesh ouputs/Block-D_1-test/mesh/tsdf_fusion_post.ply --use_wandb & 

# CUDA_VISIBLE_DEVICES=2 python train.py --eval -s data/small_city/street/pose/block_D_2 -m ouputs/Block-D_2-Horizon-ours \
#     --ply_path data/small_city/aerial/small_city_pointcloud/point_cloud_ds20/aerial/Block_D.ply \
#     --ply_mesh ouputs/Block-D_2-test/mesh/tsdf_fusion_post.ply --use_wandb & 
# python render_real.py -m ouputs/hier_smallcity_ours_base_1_02 --ply_path data/hirarchy/small_city/camera_calibration/rectified/sparse/0/points3D.ply --ply_mesh data/hirarchy/small_city/camera_calibration/rectified/tsdf_fusion_post_smallcity.ply --skip_train


# python metrics.py  -m ouputs/hier_smallcity_ours_base_1_02 -s data/hirarchy/small_city/camera_calibration/rectified/masks

# CUDA_VISIBLE_DEVICES=3 python train.py --eval -s data/small_city/street/pose/block_E -m ouputs/Block-D_E-Horizon-ours-smesh \
#     --ply_path data/small_city/aerial/small_city_pointcloud/point_cloud_ds20/aerial/Block_E.ply \
#     --ply_mesh block_E_from_mesh.ply --use_wandb 




# export CUDA_VISIBLE_DEVICES=0
# pip install submodules/diff-gaussian-rasterization
# python train.py --eval -s data/alameda -m outputs/alameda-a100_octree0005_thred004  --resolution 2  --densify_grad_threshold 0.00004
# python render.py -m output/nyc-a100_octree0001_thred08
# python metrics.py -m output/nyc-a100_octree0001_thred08


# python render.py -m outputs/nyc-a100_octree0005_thred008

# export NCCL_ALGO=Ring
# export NCCL_MAX_NCHANNELS=16
# export NCCL_MIN_NCHANNELS=16export NCCL_DEBUG=INFO
# export NCCL_TOPO_FILE=/home/bingxing2/apps/nccl/conf/dump.xml

# export NCCL_IB_HCA=mlx5_0,mlx5_2
# export NCCL_IB_GID_INDEX=3
# export NCCL_IB_TIMEOUT=23
# export NCCL_IB_RETRY_CNT=7