#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import os
from os import makedirs
import torch
import numpy as np

import subprocess
cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))

os.system('echo $CUDA_VISIBLE_DEVICES')
import open3d as o3d
from scene import Scene
import json
import time
from gaussian_renderer import render, prefilter_voxel
import torchvision
from tqdm import tqdm
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import cv2
import re
import OpenEXR, Imath
# import numpy as np

def show_depth_preview(depth_m, mask, path, invert=True, q=(0.02, 0.98)):
    import torch, numpy as np
    import matplotlib.pyplot as plt
    """把有效深度按分位数拉伸到[0,1]并预览。"""
    if not mask.any():
        print("No visible pixels.")
        return
    valid = depth_m[mask]

    # 选一个稳定的显示区间（去掉极端值）
    lo, hi = torch.quantile(valid, torch.tensor([q[0], q[1]], device=valid.device))
    dep = depth_m.clamp(min=float(lo), max=float(hi))
    # dep = depth_m
    if invert:                     # 近处亮、远处暗（更符合人眼）
        dep = hi - (dep - lo)      # 也可以用 1/(dep+eps)，但这个更线性

    # 归一化到[0,1]，无效处设为0（黑）
    dep01 = (dep - dep[mask].min()) / (dep[mask].max() - dep[mask].min() + 1e-12)
    dep01 = torch.where(mask, dep01, torch.zeros_like(dep01))

    plt.figure(figsize=(17,15))
    plt.imshow(dep01.detach().cpu().numpy(), cmap="gray")  # 或 "gray"
    plt.title("Depth preview (normalized)")
    plt.axis("off")
    plt.colorbar()
    plt.savefig(path)



def read_exr_to_tensor(path):
    # 打开 EXR 文件
    exr_file = OpenEXR.InputFile(path)
    header = exr_file.header()
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    # 获取所有通道名（常见深度通道是 'Z'，有些会放在 'R'）
    channels = list(header['channels'].keys())
    if 'Z' in channels:
        channel_name = 'Z'
    elif 'R' in channels:
        channel_name = 'R'
    else:
        channel_name = channels[0]  # 如果都不是，就取第一个

    # 读取指定通道
    pt = Imath.PixelType(Imath.PixelType.FLOAT)  # float32
    depth_str = exr_file.channel(channel_name, pt)

    # 转成 numpy array 并 reshape
    depth_np = np.frombuffer(depth_str, dtype=np.float32).reshape(height, width)

    # 转成 torch.Tensor
    depth_tensor = torch.from_numpy(depth_np).float()  # 保持 float32
    return depth_tensor

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, show_level, ape_code, max_depth=5.0, volume=None, use_depth_filter=False):

    depths_tsdf_fusion = []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        # path = view.image_path.replace("train/small_city_road_horizon/", "depth/small_city_road_horizon_depth/")
        path= re.sub(
        r"train/(block_\d+)",
        r"depth/aerial/train/\1_depth",
        view.image_path)
        # path
        path = path.replace("png", "exr")
        depth = read_exr_to_tensor(path)
        depth_tsdf = depth.clone()
        # img = o3d.io.read_image(path)      # 会得到浮点图（取决于文件）


        if use_depth_filter:
            view_dir = torch.nn.functional.normalize(view.get_rays(), p=2, dim=-1)
            depth_normal = out["depth_normal"].permute(1,2,0)
            depth_normal = torch.nn.functional.normalize(depth_normal, p=2, dim=-1)
            dot = torch.sum(view_dir*depth_normal, dim=-1).abs()
            angle = torch.acos(dot)
            mask = angle > (80.0 / 180 * 3.14159)
            depth_tsdf[mask] = 0

        depths_tsdf_fusion.append(depth_tsdf.squeeze().cpu())
        Fx = view.Fx
        Fy = view.Fy
        Cx = view.Cx
        Cy = view.Cy
        H, W = depth.shape


    if volume is not None:
        depths_tsdf_fusion = torch.stack(depths_tsdf_fusion, dim=0)
        for idx, view in enumerate(tqdm(views, desc="TSDF Fusion progress")):
            ref_depth = depths_tsdf_fusion[idx].cuda()
            # mask = ref_depth>20000
            # ref_depth[mask] = 0
            # mask = torch.ones_like(depth, dtype=torch.bool,)
            # show_depth_preview(ref_depth, mask, os.path.join(model_path, f"depth_{name}_{idx}.png"), invert=True, q=(0.02, 0.98))
            ref_depth = ref_depth.detach().cpu().numpy()
            
            pose = np.identity(4)
            pose[:3,:3] = view.R.transpose(-1,-2)
            pose[:3, 3] = view.T
            color_np = np.asarray(o3d.io.read_image(view.image_path))
            if color_np.shape[2] == 4:  # RGBA → RGB
                color_np = color_np[:, :, :3].copy()  # 关键：copy 让数据变成 contiguous
            color = o3d.geometry.Image(color_np)
            depth = o3d.geometry.Image((ref_depth/10000).astype(np.float32))
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color, depth, depth_scale=1.0, depth_trunc=max_depth, convert_rgb_to_intensity=False)
            volume.integrate(
                rgbd,
                o3d.camera.PinholeCameraIntrinsic(W, H, Fx, Fy, Cx, Cy),
                pose)
            
def clean_mesh(mesh, min_len=1000):
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters, cluster_n_triangles, cluster_area = (mesh.cluster_connected_triangles())
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < min_len
    mesh_0 = copy.deepcopy(mesh)
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    return mesh_0

def post_process_mesh(mesh, cluster_to_keep=1):
    """
    Post-process a mesh to filter out floaters and disconnected parts
    """
    import copy
    print("post processing the mesh to have {} clusterscluster_to_kep".format(cluster_to_keep))
    mesh_0 = copy.deepcopy(mesh)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            triangle_clusters, cluster_n_triangles, cluster_area = (mesh_0.cluster_connected_triangles())

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]
    n_cluster = max(n_cluster, 50) # filter meshes smaller than 50
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()
    print("num vertices raw {}".format(len(mesh.vertices)))
    print("num vertices post {}".format(len(mesh_0.vertices)))
    return mesh_0
     
def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, show_level : bool, ape_code : int, max_depth : float, voxel_size : float, num_cluster: int, use_depth_filter : bool, ply_path=None, ply_mesh=None):
    with torch.no_grad():
        gaussians = GaussianModel(
            dataset.feat_dim, dataset.n_offsets, dataset.fork, dataset.use_feat_bank, dataset.appearance_dim, 
            dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist, dataset.add_level, 
            dataset.visible_threshold, dataset.dist2level, dataset.base_layer, dataset.progressive, dataset.extend
        )
        scene = Scene(dataset, gaussians, load_iteration=None,  ply_path=ply_path, shuffle=False, resolution_scales=dataset.resolution_scales, render_mesh = True)

        if dataset.random_background:
            bg_color = [np.random.random(),np.random.random(),np.random.random()] 
        elif dataset.white_background:
            bg_color = [1.0, 1.0, 1.0]
        else:
            bg_color = [0.0, 0.0, 0.0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        if not os.path.exists(dataset.model_path):
            os.makedirs(dataset.model_path)

        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_size,
            sdf_trunc=4.0*voxel_size,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, show_level, ape_code,
                       max_depth=max_depth, volume=volume, use_depth_filter=use_depth_filter)
            print(f"extract_triangle_mesh")
            mesh = volume.extract_triangle_mesh()

            path = os.path.join(dataset.model_path, "mesh")
            os.makedirs(path, exist_ok=True)
            
            o3d.io.write_triangle_mesh(os.path.join(path, "tsdf_fusion.ply"), mesh, 
                                       write_triangle_uvs=True, write_vertex_colors=True, write_vertex_normals=True)
            
            mesh = post_process_mesh(mesh, num_cluster)
            o3d.io.write_triangle_mesh(os.path.join(path, "tsdf_fusion_post.ply"), mesh, 
                                       write_triangle_uvs=True, write_vertex_colors=True, write_vertex_normals=True)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, show_level, ape_code)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--ape", default=10, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--show_level", action="store_true")
    parser.add_argument("--max_depth", default=5.0, type=float)
    parser.add_argument("--voxel_size", default=0.004, type=float)
    parser.add_argument("--num_cluster", default=1, type=int)
    parser.add_argument("--use_depth_filter", action="store_true")
    parser.add_argument("--ply_path", type=str, default=None)
    # parser.add_argument("--ply_mesh", type=str, default=None)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.show_level, args.ape, args.max_depth, args.voxel_size, args.num_cluster, args.use_depth_filter, args.ply_path)
    
