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
import numpy as np

import subprocess
cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))

os.system('echo $CUDA_VISIBLE_DEVICES')


import torch
import torchvision
import json
import wandb
import time
from os import makedirs
import shutil
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as tf
import lpips
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import prefilter_voxel, render, network_gui,  mesh_depth_render
import sys
from scene import Scene, GaussianModel
from Mesh2DepthHelper import DepthRenderer, Load_ply_resource
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import torch.nn.functional as F
# torch.set_num_threads(32)
lpips_fn = lpips.LPIPS(net='vgg').to('cuda')

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
    print("found tf board")
except ImportError:
    TENSORBOARD_FOUND = False
    print("not found tf board")

def saveRuntimeCode(dst: str) -> None:
    additionalIgnorePatterns = ['.git', '.gitignore']
    ignorePatterns = set()
    ROOT = '.'
    with open(os.path.join(ROOT, '.gitignore')) as gitIgnoreFile:
        for line in gitIgnoreFile:
            if not line.startswith('#'):
                if line.endswith('\n'):
                    line = line[:-1]
                if line.endswith('/'):
                    line = line[:-1]
                ignorePatterns.add(line)
    ignorePatterns = list(ignorePatterns)
    for additionalPattern in additionalIgnorePatterns:
        ignorePatterns.append(additionalPattern)

    log_dir = Path(__file__).resolve().parent

    shutil.copytree(log_dir, dst, ignore=shutil.ignore_patterns(*ignorePatterns))
    
    print('Backup Finished!')


import math

def depth_tolerance(iteration, total_iters, max_tol=1.0, min_tol=0.0, mode='cosine'):
    """
    根据迭代数动态调整 depth 容忍度
    iteration: 当前迭代数
    total_iters: 总迭代数
    max_tol: 最大容忍度
    min_tol: 最小容忍度
    mode: 'linear' 或 'cosine'
    """
    progress = min(1.0, iteration / total_iters)

    if mode == 'linear':
        tol = min_tol + (max_tol - min_tol) * progress
    elif mode == 'cosine':
        # 先快后慢，符合 curriculum learning
        tol = min_tol + (max_tol - min_tol) * (1 - math.cos(math.pi * progress)) / 2
    else:
        raise ValueError("mode must be 'linear' or 'cosine'")

    return tol


def mesh_render(dataset, opt, pipe, dataset_name, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, wandb=None, logger=None, ply_path=None, mesh_path=None, depth_npy_dir=None):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(
        dataset.feat_dim, dataset.n_offsets, dataset.fork, dataset.use_feat_bank, dataset.appearance_dim, 
        dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist, dataset.add_level, 
        dataset.visible_threshold, dataset.dist2level, dataset.base_layer, dataset.progressive, dataset.extend
    )
    scene = Scene(dataset, gaussians, ply_path=ply_path, shuffle=False, logger=logger, resolution_scales=dataset.resolution_scales, mesh_path= mesh_path)

    # 初始化 DepthRenderer，避免每次迭代都重新创建
    device = "cuda"
    depth_renderer = DepthRenderer(device=device)

    viewpoint_stack = None

    viewpoint_stack = scene.getTrainCameras().copy()
    mesh = Load_ply_resource(mesh_path,'cuda')
    if depth_npy_dir is None:
        depth_npy_dir = os.path.join(dataset.model_path, "mesh_depth_npy")
    os.makedirs(depth_npy_dir, exist_ok=True)

    for viewpoint_cam in viewpoint_stack:        
        # network gui not available in octree-gs yet        
        # Pick a random Camera
        

        depth_m =  mesh_depth_render(viewpoint_cam, renderer = depth_renderer, mesh=mesh)
        depth_np = depth_m.detach().cpu().numpy() if torch.is_tensor(depth_m) else np.asarray(depth_m)
        depth_save_path = os.path.join(depth_npy_dir, f"{viewpoint_cam.image_name}.npy")
        np.save(depth_save_path, depth_np)



        


def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer




def get_logger(path):
    import logging

    logger = logging.getLogger()
    logger.setLevel(logging.INFO) 
    fileinfo = logging.FileHandler(os.path.join(path, "outputs.log"))
    fileinfo.setLevel(logging.INFO) 
    controlshow = logging.StreamHandler()
    controlshow.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fileinfo.setFormatter(formatter)
    controlshow.setFormatter(formatter)

    logger.addHandler(fileinfo)
    logger.addHandler(controlshow)

    return logger

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--warmup', action='store_true', default=False)
    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[-1])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000, 50_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--gpu", type=str, default = '-1')
    parser.add_argument("--ply_path", type=str, default=None)
    parser.add_argument("--ply_mesh", type=str, default=None)
    parser.add_argument("--depth_npy_dir", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])

    # enable logging
    model_path = args.model_path
    os.makedirs(model_path, exist_ok=True)

    logger = get_logger(model_path)

    logger.info(f'args: {args}')

    if args.test_iterations[0] == -1:
        args.test_iterations = [i for i in range(10000, args.iterations + 1, 10000)]
    if len(args.test_iterations) == 0 or args.test_iterations[-1] != args.iterations:
        args.test_iterations.append(args.iterations)
    print(args.test_iterations)

    if args.save_iterations[0] == -1:
        args.save_iterations = [i for i in range(10000, args.iterations + 1, 10000)]
    if len(args.save_iterations) == 0 or args.save_iterations[-1] != args.iterations:
        args.save_iterations.append(args.iterations)
    print(args.save_iterations)

    if args.gpu != '-1':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        os.system("echo $CUDA_VISIBLE_DEVICES")
        logger.info(f'using GPU {args.gpu}')

    # try:
    #     saveRuntimeCode(os.path.join(args.model_path, 'backup'))
    # except:
    #     logger.info(f'save code failed~')
        
    dataset = args.source_path.split('/')[-1]
    exp_name = args.model_path.split('/')[-1]
    
    if args.use_wandb:
        wandb.login(key='1a21dba66d9736777e51aa1700ab09d6623a9183')
        wandb.login(verify=False) 
        run = wandb.init(
            # Set the project where this run will be logged
            project=f"least-gs",
            name=exp_name,
            # Track hyperparameters and run metadata
            settings=wandb.Settings(start_method="fork"),
            config=vars(args)
        )
    else:
        wandb = None
    
    logger.info("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    # training
    if args.depth_npy_dir is None:
        args.depth_npy_dir = os.path.join(args.model_path, "mesh_depth_npy")
    mesh_render(lp.extract(args), op.extract(args), pp.extract(args), dataset,  args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, wandb, logger, args.ply_path, mesh_path=args.ply_mesh, depth_npy_dir=args.depth_npy_dir)


