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


# os.environ["PYOPENGL_PLATFORM"] = "egl"
# os.environ["FAST_GAUSS_FORCE_ONLINE"] = "1"
# os.environ["FAST_GAUSS_OFFLINE_WRITEBACK"] = "0"


from os import makedirs
import torch
import numpy as np
import sys
import subprocess
cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))

os.system('echo $CUDA_VISIBLE_DEVICES')
from Mesh2DepthHelper import DepthRenderer,Load_ply_resource,Build_Ply_Render_Camera_Parameters_colmap,Build_Ply_Render_Camera_Parameters_default,show_depth_preview, Build_Ply_Render_Camera_Parameters_colmap_correct
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

import OpenGL.GL as gl


sys.path.insert(0, "python")
sys.path.insert(0, "build-py/_bin/Release")
import sys
from vk2torch_renderer import VK2TorchRenderer
from vk2torch_utils import save_depth_png


# Configuration
width, height = 1024, 690
# scene_file = "./_downloaded_resources/house_new.glb"  # Relative to asset root
asset_root = os.getcwd()  # to the directory
output_dir = "basic_render_output"
scene_file = "/home/yyg/Desktop/vk_lod_clusters/small_city_reduced.glb"  # Relative to asset root

# scene_file = "/home/yyg/Downloads/code/Mesh_occGS/Block_E_Reduced_mesh.glb"
# scene_file = "/home/yyg/Downloads/code/Mesh_occGS/city_street_new.glb"

# scene_file = "/home/yyg/Desktop/vk_lod_clusters/Block-D_1-test.glb" 

asset_root = "/home/yyg/Desktop/vk_lod_clusters/"  # to the directory

@torch.inference_mode()
def render_set(model_path, name, iteration, views, gaussians, pipeline, background, show_level, ape_code):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    makedirs(render_path, exist_ok=True)
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    makedirs(gts_path, exist_ok=True)
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")
    makedirs(depth_path, exist_ok=True)
    if show_level:
        render_level_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_level")
        makedirs(render_level_path, exist_ok=True)

    t_list = []
    per_view_dict = {}
    per_view_level_dict = {}
    renderer = VK2TorchRenderer(width=width, height=height, scene_file=scene_file, asset_root=asset_root) 
    print(f"Renderer info: {renderer.get_info()}")
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    anchor_number = 0
    for j in range(10):
        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):

            # torch.cuda.synchronize(); t0 = time.time()
            start.record()
            gaussians.set_anchor_mask(view.camera_center, iteration, view.resolution_scale)

            # Create renderer



            # Example 1: Simple camera positioned in front of scene
            

            voxel_visible_mask ,depth_m = prefilter_voxel(view, renderer, gaussians, pipeline, background)
            render_pkg = render(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask, ape_code=ape_code)
            end.record()
            torch.cuda.synchronize(); 
            # t1 = time.time()
            # t_list.append(t1-t0)
            elapsed_ms = start.elapsed_time(end) 
            if j >3:
                t_list.append(elapsed_ms)
            rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
            visible_count = render_pkg["visibility_filter"].sum()  
            per_view_dict['{0:05d}'.format(idx)+".png"] = visible_count.item()
            anchor_number += voxel_visible_mask.sum().item()

            save_File = False
            if save_File and j==9:
                gt = view.original_image[0:3, :, :]
                torchvision.utils.save_image(rendering, os.path.join(render_path,view.image_name + ".png"))
                torchvision.utils.save_image(gt, os.path.join(gts_path,view.image_name + ".png"))
                depth = depth_m.clone()
                depth = depth.cpu().numpy()
                # print(f"Rendered depth shape: {depth.shape}")
                # print(f"Depth range: [{depth.min():.3f}, {depth.max():.3f}]")

                # Save depth image
                depth_file = os.path.join(depth_path, view.image_name + ".png")

                save_depth_png(depth, depth_file)

            # show_depth_preview(depth_m, mask_inf,os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"), invert=False, q=(0.02, 0.98))
            if show_level:
                for cur_level in range(gaussians.levels):
                    gaussians.set_anchor_mask_perlevel(view.camera_center, view.resolution_scale, cur_level)
                    voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background)
                    render_pkg = render(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask, ape_code=ape_code)
                    
                    rendering = render_pkg["render"]
                    visible_count = render_pkg["visibility_filter"].sum()
                    
                    torchvision.utils.save_image(rendering, os.path.join(render_level_path, '{0:05d}_LOD{1:d}'.format(idx, cur_level) + ".png"))
                    per_view_level_dict['{0:05d}_LOD{1:d}'.format(idx, cur_level) + ".png"] = visible_count.item()

    t = np.array(t_list[5:])
    fps = 1000.0 / t.mean()
    print(f'Test FPS: \033[1;35m{fps:.5f}\033[0m')
    print(anchor_number/len(views))

    with open(os.path.join(model_path, name, "ours_{}".format(iteration), "per_view_count.json"), 'w') as fp:
        json.dump(per_view_dict, fp, indent=True) 
    if show_level:
        with open(os.path.join(model_path, name, "ours_{}".format(iteration), "per_view_count_level.json"), 'w') as fp:
            json.dump(per_view_level_dict, fp, indent=True)     
     
def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, show_level : bool, ape_code : int, ply_path=None):
    with torch.no_grad():
        gaussians = GaussianModel(
            dataset.feat_dim, dataset.n_offsets, dataset.fork, dataset.use_feat_bank, dataset.appearance_dim, 
            dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist, dataset.add_level, 
            dataset.visible_threshold, dataset.dist2level, dataset.base_layer, dataset.progressive, dataset.extend
        )
        scene = Scene(dataset, gaussians, load_iteration=iteration,  ply_path=ply_path, shuffle=False, resolution_scales=dataset.resolution_scales)
        gaussians.eval()
        gaussians.plot_levels()
        if dataset.random_background:
            bg_color = [np.random.random(),np.random.random(),np.random.random()] 
        elif dataset.white_background:
            bg_color = [1.0, 1.0, 1.0]
        else:
            bg_color = [0.0, 0.0, 0.0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        if not os.path.exists(dataset.model_path):
            os.makedirs(dataset.model_path)
        
        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, show_level, ape_code)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, show_level, ape_code)

if __name__ == "__main__":

    # W, H = 1000, 1000  # 你想测的分辨率

    # fb_w = W
    # fb_h = H
    # gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
    # gl.glViewport(0, 0, fb_w, fb_h)
    # gl.glScissor(0, 0, fb_w, fb_h)
    # print('')

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
    parser.add_argument("--ply_path", type=str, default=None)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.show_level, args.ape, args.ply_path)
    
