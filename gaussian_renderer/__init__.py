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
import torch
from einops import repeat

import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
# from diff_gaussian_rasterization_cull_filter import GaussianRasterizationSettings as GaussianRasterizationSettings_fillter
# from diff_gaussian_rasterization_cull_filter import GaussianRasterizer as GaussianRasterizer_filter
from Mesh2DepthHelper import DepthRenderer,Load_ply_resource,Build_Ply_Render_Camera_Parameters_colmap,Build_Ply_Render_Camera_Parameters_default,show_depth_preview, Build_Ply_Render_Camera_Parameters_colmap_correct
import cv2
import time
import open3d as o3d
import numpy as np


def save_pts_world_as_ply(pts_world, save_path="pts_world.ply", color=[1, 0, 0]):
    """
    保存世界坐标点为 PLY 点云
    pts_world: torch.Tensor (N,3)
    save_path: 输出文件名
    color: 点颜色 (R,G,B)，范围 [0,1]
    """
    if isinstance(pts_world, torch.Tensor):
        pts_world = pts_world.detach().cpu().numpy()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_world)

    # 给每个点设置同样的颜色
    colors = np.tile(np.array(color).reshape(1,3), (pts_world.shape[0], 1))
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_point_cloud(save_path, pcd)
    print(f"✅ Saved {pts_world.shape[0]} points to {save_path}")
    

def build_rotation(r):
    norm = torch.sqrt(
        r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3]
    )

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device="cuda")

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R




def in_frustum_mask_depth_mask(points, viewmatrix, projmatrix, far: float=100.0, image_height: int=512, image_width: int=512, bound: float=1.3):
    """This function is based on `in_frustum` in auxiliary.h of diff-gaussian-rasterization
    
    Args:
        points (torch.Tensor): gaussian center that is a Tensor of shape (#points, 3)
        viewmatrix (torch.Tensor): world-to-camera matrix that is a column-major (=transposed) Tensor of shape (4, 4)
        projmatrix (torch.Tensor): projection matrix that is a column-major (=transposed) Tensor of shape (4, 4)

    Returns:
        masks (torch.Tensor): binary mask that is a Tensor of shape (#points,)
    """
    p_hom = projmatrix.T[None, :4, :3] @ points.reshape(-1, 3, 1) + projmatrix.T[None, :4, -1:]
    p_w = 1 / (p_hom[:, -1] + 1e-7)
    p_proj = p_hom[:, :3, 0] * p_w
    p_view = viewmatrix.T[None, :3, :3] @ points.reshape(-1, 3, 1) + viewmatrix.T[None, :3, -1:]

    image_width = far.shape[1] 
    image_height = far.shape[0] 

    x = ((p_proj[:, 0]+1)*image_width/2.0).long()
    y = ((p_proj[:, 1]+1)*image_height/2.0).long()

    mask_1 = (x < image_width) & (x >= 0)
    mask_2 = (y < image_height) & (y >= 0)

    mask = mask_1 & mask_2

    depth_bound = torch.ones_like(p_proj[:, 0], device=x.device)*1000

    x = x[mask]

    y = y[mask]
    far = far.cuda()
    # far_mask = far > far.mean()
    # far[far_mask] = far[far_mask]*100
    depth_bound[mask] = far[y,x] * 1.2 + 1



    return p_view[:, -1, 0] < depth_bound







def generate_neural_gaussians(viewpoint_camera, pc : GaussianModel, visible_mask=None, is_training=False,  ape_code=-1):
    ## view frustum filtering for acceleration    
    if visible_mask is None:
        visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device = pc.get_anchor.device)

    anchor = pc.get_anchor[visible_mask]
    # save_pts_world_as_ply(anchor, "output/pts_world.ply")
    feat = pc.get_anchor_feat[visible_mask]
    level = pc.get_level[visible_mask]
    grid_offsets = pc._offset[visible_mask]
    grid_scaling = pc.get_scaling[visible_mask]

    ## get view properties for anchor
    ob_view = anchor - viewpoint_camera.camera_center
    # dist
    ob_dist = ob_view.norm(dim=1, keepdim=True)
    # view
    ob_view = ob_view / ob_dist

    # camera_parameters = Build_Ply_Render_Camera_Parameters_colmap_correct(viewpoint_camera.Fx, viewpoint_camera.Fy, viewpoint_camera.Cx, viewpoint_camera.Cy, viewpoint_camera.image_width, viewpoint_camera.image_height, 0.01, 100.0, \
    #     viewpoint_camera.R.transpose(-1,-2), viewpoint_camera.T, 'cuda')

    # device = "cuda"

    # mesh = pc.mesh


    # renderer = DepthRenderer(device=device)
    # depth_m, mask = renderer.render_depth_batched(
    #     mesh = mesh,camera_parameters=camera_parameters,  max_tris_per_pass=1_000_000_0, flip_y=True,linear_depth=True
    # )
    # depth_m= None

    # # mask_inf = mask
    # mask_inf = None

    # mask_depth = in_frustum_mask_depth_mask(anchor, viewpoint_camera.world_view_transform, viewpoint_camera.full_proj_transform, depth_m, int(viewpoint_camera.image_height), int(viewpoint_camera.image_width))   

    # mask_depth = torch.ones(anchor.shape[0], dtype=torch.bool, device=anchor.device)

    ob_view = ob_view
    level = level
    feat = feat
    anchor = anchor
    grid_offsets = grid_offsets
    grid_scaling = grid_scaling
    ob_dist = ob_dist


    ## view-adaptive feature
    if pc.use_feat_bank:
        if pc.add_level:
            cat_view = torch.cat([ob_view, level], dim=1)
        else:
            cat_view = ob_view
        
        bank_weight = pc.get_featurebank_mlp(cat_view).unsqueeze(dim=1) # [n, 1, 3]

        ## multi-resolution feat
        feat = feat.unsqueeze(dim=-1)
        feat = feat[:,::4, :1].repeat([1,4,1])*bank_weight[:,:,:1] + \
            feat[:,::2, :1].repeat([1,2,1])*bank_weight[:,:,1:2] + \
            feat[:,::1, :1]*bank_weight[:,:,2:]
        feat = feat.squeeze(dim=-1) # [n, c]

    if pc.add_level:
        cat_local_view = torch.cat([feat, ob_view, ob_dist, level], dim=1) # [N, c+3+1+1]
        cat_local_view_wodist = torch.cat([feat, ob_view, level], dim=1) # [N, c+3+1]
    else:
        cat_local_view = torch.cat([feat, ob_view, ob_dist], dim=1) # [N, c+3+1]
        cat_local_view_wodist = torch.cat([feat, ob_view], dim=1) # [N, c+3]

    if pc.appearance_dim > 0:
        if is_training or ape_code < 0:
            camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * viewpoint_camera.uid
            appearance = pc.get_appearance(camera_indicies)
        else:
            camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * ape_code[0]
            appearance = pc.get_appearance(camera_indicies)
            
    # get offset's opacity
    if pc.add_opacity_dist:
        neural_opacity = pc.get_opacity_mlp(cat_local_view) # [N, k]
    else:
        neural_opacity = pc.get_opacity_mlp(cat_local_view_wodist)
    
    if pc.dist2level=="progressive":
        prog = pc._prog_ratio[visible_mask]
        transition_mask = pc.transition_mask[visible_mask]
        prog[~transition_mask] = 1.0
        neural_opacity = neural_opacity * prog

    # opacity mask generation
    neural_opacity = neural_opacity.reshape([-1, 1])
    mask = (neural_opacity>0.0)
    mask = mask.view(-1)

    # select opacity 
    opacity = neural_opacity[mask]

    # get offset's color
    if pc.appearance_dim > 0:
        if pc.add_color_dist:
            color = pc.get_color_mlp(torch.cat([cat_local_view, appearance], dim=1))
        else:
            color = pc.get_color_mlp(torch.cat([cat_local_view_wodist, appearance], dim=1))
    else:
        if pc.add_color_dist:
            color = pc.get_color_mlp(cat_local_view)
        else:
            color = pc.get_color_mlp(cat_local_view_wodist)
    color = color.reshape([anchor.shape[0]*pc.n_offsets, 3])# [mask]

    # get offset's cov
    if pc.add_cov_dist:
        scale_rot = pc.get_cov_mlp(cat_local_view)
    else:
        scale_rot = pc.get_cov_mlp(cat_local_view_wodist)
    scale_rot = scale_rot.reshape([anchor.shape[0]*pc.n_offsets, 7]) # [mask]
    
    # offsets
    offsets = grid_offsets.view([-1, 3]) # [mask]
    
    # combine for parallel masking
    concatenated = torch.cat([grid_scaling, anchor], dim=-1)
    concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=pc.n_offsets)
    concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets], dim=-1)
    masked = concatenated_all[mask]
    scaling_repeat, repeat_anchor, color, scale_rot, offsets = masked.split([6, 3, 3, 7, 3], dim=-1)
    
    # post-process cov
    scaling = scaling_repeat[:,3:] * torch.sigmoid(scale_rot[:,:3]) # * (1+torch.sigmoid(repeat_dist))
    rot = pc.rotation_activation(scale_rot[:,3:7])
    
    # post-process offsets to get centers for gaussians
    offsets = offsets * scaling_repeat[:,:3]
    xyz = repeat_anchor + offsets 

    if is_training:
        return xyz, color, opacity, scaling, rot, neural_opacity, mask 
    else:
        return xyz, color, opacity, scaling, rot, mask

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier=1.0, visible_mask=None, retain_grad=False, ape_code=-1):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # start = time.time()
    is_training = pc.get_color_mlp.training
        
    if is_training:
        xyz, color, opacity, scaling, rot, neural_opacity, mask = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=is_training)
    else:
        xyz, color, opacity, scaling, rot, mask = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=is_training, ape_code=ape_code)

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(xyz, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    if retain_grad:
        try:
            screenspace_points.retain_grad()
        except:
            pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = xyz,
        means2D = screenspace_points,
        shs = None,
        colors_precomp = color,
        opacities = opacity,
        scales = scaling,
        rotations = rot,
        cov3D_precomp = None)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # end = time.time()
    # print("Rendering time: ", end - start)
    if is_training:
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "selection_mask": mask,
                "neural_opacity": neural_opacity,
                "scaling": scaling,
                # "mask_mesh": mask_depth
                }
    else:
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "selection_mask": mask,
                # "mask_mesh": mask_inf,
                # "depth_map": depth_m
                }


def prefilter_voxel_cull(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings_fillter(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer_filter = GaussianRasterizer_filter(raster_settings=raster_settings)
    means3D = pc.get_anchor[pc._anchor_mask]

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling[pc._anchor_mask]
        rotations = pc.get_rotation[pc._anchor_mask]
        opacity = pc.get_opacity[pc._anchor_mask]

    mask_cull, radii = rasterizer_filter(
        means3D = means3D,
        shs = None,
        opacities = opacity,
        scales = scales[:,:3],
        rotations = rotations,
        cov3D_precomp = None)
    
    visible_mask = pc._anchor_mask.clone()
    visible_mask[pc._anchor_mask] = mask_cull.squeeze(-1)
    return visible_mask


def mesh_depth_render(viewpoint_camera, renderer = None, mesh=None):
    device = "cuda"
    # Set up rasterization configuration
    if renderer is None:
        renderer = DepthRenderer(device=device)
    # start = time.time()

    camera_parameters = Build_Ply_Render_Camera_Parameters_colmap_correct(viewpoint_camera.Fx, viewpoint_camera.Fy, viewpoint_camera.Cx, viewpoint_camera.Cy, viewpoint_camera.image_width, viewpoint_camera.image_height, 0.01, 100.0, \
    viewpoint_camera.R.transpose(-1,-2), viewpoint_camera.T, 'cuda')


    # mesh = pc.mesh

    depth_m, mask_inf = renderer.render_depth_batched(
        mesh = mesh,camera_parameters=camera_parameters,  max_tris_per_pass=1_000_000_00, flip_y=True,linear_depth=True
    )

    return depth_m


def prefilter_voxel(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, tol = None, renderer = None, depth_map = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    depth_device = bg_color.device
    if depth_map is None:
        depth_m = torch.full(
            (int(viewpoint_camera.image_height), int(viewpoint_camera.image_width)),
            1e8,
            dtype=torch.float32,
            device=depth_device,
        )
    else:
        depth_m = torch.as_tensor(depth_map, dtype=torch.float32, device=depth_device).squeeze()

    if depth_m.shape != (int(viewpoint_camera.image_height), int(viewpoint_camera.image_width)):
        raise ValueError(
            f"Depth map shape mismatch for {viewpoint_camera.image_name}: "
            f"got {tuple(depth_m.shape)}, expected {(int(viewpoint_camera.image_height), int(viewpoint_camera.image_width))}"
        )

    if tol is not None:
        depth_m = depth_m + tol
    # depth_m = torch.ones_like(depth_m, device=depth_m.device) * 10000
    # end = time.time()
    # print("Depth rendering time: ", end - start)
    # depth_m = depth_m.clone()
    # depth_m[mask_inf == 0] = 1000.0

    # start = time.time()
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        depth_mesh=depth_m,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    means3D = pc.get_anchor

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling[pc._anchor_mask]
        rotations = pc.get_rotation[pc._anchor_mask]

    radii_pure = rasterizer.visible_filter(means3D = means3D,
        scales = scales[:,:3],
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,
        point_mask = pc._anchor_mask)
    
    visible_mask = (radii_pure > 0)#&pc._anchor_mask        
    # end = time.time()
    # print("Prefiltering time: ", end - start)

    return visible_mask, depth_m
