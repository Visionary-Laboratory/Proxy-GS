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
from diff_gaussian_rasterization import GaussianRasterizationSettings as GaussianRasterizationSettings_filter
from diff_gaussian_rasterization import GaussianRasterizer  as GaussianRasterizer_filter
from fast_gauss import GaussianRasterizationSettings as GaussianRasterizationSettings_hard
from fast_gauss import GaussianRasterizer as GaussianRasterizer_hard


from scene.gaussian_model import GaussianModel
# from diff_gaussian_rasterization_cull_filter import GaussianRasterizationSettings as GaussianRasterizationSettings_fillter
# from diff_gaussian_rasterization_cull_filter import GaussianRasterizer as GaussianRasterizer_filter
from Mesh2DepthHelper import DepthRenderer,Load_ply_resource,Build_Ply_Render_Camera_Parameters_colmap,Build_Ply_Render_Camera_Parameters_default,show_depth_preview, Build_Ply_Render_Camera_Parameters_colmap_correct
import cv2
import time

def _cuda_timing_start():
    if torch.cuda.is_available():
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()  # 清空队列，防止前序算子影响
        s.record()
        return ("cuda", s, e)
    else:
        return ("cpu", time.perf_counter(), None)

def _cuda_timing_end(stamp):
    mode, start, end = stamp
    if mode == "cuda":
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end)  # ms
    else:
        return (time.perf_counter() - start) * 1000.0  # ms



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






@torch.inference_mode()
def generate_neural_gaussians(viewpoint_camera, pc : GaussianModel, visible_mask=None, is_training=False,  ape_code=-1):
    ## view frustum filtering for acceleration    
    if visible_mask is None:
        visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device = pc.get_anchor.device)

    anchor = pc.get_anchor[visible_mask]
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

    ob_view = ob_view
    level = level
    feat = feat
    anchor = anchor
    grid_offsets = grid_offsets
    grid_scaling = grid_scaling
    ob_dist = ob_dist


    ## view-adaptive feature
    # false
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


    # false
    if pc.add_level:
        cat_local_view = torch.cat([feat, ob_view, ob_dist, level], dim=1) # [N, c+3+1+1]
        cat_local_view_wodist = torch.cat([feat, ob_view, level], dim=1) # [N, c+3+1]
    else:
        1
        cat_local_view = torch.cat([feat, ob_view, ob_dist], dim=1) # [N, c+3+1]
        cat_local_view_wodist = torch.cat([feat, ob_view], dim=1) # [N, c+3]

    if pc.appearance_dim > 0:
        if is_training or ape_code < 0:
            camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * viewpoint_camera.uid
            appearance = pc.get_appearance(camera_indicies)
        else:
            camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * ape_code[0]
            appearance = pc.get_appearance(camera_indicies)
            
    #false       
    # get offset's opacity
    if pc.add_opacity_dist:
        neural_opacity = pc.get_opacity_mlp(cat_local_view) # [N, k]
    else:
        neural_opacity = pc.get_opacity_mlp(cat_local_view_wodist)
    
    #false
    if pc.dist2level=="progressive":
        print("progressive")
        # save no reached
        prog = pc._prog_ratio[visible_mask]
        transition_mask = pc.transition_mask[visible_mask]
        prog[~transition_mask] = 1.0
        neural_opacity = neural_opacity * prog

    # opacity mask generation
    neural_opacity = neural_opacity.reshape([-1, 1])
    mask = (neural_opacity>0.0)
    mask = mask.view(-1)

    # select opacity 
    opacity = neural_opacity[mask, 0:1].contiguous() 

    # get offset's color
    # false
    if pc.appearance_dim > 0:
        if pc.add_color_dist:
            color = pc.get_color_mlp(torch.cat([cat_local_view, appearance], dim=1))
        else:
            color = pc.get_color_mlp(torch.cat([cat_local_view_wodist, appearance], dim=1))
    else:
        if pc.add_color_dist: #false
            color = pc.get_color_mlp(cat_local_view)
        else:
            color = pc.get_color_mlp(cat_local_view_wodist)
    color = color.reshape([anchor.shape[0]*pc.n_offsets, 3])# [mask]
    

    #false
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


from contextlib import contextmanager
import time
import torch
from einops import repeat

@torch.inference_mode()
def generate_neural_gaussians_new(
    viewpoint_camera,
    pc: GaussianModel,
    visible_mask=None,
    is_training=False,
    ape_code=-1,
    *,
    profile: bool = True,
    return_timings: bool = False,
):
    """
    profile=True 时启用分段计时；return_timings=True 时在返回值最后追加 timings(dict, ms)。
    其余返回结构与原函数保持一致。
    """

    # -------------------- 计时辅助 --------------------
    timings = {}

    @contextmanager
    def section(name: str):
        if not profile:
            yield
            return
        _stamp = _cuda_timing_start()
        try:
            yield
        finally:
            dt = _cuda_timing_end(_stamp)
            timings[name] = timings.get(name, 0.0) + float(dt)

    if profile:
        _total_stamp = _cuda_timing_start()

    # -------------------- 可见性掩码 / gather --------------------
    with section("visible_mask_default"):
        if visible_mask is None:
            visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device=pc.get_anchor.device)

    with section("gather_inputs"):
        anchor = pc.get_anchor[visible_mask]
        feat = pc.get_anchor_feat[visible_mask]
        level = pc.get_level[visible_mask]
        grid_offsets = pc._offset[visible_mask]
        grid_scaling = pc.get_scaling[visible_mask]

    # -------------------- 视图属性（方向+距离） --------------------
    with section("view_properties"):
        ob_view = anchor - viewpoint_camera.camera_center
        ob_dist = ob_view.norm(dim=1, keepdim=True)
        ob_view = ob_view / (ob_dist + 1e-12)

    # -------------------- 视图自适应特征（feat bank） --------------------
    if pc.use_feat_bank:
        with section("feature_bank"):
            if pc.add_level:
                cat_view = torch.cat([ob_view, level], dim=1)
            else:
                cat_view = ob_view
            bank_weight = pc.get_featurebank_mlp(cat_view).unsqueeze(1)  # [N,1,3]

            f = feat.unsqueeze(-1)  # [N,C,1]
            feat = f[:, ::4, :1].repeat([1, 4, 1]) * bank_weight[:, :, :1] + \
                   f[:, ::2, :1].repeat([1, 2, 1]) * bank_weight[:, :, 1:2] + \
                   f[:, ::1, :1] * bank_weight[:, :, 2:]
            feat = feat.squeeze(-1)  # [N,C]

    # -------------------- 拼接输入（含/不含 dist, level） --------------------
    with section("concat_features"):
        if pc.add_level:
            cat_local_view = torch.cat([feat, ob_view, ob_dist, level], dim=1)      # [N, c+3+1+1]
            cat_local_view_wodist = torch.cat([feat, ob_view, level], dim=1)        # [N, c+3+1]
        else:
            # cat_local_view = torch.cat([feat, ob_view, ob_dist], dim=1)             # [N, c+3+1]
            cat_local_view_wodist = torch.cat([feat, ob_view], dim=1)               # [N, c+3]

    # -------------------- 取 appearance --------------------
    appearance = None
    # not reached
    if pc.appearance_dim > 0:
        with section("appearance_gather"):
            if is_training or ape_code < 0:
                camera_indicies = torch.ones_like(cat_local_view[:, 0], dtype=torch.long, device=cat_local_view.device) * viewpoint_camera.uid
            else:
                camera_indicies = torch.ones_like(cat_local_view[:, 0], dtype=torch.long, device=cat_local_view.device) * int(ape_code[0] if torch.is_tensor(ape_code) else ape_code)
            appearance = pc.get_appearance(camera_indicies)  # [N, A]

    # -------------------- Opacity MLP --------------------
    with section("opacity_mlp"):
        if pc.add_opacity_dist:
            neural_opacity = pc.get_opacity_mlp(cat_local_view)  # [N,k]
        else:
            neural_opacity = pc.get_opacity_mlp(cat_local_view_wodist)  # [N,k]

    # 渐进 level 权重
    if pc.dist2level == "progressive":
        with section("progressive_scale"):
            prog = pc._prog_ratio[visible_mask]
            transition_mask = pc.transition_mask[visible_mask]
            prog = prog.clone()
            prog[~transition_mask] = 1.0
            neural_opacity = neural_opacity * prog

    # -------------------- 生成 opacity 掩码 --------------------
    with section("opacity_masking"):
        neural_opacity = neural_opacity.reshape([-1, 1])          # [N*k,1]
        mask = (neural_opacity > 0.0).view(-1)                    # [N*k]
        opacity = neural_opacity[mask]                            # [M,1]

    # -------------------- Color MLP --------------------
    with section("color_mlp"):
        if pc.appearance_dim > 0 and appearance is not None:
            if pc.add_color_dist:
                color = pc.get_color_mlp(torch.cat([cat_local_view, appearance], dim=1))
            else:
                color = pc.get_color_mlp(torch.cat([cat_local_view_wodist, appearance], dim=1))
        else:
            if pc.add_color_dist:
                color = pc.get_color_mlp(cat_local_view)
            else:
                color = pc.get_color_mlp(cat_local_view_wodist)
        color = color.reshape([anchor.shape[0] * pc.n_offsets, 3])  # [N*k,3]

    # -------------------- Cov MLP --------------------
    with section("cov_mlp"):
        if pc.add_cov_dist:
            scale_rot = pc.get_cov_mlp(cat_local_view)
        else:
            scale_rot = pc.get_cov_mlp(cat_local_view_wodist)
        scale_rot = scale_rot.reshape([anchor.shape[0] * pc.n_offsets, 7])  # [N*k,7]

    # -------------------- offsets/拼接/掩码选取 --------------------
    with section("offsets_reshape"):
        offsets = grid_offsets.view([-1, 3])  # [N*k,3]

    with section("concat_repeat"):
        concatenated = torch.cat([grid_scaling, anchor], dim=-1)                 # [N,9]
        concatenated_repeated = repeat(concatenated, 'n c -> (n k) c', k=pc.n_offsets)  # [N*k,9]
        concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets], dim=-1)  # [N*k,22]

    with section("boolean_mask_and_split"):
        masked = concatenated_all[mask]                                          # [M,22]
        scaling_repeat, repeat_anchor, color, scale_rot, offsets = masked.split([6, 3, 3, 7, 3], dim=-1)

    # -------------------- 后处理（cov/offset/xyz） --------------------
    with section("post_cov"):
        scaling = scaling_repeat[:, 3:] * torch.sigmoid(scale_rot[:, :3])        # [M,3]
        rot = pc.rotation_activation(scale_rot[:, 3:7])                           # [M,4] (取决于实现)

    with section("post_offsets_xyz"):
        offsets = offsets * scaling_repeat[:, :3]                                 # [M,3]
        xyz = repeat_anchor + offsets                                             # [M,3]

    # -------------------- 收尾 & 返回 --------------------
    if profile:
        timings["TOTAL"] = float(_cuda_timing_end(_total_stamp))
        print({k: f"{v:.3f} ms" for k, v in timings.items()})

    if is_training:
        out = (xyz, color, opacity, scaling, rot, neural_opacity, mask)
    else:
        out = (xyz, color, opacity, scaling, rot, mask)

    if return_timings:
        return (*out, timings)
    else:
        return out





import torch
from torch import Tensor

# —— 建议在程序初始化处设置一次 —— 
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True

def make_gaussian_decoder(
    n_offsets: int,
    add_level: bool = False,
    appearance_dim: int = 0,
    add_color_dist: bool = False,
    add_cov_dist: bool = False,
    add_opacity_dist: bool = False,
    dtype = torch.bfloat16,               # 也可用 torch.float16
    compile_mode: str = "reduce-overhead" # 或 "max-autotune"
):
    """
    返回已编译的 core 函数。把所有恒定开关 bake 进图里，便于 torch.compile 消去分支。
    """

    def _core(
        # —— 只传 Tensor & 常量，杜绝 Python 属性访问 —— 
        anchor: Tensor,            # [N,3]
        feat: Tensor,              # [N,C]
        level: Tensor,             # [N,1] (可不使用)
        grid_offsets: Tensor,      # [N,k,3]
        grid_scaling: Tensor,      # [N,6]  ([:,:3] for offset scale, [:,3:] for cov-scale)
        camera_center: Tensor,     # [1,3]
        opacity_mlp, color_mlp, cov_mlp, rot_act,   # 可调用的 Module/activation
        # 运行时可变：可见 mask / 训练开关 / 外部ape
        visible_mask: Tensor = None,
        is_training: bool = False,
        appearance_vec: Tensor = None      # [N,appearance_dim] 按需要传
    ):
        if visible_mask is None:
            visible_mask = torch.ones(anchor.shape[0], dtype=torch.bool, device=anchor.device)

        # 预筛 anchor
        anchor = anchor[visible_mask]
        feat   = feat[visible_mask]
        level  = level[visible_mask] if add_level else level
        grid_offsets = grid_offsets[visible_mask]       # [N,k,3]
        grid_scaling = grid_scaling[visible_mask]       # [N,6]

        # 视角量
        ob_view = anchor - camera_center                # [N,3]
        ob_dist = ob_view.norm(dim=1, keepdim=True)     # [N,1]
        ob_view = ob_view / (ob_dist + 1e-12)

        # 拼特征（视配置而定）
        if add_level:
            cat_lv      = torch.cat([feat, ob_view, ob_dist, level], dim=1)       # [N, C+3+1+1]
            cat_lv_wod  = torch.cat([feat, ob_view, level], dim=1)                # [N, C+3+1]
        else:
            cat_lv      = torch.cat([feat, ob_view, ob_dist], dim=1)              # [N, C+3+1]
            cat_lv_wod  = torch.cat([feat, ob_view], dim=1)                       # [N, C+3]

        # 可选 appearance
        if appearance_dim > 0 and appearance_vec is not None:
            cat_in_full    = torch.cat([cat_lv,     appearance_vec], dim=1) if add_color_dist else torch.cat([cat_lv_wod, appearance_vec], dim=1)
            cat_in_opacity = torch.cat([cat_lv,     appearance_vec], dim=1) if add_opacity_dist else torch.cat([cat_lv_wod, appearance_vec], dim=1)
            cat_in_cov     = torch.cat([cat_lv,     appearance_vec], dim=1) if add_cov_dist     else torch.cat([cat_lv_wod, appearance_vec], dim=1)
        else:
            cat_in_full    = cat_lv if add_color_dist else cat_lv_wod
            cat_in_opacity = cat_lv if add_opacity_dist else cat_lv_wod
            cat_in_cov     = cat_lv if add_cov_dist else cat_lv_wod

        # ====== 关键：先跑便宜的 opacity，再取索引，最后才跑 color/cov ======
        with torch.autocast("cuda", dtype=dtype):
            # opacity_mlp 输出 [N, k]；只用于筛
            neural_opacity = opacity_mlp(cat_in_opacity)                     # [N,k]
            op_flat = neural_opacity.reshape(-1, 1)                           # [N*k,1]
            mask    = (op_flat > 0.0).squeeze(1)                              # [N*k]
            idx     = torch.nonzero(mask, as_tuple=False).squeeze(1)          # [M]
            if idx.numel() == 0:
                # 返回空张量，维度对齐
                empty = anchor.new_zeros((0,3))
                return empty, empty, empty, empty, empty, mask

            # 解析 (anchor_idx, offset_idx)
            N = anchor.shape[0]
            k = n_offsets
            anchor_idx = torch.div(idx, k, rounding_mode='floor')             # [M]
            offset_idx = idx - anchor_idx * k                                 # [M]

            # 仅对 unique anchors 跑一次大 MLP
            uniq_anchor, inv = torch.unique(anchor_idx, sorted=True, return_inverse=True)  # uniq U, inv maps M->U

            # color_mlp：原来输出 [N, k*3]，这里只对 U 个 anchor 计算
            col_u = color_mlp(cat_in_full.index_select(0, uniq_anchor))       # [U, k*3]
            col_u = col_u.view(-1, k, 3)                                      # [U, k, 3]
            color = col_u[inv, offset_idx, :]                                 # [M, 3]

            # cov_mlp：原来输出 [N, k*7]
            cov_u = cov_mlp(cat_in_cov.index_select(0, uniq_anchor))          # [U, k*7]
            cov_u = cov_u.view(-1, k, 7)                                      # [U, k, 7]
            scale_rot = cov_u[inv, offset_idx, :]                              # [M, 7]

            # 取对应的 base 几何量（不 materialize repeat）
            repeat_anchor   = anchor.index_select(0, anchor_idx)              # [M,3]
            scaling_repeat  = grid_scaling.index_select(0, anchor_idx)        # [M,6]
            offsets_sel     = grid_offsets.view(-1, 3).index_select(0, idx)   # [M,3]
            opacity         = op_flat.index_select(0, idx)         # [M]

            # 后处理 cov
            scaling = scaling_repeat[:, 3:] * torch.sigmoid(scale_rot[:, :3]) # [M,3]
            rot     = rot_act(scale_rot[:, 3:7])                               # [M,4] (或你定义的旋转格式)

            # 偏移 -> xyz
            offsets = offsets_sel * scaling_repeat[:, :3]                      # [M,3]
            xyz     = repeat_anchor + offsets                                  # [M,3]

        if is_training:
            # 如需把未筛前的 neural_opacity 带回去，这里返回 op_flat 与 mask
            return xyz, color, opacity, scaling, rot, op_flat, mask
        else:
            return xyz, color, opacity, scaling, rot, mask

    # —— 编译 —— 
    compiled = torch.compile(_core, dynamic=True, mode=compile_mode)
    return compiled




decode = None



# def makeDecoder(pc):
#     # 一次性构建（比如在 GaussianModel 初始化后）

@torch.inference_mode()
def generate_neural_gaussians_fast(viewpoint_camera, pc, visible_mask=None, is_training=False, ape_code=-1):
    # 把所有 Tensor & 模块先“解引用”，避免在 compiled 函数里反复 Python 取属性导致图断裂
    global decode
    return decode(
        pc.get_anchor,                # [N,3]
        pc.get_anchor_feat,           # [N,C]
        pc.get_level,                 # [N,1]
        pc._offset,                   # [N,k,3]
        pc.get_scaling,               # [N,6]
        viewpoint_camera.camera_center[None, :],  # [1,3]
        pc.get_opacity_mlp,
        pc.get_color_mlp,
        pc.get_cov_mlp,
        pc.rotation_activation,
        visible_mask=visible_mask,
        is_training=is_training,
        appearance_vec=None           # 如有 appearance，请计算并传入 [N, D]
    )







def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier=1.0, visible_mask=None, retain_grad=False, ape_code=-1):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # start = time.time()
    global decode
    if decode is None:
        
        decode = make_gaussian_decoder(
            n_offsets=pc.n_offsets,
            add_level=False,            # 你的实际恒定开关
            appearance_dim=0,
            add_color_dist=False,
            add_cov_dist=False,
            add_opacity_dist=False,
            dtype=torch.float32,       # A100/H100 建议 bf16；4090 可试 fp16
            compile_mode="max-autotune"
        )

    is_training = pc.get_color_mlp.training
        
    if is_training:
        xyz, color, opacity, scaling, rot, neural_opacity, mask = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=is_training)
    else:
        xyz, color, opacity, scaling, rot, mask = generate_neural_gaussians_fast(viewpoint_camera, pc, visible_mask, is_training=is_training, ape_code=ape_code)

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
    
    use_hard_raster = False
    if use_hard_raster:
        raster_settings = GaussianRasterizationSettings_hard(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color.detach().cpu(),
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform.detach().cpu(),
            projmatrix=viewpoint_camera.full_proj_transform.detach().cpu(),
            sh_degree=1,
            campos=viewpoint_camera.camera_center.detach().cpu(),
            prefiltered=False,
            debug=pipe.debug
        )

        rasterizer = GaussianRasterizer_hard(raster_settings=raster_settings,
                                        init_buffer_size = 117897,
                                        # dtype=torch.float32,
                                        # tex_dtype=torch.float32,
                                        offline_writeback = False)


        rendered_image, radii = rasterizer(
            means3D = xyz,
            means2D = screenspace_points,
            shs = None,
            colors_precomp = color,
            opacities = opacity,
            scales = scaling,
            rotations = rot,
            cov3D_precomp = None)
        
       # print({k: f"{v:.3f} ms" for k, v in rasterizer.last_timing.items()})

    else:
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 

        raster_settings = GaussianRasterizationSettings_filter(
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
        rasterizer = GaussianRasterizer_filter(raster_settings=raster_settings)

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
    # print(viewpoint_camera.image_path)



    if ('rendered_image' not in locals()
        or locals()['rendered_image'] is None
        or (isinstance(rendered_image, torch.Tensor) and rendered_image.numel() == 0)):
        rendered_image = torch.zeros((1, 3, 1, 1), dtype=torch.float32, device='cuda')


    # —— 3) radii 兜底：用长度为 1 的小向量，占位且 float32
    if ('radii' not in locals()
        or locals()['radii'] is None
        or (isinstance(radii, torch.Tensor) and radii.numel() == 0)):
        radii = torch.zeros((1,), dtype=torch.float32, device='cuda')

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







import time
import math
# import torch


def prefilter_voxel_old(viewpoint_camera, renderer, pc: "GaussianModel", pipe, bg_color: torch.Tensor,
                    scaling_modifier=1.0, override_color=None):
    """
    Render the scene and prefilter visible anchors.

    Prints timing for:
      - depth rendering
      - gather/slice
      - filter (visible_filter)
    """
    # === 1) 渲染 depth ===
    t0 = _cuda_timing_start()
    depth_m = renderer.render(
        camera_R=viewpoint_camera.R,
        camera_T=viewpoint_camera.T,
        fx=viewpoint_camera.Fx, fy=viewpoint_camera.Fy,
        cx=viewpoint_camera.Cx, cy=viewpoint_camera.Cy,
        znear=0.01, zfar=1000.0
    )
    depth_ms = _cuda_timing_end(t0)

    # === 2) 准备 rasterizer + gather/切片 ===
    # （构建 settings/rasterizer 基本是 CPU 操作，开销小，这里不单独计时）
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    raster_settings = GaussianRasterizationSettings_filter(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx, tanfovy=tanfovy,
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
    rasterizer = GaussianRasterizer_filter(raster_settings=raster_settings)

    t1 = _cuda_timing_start()
    mask = pc._anchor_mask

    # means3D = pc.get_anchor[mask]  # gather 1

    # scales = None
    # rotations = None
    # cov3D_precomp = None
    # if pipe.compute_cov3D_python:
    #     cov3D_precomp = pc.get_covariance(scaling_modifier)  # 这一步通常也在 GPU 上
    # else:
    #     scales = pc.get_scaling[mask]      # gather 2
    #     rotations = pc.get_rotation[mask]  # gather 3

    means3D = pc.get_anchor  # gather 1

    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:

        cov3D_precomp = pc.get_covariance(scaling_modifier)  # 这一步通常也在 GPU 上
    else:
        scales = pc.get_scaling      # gather 2
        rotations = pc.get_rotation  # gather 3

    gather_ms = _cuda_timing_end(t1)

    # === 3) filter（visible_filter）===
    t2 = _cuda_timing_start()
    radii_pure = rasterizer.visible_filter(means3D = means3D,
        scales = scales[:,:3],
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,
        point_mask = pc._anchor_mask)
    
    # visible_mask = pc._anchor_mask.clone()
    visible_mask = (radii_pure > 0)
    filter_ms = _cuda_timing_end(t2)

    # 根据 filter 结果更新可见 mask
    # visible_mask = pc._anchor_mask.clone()
    # visible_mask[mask] = (radii_pure > 0)
    # visible_mask = (radii_pure > 0)&pc._anchor_mask
    # === 打印结果 ===
    n_selected = int(means3D.shape[0])
    print(f"[prefilter_voxel] depth={depth_ms:.3f} ms | gather={gather_ms:.3f} ms | filter={filter_ms:.3f} ms | N={n_selected} | used = {visible_mask.sum()}")

    return visible_mask, depth_m




def prefilter_voxel(viewpoint_camera, renderer, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # print("prefilter_voxel")
    device = "cuda"
    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    # start = time.time()

    camera_parameters = Build_Ply_Render_Camera_Parameters_colmap_correct(viewpoint_camera.Fx, viewpoint_camera.Fy, viewpoint_camera.Cx, viewpoint_camera.Cy, viewpoint_camera.image_width, viewpoint_camera.image_height, 0.01, 100.0, \
    viewpoint_camera.R.transpose(-1,-2), viewpoint_camera.T, 'cuda')





    depth_m = renderer.render(
        camera_R=viewpoint_camera.R,
        camera_T=viewpoint_camera.T,
        fx=viewpoint_camera.Fx, fy=viewpoint_camera.Fy, cx=viewpoint_camera.Cx, cy=viewpoint_camera.Cy,znear=0.01,zfar=1000.0
    )


    # start = time.time()
    raster_settings = GaussianRasterizationSettings_filter(
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

    rasterizer = GaussianRasterizer_filter(raster_settings=raster_settings)
    # print(pc.get_anchor.shape)
    means3D = pc.get_anchor#[pc._anchor_mask]

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    # if pipe.compute_cov3D_python:
    #     cov3D_precomp = pc.get_covariance(scaling_modifier)
    # else:
    #     scales = pc.get_scaling[pc._anchor_mask]
    #     rotations = pc.get_rotation[pc._anchor_mask]


    scales = pc.get_scaling
    rotations = pc.get_rotation



    radii_pure = rasterizer.visible_filter(means3D = means3D,
        scales = scales[:,:3],
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,
        point_mask = pc._anchor_mask)
    
    # visible_mask = pc._anchor_mask.clone()
    visible_mask = (radii_pure > 0)#&pc._anchor_mask

    # end = time.time()
    # print("Prefiltering time: ", end - start)

    return visible_mask, depth_m