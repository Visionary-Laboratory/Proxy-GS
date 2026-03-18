#!/usr/bin/env python3
"""
VK2Torch Utilities Module

Helper functions for camera matrices, depth processing, and image utilities.
Extracted from pybind_timeline_roundtrip.py for reusability.
"""

import numpy as np
import os
from typing import Tuple, Union, Optional

def _frustum_offcenter_rh_zo(l: float, r: float, b: float, t: float, n: float, f: float) -> np.ndarray:
    """
    Create frustum projection matrix (right-handed, depth 0..1).
    Compatible with glm::perspectiveRH_ZO.
    """
    P = np.array([
        [2*n/(r-l),      0.0,      (r+l)/(r-l),          0.0],
        [0.0,        2*n/(t-b),    (t+b)/(t-b),          0.0],
        [0.0,            0.0,          f/(n-f),    (f*n)/(n-f)],
        [0.0,            0.0,            -1.0,          0.0],
    ], dtype=np.float32)
    return P

def proj_from_intrinsics_vulkan(fx: float, fy: float, cx: float, cy: float, 
                                W: int, H: int, znear: float, zfar: float, 
                                *, flip_y: bool = True) -> np.ndarray:
    """
    Create projection matrix from camera intrinsics for Vulkan.
    
    Args:
        fx, fy: Focal lengths in pixels
        cx, cy: Principal point in pixels  
        W, H: Image dimensions
        znear, zfar: Near and far plane distances
        flip_y: Flip Y axis for Vulkan coordinate system
        
    Returns:
        4x4 projection matrix as numpy array
    """
    # Frustum boundaries at near plane (camera coordinate system: y↑, z→)
    l = -znear * (cx)      / fx
    r =  znear * (W - cx)  / fx
    t =  znear * (cy)      / fy
    b = -znear * (H - cy)  / fy
    
    P = _frustum_offcenter_rh_zo(l, r, b, t, znear, zfar)
    
    if flip_y:  # Vulkan convention: flip Y in projection
        P[1, :] *= -1.0
    return P

def to_vulkan_viewproj_match_nvdiffrast(
    R_ocv: np.ndarray, T_ocv: np.ndarray,  # world->cam, OpenCV/Colmap convention
    fx: float, fy: float, cx: float, cy: float, W: int, H: int, 
    znear: float, zfar: float,
    *,
    nvdiffrast_world_is_z_up: bool = True,   # If nvdiffrast uses Z-up world
    your_world_is_y_up: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert camera parameters to Vulkan view/projection matrices.
    
    Args:
        R_ocv, T_ocv: Rotation matrix and translation vector (world->camera)
        fx, fy, cx, cy: Camera intrinsics
        W, H: Image dimensions
        znear, zfar: Depth range
        nvdiffrast_world_is_z_up: If input uses Z-up coordinate system
        your_world_is_y_up: If output should use Y-up coordinate system
        
    Returns:
        Tuple of (view_matrix_flat, proj_matrix_flat) as 16-element arrays
    """
    R_ocv = np.swapaxes(R_ocv, -1, -2)
    R = np.asarray(R_ocv, np.float32)
    t = np.asarray(T_ocv, np.float32).reshape(3, 1)

    # Optional basis transformation: Z-up world -> Y-up world
    # A = Rx(+90°): (x, y, z)_nv -> (x, z, -y)_your
    # For world->cam: change world basis => R' = R * A^{-1} = R * A^T
    if nvdiffrast_world_is_z_up and your_world_is_y_up:
        A = np.array([[1, 0,  0],
                      [0, 0,  1],
                      [0, -1, 0]], dtype=np.float32)  # Rx(+90°)
        R = R @ A.T

    # OpenCV/Colmap(x→, y↓, z→) -> GL/Vulkan camera system(x→, y↑, z←)
    S = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
    R_cam = S @ R
    t_cam = S @ t

    view = np.eye(4, dtype=np.float32)
    view[:3, :3] = R_cam
    view[:3, 3] = t_cam.ravel()

    # Projection: flip Y once for Vulkan display compatibility
    proj = proj_from_intrinsics_vulkan(fx, fy, cx, cy, W, H, znear, zfar, flip_y=True)

    # Return as column-major flattened arrays (compatible with glm::mat4)
    return view.T.ravel(), proj.T.ravel()

def create_orbital_camera(frame_num: int, radius: float = 5.0, height: float = 2.0, 
                         center: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create camera parameters for orbital motion around a center point.
    
    Args:
        frame_num: Current frame number (affects orbital position)
        radius: Orbital radius from center
        height: Camera height above center
        center: Center point to orbit around (default: origin)
        
    Returns:
        Tuple of (rotation_matrix, translation_vector)
    """
    if center is None:
        center = np.array([0.0, 0.0, 0.0])
    
    # Orbital angle based on frame
    angle = frame_num * 0.1  # Adjust speed as needed
    
    # Camera position in orbit
    cam_x = center[0] + radius * np.cos(angle)
    cam_y = center[1] + height
    cam_z = center[2] + radius * np.sin(angle)
    
    camera_pos = np.array([cam_x, cam_y, cam_z])
    
    # Look-at matrix calculation
    target = center
    up = np.array([0.0, 1.0, 0.0])
    
    # Forward vector (camera -> target)
    forward = target - camera_pos
    forward = forward / np.linalg.norm(forward)
    
    # Right vector
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    
    # Up vector (recompute for orthogonality)
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)
    
    # Create rotation matrix (world -> camera)
    R = np.array([right, -up, -forward], dtype=np.float32)
    T = -R @ camera_pos.astype(np.float32)
    
    return R, T

def save_depth_png(
    depth: Union[np.ndarray, 'torch.Tensor'],
    filename: str,
    *,
    normalize: bool = True,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    invert: bool = False,
    robust_percentile: float = 0.5,
    bitdepth: int = 16
) -> bool:
    """
    Save single-channel depth data as PNG.
    
    Args:
        depth: Depth data as numpy array or torch tensor
        filename: Output filename
        normalize: Whether to normalize depth to [0,1] range
        min_val, max_val: Manual depth range (overrides auto-calculation)
        invert: Whether to invert depth (1-x) for reversed-Z
        robust_percentile: Percentile for robust range estimation
        bitdepth: PNG bit depth (8 or 16)
        
    Returns:
        True if save was successful, False otherwise
    """
    try:
        # Try to import OpenCV
        try:
            import cv2
            has_cv2 = True
        except ImportError:
            has_cv2 = False
            
        # Convert to numpy
        if hasattr(depth, 'detach'):  # torch tensor
            depth_np = depth.detach().cpu().numpy()
        else:
            depth_np = np.asarray(depth)

        # Squeeze to (H, W)
        if depth_np.ndim == 3:
            if depth_np.shape[0] == 1:
                depth_np = depth_np[0]
            elif depth_np.shape[2] == 1:
                depth_np = depth_np[:, :, 0]
            else:
                raise ValueError(f"Depth tensor must be single-channel; got shape {depth_np.shape}")
        elif depth_np.ndim != 2:
            raise ValueError(f"Depth tensor must be 2D or single-channel 3D; got ndim={depth_np.ndim}")

        depth_np = depth_np.astype(np.float32)

        # Handle NaN/Inf
        finite_mask = np.isfinite(depth_np)
        if not np.any(finite_mask):
            print("Warning: Depth has no finite values.")
            return False

        # Calculate normalization range
        if normalize:
            finite_depth = depth_np[finite_mask]
            if min_val is None or max_val is None:
                if robust_percentile > 0:
                    lo = np.percentile(finite_depth, robust_percentile)
                    hi = np.percentile(finite_depth, 100 - robust_percentile)
                else:
                    lo, hi = np.min(finite_depth), np.max(finite_depth)
                
                if min_val is not None:
                    lo = min_val
                if max_val is not None:
                    hi = max_val
            else:
                lo, hi = min_val, max_val

            # Normalize
            if hi > lo:
                depth_np = np.clip((depth_np - lo) / (hi - lo), 0, 1)
            else:
                depth_np = np.ones_like(depth_np) * 0.5

        # Invert if requested
        if invert:
            depth_np = 1.0 - depth_np

        # Convert to appropriate integer type
        if bitdepth == 16:
            depth_img = (depth_np * 65535).astype(np.uint16)
        else:
            depth_img = (depth_np * 255).astype(np.uint8)

        # Create directory if needed
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)

        # Save using OpenCV if available, otherwise use PIL
        if has_cv2:
            success = cv2.imwrite(filename, depth_img)
            return success
        else:
            try:
                from PIL import Image
                if bitdepth == 16:
                    img = Image.fromarray(depth_img, mode='I;16')
                else:
                    img = Image.fromarray(depth_img, mode='L')
                img.save(filename)
                return True
            except ImportError:
                print("Warning: Neither OpenCV nor PIL available for saving images")
                return False

    except Exception as e:
        print(f"Error saving depth PNG: {e}")
        return False

def make_pitched_cupy_array(device_ptr: int, pitch: int, width: int, height: int, dtype: np.dtype):
    """
    Create a pitched CuPy array from device pointer.
    
    Args:
        device_ptr: CUDA device pointer value
        pitch: Row pitch in bytes
        width: Image width
        height: Image height
        dtype: NumPy data type
        
    Returns:
        CuPy array with proper pitch handling
    """
    try:
        import cupy as cp
        
        # Calculate total size
        total_bytes = pitch * height
        
        # Create memptr from device pointer
        memptr = cp.cuda.MemoryPointer(
            cp.cuda.UnownedMemory(device_ptr, total_bytes, owner=None), 
            offset=0
        )
        
        # Calculate elements per row (accounting for padding)
        dtype_size = np.dtype(dtype).itemsize
        elements_per_row = pitch // dtype_size
        
        # Create pitched array
        pitched_array = cp.ndarray(
            shape=(height, elements_per_row),
            dtype=dtype,
            memptr=memptr
        )
        
        return pitched_array
        
    except ImportError:
        raise RuntimeError("CuPy is required for pitched array creation")