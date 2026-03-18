#!/usr/bin/env python3
"""
VK2Torch Renderer Module

A reusable Python class for rendering 3D scenes using the VK2Torch extension.
Provides zero-copy depth buffer access via CUDA integration.
"""

import os
import sys
import time
import numpy as np
from typing import Tuple, Union, Optional, List
from pathlib import Path

# Add parent directory to path for module imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir.parent))

try:
    import cupy as cp
    import torch
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False
    cp = None
    torch = None

# Import VK2Torch modules
try:
    # Add build directory to path
    build_paths = [
        "./build-py/_bin/Release",
        "./_bin/Release",
    ]
    for path in build_paths:
        full_path = current_dir.parent / path
        if full_path.exists():
            sys.path.insert(0, str(full_path))
            break
    
    import vk2torch_ext as ext
    from vk2torch_cuda import (
        import_ext_memory_fd, import_timeline_semaphore_fd,
        make_pitched_cupy_array, depth_d24_to_float
    )
    HAS_VK2TORCH = True
except ImportError as e:
    HAS_VK2TORCH = False
    ext = None

def depth01_to_linear(depth01, znear, zfar):
    # # 适用于 Vulkan/D3D 的 0..1 深度（非 reversed-Z）
    # return (znear * zfar) / (zfar - depth01 * (zfar - znear))
    mask_inf = (depth01 == 1.0)
    depth01 = 2.0 * (depth01 - 0.5)
    depth01 = (2.0 * znear * zfar) / (zfar + znear - depth01 * (zfar - znear))
    depth01.masked_fill_(mask_inf, torch.inf)
    return depth01


# --- 放在文件顶层（import torch 后） ---
try:
    import torch

    _D24_MASK = torch.iinfo(torch.uint32).max >> 8  # 0x00FFFFFF = 16777215

    def _d24_to_linear(u32_depth: torch.Tensor, znear: float, zfar: float) -> torch.Tensor:
        # u32_depth: torch.uint32，形状与贴图一致（可带 pitch/stride）
        d24 = u32_depth & _D24_MASK
        # 直接在 Torch 里做 UNORM 映射
        depth01 = d24.to(torch.float32) * (1.0 / 16777215.0)
        # 1.0 代表“无穷远/无命中”视作 inf
        mask_inf = d24 == _D24_MASK
        x = depth01 * 2.0 - 1.0
        denom = (zfar + znear) - x * (zfar - znear)
        out = (2.0 * znear * zfar) / denom
        return torch.where(mask_inf, torch.full_like(out, float("inf")), out)

    d24_to_linear_compiled = torch.compile(
        _d24_to_linear, fullgraph=True, dynamic=False, mode="max-autotune"
    )
except Exception:
    d24_to_linear_compiled = None



# ---- add: compiled decoder for D24 on int32 ----
try:
    import torch
    def _d24i_to_linear(i32_depth: torch.Tensor, znear: float, zfar: float) -> torch.Tensor:
        # i32_depth: torch.int32 视图（由 D24S8 的 uint32 重解释而来）
        d24 = torch.bitwise_and(i32_depth, 0x00FFFFFF)          # 取低 24 位
        depth01 = d24.to(torch.float32) * (1.0 / 16777215.0)    # UNORM24 -> [0,1]
        mask_inf = d24 == 0x00FFFFFF                            # 1.0 表示“无命中”
        x = depth01 * 2.0 - 1.0
        denom = (zfar + znear) - x * (zfar - znear)
        out = (2.0 * znear * zfar) / denom
        return torch.where(mask_inf, torch.full_like(out, float("inf")), out)

    d24i_to_linear_compiled = torch.compile(
        _d24i_to_linear, fullgraph=True, dynamic=False, mode="max-autotune"
    )
except Exception:
    d24i_to_linear_compiled = None



# Import utilities
from vk2torch_utils import (
    to_vulkan_viewproj_match_nvdiffrast, 
    create_orbital_camera,
    save_depth_png,
    make_pitched_cupy_array as util_make_pitched_array
)

class VK2TorchRenderer:
    """
    High-level renderer class for VK2Torch integration.
    
    Provides an easy-to-use interface for rendering 3D scenes with programmable
    camera control and zero-copy depth buffer access.
    """
    
    def __init__(self, 
                 width: int, 
                 height: int,
                 scene_file: Union[str, Path],
                 asset_root: Union[str, Path], 
                 ):
        """
        Initialize the VK2Torch renderer.
        
        Args:
            width, height: Render target dimensions
            scene_file: Path to scene file (relative to asset_root)
            asset_root: Root directory for assets (shaders, models, etc.)
        """
        # Validate dependencies
        if not HAS_CUDA:
            raise RuntimeError("CUDA libraries (CuPy, PyTorch) are required")
        if not HAS_VK2TORCH:
            raise RuntimeError("VK2Torch extension not found. Please build with: "
                             "cmake -S . -B build-py -DBUILD_PYTHON_EXT=ON && "
                             "cmake --build build-py --config Release")
        
        # Store configuration
        self.width = width
        self.height = height
        self.scene_file = Path(scene_file)
        self.asset_root = Path(asset_root)
        
        # State tracking
        self._app = None
        self._initialized = False
        self._cuda_resources_ready = False
        self._frame_count = 0
        
        # CUDA resources (initialized lazily)
        self._ext_mem = None
        self._dev_ptr = None
        self._sem_scene = None
        self._sem_camera = None
        self._sem_frame = None
        self._depth_array = None
        
        # Interop info
        self._interop_info = None
        
        print(f"VK2TorchRenderer initialized: {width}x{height}, scene: {scene_file}")
    
        if not self._initialized:
            self._init_vulkan_app()
            
        if not self._cuda_resources_ready:
            self._init_cuda_resources()
            
        # Initialize Vulkan app for headless rendering
 
        self._app.headless_init()
        # Compile depth conversion operator
        _ = depth_d24_to_float(self._depth_array)
        cp.cuda.Stream.null.synchronize()
        time.sleep(0.1)  # Small delay for initialization

    def _init_vulkan_app(self):
        """Initialize the Vulkan application."""
        if self._app is not None:
            return
            
        print(f"Creating Vulkan app with scene: {self.scene_file}")
        print(f"Asset root: {self.asset_root}")
        
        # Create Vk2TorchApp
        self._app = ext.Vk2TorchApp(
            self.width, 
            self.height, 
            True,
            str(self.scene_file),
            str(self.asset_root)
        )
        
        print("✅ Vulkan app created successfully")
        self._initialized = True
    
    def _init_cuda_resources(self):
        """Initialize CUDA external memory resources."""
        if self._cuda_resources_ready or self._app is None:
            return
            
        print("Initializing CUDA resources...")
        
        # Get interop info
        self._interop_info = self._app.get_interop_info()
        time.sleep(1.0)
        # Import external memory
        buffer_size = int(self._interop_info['height']) * int(self._interop_info['row_pitch_bytes'])
        self._ext_mem, self._dev_ptr = import_ext_memory_fd(
            int(self._interop_info['depth_mem_fd']), 
            buffer_size
        )
        
        # Import timeline semaphores
        self._sem_scene = import_timeline_semaphore_fd(int(self._interop_info['scene_ready_sem_fd']))
        self._sem_camera = import_timeline_semaphore_fd(int(self._interop_info['camera_ready_sem_fd']))
        self._sem_frame = import_timeline_semaphore_fd(int(self._interop_info['frame_done_sem_fd']))
        
        # Create depth buffer view
        width = int(self._interop_info['width'])
        height = int(self._interop_info['height'])
        pitch = int(self._interop_info['row_pitch_bytes'])
        
        # Create pitched array
        try:
            # Try using the imported function first
            u32_pitched = make_pitched_cupy_array(
                self._dev_ptr.value, pitch, width, height, np.uint32
            )
        except:
            # Fallback to utility function
            u32_pitched = util_make_pitched_array(
                self._dev_ptr.value, pitch, width, height, np.uint32
            )
        
        # Slice to actual image dimensions
        self._depth_array = u32_pitched[:, :width]
        
        print(f"✅ CUDA resources initialized: {width}x{height}, pitch={pitch}")
        self._cuda_resources_ready = True
    
    def render(self,
              camera_R: Optional[np.ndarray] = None,
              camera_T: Optional[np.ndarray] = None,
              *,
              fx: Optional[float] = None,
              fy: Optional[float] = None,
              cx: Optional[float] = None,
              cy: Optional[float] = None,
              znear: Optional[float] = 0.1,
              zfar: Optional[float] = 1000
              ):
        """
        Render a frame with the specified camera parameters.
        
        Args:
            camera_R: 3x3 rotation matrix (world -> camera)
            camera_T: 3-element translation vector (world -> camera)
            fx, fy: Camera focal lengths (pixels)
            cx, cy: Camera principal point (pixels)
            orbital_frame: If provided, use orbital motion instead of R/T
            orbital_radius: Radius for orbital motion
            orbital_height: Height for orbital motion
            return_cpu: Return result as NumPy array (CPU) instead of CuPy (GPU)
            
        Returns:
            Depth buffer as CuPy array (GPU) or NumPy array (CPU)
        """
        # Initialize if needed

        
            
        # Convert to Vulkan matrices
        view_flat, proj_flat = to_vulkan_viewproj_match_nvdiffrast(
            camera_R, camera_T, fx, fy, cx, cy, 
            self.width, self.height, znear, zfar
        )
        
        proj_matrix = proj_flat.reshape([4, 4])
        view_matrix = view_flat.reshape([4, 4])
        
        # Set camera matrices and render
        self._app.set_camera_matrices(proj_matrix, view_matrix)
        self._app.headless_step()
        
        # Convert depth buffer
        # depth_float = depth_d24_to_float(self._depth_array)
        
        self._frame_count += 1
        
        # 从 CuPy 的 uint32 视图成 int32（零拷贝重解释），再交给 Torch
        i32_view = self._depth_array.view(cp.int32)                 # CuPy int32 视图（仍指向外部显存）
        i32_torch_ext = torch.utils.dlpack.from_dlpack(i32_view)    # Torch 张量，但存储仍然是“外来的”

        # 关键：强制把存储变成 Torch 自己分配的 CUDA 显存（避免 Triton 指针校验失败）
        # 任选其一：
        i32_torch = i32_torch_ext.contiguous()                      # 常用写法，分配+拷贝（GPU 内）
        # i32_torch = i32_torch_ext.to(torch.int32, device="cuda", copy=True)  # 等价写法

        # 断言一下（调试用）
        # assert i32_torch.is_cuda and i32_torch.dtype == torch.int32
        # assert i32_torch.stride() == (i32_torch.shape[1], 1)  # 应该已变成连续布局

        # 编译核：D24S8 -> 线性深度
        if d24i_to_linear_compiled is not None:
            depth_torch = d24i_to_linear_compiled(i32_torch, znear, zfar)
        else:
            depth_torch = _d24i_to_linear(i32_torch, znear, zfar)


        return depth_torch

    
    def render_batch(self,
                    camera_params_list: List[dict],
                    *,
                    return_cpu: bool = False,
                    show_progress: bool = True) -> List[Union[cp.ndarray, np.ndarray]]:
        """
        Render multiple frames with different camera parameters.
        
        Args:
            camera_params_list: List of dictionaries with camera parameters
            return_cpu: Return results as NumPy arrays instead of CuPy
            show_progress: Show progress during batch rendering
            
        Returns:
            List of depth arrays
        """
        results = []
        total_frames = len(camera_params_list)
        
        start_time = time.time()
        
        for i, params in enumerate(camera_params_list):
            depth = self.render(return_cpu=return_cpu, **params)
            results.append(depth)
            
            if show_progress and (i % 10 == 0 or i == total_frames - 1):
                elapsed = time.time() - start_time
                fps = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (total_frames - i - 1) / fps if fps > 0 else 0
                progress = (i + 1) / total_frames * 100
                print(f"Progress: {i+1}/{total_frames} ({progress:.1f}%) - "
                      f"{fps:.1f} FPS - ETA: {eta:.1f}s")
        
        return results
    
    def get_depth_cpu(self) -> np.ndarray:
        """Get the current depth buffer as a CPU NumPy array."""
        if not self._cuda_resources_ready:
            raise RuntimeError("CUDA resources not initialized. Call render() first.")
        
        depth_float = depth_d24_to_float(self._depth_array)
        return cp.asnumpy(depth_float)
    
    def save_depth(self, 
                   filename: Union[str, Path],
                   *,
                   normalize: bool = True,
                   format: str = 'png') -> bool:
        """
        Save the current depth buffer to file.
        
        Args:
            filename: Output filename
            normalize: Normalize depth values to [0,1] range
            format: Output format ('png' or 'npy')
            
        Returns:
            True if save was successful
        """
        if not self._cuda_resources_ready:
            raise RuntimeError("CUDA resources not initialized. Call render() first.")
        
        depth_cpu = self.get_depth_cpu()
        
        if format.lower() == 'npy':
            np.save(filename, depth_cpu)
            return True
        else:
            return save_depth_png(depth_cpu, str(filename), normalize=normalize)
    
    def create_orbital_sequence(self,
                              num_frames: int,
                              *,
                              radius: float = 5.0,
                              height: float = 2.0,
                              fx: Optional[float] = None,
                              fy: Optional[float] = None,
                              return_cpu: bool = False) -> List[Union[cp.ndarray, np.ndarray]]:
        """
        Create an orbital camera sequence.
        
        Args:
            num_frames: Number of frames to render
            radius: Orbital radius
            height: Camera height
            fx, fy: Camera focal lengths
            return_cpu: Return as NumPy arrays
            
        Returns:
            List of depth arrays
        """
        camera_params = []
        
        for frame in range(num_frames):
            params = {
                'orbital_frame': frame,
                'orbital_radius': radius,
                'orbital_height': height
            }
            
            if fx is not None:
                params['fx'] = fx
            if fy is not None:
                params['fy'] = fy
                
            camera_params.append(params)
        
        return self.render_batch(camera_params, return_cpu=return_cpu)
    
    def get_info(self) -> dict:
        """Get information about the renderer and current state."""
        return {
            'width': self.width,
            'height': self.height,
            'scene_file': str(self.scene_file),
            'asset_root': str(self.asset_root),
            'initialized': self._initialized,
            'cuda_ready': self._cuda_resources_ready,
            'frame_count': self._frame_count,
            'interop_info': self._interop_info
        }
    
    def close(self):
        """Clean up resources."""
        if self._app is not None:
            try:
                self._app.stop()
            except:
                pass
            self._app = None
        
        # Reset state
        self._initialized = False
        self._cuda_resources_ready = False
        self._frame_count = 0
        
        # Clear CUDA resources
        self._ext_mem = None
        self._dev_ptr = None
        self._sem_scene = None
        self._sem_camera = None
        self._sem_frame = None
        self._depth_array = None
        
        print("VK2TorchRenderer resources cleaned up")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def __del__(self):
        """Destructor - ensure cleanup."""
        try:
            self.close()
        except:
            pass


# Convenience functions for quick usage
def quick_render(scene_file: str,
                asset_root: str = ".",
                *,
                width: int = 1000,
                height: int = 1000,
                orbital_frames: int = 10,
                save_dir: str = "output") -> List[np.ndarray]:
    """
    Quick rendering function for simple use cases.
    
    Args:
        scene_file: Path to scene file
        asset_root: Asset root directory
        width, height: Render dimensions
        orbital_frames: Number of orbital motion frames
        save_dir: Directory to save depth images
        
    Returns:
        List of depth arrays
    """
    os.makedirs(save_dir, exist_ok=True)
    
    with VK2TorchRenderer(width, height, scene_file, asset_root) as renderer:
        depths = renderer.create_orbital_sequence(
            orbital_frames, return_cpu=True
        )
        
        # Save depth images
        for i, depth in enumerate(depths):
            filename = os.path.join(save_dir, f"depth_{i:04d}.png")
            save_depth_png(depth, filename)
        
        print(f"Saved {len(depths)} depth images to {save_dir}")
        return depths


