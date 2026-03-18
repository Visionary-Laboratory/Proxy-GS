#!/usr/bin/env python3
"""
Pybind11 Timeline Roundtrip Example
端到端时间线信号量握手版本

This example demonstrates the complete zero-copy depth rendering pipeline using:
1. pybind11 extension for direct Vulkan integration (no sockets)
2. Timeline semaphore coordination for frame synchronization  
3. CUDA external memory import for zero-copy depth access
4. Orbital camera motion with 1000 frame capture

Integration Flow:
1. Python creates Vk2TorchApp → Vulkan initializes → signalSceneReady(1)
2. Python imports all FDs → CUDA external memory/semaphore setup
3. Python waits scene_ready ≥ 1 → Vulkan ready for camera data
4. Frame loop: set_camera_matrices() → signal camera_ready=N → wait frame_done=N → decode depth
"""

import os
import sys
import time
import math
import numpy as np



import numpy as np

def _frustum_offcenter_rh_zo(l, r, b, t, n, f):
    # 右手、深度 0..1（与 glm::perspectiveRH_ZO 一致）
    P = np.array([
        [2*n/(r-l),      0.0,      (r+l)/(r-l),          0.0],
        [0.0,        2*n/(t-b),    (t+b)/(t-b),          0.0],
        [0.0,            0.0,          f/(n-f),    (f*n)/(n-f)],
        [0.0,            0.0,            -1.0,          0.0],
    ], dtype=np.float32)
    return P

def proj_from_intrinsics_vulkan(fx, fy, cx, cy, W, H, znear, zfar, *, flip_y=True):
    # 近裁面上的 frustum 边界（相机坐标系 y↑、z 向里前提下）
    l = -znear * (cx)      / fx
    r =  znear * (W - cx)  / fx
    t =  znear * (cy)      / fy
    b = -znear * (H - cy)  / fy
    P = _frustum_offcenter_rh_zo(l, r, b, t, znear, zfar)
    if flip_y:                      # ★ Vulkan 常用：在投影里翻一次 Y
        P[1, :] *= -1.0
    return P

def to_vulkan_viewproj_match_nvdiffrast(
    R_ocv, T_ocv,           # 同一组输入 R,T（world->cam，OpenCV/Colmap 约定）
    fx, fy, cx, cy, W, H, znear, zfar,
    *,
    nvdiffrast_world_is_z_up=True,   # 如果 nv 那边是 Z-up，而你的世界/Y-up，需要做基变换
    your_world_is_y_up=False
):
    R_ocv = np.swapaxes(R_ocv, -1, -2)
    R = np.asarray(R_ocv, np.float32)
    t = np.asarray(T_ocv, np.float32).reshape(3,1)

    # (可选) 基变换：把 Z-up 的世界坐标“翻译”为你这边的 Y-up
    # A = Rx(+90°) : (x, y, z)_nv -> (x, z, -y)_your
    # 对 world->cam：改变“世界基” => R' = R * A^{-1} = R * A^T
    if nvdiffrast_world_is_z_up and your_world_is_y_up:
        A = np.array([[1,0,0],
                      [0,0,1],
                      [0,-1,0]], dtype=np.float32)  # Rx(+90°)
        R = R @ A.T
        # t 不需要绕原点的基变换；如果你的世界原点与 nv 的不一致，再单独处理平移

    # OpenCV/Colmap(x→,y↓,z→) -> GL/Vulkan 相机系(x→,y↑,z里)
    S = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
    R_cam = S @ R
    t_cam = S @ t

    view = np.eye(4, dtype=np.float32)
    view[:3,:3] = R_cam
    view[:3, 3] = t_cam.ravel()

    # 投影：为了与 nvdiffrast（OpenGL 栈）对齐且在 Vulkan 正显，翻一次 Y
    proj = proj_from_intrinsics_vulkan(fx, fy, cx, cy, W, H, znear, zfar, flip_y=True)

    # 以列主序展平给 Vulkan（和 glm::mat4(...) 构造一致）
    return view.T.ravel(), proj.T.ravel()




# Add parent directory to path for vk2torch_cuda import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import cupy as cp
    import torch
    print(f"✅ CuPy {cp.__version__}, PyTorch {torch.__version__}")
except ImportError as e:
    print(f"❌ Required library missing: {e}")
    print("Please install: conda activate vk2torch")
    sys.exit(1)

# Import our modules
try:
    # Add build directory to path for vk2torch_ext
    build_paths = [
        "build-py/_bin/Release",
        # "_bin/Release", 
        # "python/_bin/Release"
    ]
    for path in build_paths:
        if os.path.exists(path):
            sys.path.insert(0, path)
            break
    
    import vk2torch_ext as ext
    from vk2torch_cuda import (
        import_ext_memory_fd, import_timeline_semaphore_fd,
        wait_timeline, signal_timeline,
        make_pitched_cupy_array, depth_d24_to_float
    )
    print("✅ VK2Torch modules imported successfully")
    
except ImportError as e:
    print(f"❌ VK2Torch module import failed: {e}")
    print("Please build the pybind11 extension first:")
    print("  cmake -S . -B build-py -DCMAKE_TOOLCHAIN_FILE=toolchains/system_no_conda.cmake -DBUILD_PYTHON_EXT=ON")
    print("  cmake --build build-py --config Release -j4")
    sys.exit(1)

# Configuration
W, H = 1000, 1000
ASSET_ROOT = os.getcwd()  # Current working directory
N_FRAMES = 20

def make_camera_matrices(frame_num: int) -> tuple:

    R = np.array([[ 0.98822485,  0.11374114, -0.10234546],
                [-0.11979481,  0.99127023, -0.05506844],
                [ 0.09518846,  0.06668046,  0.99322348]], dtype=np.float32)
    T = np.array([-2.80552141, -1.27673587,  3.06543639 + (frame_num-1) * 0.99], dtype=np.float32)


    Fx = 1208.1880959114053
    Fy = 1209.669871748316

    W  = 1000
    H  = 1000
    Cx = W / 2
    Cy = H / 2
    znear, zfar = 0.1, 1000.0

    view_flat, proj_flat = to_vulkan_viewproj_match_nvdiffrast(
        R, T, Fx, Fy, Cx, Cy, W, H, znear, zfar
    )

    
    return proj_flat.reshape([4,4]), view_flat.reshape([4,4])

def main():
    """Main timeline semaphore roundtrip example"""
    
    print("🎬 Pybind11 Timeline Roundtrip Example")
    print("=" * 60)
    
    try:
        # 1) Create Vk2TorchApp with real 3D rendering
        print(f"📱 Creating Vk2TorchApp ({W}x{H}) with asset root: {ASSET_ROOT}")
        app = ext.Vk2TorchApp(W, H, True, "./_downloaded_resources/matrix_city_new_zup.glb", ASSET_ROOT)  # raster=True, no scene override, use asset root
        print("✅ Vk2TorchApp created successfully")
        
        # 2) Get complete interop info with all FDs (duplicated for Python ownership)
        print("📋 Getting interop export information...")
        info = app.get_interop_info()
        print("✅ Interop info retrieved:")
        for key, value in info.items():
            if 'fd' in key:
                print(f"    {key}: {value} (FD)")
            else:
                print(f"    {key}: {value}")
        
        # 3) Import all CUDA resources from file descriptors
        print("🔗 Importing CUDA external resources...")

        time.sleep(1.0)

        # Import external memory and get device pointer

        buffer_size = int(info['height']) * int(info['row_pitch_bytes'])
        ext_mem, dev_ptr = import_ext_memory_fd(int(info['depth_mem_fd']), buffer_size)
        print(f"✅ Depth memory imported: {buffer_size} bytes")
        
        # Import timeline semaphores  
        
        sem_scene = import_timeline_semaphore_fd(int(info['scene_ready_sem_fd']))
        sem_camera = import_timeline_semaphore_fd(int(info['camera_ready_sem_fd']))
        sem_frame = import_timeline_semaphore_fd(int(info['frame_done_sem_fd']))
        # print("✅ Timeline semaphores imported")
        
        # NO need to Close FDs !!!!
        # for fd_key in ['depth_mem_fd', 'scene_ready_sem_fd', 'camera_ready_sem_fd', 'frame_done_sem_fd']:
        #     fd_val = int(info[fd_key])
        #     if fd_val >= 0:
        #         os.close(fd_val)
        
        width = int(info['width'])
        height = int(info['height']) 
        pitch = int(info['row_pitch_bytes'])
        
        # 4) Wait for Vulkan scene ready (scene_ready >= 1)
        print("⏳ Waiting for Vulkan scene initialization...")
        stream = cp.cuda.Stream(non_blocking=True)
        
        # with stream:
        #     wait_timeline(sem_scene, 1, stream.ptr)
        cp.cuda.Stream.null.synchronize()
        # print("✅ Scene ready - Vulkan rendering pipeline initialized")
        
        # 5) Set up CuPy view of depth buffer (pitched for proper row alignment)
        # print(f"🖼️  Creating depth buffer view: {width}x{height}, pitch={pitch}")
        
        # Create pitched array view (full width including padding)
        u32_pitched = make_pitched_cupy_array(dev_ptr.value, pitch, width, height, np.uint32)
        
        # Slice to actual image dimensions (remove row padding)
        u32 = u32_pitched[:, :width] 
        _ = depth_d24_to_float(u32)
    
        os.makedirs("out_depth", exist_ok=True)    
        
        app.headless_init()
        # compile operator
        
        cp.cuda.Stream.null.synchronize()
        time.sleep(1.0)
        start_time = time.time()
        frame_times = []
        
        for frame_num in range(1, N_FRAMES + 1):
            frame_start = time.time()
            
            # 6.1) Generate camera matrices for orbital motion
            proj_matrix, view_matrix = make_camera_matrices(frame_num)
            
            # 6.2) Set camera matrices via pybind11 (triggers camera ready signal internally)
            app.set_camera_matrices(proj_matrix, view_matrix)
            
            
            app.headless_step()
            # time.sleep(1.0)
            

            
            # 6.5) Process and save depth frame
            depth_float = depth_d24_to_float(u32)  # Convert D24 to float32 [0,1]



            frame_end = time.time()
            frame_time_ms = (frame_end - frame_start) * 1000
            frame_times.append(frame_time_ms)
            # print(f"Frame {frame_num:4d}: {frame_time_ms:6.2f}ms | depth shape: {depth_float.shape}")
            
            # Save selected frames
            if frame_num % 100 == 0 or frame_num <= 10 or frame_num > N_FRAMES - 10:
                # Copy to CPU for saving
                depth_cpu = cp.asnumpy(depth_float)
                save_depth_png(depth_cpu,f"out_depth/depth_{frame_num:04d}.png")
                np.save(f"out_depth/depth_{frame_num:04d}.npy", depth_cpu)
                
            #     # Calculate statistics
            #     valid_mask = depth_cpu > 0.0
            #     if np.any(valid_mask):
            #         min_depth = np.min(depth_cpu[valid_mask])
            #         max_depth = np.max(depth_cpu[valid_mask])
            #         mean_depth = np.mean(depth_cpu[valid_mask])
            #         print(f"📸 Frame {frame_num:4d}: depth range [{min_depth:.3f}, {max_depth:.3f}], mean={mean_depth:.3f} - saved")
            #     else:
            #         print(f"📸 Frame {frame_num:4d}: no valid depth data - saved")
            
            # Progress indicator with frame timing stats
            if frame_num % 50 == 0:
                elapsed = time.time() - start_time
                fps = frame_num / elapsed
                eta = (N_FRAMES - frame_num) / fps if fps > 0 else 0
                
                # Calculate frame timing statistics
                recent_frames = frame_times[-50:] if len(frame_times) >= 50 else frame_times
                avg_frame_time = np.mean(recent_frames)
                min_frame_time = np.min(recent_frames)
                max_frame_time = np.max(recent_frames)
                
                print(f"⏱️  Progress: {frame_num}/{N_FRAMES} ({100*frame_num/N_FRAMES:.1f}%) - {fps:.1f} FPS - ETA: {eta:.1f}s")
                print(f"    Frame timing (last {len(recent_frames)}): avg={avg_frame_time:.1f}ms, min={min_frame_time:.1f}ms, max={max_frame_time:.1f}ms")
        
        # Final synchronization
        cp.cuda.Stream.null.synchronize()
        
        # 7) Summary and cleanup
        total_time = time.time() - start_time
        avg_fps = N_FRAMES / total_time
        
        

        # Calculate overall frame timing statistics
        if frame_times:
            avg_frame_time = np.mean(frame_times)
            min_frame_time = np.min(frame_times)
            max_frame_time = np.max(frame_times)
            p50_frame_time = np.percentile(frame_times, 50)
            p95_frame_time = np.percentile(frame_times, 95)
            p99_frame_time = np.percentile(frame_times, 99)
        
        print("=" * 60)
        print("🎉 Timeline Roundtrip Complete!")
        print(f"📊 Performance Summary:")
        print(f"    • Total frames: {N_FRAMES}")
        print(f"    • Total time: {total_time:.2f} seconds")
        print(f"    • Average FPS: {avg_fps:.1f}")
        print(f"📏 Frame Timing Statistics:")
        if frame_times:
            print(f"    • Average: {avg_frame_time:.2f}ms ({1000/avg_frame_time:.1f} FPS)")
            print(f"    • Minimum: {min_frame_time:.2f}ms ({1000/min_frame_time:.1f} FPS)")
            print(f"    • Maximum: {max_frame_time:.2f}ms ({1000/max_frame_time:.1f} FPS)")
            print(f"    • Median (P50): {p50_frame_time:.2f}ms")
            print(f"    • P95: {p95_frame_time:.2f}ms")
            print(f"    • P99: {p99_frame_time:.2f}ms")
        else:
            print("    • No frame timing data available")
        print(f"📁 Depth frames saved to: out_depth/")
        print(f"    • Saved frames: {len([f for f in os.listdir('out_depth') if f.endswith('.npy')])}")
        
        # Stop application
        print("🛑 Stopping Vk2TorchApp...")
        app.stop()
        print("✅ Application stopped")
        
    except Exception as e:
        print(f"❌ Timeline roundtrip failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def save_depth_png(
    depth,                  # torch.Tensor 或 numpy.ndarray，形状可为 (H,W), (1,H,W), (H,W,1)
    filename: str,
    *,
    normalize: bool = True,
    min_val: float | None = None,
    max_val: float | None = None,
    invert: bool = False,           # 若你用 reversed-Z（近=大），想让“近更亮”，可设 True
    robust_percentile: float = 0.5, # 百分位裁剪，0.5 表示 [0.5%, 99.5%]
    bitdepth: int = 16              # 16 或 8；建议 16
) -> bool:
    """
    将单通道深度保存为 PNG。返回 True/False 表示是否保存成功。
    - normalize=True 时：用 (min,max) 将深度线性映射到 [0,1]（会先按百分位裁剪减少异常值影响）。
    - min_val/max_val 可手动覆盖自动范围。
    - invert=True 则做 1 - x（常用于 reversed-Z: 近=大，想让近=亮）。
    - bitdepth: 16（推荐）或 8。
    """
    try:
        import numpy as np
        # 尽量用已存在的环境变量/对象
        has_cv = globals().get("HAS_OPENCV", False)
        if has_cv:
            import cv2
        else:
            cv2 = None

        # 1) 拿到 numpy，去掉多余维度
        if "torch" in str(type(depth)):  # 粗略判断是否为 torch.Tensor
            # 防止梯度/显存问题
            depth_np = depth.detach().to("cpu").numpy()
        else:
            depth_np = np.asarray(depth)

        # squeeze 到 (H,W)
        if depth_np.ndim == 3:
            # 允许 [1,H,W] 或 [H,W,1]
            if depth_np.shape[0] == 1:
                depth_np = depth_np[0]
            elif depth_np.shape[2] == 1:
                depth_np = depth_np[:, :, 0]
            else:
                raise ValueError(f"Depth tensor must be single-channel; got shape {depth_np.shape}")
        elif depth_np.ndim != 2:
            raise ValueError(f"Depth tensor must be 2D or single-channel 3D; got ndim={depth_np.ndim}")

        depth_np = np.asanyarray(depth_np).astype(np.float32, copy=False)

        # 2) 处理 NaN/Inf
        finite_mask = np.isfinite(depth_np)
        if not np.any(finite_mask):
            if 'logger' in globals():
                logger.error("Depth has no finite values.")
            return False

        # 3) 计算归一化范围
        lo, hi = (min_val, max_val)
        if normalize:
            vals = depth_np[finite_mask]
            # 百分位裁剪（减少极端值影响）
            p = float(robust_percentile)
            if lo is None:
                lo = np.percentile(vals, p) if p > 0 else float(np.min(vals))
            if hi is None:
                hi = np.percentile(vals, 100.0 - p) if p > 0 else float(np.max(vals))
        else:
            # 不做归一化就直接 clamp 到 [0,1]
            lo = 0.0 if lo is None else lo
            hi = 1.0 if hi is None else hi

        # 防止 lo==hi
        if hi <= lo:
            # 退化情况：全图近似常数
            if 'logger' in globals():
                logger.warning(f"Depth range collapsed (lo={lo}, hi={hi}), producing a constant image.")
            norm = np.zeros_like(depth_np, dtype=np.float32)
        else:
            norm = (depth_np - lo) / (hi - lo)

        # 4) 反转（常用于 reversed-Z：近=大 → 近更亮）
        if invert:
            norm = 1.0 - norm

        # 5) 裁剪到 [0,1]，并将非有限值置 0
        norm[~finite_mask] = 0.0
        norm = np.clip(norm, 0.0, 1.0)

        # 6) 转整数并保存
        if bitdepth == 16:
            img = np.round(norm * 65535.0).astype(np.uint16)
        elif bitdepth == 8:
            img = np.round(norm * 255.0).astype(np.uint8)
        else:
            raise ValueError("bitdepth must be 8 or 16")

        # OpenCV 保存（shape 必须是 (H,W)）
        if has_cv and cv2 is not None:
            ok = cv2.imwrite(filename, img)
            if ok:
                if 'logger' in globals():
                    logger.info(f"Saved depth to {filename} ({bitdepth}-bit PNG)")
                return True
        else:
            # PIL 兜底
            try:
                from PIL import Image
                if bitdepth == 16:
                    pil_img = Image.fromarray(img, mode="I;16")
                else:
                    pil_img = Image.fromarray(img, mode="L")
                pil_img.save(filename)
                if 'logger' in globals():
                    logger.info(f"Saved depth to {filename} using PIL ({bitdepth}-bit)")
                return True
            except ImportError:
                if 'logger' in globals():
                    logger.error("No image library available (OpenCV or PIL)")
                return False

        return False

    except Exception as e:
        if 'logger' in globals():
            logger.error(f"Failed to save depth PNG: {e}")
        return False



if __name__ == "__main__":
    print("VK2Torch Pybind11 Timeline Roundtrip")
    print("零拷贝深度渲染时间线信号量端到端测试")
    print("=" * 80)
    
    # Verify environment
    print("🔍 Environment check:")
    
    # Check CUDA availability
    if not cp.cuda.is_available():
        print("❌ CUDA not available")
        sys.exit(1)
    print(f"✅ CUDA available - device count: {cp.cuda.runtime.getDeviceCount()}")
    
    # Check PyTorch CUDA
    if not torch.cuda.is_available():
        print("❌ PyTorch CUDA not available")
        sys.exit(1)
    print(f"✅ PyTorch CUDA available - device: {torch.cuda.get_device_name()}")
    
    print()
    main()
    
