#!/usr/bin/env python3
"""
VK2Torch CUDA Pipeline Usage Example
Demonstrates how to use vk2torch_ext with vk2torch_cuda for zero-copy GPU operations

This example shows:
1. Setting up Vulkan rendering with camera override
2. Exporting depth buffers via external memory
3. Importing to CUDA and creating PyTorch tensors
4. Timeline semaphore synchronization
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add paths for modules
ext_dir = Path(__file__).parent.parent / "_bin" / "Release"
python_dir = Path(__file__).parent
sys.path.insert(0, str(ext_dir))
sys.path.insert(0, str(python_dir))

def create_view_projection_matrices(distance=3.0, yaw=0.0, pitch=0.0):
    """Create view and projection matrices for camera positioning"""
    
    # Camera position
    x = distance * np.cos(pitch) * np.sin(yaw)
    y = distance * np.sin(pitch)  
    z = distance * np.cos(pitch) * np.cos(yaw)
    camera_pos = np.array([x, y, z], dtype=np.float32)
    
    # Look at origin
    target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    
    # Create view matrix
    forward = target - camera_pos
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    up_corrected = np.cross(right, forward)
    
    view_matrix = np.eye(4, dtype=np.float32)
    view_matrix[0, :3] = right
    view_matrix[1, :3] = up_corrected
    view_matrix[2, :3] = -forward
    view_matrix[:3, 3] = -np.dot(view_matrix[:3, :3], camera_pos)
    
    # Create projection matrix
    fov = np.radians(45.0)
    aspect = 1.0
    near = 0.1
    far = 100.0
    
    f = 1.0 / np.tan(fov / 2.0)
    proj_matrix = np.zeros((4, 4), dtype=np.float32)
    proj_matrix[0, 0] = f / aspect
    proj_matrix[1, 1] = f
    proj_matrix[2, 2] = (far + near) / (near - far)
    proj_matrix[2, 3] = (2.0 * far * near) / (near - far)
    proj_matrix[3, 2] = -1.0
    
    return view_matrix, proj_matrix

def example_basic_usage():
    """Basic usage example without full Vulkan context"""
    
    print("=" * 60)
    print("📝 VK2Torch CUDA Basic Usage Example")
    print("=" * 60)
    
    try:
        import vk2torch_cuda
        import cupy as cp
        import torch
        
        print("✅ All modules imported successfully")
        print(f"✅ CUDA available: {vk2torch_cuda.test_cuda_availability()}")
        print(f"✅ CUDA driver: {vk2torch_cuda.get_cuda_driver_version()}")
        
        # Example: D24 depth buffer processing
        print("\n🔍 Example: D24 Depth Buffer Processing")
        
        # Simulate D24 depth data (24-bit depth in upper bits of uint32)
        width, height = 256, 256
        # Create random depth values, shifted to upper 24 bits
        depth_24bit = np.random.randint(0, 0xFFFFFF, (height, width), dtype=np.uint32)
        depth_d24 = depth_24bit << 8  # Shift to upper 24 bits (D24 format)
        
        # Convert to CuPy array (simulate GPU memory)
        depth_d24_gpu = cp.array(depth_d24)
        print(f"📊 D24 buffer: {depth_d24_gpu.shape} {depth_d24_gpu.dtype}")
        
        # Convert D24 to float32
        depth_float = vk2torch_cuda.depth_d24_to_float(depth_d24_gpu)
        print(f"📊 Float depth: {depth_float.shape} {depth_float.dtype}")
        print(f"📊 Depth range: {depth_float.min():.6f} to {depth_float.max():.6f}")
        
        # Convert to PyTorch tensor (zero-copy)
        depth_tensor = vk2torch_cuda.to_torch(depth_float)
        print(f"📊 PyTorch tensor: {depth_tensor.shape} on {depth_tensor.device}")
        
        # Verify zero-copy
        if depth_float.data.ptr == depth_tensor.data_ptr():
            print("✅ Zero-copy conversion confirmed")
        
        # Example tensor operations
        depth_stats = {
            'min': depth_tensor.min().item(),
            'max': depth_tensor.max().item(), 
            'mean': depth_tensor.mean().item(),
            'std': depth_tensor.std().item()
        }
        
        print("📊 Depth statistics:")
        for key, value in depth_stats.items():
            print(f"   {key}: {value:.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic usage example failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def example_camera_matrices():
    """Example of camera matrix creation for different viewpoints"""
    
    print("\n" + "=" * 60)
    print("📹 Camera Matrix Creation Example")
    print("=" * 60)
    
    try:
        # Different camera positions
        viewpoints = [
            ("Front View", 3.0, 0.0, 0.0),
            ("Side View", 3.5, np.pi/2, 0.0),
            ("Top-Down", 4.0, 0.0, np.pi/3),
            ("Angled", 3.2, np.pi/4, np.pi/6),
        ]
        
        for name, distance, yaw, pitch in viewpoints:
            view_matrix, proj_matrix = create_view_projection_matrices(distance, yaw, pitch)
            
            # Extract camera position from view matrix
            camera_pos = -view_matrix[:3, :3].T @ view_matrix[:3, 3]
            
            print(f"\n📹 {name}:")
            print(f"   Distance: {distance}")
            print(f"   Yaw: {np.degrees(yaw):.1f}°, Pitch: {np.degrees(pitch):.1f}°")
            print(f"   Camera Position: ({camera_pos[0]:.2f}, {camera_pos[1]:.2f}, {camera_pos[2]:.2f})")
            print(f"   View Matrix Shape: {view_matrix.shape}")
            print(f"   Projection Matrix Shape: {proj_matrix.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Camera matrix example failed: {e}")
        return False

def example_integration_overview():
    """Overview of complete integration pipeline"""
    
    print("\n" + "=" * 60)
    print("🔄 Complete Integration Pipeline Overview")
    print("=" * 60)
    
    print("""
📋 Complete VK2Torch Integration Steps:

1. 🎮 Vulkan Rendering Setup:
   • Create Vk2TorchApp with scene and dimensions
   • Set camera matrices using set_camera()
   • Render frames with external memory export
   
2. 🔗 External Memory Export:
   • Vulkan creates exportable depth buffer
   • Timeline semaphore for frame synchronization
   • File descriptors passed for zero-copy access
   
3. 🚀 CUDA Import:
   • Import external memory via file descriptor
   • Import timeline semaphore for synchronization
   • Map GPU memory for direct access
   
4. 🧮 Data Processing:
   • Convert D24 depth to float32 format
   • Create CuPy arrays with custom row pitch
   • Convert to PyTorch tensors via DLPack
   
5. 📊 Analysis & ML:
   • Zero-copy tensor operations
   • Direct GPU memory access
   • No CPU roundtrips for performance

🔧 Key Functions:
   • vk2torch_cuda.create_zero_copy_depth_tensor() - Complete pipeline
   • vk2torch_cuda.import_ext_memory_fd() - Memory import
   • vk2torch_cuda.wait_timeline() - Synchronization
   • vk2torch_cuda.to_torch() - Zero-copy tensor creation

💡 Usage Pattern:
   ```python
   # Set up Vulkan rendering
   app = vk2torch_ext.Vk2TorchApp(width, height, ...)
   app.set_camera(frame_num, view_matrix, proj_matrix)
   
   # Get export info
   export_info = app.get_depth_export_info()
   
   # Create zero-copy tensor
   depth_tensor, ext_mem = vk2torch_cuda.create_zero_copy_depth_tensor(
       export_info['depth_fd'], width, height, pitch
   )
   
   # Process tensor directly on GPU
   result = torch.nn.functional.conv2d(depth_tensor.unsqueeze(0), kernel)
   ```
""")
    
    return True

def main():
    """Run all examples"""
    
    print("🎯 VK2Torch CUDA Integration Examples")
    
    examples = [
        ("Basic CUDA Usage", example_basic_usage),
        ("Camera Matrix Creation", example_camera_matrices),
        ("Integration Overview", example_integration_overview),
    ]
    
    results = []
    for name, example_func in examples:
        try:
            result = example_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ Example {name} failed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 Examples Summary")
    print("=" * 60)
    
    for name, result in results:
        status = "✅ SUCCESS" if result else "❌ FAILED"
        print(f"{name}: {status}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"\nOverall: {passed}/{total} examples completed successfully")
    
    if passed == total:
        print("\n🎉 All examples completed! Ready to use VK2Torch CUDA integration.")
    
    print("\n💡 Next Steps:")
    print("1. Use depth_roundtrip.py for complete end-to-end testing")
    print("2. Integrate with your own Vulkan applications")
    print("3. Build ML/AI pipelines with zero-copy GPU tensors")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)