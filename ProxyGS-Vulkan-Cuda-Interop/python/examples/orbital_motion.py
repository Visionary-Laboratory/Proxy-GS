#!/usr/bin/env python3
"""
Orbital Motion VK2Torch Example

Demonstrates smooth orbital camera motion around a 3D scene,
similar to the original pybind_timeline_roundtrip.py but with
a clean modular interface.
"""

import os
import sys
import time
import numpy as np
from pathlib import Path

# Add parent directory to import our modules
sys.path.append(str(Path(__file__).parent.parent))

from vk2torch_renderer import VK2TorchRenderer
from vk2torch_utils import save_depth_png

def create_smooth_orbital_path(num_frames: int, 
                               radius: float = 5.0, 
                               height_variation: float = 1.0,
                               base_height: float = 2.0) -> list:
    """
    Create smooth orbital camera parameters with height variation.
    
    Args:
        num_frames: Total number of frames
        radius: Orbital radius from center
        height_variation: Amount of height variation (0 = constant height)
        base_height: Base camera height
        
    Returns:
        List of camera parameter dictionaries
    """
    camera_params = []
    
    for frame in range(num_frames):
        # Orbital angle
        angle = (frame / num_frames) * 2 * np.pi
        
        # Position with smooth height variation
        height = base_height + height_variation * np.sin(angle * 2)  # 2 cycles per orbit
        
        # Camera position
        cam_x = radius * np.cos(angle)
        cam_y = height
        cam_z = radius * np.sin(angle)
        
        camera_pos = np.array([cam_x, cam_y, cam_z])
        target = np.array([0.0, 0.0, 0.0])  # Always look at origin
        up = np.array([0.0, 1.0, 0.0])
        
        # Create look-at matrix
        forward = target - camera_pos
        forward = forward / np.linalg.norm(forward)
        
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        
        up_corrected = np.cross(right, forward)
        
        # Rotation matrix (world to camera)
        R = np.array([right, -up_corrected, -forward], dtype=np.float32)
        T = -R @ camera_pos.astype(np.float32)
        
        camera_params.append({
            'camera_R': R,
            'camera_T': T,
            'frame_info': {
                'frame': frame,
                'angle_deg': np.degrees(angle),
                'position': camera_pos.tolist(),
                'height': height
            }
        })
    
    return camera_params

def main():
    """Orbital motion rendering example."""
    print("🌍 Orbital Motion VK2Torch Example")
    print("=" * 50)
    
    # Configuration
    width, height = 1000, 1000
    scene_file = "house_new.glb"
    asset_root = os.getcwd()
    output_dir = "orbital_motion_output"
    
    # Orbital motion parameters
    num_frames = 30
    orbital_radius = 6.0
    base_height = 2.5
    height_variation = 1.5
    
    # Camera intrinsics
    fx = fy = 1200.0  # Focal length
    cx = width / 2
    cy = height / 2
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        print(f"📋 Configuration:")
        print(f"  Render size: {width}x{height}")
        print(f"  Scene: {scene_file}")
        print(f"  Frames: {num_frames}")
        print(f"  Orbital radius: {orbital_radius}")
        print(f"  Height: {base_height} ± {height_variation}")
        
        # Create renderer
        with VK2TorchRenderer(
            width=width,
            height=height,
            scene_file=scene_file,
            asset_root=asset_root
        ) as renderer:
            
            # Create orbital camera path
            print("\n🔄 Generating orbital camera path...")
            camera_params = create_smooth_orbital_path(
                num_frames, orbital_radius, height_variation, base_height
            )
            
            # Render frames with timing
            print(f"\n🎬 Rendering {num_frames} frames...")
            
            depths = []
            frame_times = []
            start_time = time.time()
            
            for i, params in enumerate(camera_params):
                frame_start = time.time()
                
                # Render frame
                depth = renderer.render(
                    camera_R=params['camera_R'],
                    camera_T=params['camera_T'],
                    fx=fx, fy=fy, cx=cx, cy=cy,
                    return_cpu=True
                )
                
                frame_end = time.time()
                frame_time = (frame_end - frame_start) * 1000  # ms
                frame_times.append(frame_time)
                
                depths.append(depth)
                
                # Save frame
                frame_filename = f"frame_{i:04d}.png"
                frame_path = os.path.join(output_dir, frame_filename)
                save_depth_png(depth, frame_path)
                
                # Also save raw data for selected frames
                if i % 5 == 0 or i < 3 or i >= num_frames - 3:
                    raw_filename = f"frame_{i:04d}.npy"
                    raw_path = os.path.join(output_dir, raw_filename)
                    np.save(raw_path, depth)
                
                # Progress update
                info = params['frame_info']
                print(f"  Frame {i+1:3d}/{num_frames}: "
                      f"{frame_time:6.2f}ms | "
                      f"angle={info['angle_deg']:6.1f}° | "
                      f"height={info['height']:5.2f} | "
                      f"depth_range=[{depth.min():.3f}, {depth.max():.3f}]")
                
                # Progress indicator every 10 frames
                if i > 0 and (i + 1) % 10 == 0:
                    elapsed = time.time() - start_time
                    avg_fps = (i + 1) / elapsed
                    eta = (num_frames - i - 1) / avg_fps if avg_fps > 0 else 0
                    print(f"    Progress: {(i+1)/num_frames*100:.1f}% | "
                          f"Avg FPS: {avg_fps:.1f} | ETA: {eta:.1f}s")
            
            # Final statistics
            total_time = time.time() - start_time
            avg_fps = num_frames / total_time
            
            frame_times = np.array(frame_times)
            avg_frame_time = np.mean(frame_times)
            min_frame_time = np.min(frame_times)
            max_frame_time = np.max(frame_times)
            
            print(f"\n📊 Rendering Statistics:")
            print(f"  Total time: {total_time:.2f} seconds")
            print(f"  Average FPS: {avg_fps:.1f}")
            print(f"  Frame timing:")
            print(f"    Average: {avg_frame_time:.2f}ms")
            print(f"    Range: [{min_frame_time:.2f}, {max_frame_time:.2f}]ms")
            
            # Create info file
            info_file = os.path.join(output_dir, "render_info.txt")
            with open(info_file, 'w') as f:
                f.write("Orbital Motion Rendering Info\n")
                f.write("=" * 30 + "\n\n")
                f.write(f"Scene: {scene_file}\n")
                f.write(f"Resolution: {width}x{height}\n")
                f.write(f"Frames: {num_frames}\n")
                f.write(f"Orbital radius: {orbital_radius}\n")
                f.write(f"Height: {base_height} ± {height_variation}\n")
                f.write(f"Camera: fx={fx}, fy={fy}\n\n")
                f.write(f"Performance:\n")
                f.write(f"  Total time: {total_time:.2f}s\n")
                f.write(f"  Average FPS: {avg_fps:.1f}\n")
                f.write(f"  Avg frame time: {avg_frame_time:.2f}ms\n")
                f.write(f"  Frame time range: [{min_frame_time:.2f}, {max_frame_time:.2f}]ms\n\n")
                f.write("Frame Details:\n")
                for i, params in enumerate(camera_params):
                    info = params['frame_info']
                    f.write(f"  Frame {i:3d}: angle={info['angle_deg']:6.1f}°, "
                           f"height={info['height']:5.2f}, time={frame_times[i]:6.2f}ms\n")
            
            print(f"\n🎉 Orbital motion complete!")
            print(f"📁 Files saved to: {output_dir}")
            print(f"  {num_frames} depth images (PNG)")
            print(f"  {len([f for f in os.listdir(output_dir) if f.endswith('.npy')])} raw depth files (NPY)")
            print(f"  1 info file (TXT)")
            print(f"\n💡 To create a video from the frames, you can use:")
            print(f"  ffmpeg -framerate 10 -i {output_dir}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p orbital_motion.mp4")
            
    except Exception as e:
        print(f"❌ Orbital motion rendering failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())