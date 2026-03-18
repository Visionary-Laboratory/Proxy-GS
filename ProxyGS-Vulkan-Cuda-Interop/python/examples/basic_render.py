#!/usr/bin/env python3
"""
Basic VK2Torch Rendering Example

Demonstrates simple scene rendering with custom camera parameters.
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add parent directory to import our modules
sys.path.append(str(Path(__file__).parent.parent))

from vk2torch_renderer import VK2TorchRenderer
from vk2torch_utils import save_depth_png




def main():
    """Basic rendering example."""
    print("🎬 Basic VK2Torch Rendering Example")
    print("=" * 50)
    
    # Configuration
    width, height = 1000, 1000
    scene_file = "house_new.glb"  # Relative to asset root
    asset_root = os.getcwd()  # Current working directory
    output_dir = "basic_render_output"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Create renderer
        with VK2TorchRenderer(
            width=width,
            height=height, 
            scene_file=scene_file,
            asset_root=asset_root
        ) as renderer:
            
            print(f"Renderer info: {renderer.get_info()}")
            
            # Example 1: Simple camera positioned in front of scene
            print("\n📸 Rendering with fixed camera...")
            
            # Define camera parameters
            # Camera positioned at (0, 2, 5) looking at origin
            camera_R = np.array([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0], 
                [0.0, 0.0, 1.0]
            ], dtype=np.float32)
            
            camera_T = np.array([0.0, -2.0, -5.0], dtype=np.float32)
            
            # Camera intrinsics
            fx = fy = 1000.0  # Focal length
            cx = width / 2    # Principal point X
            cy = height / 2   # Principal point Y
            
            # Render
            depth = renderer.render(
                camera_R=camera_R,
                camera_T=camera_T,
                fx=fx, fy=fy, cx=cx, cy=cy,
                return_cpu=True
            )
            
            print(f"Rendered depth shape: {depth.shape}")
            print(f"Depth range: [{depth.min():.3f}, {depth.max():.3f}]")
            
            # Save depth image
            depth_file = os.path.join(output_dir, "depth_fixed.png")
            if save_depth_png(depth, depth_file):
                print(f"✅ Depth saved to: {depth_file}")
            
            # Save raw depth data
            np.save(os.path.join(output_dir, "depth_fixed.npy"), depth)
            
            # Example 2: Multiple viewpoints around the scene
            print("\n🔄 Rendering multiple viewpoints...")
            
            viewpoints = [
                # (distance, angle_degrees, height)
                (5.0, 0, 2.0),      # Front
                (5.0, 90, 2.0),     # Right side
                (5.0, 180, 2.0),    # Back
                (5.0, 270, 2.0),    # Left side
                (3.0, 45, 4.0),     # Close, elevated
                (8.0, 30, 1.0),     # Far, low
            ]
            
            for i, (distance, angle_deg, height) in enumerate(viewpoints):
                # Convert to camera position
                angle_rad = np.radians(angle_deg)
                cam_x = distance * np.cos(angle_rad)
                cam_z = distance * np.sin(angle_rad)
                cam_y = height
                
                # Camera position
                camera_pos = np.array([cam_x, cam_y, cam_z])
                target = np.array([0.0, 0.0, 0.0])  # Look at origin
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
                
                # Render
                depth = renderer.render(
                    camera_R=R, camera_T=T,
                    fx=fx, fy=fy, cx=cx, cy=cy,
                    return_cpu=True
                )
                
                # Save
                filename = f"depth_view_{i:02d}_{angle_deg:03d}deg.png"
                depth_file = os.path.join(output_dir, filename)
                save_depth_png(depth, depth_file)
                
                print(f"  View {i+1}/{len(viewpoints)}: {angle_deg}° - saved to {filename}")
            
            # Example 3: Using orbital motion helper
            print("\n🌍 Rendering orbital sequence...")
            
            orbital_depths = renderer.create_orbital_sequence(
                num_frames=8,
                radius=6.0,
                height=3.0,
                fx=1200,  # Slightly longer focal length
                return_cpu=True
            )
            
            for i, depth in enumerate(orbital_depths):
                filename = f"depth_orbital_{i:03d}.png"
                depth_file = os.path.join(output_dir, filename)
                save_depth_png(depth, depth_file)
            
            print(f"✅ Saved {len(orbital_depths)} orbital frames")
            
            print(f"\n🎉 Basic rendering complete!")
            print(f"📁 All outputs saved to: {output_dir}")
            print(f"📊 Total files: {len(os.listdir(output_dir))}")
            
    except Exception as e:
        print(f"❌ Rendering failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())