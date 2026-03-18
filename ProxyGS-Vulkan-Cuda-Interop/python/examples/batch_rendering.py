#!/usr/bin/env python3
"""
Batch Rendering VK2Torch Example

Demonstrates efficient batch processing of multiple camera configurations
for dataset generation, camera calibration, or large-scale rendering tasks.
"""

import os
import sys
import time
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to import our modules
sys.path.append(str(Path(__file__).parent.parent))

from vk2torch_renderer import VK2TorchRenderer
from vk2torch_utils import save_depth_png

def generate_random_cameras(num_cameras: int,
                          center: np.ndarray = None,
                          radius_range: tuple = (3.0, 8.0),
                          height_range: tuple = (0.5, 4.0),
                          seed: int = 42) -> List[Dict[str, Any]]:
    """
    Generate random camera positions around a scene.
    
    Args:
        num_cameras: Number of camera poses to generate
        center: Scene center point
        radius_range: (min_radius, max_radius) from center
        height_range: (min_height, max_height) camera positions
        seed: Random seed for reproducibility
        
    Returns:
        List of camera parameter dictionaries
    """
    if center is None:
        center = np.array([0.0, 0.0, 0.0])
    
    np.random.seed(seed)
    camera_params = []
    
    for i in range(num_cameras):
        # Random spherical coordinates
        radius = np.random.uniform(radius_range[0], radius_range[1])
        theta = np.random.uniform(0, 2 * np.pi)  # Azimuth
        phi = np.random.uniform(0.3, np.pi - 0.3)  # Elevation (avoid top/bottom poles)
        
        # Convert to Cartesian coordinates
        x = radius * np.sin(phi) * np.cos(theta)
        z = radius * np.sin(phi) * np.sin(theta)
        y = radius * np.cos(phi)
        
        # Adjust height to specified range
        y = np.random.uniform(height_range[0], height_range[1])
        
        camera_pos = center + np.array([x, y, z])
        target = center  # Always look at scene center
        up = np.array([0.0, 1.0, 0.0])
        
        # Create look-at matrix
        forward = target - camera_pos
        forward = forward / np.linalg.norm(forward)
        
        right = np.cross(forward, up)
        if np.linalg.norm(right) > 1e-6:
            right = right / np.linalg.norm(right)
        else:
            # Handle degenerate case (camera pointing straight up/down)
            right = np.array([1.0, 0.0, 0.0])
        
        up_corrected = np.cross(right, forward)
        
        # Rotation matrix (world to camera)
        R = np.array([right, -up_corrected, -forward], dtype=np.float32)
        T = -R @ camera_pos.astype(np.float32)
        
        # Random camera intrinsics variations
        base_focal = 1000.0
        focal_variation = np.random.uniform(0.8, 1.2)
        fx = fy = base_focal * focal_variation
        
        camera_params.append({
            'camera_R': R,
            'camera_T': T,
            'fx': fx,
            'fy': fy,
            'metadata': {
                'id': i,
                'position': camera_pos.tolist(),
                'target': target.tolist(),
                'radius': float(radius),
                'theta_deg': float(np.degrees(theta)),
                'phi_deg': float(np.degrees(phi)),
                'focal_length': float(fx)
            }
        })
    
    return camera_params

def generate_grid_cameras(grid_size: tuple = (5, 5),
                         radius: float = 5.0,
                         height_range: tuple = (1.0, 3.0)) -> List[Dict[str, Any]]:
    """
    Generate cameras in a regular grid pattern around the scene.
    
    Args:
        grid_size: (azimuth_steps, elevation_steps)
        radius: Distance from scene center
        height_range: (min_height, max_height)
        
    Returns:
        List of camera parameter dictionaries
    """
    camera_params = []
    az_steps, el_steps = grid_size
    
    camera_id = 0
    for i in range(az_steps):
        azimuth = (i / az_steps) * 2 * np.pi
        
        for j in range(el_steps):
            # Height interpolation
            height = height_range[0] + (j / (el_steps - 1)) * (height_range[1] - height_range[0])
            
            # Camera position
            x = radius * np.cos(azimuth)
            z = radius * np.sin(azimuth)
            y = height
            
            camera_pos = np.array([x, y, z])
            target = np.array([0.0, 0.0, 0.0])
            up = np.array([0.0, 1.0, 0.0])
            
            # Create look-at matrix
            forward = target - camera_pos
            forward = forward / np.linalg.norm(forward)
            
            right = np.cross(forward, up)
            right = right / np.linalg.norm(right)
            
            up_corrected = np.cross(right, forward)
            
            # Rotation matrix
            R = np.array([right, -up_corrected, -forward], dtype=np.float32)
            T = -R @ camera_pos.astype(np.float32)
            
            camera_params.append({
                'camera_R': R,
                'camera_T': T,
                'fx': 1000.0,
                'fy': 1000.0,
                'metadata': {
                    'id': camera_id,
                    'grid_pos': [i, j],
                    'position': camera_pos.tolist(),
                    'azimuth_deg': float(np.degrees(azimuth)),
                    'height': float(height)
                }
            })
            camera_id += 1
    
    return camera_params

def save_batch_results(depths: List[np.ndarray],
                      camera_params: List[Dict],
                      output_dir: str,
                      *,
                      save_images: bool = True,
                      save_raw: bool = True,
                      save_metadata: bool = True) -> None:
    """Save batch rendering results with metadata."""
    
    if save_images:
        image_dir = os.path.join(output_dir, "images")
        os.makedirs(image_dir, exist_ok=True)
        
        for i, depth in enumerate(depths):
            filename = f"depth_{i:04d}.png"
            filepath = os.path.join(image_dir, filename)
            save_depth_png(depth, filepath)
    
    if save_raw:
        raw_dir = os.path.join(output_dir, "raw")
        os.makedirs(raw_dir, exist_ok=True)
        
        for i, depth in enumerate(depths):
            filename = f"depth_{i:04d}.npy"
            filepath = os.path.join(raw_dir, filename)
            np.save(filepath, depth)
    
    if save_metadata:
        # Convert numpy arrays to lists for JSON serialization
        serializable_params = []
        for params in camera_params:
            serializable = {
                'camera_R': params['camera_R'].tolist(),
                'camera_T': params['camera_T'].tolist(),
                'fx': float(params['fx']),
                'fy': float(params['fy']),
                'metadata': params['metadata']
            }
            serializable_params.append(serializable)
        
        metadata_file = os.path.join(output_dir, "cameras.json")
        with open(metadata_file, 'w') as f:
            json.dump({
                'num_cameras': len(camera_params),
                'cameras': serializable_params
            }, f, indent=2)

def main():
    """Batch rendering example."""
    print("📦 Batch Rendering VK2Torch Example")
    print("=" * 50)
    
    # Configuration
    width, height = 800, 800  # Smaller for faster batch processing
    scene_file = "house_new.glb"
    asset_root = os.getcwd()
    output_base = "batch_rendering_output"
    
    try:
        # Create renderer
        with VK2TorchRenderer(
            width=width,
            height=height,
            scene_file=scene_file,
            asset_root=asset_root
        ) as renderer:
            
            print(f"Renderer initialized: {renderer.get_info()}")
            
            # Example 1: Random camera positions
            print("\n🎲 Batch 1: Random camera positions")
            output_dir1 = os.path.join(output_base, "random_cameras")
            os.makedirs(output_dir1, exist_ok=True)
            
            random_cameras = generate_random_cameras(
                num_cameras=20,
                radius_range=(4.0, 7.0),
                height_range=(1.0, 4.0)
            )
            
            print(f"Generated {len(random_cameras)} random cameras")
            
            start_time = time.time()
            random_depths = renderer.render_batch(
                random_cameras,
                return_cpu=True,
                show_progress=True
            )
            batch1_time = time.time() - start_time
            
            save_batch_results(random_depths, random_cameras, output_dir1)
            print(f"✅ Random batch complete: {batch1_time:.2f}s, "
                  f"{len(random_cameras)/batch1_time:.1f} FPS")
            
            # Example 2: Grid pattern cameras
            print("\n🔲 Batch 2: Grid pattern cameras")
            output_dir2 = os.path.join(output_base, "grid_cameras")
            os.makedirs(output_dir2, exist_ok=True)
            
            grid_cameras = generate_grid_cameras(
                grid_size=(6, 4),  # 6 azimuth × 4 elevation = 24 cameras
                radius=5.5,
                height_range=(0.5, 3.5)
            )
            
            print(f"Generated {len(grid_cameras)} grid cameras")
            
            start_time = time.time()
            grid_depths = renderer.render_batch(
                grid_cameras,
                return_cpu=True,
                show_progress=True
            )
            batch2_time = time.time() - start_time
            
            save_batch_results(grid_depths, grid_cameras, output_dir2)
            print(f"✅ Grid batch complete: {batch2_time:.2f}s, "
                  f"{len(grid_cameras)/batch2_time:.1f} FPS")
            
            # Example 3: Mixed batch (different focal lengths)
            print("\n🔍 Batch 3: Mixed focal lengths")
            output_dir3 = os.path.join(output_base, "mixed_focal")
            os.makedirs(output_dir3, exist_ok=True)
            
            mixed_cameras = []
            focal_lengths = [600, 800, 1000, 1200, 1500]  # Different focal lengths
            
            for i, focal in enumerate(focal_lengths):
                # Same position, different focal length
                angle = (i / len(focal_lengths)) * 2 * np.pi
                radius = 6.0
                
                camera_pos = np.array([
                    radius * np.cos(angle),
                    2.0,
                    radius * np.sin(angle)
                ])
                
                target = np.array([0.0, 0.0, 0.0])
                up = np.array([0.0, 1.0, 0.0])
                
                forward = target - camera_pos
                forward = forward / np.linalg.norm(forward)
                
                right = np.cross(forward, up)
                right = right / np.linalg.norm(right)
                
                up_corrected = np.cross(right, forward)
                
                R = np.array([right, -up_corrected, -forward], dtype=np.float32)
                T = -R @ camera_pos.astype(np.float32)
                
                mixed_cameras.append({
                    'camera_R': R,
                    'camera_T': T,
                    'fx': float(focal),
                    'fy': float(focal),
                    'metadata': {
                        'id': i,
                        'focal_length': focal,
                        'position': camera_pos.tolist(),
                        'angle_deg': float(np.degrees(angle))
                    }
                })
            
            start_time = time.time()
            mixed_depths = renderer.render_batch(
                mixed_cameras,
                return_cpu=True,
                show_progress=True
            )
            batch3_time = time.time() - start_time
            
            save_batch_results(mixed_depths, mixed_cameras, output_dir3)
            print(f"✅ Mixed batch complete: {batch3_time:.2f}s, "
                  f"{len(mixed_cameras)/batch3_time:.1f} FPS")
            
            # Summary
            total_frames = len(random_cameras) + len(grid_cameras) + len(mixed_cameras)
            total_time = batch1_time + batch2_time + batch3_time
            overall_fps = total_frames / total_time
            
            print(f"\n📊 Batch Rendering Summary:")
            print(f"  Total frames rendered: {total_frames}")
            print(f"  Total time: {total_time:.2f}s")
            print(f"  Overall FPS: {overall_fps:.1f}")
            print(f"  Batch performance:")
            print(f"    Random cameras: {len(random_cameras)} frames, {len(random_cameras)/batch1_time:.1f} FPS")
            print(f"    Grid cameras: {len(grid_cameras)} frames, {len(grid_cameras)/batch2_time:.1f} FPS")
            print(f"    Mixed cameras: {len(mixed_cameras)} frames, {len(mixed_cameras)/batch3_time:.1f} FPS")
            
            print(f"\n📁 Output saved to: {output_base}")
            print(f"  Each batch includes: PNG images, NPY raw data, JSON metadata")
            
            # Create overall summary
            summary_file = os.path.join(output_base, "batch_summary.json")
            summary_data = {
                'total_frames': total_frames,
                'total_time_seconds': total_time,
                'overall_fps': overall_fps,
                'batches': {
                    'random_cameras': {
                        'count': len(random_cameras),
                        'time_seconds': batch1_time,
                        'fps': len(random_cameras) / batch1_time
                    },
                    'grid_cameras': {
                        'count': len(grid_cameras),
                        'time_seconds': batch2_time,
                        'fps': len(grid_cameras) / batch2_time
                    },
                    'mixed_focal': {
                        'count': len(mixed_cameras),
                        'time_seconds': batch3_time,
                        'fps': len(mixed_cameras) / batch3_time
                    }
                },
                'configuration': {
                    'resolution': [width, height],
                    'scene_file': scene_file
                }
            }
            
            with open(summary_file, 'w') as f:
                json.dump(summary_data, f, indent=2)
            
            print(f"📄 Summary saved to: batch_summary.json")
            
    except Exception as e:
        print(f"❌ Batch rendering failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())