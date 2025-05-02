import open3d as o3d
import numpy as np
import os

# === Paths ===
input_dir = "/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/Smoothed_Centerlines"
output_dir = "/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/Smoothed_Centerlines"
os.makedirs(output_dir, exist_ok=True)

# === Parameters ===
voxel_size = 0.01  # adjust for finer or coarser smoothing

refined_files = [f for f in os.listdir(input_dir) if f.endswith("_refined.ply")]

for file_name in refined_files:
    input_path = os.path.join(input_dir, file_name)
    output_path = os.path.join(output_dir, file_name.replace("_refined.ply", "_mls.ply"))

    print(f" Simulated MLS Smoothing: {file_name}")
    pcd = o3d.io.read_point_cloud(input_path)

    # Estimate normals (required for MLS-like methods)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
    
    # Downsample + smooth using voxel grid
    pcd_smoothed = pcd.voxel_down_sample(voxel_size=voxel_size)

    o3d.io.write_point_cloud(output_path, pcd_smoothed)
    print(f" Saved: {output_path}")

print(" All files smoothed and saved.")
