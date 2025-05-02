import open3d as o3d
import numpy as np
import os

# ============================
# Load Centerline Data
# ============================

def load_ply_file(file_path):
    """ Load a PLY file and return its point cloud as a NumPy array. """
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None
    pcd = o3d.io.read_point_cloud(file_path)
    return pcd

# Worst performing centerlines that need MLS smoothing
refined_centerline_files = [
   # "/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/KALMAN & MULIIWAY/centerline_4_0005.000.ply",  # Example refined file after Kalman Filter
    "/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/KALMAN & MULIIWAY/centerline_5_0015.500.ply",
    #"/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/KALMAN & MULIIWAY/centerline_6_0006.500.ply",
    #"/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/KALMAN & MULIIWAY/centerline_7_0029.500.ply",
    #"/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/KALMAN & MULIIWAY/centerline_8_0007.500.ply",
    #"/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/KALMAN & MULIIWAY/centerline_9_0000.000.ply",
]

# ============================
# Apply Moving Least Squares (MLS) Smoothing
# ============================

def apply_mls_smoothing(pcd, search_radius=0.05):
    """ Apply Moving Least Squares (MLS) to refine the centerline. """
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=search_radius, max_nn=30))
    
    # Perform MLS smoothing with projection
    smoothed_pcd = pcd.smooth_point_cloud(voxel_size=0.02, iteration=3)
    
    return smoothed_pcd

    # Perform MLS smoothing
    mls = o3d.geometry.PointCloud()
    mls.points = pcd.points
    mls.estimate_normals()
    
    return mls

# ============================
# Process and Save MLS-Smoothed Centerlines
# ============================

for file in refined_centerline_files:
    print(f"🔄 Processing: {file}")
    
    pcd = load_ply_file(file)
    if pcd is None or len(pcd.points) == 0:
        print(f"❌ Skipping {file}, not enough points.")
        continue
    
    # Apply MLS smoothing
    smoothed_pcd = apply_mls_smoothing(pcd)
    
    # Save the MLS-smoothed centerline
    mls_file_path = file.replace("_refined.ply", "_mls.ply")
    o3d.io.write_point_cloud(mls_file_path, smoothed_pcd)
    print(f"✅ MLS-smoothed centerline saved: {mls_file_path}")

print("🎯 MLS Smoothing Completed!")
