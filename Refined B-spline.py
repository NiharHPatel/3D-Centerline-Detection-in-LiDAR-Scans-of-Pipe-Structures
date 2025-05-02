import open3d as o3d
import numpy as np
from scipy.interpolate import splprep, splev
import os

# === Paths ===
input_dir = "/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/Extracted_Centerlines"
output_dir = "/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/Smoothed_Centerlines"
os.makedirs(output_dir, exist_ok=True)

# === Refined B-spline Function ===
def refined_b_spline(points, smoothness=2.0, degree=3, num_points=1000):
    if len(points) < degree + 1:
        return points
    points = points[np.argsort(points[:, 2])]  # Sort by Z
    try:
        tck, _ = splprep(points.T, s=smoothness, k=degree)
        u_fine = np.linspace(0, 1, num_points)
        return np.array(splev(u_fine, tck)).T
    except:
        return points

# === Batch Process All Centerlines ===
ply_files = [f for f in os.listdir(input_dir) if f.endswith(".ply")]

for file_name in ply_files:
    input_path = os.path.join(input_dir, file_name)
    output_path = os.path.join(output_dir, file_name.replace(".ply", "_refined.ply"))

    print(f"Processing: {file_name}")
    pcd = o3d.io.read_point_cloud(input_path)
    points = np.asarray(pcd.points)

    if len(points) < 10:
        print(f" Skipped (not enough points): {file_name}")
        continue

    smoothed_points = refined_b_spline(points, smoothness=2.0)

    # Save refined result
    refined_pcd = o3d.geometry.PointCloud()
    refined_pcd.points = o3d.utility.Vector3dVector(smoothed_points)
    o3d.io.write_point_cloud(output_path, refined_pcd)
    print(f" Saved refined: {output_path}")

print(" All centerlines processed and saved.")
