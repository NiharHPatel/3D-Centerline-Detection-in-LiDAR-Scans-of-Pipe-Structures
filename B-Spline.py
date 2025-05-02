import open3d as o3d
import numpy as np
from scipy.interpolate import splprep, splev
import os

# Set the directory where the extracted centerline files are stored
centerline_dir = r"/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/Extracted_Centerlines"

# List all centerline PLY files
centerline_files = sorted([os.path.join(centerline_dir, f) for f in os.listdir(centerline_dir) if f.endswith(".ply")])

# Process each centerline separately
for file in centerline_files:
    print(f"🔹 Processing: {file}")

    # Load centerline
    centerline_pcd = o3d.io.read_point_cloud(file)
    points = np.asarray(centerline_pcd.points) 

    # Ensure enough points exist
    if len(points) < 10:
        print(f" Skipping {file}: Not enough points for B-Spline fitting.")
        continue

    # **Improved Sorting Step**
    # Sort by Z first, then by X-Y to ensure proper continuity
    sorted_indices = np.lexsort((points[:, 0], points[:, 1], points[:, 2]))
    points = points[sorted_indices]

    # Remove duplicates (helps prevent looping issues)
    points = np.unique(points, axis=0)

    # Extract X, Y, Z coordinates
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    # Apply B-Spline curve fitting
    try:
        tck, u = splprep([x, y, z], s=0.2, k=3)  # Increased smoothing (s=0.2)
        u_fine = np.linspace(0, 1, len(points) * 5)
        smoothed_points = np.array(splev(u_fine, tck)).T
    except Exception as e:
        print(f" B-Spline Fitting Failed for {file}: {e}")
        continue

    # Convert smoothed centerline to Open3D PointCloud
    smoothed_pcd = o3d.geometry.PointCloud()
    smoothed_pcd.points = o3d.utility.Vector3dVector(smoothed_points)
    smoothed_pcd.paint_uniform_color([1, 0, 0])  # Red for visualization

    # Save smoothed centerline
    smoothed_file = file.replace(".ply", "_smoothed.ply")
    o3d.io.write_point_cloud(smoothed_file, smoothed_pcd)

    # Visualize smoothed centerline
    o3d.visualization.draw_geometries([smoothed_pcd])

    print(f" Smoothed centerline saved: {smoothed_file}")

print(" All centerlines processed separately.")
