import open3d as o3d
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.signal import medfilt
import os

# Set the directory where the extracted centerline files are stored
centerline_dir = r"/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/Extracted_Centerlines"

# List all centerline PLY files
centerline_files = sorted([os.path.join(centerline_dir, f) for f in os.listdir(centerline_dir) if f.endswith(".ply")])

# Function to apply MLS smoothing before B-Spline fitting
def apply_mls_smoothing(pcd, search_radius=0.05, polynomial_order=2):
    """Apply Moving Least Squares (MLS) smoothing using Open3D."""
    mls = o3d.geometry.PointCloud()
    mls.points = pcd.points
    mls = o3d.geometry.PointCloud().create_from_point_cloud_poisson(mls, search_radius)
    return mls

# Process each centerline separately
for file in centerline_files:
    print(f"🔹 Processing: {file}")

    # Load centerline
    centerline_pcd = o3d.io.read_point_cloud(file)
    points = np.asarray(centerline_pcd.points)

    # Ensure enough points exist
    if len(points) < 10:
        print(f"❌ Skipping {file}: Not enough points for B-Spline fitting.")
        continue

    # Apply MLS smoothing to reduce noise
    smoothed_pcd = apply_mls_smoothing(centerline_pcd)
    points = np.asarray(smoothed_pcd.points)

    # Sort by Z-axis to maintain order
    sorted_indices = np.argsort(points[:, 2])
    points = points[sorted_indices]

    # Remove duplicates to avoid self-intersections
    points = np.unique(points, axis=0)

    # Extract X, Y, Z coordinates
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    # Apply median filtering to smooth out noise
    x, y, z = medfilt(x, kernel_size=5), medfilt(y, kernel_size=5), medfilt(z, kernel_size=5)

    # Apply B-Spline curve fitting
    try:
        tck, u = splprep([x, y, z], s=1.0, k=3)  # Increased s=1.0 for smoother results
        u_fine = np.linspace(0, 1, len(points) * 5)
        smoothed_points = np.array(splev(u_fine, tck)).T
    except Exception as e:
        print(f"❌ B-Spline Fitting Failed for {file}: {e}")
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

    print(f"✅ Smoothed centerline saved: {smoothed_file}")

print("✅ All centerlines processed separately with MLS + B-Spline smoothing.")
