import open3d as o3d
import numpy as np
import os
from scipy.spatial import ConvexHull
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt

# List of PLY file paths
ply_files = [
    r"C:\Users\nadir\OneDrive\Desktop\NIHAR PROJECT\PLY Examples\1_0058.000.ply",
    r"C:\Users\nadir\OneDrive\Desktop\NIHAR PROJECT\PLY Examples\2_0002.500.ply",
    r"C:\Users\nadir\OneDrive\Desktop\NIHAR PROJECT\PLY Examples\3_0000.500.ply",
    r"C:\Users\nadir\OneDrive\Desktop\NIHAR PROJECT\PLY Examples\4_0005.000.ply",
    r"C:\Users\nadir\OneDrive\Desktop\NIHAR PROJECT\PLY Examples\5_0015.500.ply",
    r"C:\Users\nadir\OneDrive\Desktop\NIHAR PROJECT\PLY Examples\6_0006.500.ply",
    r"C:\Users\nadir\OneDrive\Desktop\NIHAR PROJECT\PLY Examples\7_0029.500.ply",
    r"C:\Users\nadir\OneDrive\Desktop\NIHAR PROJECT\PLY Examples\8_0007.500.ply",
    r"C:\Users\nadir\OneDrive\Desktop\NIHAR PROJECT\PLY Examples\9_0000.000.ply",
]

# Function to preprocess the point cloud
def preprocess_point_cloud(pcd, voxel_size=0.02):
    """
    Preprocess the point cloud with statistical outlier removal and downsampling.
    """
    # Remove outliers
    pcd_filtered, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    # Downsample
    return pcd_filtered.voxel_down_sample(voxel_size=voxel_size)

# Function to slice the point cloud along the Z-axis
def slice_point_cloud(pcd, z_interval=0.1):
    """
    Slice the point cloud into sections along the Z-axis.
    """
    points = np.asarray(pcd.points)
    min_z, max_z = points[:, 2].min(), points[:, 2].max()
    slices = []
    for z in np.arange(min_z, max_z, z_interval):
        mask = (points[:, 2] >= z) & (points[:, 2] < z + z_interval)
        slice_points = points[mask]
        if len(slice_points) > 0:
            slices.append(slice_points)
    return slices

# Function to compute centroids for each slice
def compute_centroids(z_slices):
    centroids = []
    for slice_points in z_slices:
        xy_points = slice_points[:, :2]
        if len(xy_points) >= 3:  # ConvexHull requires at least 3 points
            hull = ConvexHull(xy_points)
            boundary_points = xy_points[hull.vertices]
            centroid = np.mean(boundary_points, axis=0)
            centroids.append([centroid[0], centroid[1], np.mean(slice_points[:, 2])])
    return np.array(centroids)

# Function to fit a centerline using a spline
def fit_centerline(centroids, smoothness=500):
    """
    Fit a 3D spline through the centroids.
    """
    tck, u = splprep(centroids.T, s=smoothness)
    u_fine = np.linspace(0, 1, 500)
    return np.array(splev(u_fine, tck)).T

# Process each PLY file
for file in ply_files:
    if not os.path.exists(file):
        print(f"File not found: {file}")
        continue
    
    # Load the point cloud
    pcd = o3d.io.read_point_cloud(file)
    if len(pcd.points) == 0:
        print(f"Skipping empty file: {file}")
        continue
    
    print(f"Processing: {file}")
    
    # Step 1: Preprocess the point cloud
    filtered_pcd = preprocess_point_cloud(pcd)
    
    # Step 2: Slice the point cloud
    z_slices = slice_point_cloud(filtered_pcd, z_interval=0.1)
    
    # Step 3: Compute centroids
    slice_centroids = compute_centroids(z_slices)
    
    # Step 4: Fit the centerline
    if len(slice_centroids) >= 3:  # Ensure there are enough centroids to fit a spline
        centerline = fit_centerline(slice_centroids, smoothness=500)
    else:
        print(f"Insufficient centroids for spline fitting in: {file}")
        continue
    
    # Step 5: Visualize the point cloud and centerline
    centerline_pcd = o3d.geometry.PointCloud()
    centerline_pcd.points = o3d.utility.Vector3dVector(centerline)
    centerline_pcd.paint_uniform_color([1, 0, 0])  # Red for centerline
    
    filtered_pcd.paint_uniform_color([0.7, 0.7, 0.7])  # Grey for point cloud
    o3d.visualization.draw_geometries(
        [filtered_pcd, centerline_pcd], 
        window_name=f"Processed: {os.path.basename(file)}"
    )
    
    # Step 6: Plot the centerline in 3D
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(slice_centroids[:, 0], slice_centroids[:, 1], slice_centroids[:, 2], c="green", label="Centroids")
    ax.plot(centerline[:, 0], centerline[:, 1], centerline[:, 2], c="red", label="Fitted Centerline")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.title(f"Centerline for {os.path.basename(file)}")
    plt.show()

print("Processing complete for all files.")
