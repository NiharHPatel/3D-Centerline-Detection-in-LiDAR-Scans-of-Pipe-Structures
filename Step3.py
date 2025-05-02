import open3d as o3d
import numpy as np
from scipy.spatial import ConvexHull
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
import os
from sklearn.cluster import DBSCAN
from matplotlib.widgets import Slider

# List of PLY file paths
ply_files = [
    r"/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/PLY Examples/1_0058.000.ply",
    r"/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/PLY Examples/2_0002.500.ply",
    r"/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/PLY Examples/3_0000.500.ply",
    r"/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/PLY Examples/4_0005.000.ply",
    r"/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/PLY Examples/5_0015.500.ply",
    r"/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/PLY Examples/6_0006.500.ply",
    r"/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/PLY Examples/7_0029.500.ply",
    r"/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/PLY Examples/8_0007.500.ply",
    r"/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/PLY Examples/9_0000.000.ply"
    
]

# Default parameters
default_smooth_factor = 500
default_cluster_eps = 1.0
default_cluster_min_samples = 3

def preprocess_point_cloud(pcd, voxel_size=0.02, nb_neighbors=20, std_ratio=2.0):
    """Preprocess the point cloud by downsampling and removing noise."""
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    cl, ind = pcd_down.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return pcd_down.select_by_index(ind)

def estimate_normals(pcd):
    """Compute normal vectors for the point cloud."""
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    return pcd

def slice_point_cloud(pcd, z_interval=0.1):
    """Slice the point cloud into sections along the Z-axis."""
    points = np.asarray(pcd.points)
    min_z, max_z = points[:, 2].min(), points[:, 2].max()
    slices = []
    for z in np.arange(min_z, max_z, z_interval):
        mask = (points[:, 2] >= z) & (points[:, 2] < z + z_interval)
        slice_points = points[mask]
        if len(slice_points) > 0:
            slices.append(slice_points)
    return slices

def compute_centroids(z_slices, eps=default_cluster_eps, min_samples=default_cluster_min_samples):
    """Compute centroids for each Z-slice and apply DBSCAN clustering."""
    centroids = []
    for slice_points in z_slices:
        xy_points = slice_points[:, :2]
        if len(xy_points) >= 3:
            hull = ConvexHull(xy_points)
            boundary_points = xy_points[hull.vertices]
            centroid = np.mean(boundary_points, axis=0)
            centroids.append([centroid[0], centroid[1], np.mean(slice_points[:, 2])])

    centroids = np.array(centroids)

    # Apply DBSCAN clustering to remove noise
    if len(centroids) > 5:
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(centroids)
        centroids = centroids[clustering.labels_ != -1]  # Remove outliers

    return centroids

def adaptive_spline_smoothing(centroids, smooth_factor=default_smooth_factor):
    """Dynamically adjust spline smoothing based on pipe curvature."""
    if len(centroids) < 3:
        return None  # Not enough points to fit a spline

    tck, u = splprep(centroids.T, s=smooth_factor)
    u_fine = np.linspace(0, 1, 500)
    return np.array(splev(u_fine, tck)).T

def visualize_interactive(file):
    """Interactive visualization with Open3D and Matplotlib sliders."""
    global default_smooth_factor, default_cluster_eps, default_cluster_min_samples

    pcd = o3d.io.read_point_cloud(file)
    pcd_filtered = preprocess_point_cloud(pcd)
    pcd_filtered = estimate_normals(pcd_filtered)

    z_slices = slice_point_cloud(pcd_filtered, z_interval=0.1)
    centroids = compute_centroids(z_slices, eps=default_cluster_eps, min_samples=default_cluster_min_samples)

    if len(centroids) < 3:
        print(f"Insufficient centroids for spline fitting in file: {file}")
        return

    centerline = adaptive_spline_smoothing(centroids, smooth_factor=default_smooth_factor)

    if centerline is None:
        print(f"Skipping spline fitting due to insufficient centroids in file: {file}")
        return

    # Open3D Visualization
    centerline_pcd = o3d.geometry.PointCloud()
    centerline_pcd.points = o3d.utility.Vector3dVector(centerline)
    centerline_pcd.paint_uniform_color([1, 0, 0])  # Red for centerline

    pcd_filtered.paint_uniform_color([0.7, 0.7, 0.7])  # Grey for pipe
    o3d.visualization.draw_geometries([pcd_filtered, centerline_pcd], window_name=f"Curved Pipe Centerline - {os.path.basename(file)}")

    # Matplotlib Interactive Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(left=0.25, bottom=0.25)

    scatter = ax.scatter(centroids[:, 0], centroids[:, 1], c="green", label="Centroids")
    line, = ax.plot(centerline[:, 0], centerline[:, 1], c="red", label="Fitted Centerline")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.title(f"3D Centerline for {os.path.basename(file)}")
    plt.legend()
    plt.show()

# Run the interactive visualization for each PLY file
for file in ply_files:
    visualize_interactive(file)

print("Processing complete.")
