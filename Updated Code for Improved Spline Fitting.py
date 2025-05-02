import open3d as o3d
import numpy as np
from scipy.spatial import ConvexHull
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
import os

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

def preprocess_point_cloud(pcd, voxel_size=0.02, nb_neighbors=20, std_ratio=2.0):
    """
    Preprocess the point cloud by downsampling and removing noise.
    """
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    cl, ind = pcd_down.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return pcd_down.select_by_index(ind)

def estimate_normals(pcd):
    """
    Compute normal vectors for the point cloud.
    """
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    return pcd

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

def compute_centroids(z_slices):
    """
    Compute centroids for each Z-slice.
    """
    centroids = []
    for slice_points in z_slices:
        xy_points = slice_points[:, :2]
        if len(xy_points) >= 3:
            hull = ConvexHull(xy_points)
            boundary_points = xy_points[hull.vertices]
            centroid = np.mean(boundary_points, axis=0)
            centroids.append([centroid[0], centroid[1], np.mean(slice_points[:, 2])])
    return np.array(centroids)

def adaptive_smooth_factor(centroids):
    """
    Compute an adaptive smoothing factor based on centroid spacing and curvature.
    """
    if len(centroids) < 5:
        return 10  # Low smoothing for very small datasets

    # Compute curvature as local variation in centroid distances
    diffs = np.diff(centroids, axis=0)
    curvature = np.linalg.norm(diffs, axis=1).mean()
    
    # Scale smoothing dynamically based on curvature
    return max(10, min(500, curvature * 50))

def fit_centerline(centroids):
    """
    Fit a 3D spline through the centroids with adaptive smoothing.
    """
    smooth_factor = adaptive_smooth_factor(centroids)
    tck, u = splprep(centroids.T, s=smooth_factor)
    u_fine = np.linspace(0, 1, 500)
    return np.array(splev(u_fine, tck)).T

def compute_deviation(centerline, pipe_pcd):
    """
    Compute the deviation of detected centerline from the actual pipe structure.
    """
    pipe_points = np.asarray(pipe_pcd.points)
    distances = np.linalg.norm(pipe_points[:, None, :] - centerline[None, :, :], axis=2).min(axis=1)
    return np.mean(distances)

# Process each PLY file
for file in ply_files:
    if not os.path.exists(file):
        print(f"File not found: {file}")
        continue

    print(f"Processing: {file}")
    
    pcd = o3d.io.read_point_cloud(file)
    
    if len(pcd.points) == 0:
        print(f"Skipping empty file: {file}")
        continue
    
    pcd_filtered = preprocess_point_cloud(pcd)
    pcd_filtered = estimate_normals(pcd_filtered)

    z_slices = slice_point_cloud(pcd_filtered, z_interval=0.1)
    centroids = compute_centroids(z_slices)

    if len(centroids) < 3:
        print(f"Insufficient centroids for spline fitting in file: {file}")
        continue

    centerline = fit_centerline(centroids)
    
    deviation = compute_deviation(centerline, pcd_filtered)
    print(f"Mean deviation for {file}: {deviation:.4f}")

    # Visualization
    centerline_pcd = o3d.geometry.PointCloud()
    centerline_pcd.points = o3d.utility.Vector3dVector(centerline)
    centerline_pcd.paint_uniform_color([1, 0, 0])  # Red for centerline

    pcd_filtered.paint_uniform_color([0.7, 0.7, 0.7])  # Grey for pipe
    o3d.visualization.draw_geometries([pcd_filtered, centerline_pcd], window_name=f"Refined 3D Centerline - {os.path.basename(file)}")

    # Plot centerline with centroids
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c="green", label="Centroids")
    ax.plot(centerline[:, 0], centerline[:, 1], centerline[:, 2], c="red", label="Fitted Centerline")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.title(f"Refined 3D Centerline for {os.path.basename(file)}")
    plt.show()

print("Processing complete.")
