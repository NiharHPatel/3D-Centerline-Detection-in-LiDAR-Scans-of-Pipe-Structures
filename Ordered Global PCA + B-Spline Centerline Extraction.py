import open3d as o3d
import numpy as np
from sklearn.decomposition import PCA
from scipy.interpolate import splprep, splev
import os

# ====== PARAMETERS ======
voxel_size = 0.01
spline_smoothness = 0.5
spline_points = 500

# ====== FUNCTION DEFINITIONS ======

def preprocess_pcd(pcd):
    """Downsample and denoise the point cloud."""
    pcd = pcd.voxel_down_sample(voxel_size)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.0)
    return pcd

def compute_global_pca_axis(points):
    """Get the major axis from global PCA."""
    pca = PCA(n_components=1)
    pca.fit(points)
    axis = pca.components_[0]
    mean = points.mean(axis=0)
    return axis, mean

def project_points_to_axis(points, axis, mean):
    """Project points onto the PCA axis."""
    projections = (points - mean) @ axis
    ordered = np.outer(projections, axis) + mean
    return ordered, projections

def smooth_with_b_spline(points, smooth=1.0, num_points=500):
    """Fit a smooth B-spline through the ordered points."""
    if len(points) < 4:
        return points
    tck, _ = splprep(points.T, s=smooth)
    u = np.linspace(0, 1, num_points)
    return np.array(splev(u, tck)).T

def visualize_centerline(pcd, centerline, title):
    """Visualize point cloud and centerline."""
    pcd.paint_uniform_color([0.6, 0.6, 0.6])
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(centerline)
    line_set.lines = o3d.utility.Vector2iVector([[i, i + 1] for i in range(len(centerline) - 1)])
    line_set.paint_uniform_color([0, 1, 0])
    o3d.visualization.draw_geometries([pcd, line_set], window_name=title)

# ====== PLY FILES TO PROCESS ======
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

# ====== EXECUTION ======
for file in ply_files:
    print(f"\n Processing: {os.path.basename(file)}")
    if not os.path.exists(file):
        print(" File not found!")
        continue

    pcd = o3d.io.read_point_cloud(file)
    pcd = preprocess_pcd(pcd)
    points = np.asarray(pcd.points)

    if len(points) < 10:
        print("⚠️ Not enough points.")
        continue

    axis, mean = compute_global_pca_axis(points)
    ordered_points, projections = project_points_to_axis(points, axis, mean)

    # Ensure ordering
    idx_sorted = np.argsort(projections)
    ordered_points = ordered_points[idx_sorted]

    smoothed_centerline = smooth_with_b_spline(ordered_points, smooth=spline_smoothness, num_points=spline_points)
    visualize_centerline(pcd, smoothed_centerline, os.path.basename(file))

print("\n  PCA Centerline Extraction Completed!")
