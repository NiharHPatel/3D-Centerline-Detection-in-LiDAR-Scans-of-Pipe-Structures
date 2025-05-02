import numpy as np
import open3d as o3d
from sklearn.decomposition import PCA
from scipy.interpolate import splprep, splev
from sklearn.neighbors import NearestNeighbors
import os
import matplotlib.pyplot as plt

# === Parameters ===
voxel_size = 0.01
segment_length = 0.15
spline_smoothness = 0.3
spline_points = 500
min_points_per_segment = 20
eigenvalue_ratio_threshold = 2.0
direction_change_threshold = 0.95

# === Preprocessing ===
def preprocess_pcd(pcd):
    pcd = pcd.voxel_down_sample(voxel_size)
    pcd, _ = pcd.remove_statistical_outlier(30, 1.0)
    return pcd

def segment_point_cloud(points, segment_length):
    pca = PCA(n_components=1)
    proj = pca.fit_transform(points)
    axis = pca.components_[0]
    origin = points.mean(axis=0)
    projected = (points - origin) @ axis
    bins = np.arange(projected.min(), projected.max(), segment_length)
    segments = []
    for i in range(len(bins) - 1):
        mask = (projected >= bins[i]) & (projected < bins[i+1])
        seg_points = points[mask]
        if len(seg_points) >= min_points_per_segment:
            segments.append(seg_points)
    return segments

def compute_pca_quality(segment):
    pca = PCA(n_components=3)
    pca.fit(segment)
    eigenvalues = pca.explained_variance_
    direction = pca.components_[0]
    center = segment.mean(axis=0)
    return eigenvalues, direction, center

def filter_and_chain_segments(segments):
    centers = []
    directions = []
    for seg in segments:
        eigenvalues, dir_vector, center = compute_pca_quality(seg)
        if eigenvalues[0] / eigenvalues[1] >= eigenvalue_ratio_threshold:
            if len(directions) > 0:
                cos_theta = np.dot(directions[-1], dir_vector)
                if cos_theta < direction_change_threshold:
                    continue
            centers.append(center)
            directions.append(dir_vector)
    return np.array(centers)

# === Centerline Fitting ===
def sort_centers_nearest_neighbor(centers):
    if len(centers) < 2:
        return centers
    visited = set()
    sorted_centers = [centers[0]]
    visited.add(0)
    current = centers[0]

    for _ in range(1, len(centers)):
        remaining_idxs = [i for i in range(len(centers)) if i not in visited]
        if not remaining_idxs:
            break
        distances = np.linalg.norm(centers[remaining_idxs] - current, axis=1)
        next_idx = remaining_idxs[np.argmin(distances)]
        current = centers[next_idx]
        sorted_centers.append(current)
        visited.add(next_idx)
    return np.array(sorted_centers)

def fit_b_spline(centers, s=spline_smoothness, num_points=spline_points):
    if len(centers) < 4:
        return centers
    sorted_points = sort_centers_nearest_neighbor(centers)
    tck, _ = splprep(sorted_points.T, s=s)
    u_fine = np.linspace(0, 1, num_points)
    return np.array(splev(u_fine, tck)).T

# === Visualization ===
def visualize_centerline(pcd, centerline, title):
    pcd.paint_uniform_color([0.5, 0.5, 0.5])
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(centerline)
    line_set.lines = o3d.utility.Vector2iVector([[i, i + 1] for i in range(len(centerline) - 1)])
    line_set.paint_uniform_color([0, 0.8, 0])  # green

    o3d.visualization.draw_geometries([pcd, line_set], window_name=title)

# === Main Loop ===
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

for file in ply_files:
    print(f"\n Processing: {os.path.basename(file)}")
    if not os.path.exists(file):
        print(" File not found!")
        continue

    pcd = o3d.io.read_point_cloud(file)
    pcd = preprocess_pcd(pcd)
    points = np.asarray(pcd.points)
    segments = segment_point_cloud(points, segment_length)
    centers = filter_and_chain_segments(segments)

    if len(centers) < 4:
        print(" Not enough valid centers for spline fitting.")
        continue

    smoothed = fit_b_spline(centers)
    visualize_centerline(pcd, smoothed, os.path.basename(file))

print("\n Final 3D  centerline extraction completed.")
