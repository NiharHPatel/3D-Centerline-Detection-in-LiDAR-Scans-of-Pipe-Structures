import numpy as np
import open3d as o3d
from sklearn.decomposition import PCA
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
import os
import glob

voxel_size = 0.01
segment_length = 0.15
spline_smoothness = 0.3
spline_points = 500
min_points_per_segment = 20
eigenvalue_ratio_threshold = 2.0
direction_change_threshold = 0.95

def preprocess_pcd(pcd):
    pcd = pcd.voxel_down_sample(voxel_size)
    pcd, _ = pcd.remove_statistical_outlier(30, 1.0)
    return pcd

def segment_along_global_pca(points, segment_length):
    pca = PCA(n_components=1)
    proj = pca.fit_transform(points)
    axis = pca.components_[0]
    origin = points.mean(axis=0)
    projections = (points - origin) @ axis
    bins = np.arange(projections.min(), projections.max(), segment_length)

    segments = []
    for i in range(len(bins) - 1):
        mask = (projections >= bins[i]) & (projections < bins[i + 1])
        seg = points[mask]
        if len(seg) >= min_points_per_segment:
            segments.append(seg)
    return segments

def compute_segment_center_direction(segment):
    pca = PCA(n_components=3)
    pca.fit(segment)
    eigenvalues = pca.explained_variance_
    direction = pca.components_[0]
    center = segment.mean(axis=0)
    return eigenvalues, direction, center

def filter_segments(segments):
    centers = []
    directions = []
    for segment in segments:
        eigenvalues, direction, center = compute_segment_center_direction(segment)
        if eigenvalues[0] / eigenvalues[1] >= eigenvalue_ratio_threshold:
            if directions:
                cos_theta = np.dot(directions[-1], direction)
                if cos_theta < direction_change_threshold:
                    continue
            centers.append(center)
            directions.append(direction)
    return np.array(centers)

def fit_b_spline(points, smooth=0.3, num_points=500):
    if len(points) < 4:
        return points
    tck, _ = splprep(points.T, s=smooth)
    u_fine = np.linspace(0, 1, num_points)
    return np.array(splev(u_fine, tck)).T

def visualize_result(pcd, centerline, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(*np.asarray(pcd.points).T, s=0.5, color='gray', alpha=0.2)
    ax.plot(*centerline.T, color='green', linewidth=2, label='Ali-style Centerline')
    ax.set_title(title)
    ax.legend()
    plt.show()

# === PLY FILES ===
ply_paths = sorted(glob.glob("/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/PLY Examples/*.ply"))

for path in ply_paths:
    print(f"\n📂 Processing: {os.path.basename(path)}")
    if not os.path.exists(path):
        print("❌ File not found.")
        continue

    pcd = o3d.io.read_point_cloud(path)
    pcd = preprocess_pcd(pcd)
    points = np.asarray(pcd.points)

    segments = segment_along_global_pca(points, segment_length)
    centers = filter_segments(segments)

    if len(centers) < 4:
        print("⚠️ Not enough centers for spline fitting.")
        continue

    spline = fit_b_spline(centers, smooth=spline_smoothness, num_points=spline_points)
    visualize_result(pcd, spline, os.path.basename(path))
