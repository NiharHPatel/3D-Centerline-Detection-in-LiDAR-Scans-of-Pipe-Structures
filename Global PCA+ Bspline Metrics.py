# Hybrid ROSA-RANSAC + B-Spline Centerline Extraction Pipeline with Enhanced Filtering and Spline Fixes

import open3d as o3d
import numpy as np
from sklearn.decomposition import PCA
from scipy.interpolate import splprep, splev
from scipy.spatial.distance import cdist
import os

# === PARAMETERS ===
voxel_size = 0.01
segment_length = 0.15
spline_smoothness = 0.1  # Tighter spline to avoid overshooting
spline_points = 500
min_points_per_segment = 20
eigenvalue_ratio_threshold = 3.0  # Stricter filtering for linearity
direction_change_threshold = 0.98  # More strict chaining

# === Load and Preprocess Point Cloud ===
def load_and_preprocess(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    pcd = pcd.voxel_down_sample(voxel_size)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.0)
    return pcd

# === Segment Point Cloud Along PCA Axis ===
def segment_point_cloud(points):
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

# === Filter and Chain Valid Segments ===
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

# === Nearest Neighbor Sorting of Centers ===
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

# === B-Spline Fitting ===
def fit_b_spline(centers):
    if len(centers) < 4:
        return centers
    sorted_points = sort_centers_nearest_neighbor(centers)
    tck, _ = splprep(sorted_points.T, s=spline_smoothness)
    u_fine = np.linspace(0, 1, spline_points)
    return np.array(splev(u_fine, tck)).T

# === Evaluation Metrics ===
def chamfer_distance(pc1, pc2):
    d1 = np.mean(np.min(cdist(pc1, pc2), axis=1))
    d2 = np.mean(np.min(cdist(pc2, pc1), axis=1))
    return d1 + d2

def hausdorff_distance(pc1, pc2):
    d1 = np.max(np.min(cdist(pc1, pc2), axis=1))
    d2 = np.max(np.min(cdist(pc2, pc1), axis=1))
    return max(d1, d2)

# === Visualization ===
def visualize_centerline(pcd, centerline):
    pcd.paint_uniform_color([0.5, 0.5, 0.5])
    line = o3d.geometry.LineSet()
    line.points = o3d.utility.Vector3dVector(centerline)
    line.lines = o3d.utility.Vector2iVector([[i, i + 1] for i in range(len(centerline) - 1)])
    line.paint_uniform_color([1, 0, 0])
    o3d.visualization.draw_geometries([pcd, line])

# === Pipeline Runner ===
def run_pipeline(file_path):
    print(f"\n📂 Processing: {os.path.basename(file_path)}")
    pcd = load_and_preprocess(file_path)
    points = np.asarray(pcd.points)

    segments = segment_point_cloud(points)
    centers = filter_and_chain_segments(segments)

    print(f"🧩 Valid center points used: {len(centers)}")

    if len(centers) < 4:
        print("⚠️ Not enough valid segments.")
        return

    centerline = fit_b_spline(centers)
    visualize_centerline(pcd, centerline)

    cd = chamfer_distance(points, centerline)
    hd = hausdorff_distance(points, centerline)
    print(f"✅ Chamfer Distance: {cd:.4f}")
    print(f"✅ Hausdorff Distance: {hd:.4f}")

# === Batch Run All 9 Frames ===
ply_files = [
    "/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/PLY Examples/1_0058.000.ply",
    "/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/PLY Examples/2_0002.500.ply",
    "/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/PLY Examples/3_0000.500.ply",
    "/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/PLY Examples/4_0005.000.ply",
    "/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/PLY Examples/5_0015.500.ply",
    "/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/PLY Examples/6_0006.500.ply",
    "/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/PLY Examples/7_0029.500.ply",
    "/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/PLY Examples/8_0007.500.ply",
    "/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/PLY Examples/9_0000.000.ply"
]

for path in ply_files:
    run_pipeline(path)
