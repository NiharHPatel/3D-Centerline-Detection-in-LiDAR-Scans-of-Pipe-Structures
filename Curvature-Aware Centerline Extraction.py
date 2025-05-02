import open3d as o3d
import numpy as np
from sklearn.decomposition import PCA
from scipy.interpolate import splprep, splev
import os

# ===== Parameters =====
voxel_size = 0.01
segment_length = 0.15
min_points_per_segment = 20
eigenvalue_ratio_threshold = 2.0
direction_change_threshold = 0.9
spline_smoothness = 0.3
spline_points = 500

# ===== Preprocessing =====
def preprocess_pcd(pcd):
    pcd = pcd.voxel_down_sample(voxel_size)
    pcd, _ = pcd.remove_statistical_outlier(30, 1.0)
    return pcd

# ===== Segmentation =====
def segment_point_cloud(points, segment_length):
    pca = PCA(n_components=1)
    axis = pca.fit(points).components_[0]
    origin = np.mean(points, axis=0)
    projections = (points - origin) @ axis
    bins = np.arange(np.min(projections), np.max(projections), segment_length)
    
    segments = []
    for i in range(len(bins) - 1):
        mask = (projections >= bins[i]) & (projections < bins[i+1])
        seg = points[mask]
        if len(seg) >= min_points_per_segment:
            segments.append(seg)
    return segments

# ===== PCA Filtering =====
def compute_pca_quality(segment):
    pca = PCA(n_components=3)
    pca.fit(segment)
    eigenvalues = pca.explained_variance_
    direction = pca.components_[0]
    center = np.mean(segment, axis=0)
    return eigenvalues, direction, center

def filter_and_chain_segments(segments):
    centers = []
    directions = []
    for seg in segments:
        eigenvalues, dir_vector, center = compute_pca_quality(seg)
        if eigenvalues[0] / eigenvalues[1] >= eigenvalue_ratio_threshold:
            if len(directions) > 0:
                cos_angle = np.dot(directions[-1], dir_vector)
                if cos_angle < direction_change_threshold:
                    continue
            centers.append(center)
            directions.append(dir_vector)
    return np.array(centers)

# ===== Curvature-Aware Sort =====
def curvature_aware_sort(points):
    if len(points) < 2:
        return points
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    start_idx = np.argmin(np.sum(distances, axis=1))
    sorted_points = [points[start_idx]]
    used = {start_idx}

    while len(used) < len(points):
        last = sorted_points[-1]
        candidates = [i for i in range(len(points)) if i not in used]
        dists = np.linalg.norm(points[candidates] - last, axis=1)
        next_idx = candidates[np.argmin(dists)]
        sorted_points.append(points[next_idx])
        used.add(next_idx)

    return np.array(sorted_points)

# ===== B-Spline Fitting =====
def fit_b_spline(points, s=spline_smoothness, num_points=spline_points):
    if len(points) < 4:
        return points
    tck, _ = splprep(points.T, s=s)
    u_fine = np.linspace(0, 1, num_points)
    return np.array(splev(u_fine, tck)).T

# ===== Visualization =====
def visualize_centerline(pcd, centerline, title):
    pcd.paint_uniform_color([0.5, 0.5, 0.5])
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(centerline)
    line_set.lines = o3d.utility.Vector2iVector([[i, i+1] for i in range(len(centerline)-1)])
    line_set.paint_uniform_color([1.0, 0, 0])  # red line
    o3d.visualization.draw_geometries([pcd, line_set], window_name=title)

# ===== File Paths =====
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

# ===== Main Loop =====
for file_path in ply_files:
    print(f"\n📂 Processing: {os.path.basename(file_path)}")
    if not os.path.exists(file_path):
        print("❌ File not found.")
        continue

    pcd = o3d.io.read_point_cloud(file_path)
    pcd = preprocess_pcd(pcd)
    points = np.asarray(pcd.points)

    segments = segment_point_cloud(points, segment_length)
    centers = filter_and_chain_segments(segments)

    if len(centers) < 4:
        print("⚠️ Not enough valid center points.")
        continue

    ordered = curvature_aware_sort(centers)
    smoothed_centerline = fit_b_spline(ordered)

    print(f"✅ Valid center points used: {len(ordered)}")
    visualize_centerline(pcd, smoothed_centerline, title=os.path.basename(file_path))

print("\n✅ All centerlines generated with curvature-aware sorting!")
