import open3d as o3d
import numpy as np
from sklearn.decomposition import PCA
from scipy.interpolate import splprep, splev
import os

# =====================
# Manual Input (PLY Files)
# =====================
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

# =====================
# Parameters
# =====================
voxel_size = 0.01
segment_length = 0.1  # Local window size (meters)
spline_smoothness = 0.5
spline_points = 500

# =====================
# Pipeline Functions
# =====================
def preprocess_pcd(pcd):
    pcd = pcd.voxel_down_sample(voxel_size)
    pcd, _ = pcd.remove_statistical_outlier(30, 1.0)
    return pcd

def extract_local_centers(points, segment_length):
    # Project points along PCA's first axis
    pca = PCA(n_components=1)
    projected = pca.fit_transform(points)

    # Create segments along projected space
    min_proj, max_proj = projected.min(), projected.max()
    bins = np.arange(min_proj, max_proj, segment_length)

    local_centers = []

    for i in range(len(bins) - 1):
        mask = (projected[:, 0] >= bins[i]) & (projected[:, 0] < bins[i + 1])
        segment = points[mask]
        if len(segment) > 0:
            local_center = np.mean(segment, axis=0)
            local_centers.append(local_center)

    return np.array(local_centers)

def smooth_centerline(center_points):
    tck, _ = splprep(center_points.T, s=spline_smoothness, k=3)
    u_fine = np.linspace(0, 1, spline_points)
    return np.array(splev(u_fine, tck)).T

def visualize(original, centerline):
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(centerline)
    line_set.lines = o3d.utility.Vector2iVector([[i, i+1] for i in range(len(centerline)-1)])
    line_set.paint_uniform_color([0, 1, 0])

    original.paint_uniform_color([0.7, 0.7, 0.7])

    o3d.visualization.draw_geometries([original, line_set])

# =====================
# Full Process
# =====================
for file in ply_files:
    print(f"Processing: {file}")

    if not os.path.exists(file):
        print(f"File not found: {file}")
        continue

    pcd = o3d.io.read_point_cloud(file)
    pcd = preprocess_pcd(pcd)

    points = np.asarray(pcd.points)

    if len(points) < 10:
        print(f"⚠️ Not enough points: {file}")
        continue

    local_centers = extract_local_centers(points, segment_length)
    smoothed_centerline = smooth_centerline(local_centers)

    visualize(pcd, smoothed_centerline)

print("Completed Local PCA + Spline centerline extraction!")
