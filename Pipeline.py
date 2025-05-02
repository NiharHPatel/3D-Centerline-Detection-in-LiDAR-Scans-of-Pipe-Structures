import open3d as o3d
import numpy as np
from sklearn.decomposition import PCA
from scipy.interpolate import splprep, splev
import os

# ==== MANUAL FILES ====
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

# ========== PIPELINE ==========

def preprocess_pcd(pcd):
    pcd = pcd.voxel_down_sample(0.01)
    pcd, _ = pcd.remove_statistical_outlier(30, 1.0)
    return pcd

def fit_pipe_axis(points):
    pca = PCA(n_components=1)
    pca.fit(points)
    axis_dir = pca.components_[0]
    axis_point = np.mean(points, axis=0)
    projections = np.dot(points - axis_point, axis_dir)
    centerline = np.outer(projections, axis_dir) + axis_point
    return centerline, projections

def smooth_centerline(centerline, smoothness=0.5, num_points=500):
    tck, _ = splprep(centerline.T, s=smoothness, k=3)
    u_fine = np.linspace(0, 1, num_points)
    return np.array(splev(u_fine, tck)).T

def visualize_centerline(original, centerline):
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(centerline)
    line_set.lines = o3d.utility.Vector2iVector([[i, i+1] for i in range(len(centerline)-1)])
    line_set.paint_uniform_color([0, 1, 0])
    
    original.paint_uniform_color([0.7, 0.7, 0.7])

    o3d.visualization.draw_geometries([original, line_set])

# ========== PROCESS ==========

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

    centerline, _ = fit_pipe_axis(points)
    smoothed_centerline = smooth_centerline(centerline)

    visualize_centerline(pcd, smoothed_centerline)

print(" Batch Visualization Complete!")
