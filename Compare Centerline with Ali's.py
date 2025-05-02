import open3d as o3d
import numpy as np
from scipy.interpolate import splprep, splev
from sklearn.decomposition import PCA
import os

# ======= List your 9 files manually here =======
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

# ======== PARAMETERS ========
voxel_size = 0.01
stat_nb_neighbors = 30
stat_std_ratio = 1.0
radius_nb_points = 15
radius = 0.02
spline_smoothness = 1.0
spline_points = 500

# ======== FUNCTIONS ========

def preprocess_pcd(pcd):
    pcd = pcd.voxel_down_sample(voxel_size)
    pcd, _ = pcd.remove_statistical_outlier(stat_nb_neighbors, stat_std_ratio)
    pcd, _ = pcd.remove_radius_outlier(radius_nb_points, radius)
    return pcd

def pca_sort(points):
    pca = PCA(n_components=1)
    projected = pca.fit_transform(points)
    return points[np.argsort(projected[:, 0])]

def extract_centerline(points, smoothness=spline_smoothness, num_points=spline_points):
    if len(points) < 5:
        return points
    points = pca_sort(points)
    tck, _ = splprep(points.T, s=smoothness, k=3)
    u_fine = np.linspace(0, 1, num_points)
    return np.array(splev(u_fine, tck)).T

def visualize_centerline(centerline):
    lines = [[i, i + 1] for i in range(len(centerline) - 1)]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(centerline)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.paint_uniform_color([0, 1, 0])  # Green
    o3d.visualization.draw_geometries([line_set])

# ======== PROCESS ALL MANUALLY LISTED FILES ========

for file in ply_files:
    print(f"Processing: {file}")

    if not os.path.exists(file):
        print(f"❌ File not found: {file}")
        continue

    pcd = o3d.io.read_point_cloud(file)
    pcd = preprocess_pcd(pcd)

    points = np.asarray(pcd.points)
    if len(points) < 10:
        print(f"⚠️ Not enough points in: {file}")
        continue

    centerline = extract_centerline(points)
    visualize_centerline(centerline)
    print(f"✅ Visualized: {file}")

print("🎉 All centerlines visualized!")
