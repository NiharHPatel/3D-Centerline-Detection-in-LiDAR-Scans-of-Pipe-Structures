import numpy as np
import open3d as o3d
from sklearn.decomposition import PCA
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# === PARAMETERS ===
voxel_size = 0.01
segment_length = 0.15
spline_smoothness = 0.3
spline_points = 500
min_points_per_segment = 20
eigenvalue_ratio_threshold = 2.0
direction_change_threshold = 0.95

# === FUNCTIONS ===
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

def fit_b_spline(centers, s=spline_smoothness, num_points=spline_points):
    if len(centers) < 4:
        return centers
    tck, _ = splprep(centers.T, s=s)
    u_fine = np.linspace(0, 1, num_points)
    return np.array(splev(u_fine, tck)).T

def plot_centerline(pcd, centerline, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    pts = np.asarray(pcd.points)
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=0.5, color='gray', alpha=0.5)
    ax.plot(centerline[:, 0], centerline[:, 1], centerline[:, 2], color='green', linewidth=2, label='Centerline')
    
    ax.set_title(title)
    ax.legend()
    ax.view_init(elev=20, azim=60)  # Critical: ensures 3D view!
    ax.grid(True)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# === MAIN EXECUTION ===
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

for file_path in ply_files:
    if not os.path.exists(file_path):
        print(f" File not found: {file_path}")
        continue

    print(f"📂 Processing: {os.path.basename(file_path)}")
    pcd = o3d.io.read_point_cloud(file_path)
    pcd = preprocess_pcd(pcd)
    points = np.asarray(pcd.points)
    segments = segment_point_cloud(points, segment_length)
    centers = filter_and_chain_segments(segments)
    smoothed = fit_b_spline(centers)
    plot_centerline(pcd, smoothed, title=os.path.basename(file_path))
