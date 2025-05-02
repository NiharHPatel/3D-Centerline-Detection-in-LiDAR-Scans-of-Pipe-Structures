# Final Polished Pipeline with GPR-Smoothed ROSA and Midpoint-Centered Correction

#Install Liberies
import open3d as o3d
import numpy as np
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.interpolate import splprep, splev
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import os

# === Parameters ===
voxel_size = 0.01
segment_length = 0.15
min_points_per_segment = 20
eigenvalue_ratio_threshold = 2.5
direction_change_threshold = 0.95
spline_smoothness = 1.5
spline_points = 500
rosafallback_threshold = 10

# === Preprocessing ===
def preprocess_pcd(pcd):
    pcd = pcd.voxel_down_sample(voxel_size)
    pcd, _ = pcd.remove_statistical_outlier(30, 1.0)
    return pcd

# === Segmentation ===
def segment_point_cloud(points):
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

# === PCA Filtering ===
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
                cos_theta = np.dot(directions[-1], dir_vector)
                if cos_theta < direction_change_threshold:
                    continue
            centers.append(center)
            directions.append(dir_vector)
    return np.array(centers)

# === Curvature-Aware Sorting ===
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

# === ROSA Fallback ===
def rosa_fallback(pcd):
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    centers = []
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    for i in range(0, len(points), 150):
        _, idxs, _ = kdtree.search_radius_vector_3d(pcd.points[i], 0.05)
        if len(idxs) < 10:
            continue
        local_pts = points[idxs]
        center = np.mean(local_pts, axis=0)
        centers.append(center)
    return np.array(centers)

# === GPR Smoothing ===
def fit_gpr_to_points(points, num_points=500):
    if len(points) < 4:
        return points
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    t = np.concatenate([[0], np.cumsum(distances)]).reshape(-1, 1)
    kernel = C(1.0, (1e-2, 1e2)) * RBF(1.0, (1e-2, 1e2))
    smoothed = []
    for i in range(3):
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-2)
        gp.fit(t, points[:, i])
        t_fine = np.linspace(t.min(), t.max(), num_points).reshape(-1, 1)
        pred, _ = gp.predict(t_fine, return_std=True)
        smoothed.append(pred)
    return np.vstack(smoothed).T

# === Midpoint-Center Correction ===
def midpoint_center_correction(centerline, point_cloud, radius=0.15, step=10):
    corrected = []
    pc_array = np.asarray(point_cloud.points)
    for i in range(0, len(centerline), step):
        p = centerline[i]
        direction = centerline[min(i+1, len(centerline)-1)] - centerline[max(i-1, 0)]
        direction = direction / np.linalg.norm(direction)
        normal = direction
        kdtree = o3d.geometry.KDTreeFlann(point_cloud)
        _, idxs, _ = kdtree.search_radius_vector_3d(p, radius)
        neighbors = pc_array[idxs]
        if len(neighbors) < 6:
            corrected.append(p)
            continue
        proj = neighbors - p
        proj_on_normal = (proj @ normal)[:, None] * normal
        proj_flat = proj - proj_on_normal
        cov = np.cov(proj_flat.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        cross_axis = eigvecs[:, -1]
        scalars = proj_flat @ cross_axis
        left = neighbors[scalars < 0]
        right = neighbors[scalars > 0]
        if len(left) < 3 or len(right) < 3:
            corrected.append(p)
            continue
        midpoint = (np.mean(left, axis=0) + np.mean(right, axis=0)) / 2.0
        corrected.append(midpoint)
    corrected = np.array(corrected)
    if len(corrected) >= 4:
        tck, _ = splprep(corrected.T, s=0.5)
        u_fine = np.linspace(0, 1, len(centerline))
        smooth = np.array(splev(u_fine, tck)).T
        return smooth
    else:
        return centerline

# === B-Spline Fit ===
def fit_b_spline(points):
    if len(points) < 4:
        return points
    sorted_pts = curvature_aware_sort(points)
    tck, _ = splprep(sorted_pts.T, s=spline_smoothness)
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

# === Rainbow Visualization ===
def visualize_rainbow_centerline(pcd, centerline, title):
    pcd.paint_uniform_color([0.5, 0.5, 0.5])
    line = o3d.geometry.LineSet()
    line.points = o3d.utility.Vector3dVector(centerline)
    line.lines = o3d.utility.Vector2iVector([[i, i+1] for i in range(len(centerline)-1)])
    colors = plt.cm.jet(np.linspace(0, 1, len(centerline)-1))[:, :3]
    line.colors = o3d.utility.Vector3dVector(colors.tolist())
    o3d.visualization.draw_geometries([pcd, line], window_name=title)

# === Pipeline Runner ===
def run_pipeline(file_path):
    print(f"\nProcessing: {os.path.basename(file_path)}")
    pcd = o3d.io.read_point_cloud(file_path)
    pcd = preprocess_pcd(pcd)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
    points = np.asarray(pcd.points)
    segments = segment_point_cloud(points)
    centers = filter_and_chain_segments(segments)
    print(f"🧩 Valid center points used: {len(centers)}")

    if len(centers) < rosafallback_threshold:
        print("🔁 Using ROSA fallback due to low center count")
        centers = rosa_fallback(pcd)
        print(f"🔁 Fallback points generated: {len(centers)}")
        centers = fit_gpr_to_points(centers, num_points=spline_points)

    if len(centers) < 4:
        print("⚠️ Still not enough points for spline.")
        return

    centerline = fit_b_spline(centers)
    centerline = midpoint_center_correction(centerline, pcd)
    visualize_rainbow_centerline(pcd, centerline, title=os.path.basename(file_path))
    cd = chamfer_distance(points, centerline)
    hd = hausdorff_distance(points, centerline)
    print(f"✅ Chamfer Distance: {cd:.4f}")
    print(f"✅ Hausdorff Distance: {hd:.4f}")

# === Batch Files ===
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
