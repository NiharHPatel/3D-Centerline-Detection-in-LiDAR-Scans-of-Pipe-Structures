import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev, interp1d
from sklearn.cluster import DBSCAN
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.linear_model import RANSACRegressor
from sklearn.neighbors import NearestNeighbors
from pykalman import KalmanFilter
import os

# List of PLY file paths
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

def preprocess_point_cloud(pcd, voxel_size=0.02):
    """Downsample and remove noise from the point cloud."""
    return pcd.voxel_down_sample(voxel_size=voxel_size)

def estimate_normals(pcd):
    """Compute normal vectors for the point cloud."""
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    return pcd

def adaptive_dbscan_eps(points, factor=1.5):
    """Automatically determine DBSCAN eps based on average nearest-neighbor distance."""
    nn = NearestNeighbors(n_neighbors=5)
    nn.fit(points)
    distances, _ = nn.kneighbors(points)
    avg_distance = np.mean(distances[:, 1])
    return avg_distance * factor

def fit_cylinder_ransac(pcd, distance_threshold=0.03):
    """Fit a cylinder using adaptive RANSAC."""
    points = np.asarray(pcd.points)

    if len(points) < 15:
        print("⚠️ Warning: Too few points for RANSAC. Returning raw points.")
        return pcd, np.arange(len(points))

    adaptive_trials = min(1000, max(200, int(len(points) / 5)))
    model_ransac = RANSACRegressor(residual_threshold=distance_threshold, max_trials=adaptive_trials)
    model_ransac.fit(points[:, :2], points[:, 2])  

    z_pred = model_ransac.predict(points[:, :2])
    residuals = np.abs(points[:, 2] - z_pred)
    inlier_mask = residuals < distance_threshold
    cylinder_pcd = pcd.select_by_index(np.where(inlier_mask)[0])

    return cylinder_pcd, inlier_mask

def compute_centerline(pcd):
    """Compute centerline using clustering, interpolation, and spline fitting."""
    points = np.asarray(pcd.points)

    if len(points) < 5:
        print("⚠️ Critical Warning: Centerline too short. Returning raw points.")
        return points

    dbscan_eps = adaptive_dbscan_eps(points[:, :2], factor=2.0)
    clustering = DBSCAN(eps=dbscan_eps, min_samples=10).fit(points[:, :2])
    labels = clustering.labels_

    unique_labels = np.unique(labels)
    centerline_points = []

    for label in unique_labels:
        if label == -1:
            continue
        cluster_points = points[labels == label]
        centroid = np.mean(cluster_points, axis=0)
        centerline_points.append(centroid)

    centerline_points = np.array(centerline_points)
    centerline_points = centerline_points[np.argsort(centerline_points[:, 2])]

    unique_z_values, unique_indices = np.unique(centerline_points[:, 2], return_index=True)
    centerline_points = centerline_points[unique_indices]

    if len(unique_z_values) < 2:
        print("⚠️ Error: Only one unique z-value found. Using Gaussian Process Regression.")
        min_z, max_z = np.min(points[:, 2]), np.max(points[:, 2])
        z_synthetic = np.linspace(min_z, max_z, num=10)
        avg_x, avg_y = np.mean(points[:, 0]), np.mean(points[:, 1])
        return np.column_stack((np.full_like(z_synthetic, avg_x), np.full_like(z_synthetic, avg_y), z_synthetic))

    if len(centerline_points) < 4:
        print("Warning: Not enough points for spline fitting. Using Gaussian Process Regression.")
        kernel = C(1.0, (1e-3, 1e2)) * RBF(0.5, (1e-2, 1e2))
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2)
        gpr.fit(unique_z_values.reshape(-1, 1), centerline_points[:, :2])
        z_fine = np.linspace(unique_z_values.min(), unique_z_values.max(), 500)
        xy_fine = gpr.predict(z_fine.reshape(-1, 1))
        return np.hstack((xy_fine, z_fine[:, None]))

    smoothing_factor = 1.0 if len(centerline_points) > 20 else 0.1
    spline_order = min(3, len(centerline_points) - 1)

    try:
        tck, u = splprep(centerline_points.T, s=smoothing_factor, k=spline_order)
        u_fine = np.linspace(0, 1, 500)
        return np.array(splev(u_fine, tck)).T
    except:
        print("⚠️ Spline fitting failed. Falling back to linear interpolation.")
        x_interp = interp1d(unique_z_values, centerline_points[:, 0], kind='linear', fill_value="extrapolate")
        y_interp = interp1d(unique_z_values, centerline_points[:, 1], kind='linear', fill_value="extrapolate")
        z_fine = np.linspace(unique_z_values.min(), unique_z_values.max(), 500)
        return np.vstack([x_interp(z_fine), y_interp(z_fine), z_fine]).T

def apply_kalman_filter(centerline_points):
    """Apply Kalman Filter to smooth centerline."""
    if len(centerline_points) < 2:
        return centerline_points

    kf = KalmanFilter(initial_state_mean=centerline_points[0], n_dim_obs=3)
    return kf.smooth(centerline_points)[0]

# Process each PLY file
for file in ply_files:
    if not os.path.exists(file):
        print(f"File not found: {file}")
        continue

    print(f"Processing: {file}")
    
    pcd = o3d.io.read_point_cloud(file)
    if len(pcd.points) == 0:
        continue
    
    pcd_filtered = preprocess_point_cloud(pcd)
    pcd_filtered = estimate_normals(pcd_filtered)

    cylinder_pcd, inlier_mask = fit_cylinder_ransac(pcd_filtered)
    centerline = compute_centerline(cylinder_pcd)
    smoothed_centerline = apply_kalman_filter(centerline)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(smoothed_centerline[:, 0], smoothed_centerline[:, 1], smoothed_centerline[:, 2], c="green")
    ax.plot(smoothed_centerline[:, 0], smoothed_centerline[:, 1], smoothed_centerline[:, 2], c="red")
    plt.show()

print("Processing complete.")
