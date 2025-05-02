import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from sklearn.linear_model import RANSACRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
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

def preprocess_point_cloud(pcd, voxel_size=0.01):
    return pcd.voxel_down_sample(voxel_size=voxel_size)

def estimate_normals(pcd):
    """ Compute normal vectors for the point cloud. """
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    return pcd

def adaptive_ransac_params(points):
    """ Dynamically adjust RANSAC distance threshold using nearest-neighbor distances. """
    if len(points) < 10:
        return 0.02  # Default fallback

    nn = NearestNeighbors(n_neighbors=5)
    nn.fit(points)
    distances, _ = nn.kneighbors(points)
    avg_distance = np.mean(distances[:, 1])  # Avoid the first distance (self-distance)

    return avg_distance * 1.5  # Scale factor for robustness

def fit_cylinder_ransac_robust(pcd, max_trials=3000):
    """ Fit a cylinder to the point cloud using Adaptive RANSAC with Robust Kernels. """
    points = np.asarray(pcd.points)

    if len(points) < 10:
        print("Warning: Not enough points for RANSAC. Skipping.")
        return pcd, np.ones(len(points), dtype=bool)

    adaptive_threshold = adaptive_ransac_params(points)

    model_ransac = RANSACRegressor(
        residual_threshold=adaptive_threshold,
        max_trials=max_trials,
        random_state=42
    )

    # Using only the X-Y plane to fit a model and get Z predictions
    model_ransac.fit(points[:, :2], points[:, 2])
    z_pred = model_ransac.predict(points[:, :2])
    residuals = np.abs(points[:, 2] - z_pred)

    inlier_mask = residuals < adaptive_threshold
    cylinder_pcd = pcd.select_by_index(np.where(inlier_mask)[0])

    return cylinder_pcd, inlier_mask

def compute_centerline(pcd):
    """ Compute centerline  Gaussian Process Regression for abrupt turns. """
    points = np.asarray(pcd.points)

    if len(points) < 5:
        return points  

    # Sort by Z-axis to maintain order
    points = points[np.argsort(points[:, 2])]

    unique_z_values, unique_indices = np.unique(points[:, 2], return_index=True)
    centerline_points = points[unique_indices]

    if len(centerline_points) < 4:
        print("Using Gaussian Process Regression due to low points.")
        kernel = C(1.0, (1e-3, 1e2)) * RBF(0.5, (1e-2, 1e2))
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2)
        gpr.fit(unique_z_values.reshape(-1, 1), centerline_points[:, :2])
        z_fine = np.linspace(unique_z_values.min(), unique_z_values.max(), 500)
        xy_fine = gpr.predict(z_fine.reshape(-1, 1))
        return np.hstack((xy_fine, z_fine[:, None]))

    # Use spline for smooth pipes
    try:
        tck, u = splprep(centerline_points.T, s=0.1, k=3)
        u_fine = np.linspace(0, 1, 500)
        return np.array(splev(u_fine, tck)).T
    except:
        return centerline_points

def apply_kalman_filter(centerline_points):
    """ Apply Kalman Filter for smoother centerline tracking. """
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
    pcd_filtered = preprocess_point_cloud(pcd)
    pcd_filtered = estimate_normals(pcd_filtered)

    # Fit cylinder and extract pipe points
    cylinder_pcd, inlier_mask = fit_cylinder_ransac_robust(pcd_filtered)

    # Compute centerline
    centerline = compute_centerline(cylinder_pcd)

    # Smooth the centerline
    smoothed_centerline = apply_kalman_filter(centerline)

    # Convert centerline to Open3D PointCloud
    centerline_pcd = o3d.geometry.PointCloud()
    centerline_pcd.points = o3d.utility.Vector3dVector(smoothed_centerline)
    centerline_pcd.paint_uniform_color([1, 0, 0])  # Red centerline

    # Add bounding box around detected pipe
    bbox = cylinder_pcd.get_oriented_bounding_box()
    bbox.color = (0, 1, 0)  # Green bounding box

    bbox.color = (0, 1, 0)  # Green bounding box

    # Visualization
    o3d.visualization.draw_geometries([pcd_filtered, cylinder_pcd.paint_uniform_color([0, 0, 1]), centerline_pcd, bbox])

print("Processing complete.")
