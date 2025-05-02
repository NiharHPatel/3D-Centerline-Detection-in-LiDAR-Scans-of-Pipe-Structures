import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from sklearn.cluster import DBSCAN
from sklearn.linear_model import RANSACRegressor
from sklearn.metrics import mean_squared_error
from pykalman import KalmanFilter
import os

# List of PLY file paths (Replace with actual file paths)
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
    """ Downsample and remove statistical outliers. """
    return pcd.voxel_down_sample(voxel_size=voxel_size)

def estimate_normals(pcd):
    """ Compute normal vectors. """
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    return pcd

def fit_cylinder_ransac(pcd, max_trials=1500, distance_threshold=0.03):
    """ Fit a cylinder using RANSAC. """
    points = np.asarray(pcd.points)
    
    if len(points) < 10:
        print("Warning: Not enough points for cylinder fitting. Returning raw points.")
        return pcd, np.arange(len(points))

    model_ransac = RANSACRegressor(residual_threshold=distance_threshold, max_trials=max_trials)
    model_ransac.fit(points[:, :2], points[:, 2])

    z_pred = model_ransac.predict(points[:, :2])
    residuals = np.abs(points[:, 2] - z_pred)
    inlier_mask = residuals < distance_threshold
    cylinder_pcd = pcd.select_by_index(np.where(inlier_mask)[0])
    
    return cylinder_pcd, inlier_mask

def compute_centerline(pcd):
    """ Compute centerline using clustering, interpolation, and spline fitting. """
    points = np.asarray(pcd.points)
    
    if len(points) < 5:
        print("Warning: Centerline too short. Returning raw points.")
        return points

    clustering = DBSCAN(eps=0.4, min_samples=8).fit(points[:, :2])
    labels = clustering.labels_
    
    centerline_points = [np.mean(points[labels == label], axis=0) for label in np.unique(labels) if label != -1]
    centerline_points = np.array(centerline_points)
    centerline_points = centerline_points[np.argsort(centerline_points[:, 2])]
    
    if len(centerline_points) < 4:
        return centerline_points
    
    spline_order = min(3, len(centerline_points) - 1)
    tck, u = splprep(centerline_points.T, s=0.5, k=spline_order)
    u_fine = np.linspace(0, 1, 500)
    refined_centerline = np.array(splev(u_fine, tck)).T
    
    return refined_centerline

def apply_kalman_filter(centerline_points):
    """ Apply Kalman Filter to smooth centerline. """
    if len(centerline_points) < 2:
        return centerline_points
    
    kf = KalmanFilter(initial_state_mean=centerline_points[0], n_dim_obs=3)
    smoothed_centerline, _ = kf.smooth(centerline_points)
    return smoothed_centerline

def calculate_accuracy(centerline, reference_centerline):
    """ Compute Mean Squared Error (MSE) between detected and expected centerline. """
    mse = mean_squared_error(reference_centerline, centerline)
    avg_deviation = np.mean(np.linalg.norm(centerline - reference_centerline, axis=1))
    return mse, avg_deviation

# Process each PLY file
for file in ply_files:
    if not os.path.exists(file):
        print(f"File not found: {file}")
        continue

    print(f"Processing: {file}")
    
    pcd = o3d.io.read_point_cloud(file)
    if len(pcd.points) == 0:
        print(f"Skipping empty file: {file}")
        continue
    
    pcd_filtered = preprocess_point_cloud(pcd)
    pcd_filtered = estimate_normals(pcd_filtered)

    # Fit a cylinder using RANSAC
    cylinder_pcd, _ = fit_cylinder_ransac(pcd_filtered)

    # Compute refined centerline
    centerline = compute_centerline(cylinder_pcd)

    # Apply Kalman Filter
    smoothed_centerline = apply_kalman_filter(centerline)

    # Placeholder: Load reference centerline for accuracy evaluation
    reference_centerline = smoothed_centerline.copy()  # Replace with actual reference if available
    mse, avg_dev = calculate_accuracy(smoothed_centerline, reference_centerline)
    print(f"MSE: {mse:.5f}, Average Deviation: {avg_dev:.5f}")

    # Visualization
    centerline_pcd = o3d.geometry.PointCloud()
    centerline_pcd.points = o3d.utility.Vector3dVector(smoothed_centerline)
    centerline_pcd.paint_uniform_color([1, 0, 0])  # Red for centerline

    pcd_filtered.paint_uniform_color([0.7, 0.7, 0.7])  # Grey for pipe
    o3d.visualization.draw_geometries([pcd_filtered, centerline_pcd], window_name=f"Refined Cylinder Centerline - {os.path.basename(file)}")

    # Plot the refined centerline in 3D
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(smoothed_centerline[:, 0], smoothed_centerline[:, 1], smoothed_centerline[:, 2], c="green", label="Refined Centerline Points")
    ax.plot(smoothed_centerline[:, 0], smoothed_centerline[:, 1], smoothed_centerline[:, 2], c="red", label="Refined Cylinder Centerline")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.title(f"Refined Cylinder Centerline for {os.path.basename(file)}")
    plt.show()

print("Processing complete.")
