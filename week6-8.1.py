import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from sklearn.cluster import DBSCAN
from sklearn.linear_model import RANSACRegressor
from pykalman import KalmanFilter  # For smoothing missing data
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
    """ Downsample the point cloud and remove statistical outliers. """
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    return pcd_down

def estimate_normals(pcd):
    """ Compute normal vectors for the point cloud. """
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    return pcd

def fit_cylinder_ransac(pcd, max_trials=1500, distance_threshold=0.03):
    """ Fit a cylinder to the point cloud using RANSAC. """
    points = np.asarray(pcd.points)

    # Use RANSAC to fit a linear model along Z-axis
    model_ransac = RANSACRegressor(residual_threshold=distance_threshold, max_trials=max_trials)
    model_ransac.fit(points[:, :2], points[:, 2])  # Fit X-Y plane vs Z

    z_pred = model_ransac.predict(points[:, :2])
    residuals = np.abs(points[:, 2] - z_pred)

    inlier_mask = residuals < distance_threshold
    cylinder_pcd = pcd.select_by_index(np.where(inlier_mask)[0])

    return cylinder_pcd, inlier_mask

def compute_centerline(pcd):
    """ Compute centerline using the fitted cylinder model. """
    points = np.asarray(pcd.points)

    # DBSCAN with tuned parameters for better clustering
    clustering = DBSCAN(eps=0.3, min_samples=5).fit(points[:, :2])
    labels = clustering.labels_

    unique_labels = np.unique(labels)
    centerline_points = []
    
    for label in unique_labels:
        if label == -1:
            continue  # Ignore noise
        
        cluster_points = points[labels == label]
        centroid = np.mean(cluster_points, axis=0)
        centerline_points.append(centroid)

    centerline_points = np.array(centerline_points)

    # **Check number of points before spline fitting**
    num_points = len(centerline_points)
    print(f"Number of extracted centerline points: {num_points}")

    if num_points < 4:  # Spline needs at least `k+1` points
        print(f"Warning: Not enough points for spline fitting ({num_points} points). Skipping spline.")
        return centerline_points  # Return raw points instead of failing

    # **Fix: Use lower spline order if points are too few**
    spline_order = min(3, num_points - 1)

    # Fit a B-Spline for curved sections
    tck, u = splprep(centerline_points.T, s=1, k=spline_order)
    u_fine = np.linspace(0, 1, 500)
    refined_centerline = np.array(splev(u_fine, tck)).T

    # Apply Kalman Filtering to smooth missing sections
    kf = KalmanFilter(initial_state_mean=refined_centerline[0], n_dim_obs=3)
    smoothed_centerline, _ = kf.smooth(refined_centerline)

    print("Before Kalman Filtering (First 5 Points):", refined_centerline[:5])
    print("After Kalman Filtering (First 5 Points):", smoothed_centerline[:5])

    return smoothed_centerline


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
    cylinder_pcd, inlier_mask = fit_cylinder_ransac(pcd_filtered)

    # Compute refined centerline
    centerline = compute_centerline(cylinder_pcd)

    # Visualization
    centerline_pcd = o3d.geometry.PointCloud()
    centerline_pcd.points = o3d.utility.Vector3dVector(centerline)
    centerline_pcd.paint_uniform_color([1, 0, 0])  # Red for centerline

    pcd_filtered.paint_uniform_color([0.7, 0.7, 0.7])  # Grey for pipe
    o3d.visualization.draw_geometries([pcd_filtered, centerline_pcd], window_name=f"Cylinder Fit - {os.path.basename(file)}")

    # Plot the refined centerline in 3D
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(centerline[:, 0], centerline[:, 1], centerline[:, 2], c="green", label="Refined Centerline Points")
    ax.plot(centerline[:, 0], centerline[:, 1], centerline[:, 2], c="red", label="Fitted Cylinder Centerline")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.title(f"Refined Cylinder Centerline for {os.path.basename(file)}")
    plt.show()

print("Processing complete.")
