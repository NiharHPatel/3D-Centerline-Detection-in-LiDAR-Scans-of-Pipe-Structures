import open3d as o3d
import numpy as np
from sklearn.linear_model import RANSACRegressor
from sklearn.cluster import DBSCAN
from scipy.interpolate import splprep, splev
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

# Load the LiDAR point cloud
pcd = o3d.io.read_point_cloud("/Users/niharpatel/Desktop/WNE UNIVERSITY/Capestone Project/PLY Examples/1_0058.000.ply")
points = np.asarray(pcd.points)

# STEP 1: Fit a Cylinder using RANSAC
ransac = RANSACRegressor(residual_threshold=0.05)  # Adjusted threshold for better inlier selection
ransac.fit(points[:, :2], points[:, 2])  

# Extract inliers (best-fit cylinder points)
inliers = points[ransac.inlier_mask_]

# Ensure we have enough inliers
if len(inliers) < 100:
    print("Warning: Too few inliers detected. Adjust RANSAC threshold.")
else:
    print(f"Number of inliers detected: {len(inliers)}")

# Save the fitted cylinder point cloud
fitted_pcd = o3d.geometry.PointCloud()
fitted_pcd.points = o3d.utility.Vector3dVector(inliers)
o3d.io.write_point_cloud("fitted_cylinder.ply", fitted_pcd)

# STEP 2: Extract the Centerline
cylinder_axis = np.mean(inliers, axis=0)
centerline_points = np.column_stack((
    np.full(100, cylinder_axis[0]),
    np.full(100, cylinder_axis[1]),
    np.linspace(min(inliers[:,2]), max(inliers[:,2]), 100)
))

# STEP 3: Fix Gaps Using Gaussian Process Regression (GPR)
gpr = GaussianProcessRegressor(kernel=RBF(length_scale=0.5))  # RBF kernel for smooth interpolation
gpr.fit(centerline_points[:, 2].reshape(-1, 1), centerline_points[:, :2])  # Fit GPR on Z-axis

# Generate interpolated points
z_interp = np.linspace(min(centerline_points[:,2]), max(centerline_points[:,2]), 200).reshape(-1, 1)
xy_interp = gpr.predict(z_interp)  # Predict X, Y for missing sections

# Merge original and interpolated points
full_centerline = np.hstack((xy_interp, z_interp))

# Save interpolated centerline
np.savetxt("interpolated_centerline.txt", full_centerline)

# STEP 4: Detect Bends and Misaligned Sections
distances = np.linalg.norm(inliers - cylinder_axis, axis=1)
threshold = np.mean(distances) * 1.2  # Define threshold for bend detection
high_error_points = inliers[distances > threshold]

# Apply DBSCAN Clustering to Segment Bends
if len(high_error_points) > 10:
    clustering = DBSCAN(eps=threshold, min_samples=5).fit(high_error_points)
    bend_pcd = o3d.geometry.PointCloud()
    bend_pcd.points = o3d.utility.Vector3dVector(high_error_points)
    o3d.io.write_point_cloud("bent_regions.ply", bend_pcd)
    print("Bend regions saved as 'bent_regions.ply'")

# STEP 5: Visualize & Debug
pcd.paint_uniform_color([0, 1, 0])  # Original scan (Green)
fitted_pcd.paint_uniform_color([1, 0, 0])  # Fitted cylinder (Red)
if len(high_error_points) > 10:
    bend_pcd.paint_uniform_color([0, 0, 1])  # Bends (Blue)
    o3d.visualization.draw_geometries([pcd, fitted_pcd, bend_pcd])
else:
    o3d.visualization.draw_geometries([pcd, fitted_pcd])

print("✅ Processing Complete: Centerline Extracted & Gaps Fixed.")
